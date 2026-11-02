"""ComfyUI executor -- run encode/generate on a remote ComfyUI over HTTP.

The promptless trick only needs two things from a model: turn prompts into
conditioning tensors (mining) and turn conditioning tensors into images
(generation). A stock ComfyUI with the RES4LYF pack does both through its API:

* encode:   ``CheckpointLoaderSimple -> CLIPTextEncode -> ConditioningToBase64``
            (an output node, so the pickled conditioning comes back in /history)
* generate: ``Base64ToConditioning -> KSampler -> VAEDecode -> PreviewImage``
            (the sampled tensor goes in inline as a base64 pickle; the text
            encoder only sees the empty negative prompt)

Mining through Comfy's own encoder keeps the distribution in exactly the space
Comfy's sampler consumes. The HTTP side is stdlib only; torch is imported lazily
for the (un)pickling, because RES4LYF's payload is a pickled torch conditioning.

Enabled by ``SA_COMFY_URL`` (e.g. ``http://127.0.0.1:8188``).
"""

from __future__ import annotations

import base64
import io
import json
import os
import pickle
import signal
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Optional

import numpy as np


# --scheduler (diffusers names) -> (Comfy sampler_name, Comfy scheduler).
# "default" stands in for diffusers' PNDM, which Comfy doesn't have.
SAMPLERS = {
    "default": ("dpmpp_2m", "karras"),
    "ddim": ("ddim", "ddim_uniform"),
    "euler": ("euler", "normal"),
    "euler_a": ("euler_ancestral", "normal"),
    "dpm": ("dpmpp_2m", "karras"),
}

# Prompts per encode graph / images per generate graph. Each SD1.5 conditioning
# is ~0.3 MB of base64, so these keep a request/history payload around 10 MB.
ENCODE_CHUNK = 32
GENERATE_CHUNK = 8


def comfy_url() -> Optional[str]:
    url = os.environ.get("SA_COMFY_URL", "").strip()
    return url.rstrip("/") or None


def escape_prompt(text: str) -> str:
    """Escape Comfy's ``(word:1.2)`` weighting so a prompt encodes literally, the
    way diffusers' tokenizer sees it. (``\\(``/``\\)`` are the only escapes Comfy
    undoes, so nothing else is touched.)"""
    return text.replace("(", "\\(").replace(")", "\\)")


def cond_to_b64(embed: np.ndarray, pooled: Optional[np.ndarray] = None) -> str:
    """One ``(tokens, hidden)`` embedding -> RES4LYF Base64ToConditioning payload."""
    import torch

    t = torch.from_numpy(np.ascontiguousarray(embed, dtype=np.float32))[None]
    opts = {}
    if pooled is not None:
        opts["pooled_output"] = torch.from_numpy(
            np.ascontiguousarray(pooled, dtype=np.float32))[None]
    return base64.b64encode(pickle.dumps([[t, opts]])).decode("ascii")


def _load_storage_bytes(b: bytes):
    """Safe stand-in for ``torch.storage._load_from_bytes``, which calls
    ``torch.load(..., weights_only=False)`` -- i.e. a second, unrestricted
    unpickle hidden inside the first."""
    import torch

    return torch.load(io.BytesIO(b), weights_only=True)


class _CondUnpickler(pickle.Unpickler):
    """Admit only what a pickled torch conditioning needs. Anything else -- the
    payload comes from a remote box -- is refused instead of executed."""

    def find_class(self, module: str, name: str):
        import collections

        import torch

        allowed = {
            ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            ("torch.storage", "_load_from_bytes"): _load_storage_bytes,
            ("collections", "OrderedDict"): collections.OrderedDict,
        }
        try:
            return allowed[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(
                f"comfy: refusing {module}.{name} in a conditioning payload") from None


def b64_to_cond(data: str) -> tuple[np.ndarray, dict]:
    """ConditioningToBase64 output -> ``(embed (tokens, hidden), options)``.

    Unpickled with an allow-list (:class:`_CondUnpickler`): a plain
    ``pickle.loads`` would let any listed box run code on this machine."""
    cond = _CondUnpickler(io.BytesIO(base64.b64decode(data))).load()
    t, opts = cond[0]
    return t[0].float().cpu().numpy(), opts


class ComfyError(RuntimeError):
    pass


class ComfyClient:
    """Minimal ComfyUI API client: submit a graph, wait for it, fetch outputs."""

    def __init__(self, url: str, poll: float = 1.0):
        self.url = url.rstrip("/")
        self.poll = poll
        self.client_id = uuid.uuid4().hex
        self._pending: set[str] = set()
        # The dashboard stops a job with SIGTERM; turn that into an exception so
        # run() can pull our prompt out of the shared queue on the way out.
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    # ---- raw HTTP -----------------------------------------------------------
    def _req(self, path: str, body=None, timeout: float = 120.0) -> bytes:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            self.url + path, data=data,
            headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            raise ComfyError(f"{path}: HTTP {e.code}: {e.read().decode(errors='replace')[:2000]}")

    def get_json(self, path: str):
        return json.loads(self._req(path))

    def checkpoints(self) -> list[str]:
        info = self.get_json("/object_info/CheckpointLoaderSimple")
        return list(info["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0])

    # ---- jobs ---------------------------------------------------------------
    def submit(self, graph: dict) -> str:
        res = json.loads(self._req("/prompt", {"prompt": graph, "client_id": self.client_id}))
        if res.get("node_errors"):
            raise ComfyError(f"graph rejected: {json.dumps(res['node_errors'])[:2000]}")
        pid = res["prompt_id"]
        self._pending.add(pid)
        return pid

    def cancel(self, pid: str) -> None:
        """Drop a still-queued prompt. A prompt already running is left alone:
        /interrupt is global and would kill whatever else that box is rendering."""
        try:
            self._req("/queue", {"delete": [pid]}, timeout=15)
        except Exception:
            pass
        self._pending.discard(pid)

    #: A prompt that is in neither the queue nor the history this long is gone
    #: (the box restarted: its history lives in memory). History is written a
    #: moment after a prompt leaves the queue, so this only needs to outlast that.
    LOST_AFTER = 60.0

    def wait(self, pid: str, label: str = "comfy") -> dict:
        last, gone_since = None, None
        while True:
            hist = self.get_json(f"/history/{pid}")
            if pid in hist:
                entry = hist[pid]
                status = entry.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    detail = msgs[-1][1].get("exception_message") if msgs else status
                    raise ComfyError(f"{label}: execution failed: {detail}")
                if status.get("completed", True):
                    self._pending.discard(pid)
                    return entry["outputs"]
            q = self.get_json("/queue")
            running = [p[1] for p in q.get("queue_running", [])]
            pending = [p[1] for p in q.get("queue_pending", [])]
            if pid in running:
                state = "running"
            elif pid in pending:
                state = f"queued ({len(running) + len(pending)} job(s) on the box)"
            else:
                state = "finishing"  # left the queue, history not written yet
            if state == "finishing":
                gone_since = gone_since or time.monotonic()
                if time.monotonic() - gone_since > self.LOST_AFTER:
                    self._pending.discard(pid)
                    raise ComfyError(f"{label}: prompt {pid[:8]} vanished from the box "
                                     f"(neither queued nor in history for {self.LOST_AFTER:.0f}s"
                                     " -- did ComfyUI restart?)")
            else:
                gone_since = None
            if state != last:
                print(f"[{label}] {pid[:8]} {state}", flush=True)
                last = state
            time.sleep(self.poll)

    def run(self, graph: dict, label: str = "comfy") -> dict:
        pid = self.submit(graph)
        try:
            return self.wait(pid, label)
        except BaseException:
            self.cancel(pid)
            raise

    def fetch_image(self, ref: dict):
        from PIL import Image

        q = urllib.parse.urlencode({"filename": ref["filename"],
                                    "subfolder": ref.get("subfolder", ""),
                                    "type": ref.get("type", "output")})
        img = Image.open(io.BytesIO(self._req(f"/view?{q}")))
        img.load()
        return img


def match_checkpoint(available: list[str], default: str, hint: Optional[str] = None) -> str:
    """Pick the Comfy checkpoint name to load.

    ``hint`` is whatever ``--ckpt`` the caller had -- often a *local* path that
    means nothing to the remote box -- so it is matched by its file name.
    Without a hit, ``default`` (``SA_COMFY_SD15_CKPT``) is used. Separators are
    compared loosely: a Windows box lists ``SD1.5\\x.ckpt`` for ``SD1.5/x.ckpt``.
    """
    by_norm = {c.replace("\\", "/"): c for c in available}
    if hint:
        h = hint.replace("\\", "/")
        base = h.rsplit("/", 1)[-1]
        hits = [c for k, c in by_norm.items() if k == h or k.rsplit("/", 1)[-1] == base]
        if hits:
            return hits[0]
    name = by_norm.get(default.replace("\\", "/"))
    if name is None:
        raise ComfyError(f"{default!r} is not a checkpoint on this ComfyUI; it lists {available}")
    return name


# ---- graph builders ---------------------------------------------------------
def encode_graph(ckpt: str, prompts: list[str]) -> tuple[dict, list[str]]:
    """One loader + (CLIPTextEncode -> ConditioningToBase64) per prompt.
    Returns the graph and the output-node id for each prompt, in order."""
    g = {"ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}}}
    outs = []
    for i, p in enumerate(prompts):
        g[f"enc{i}"] = {"class_type": "CLIPTextEncode",
                        "inputs": {"text": escape_prompt(p), "clip": ["ckpt", 1]}}
        g[f"b64_{i}"] = {"class_type": "ConditioningToBase64",
                         "inputs": {"conditioning": [f"enc{i}", 0]}}
        outs.append(f"b64_{i}")
    return g, outs


def generate_graph(ckpt: str, conds_b64: list[str], seeds: list[int], *,
                   steps: int, cfg: float, sampler: str, scheduler: str,
                   width: int, height: int, negative_b64: Optional[str] = None,
                   negative_text: str = "") -> tuple[dict, list[str]]:
    """One branch per conditioning (own seed), sharing the loader and negative.
    The negative is ``negative_text`` encoded (default: the empty prompt) unless
    ``negative_b64`` is given."""
    g = {
        "ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "latent": {"class_type": "EmptyLatentImage",
                   "inputs": {"width": width, "height": height, "batch_size": 1}},
    }
    if negative_b64 is None:
        g["neg"] = {"class_type": "CLIPTextEncode", "inputs": {"text": escape_prompt(negative_text), "clip": ["ckpt", 1]}}
    else:
        g["neg"] = {"class_type": "Base64ToConditioning", "inputs": {"data": negative_b64}}
    outs = []
    for i, (c, s) in enumerate(zip(conds_b64, seeds)):
        g[f"pos{i}"] = {"class_type": "Base64ToConditioning", "inputs": {"data": c}}
        g[f"ks{i}"] = {"class_type": "KSampler", "inputs": {
            "model": ["ckpt", 0], "seed": int(s), "steps": int(steps), "cfg": float(cfg),
            "sampler_name": sampler, "scheduler": scheduler,
            "positive": [f"pos{i}", 0], "negative": ["neg", 0],
            "latent_image": ["latent", 0], "denoise": 1.0}}
        g[f"dec{i}"] = {"class_type": "VAEDecode",
                        "inputs": {"samples": [f"ks{i}", 0], "vae": ["ckpt", 2]}}
        # PreviewImage writes to Comfy's temp dir -- nothing piles up in the
        # shared box's output folder; we fetch the PNG straight away.
        g[f"img{i}"] = {"class_type": "PreviewImage", "inputs": {"images": [f"dec{i}", 0]}}
        outs.append(f"img{i}")
    return g, outs
