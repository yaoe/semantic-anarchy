#!/usr/bin/env python3
"""Semantic Anarchy — explorer dashboard.

A self-contained FastAPI app that drives the project's *existing* CLI scripts
(mine_distribution / generate / temperature_sweep / sampler_sweep) as
subprocesses and surfaces them in the browser:

* Generate controls — pick backend/model/sampler/temperature/n/seed/... and run.
* Sweep grids       — temperature & sampler contact sheets.
* Mine              — (re)build a backend's conditioning distribution.
* Live job log      — streamed stdout of the running script, with a job history.
* Gallery           — every artifact in ``outputs/`` (generated images, sweep
                      sheets, mined marginal plots), newest first.

One GPU => one job at a time: submissions are queued and run by a single worker
thread. The browser polls for state; no websockets needed.

Run via ``webui/run.sh`` (binds to the Tailscale IP). Nothing here imports
torch/diffusers — the heavy work happens in the subprocesses, which use the
venv python passed in ``SA_PYTHON``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import queue
import shlex
import struct
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# --------------------------------------------------------------------------- #
# Paths / config
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parent.parent
OUTPUTS = REPO / "outputs"

# The one thing this file borrows from the package: the naming rules that map a
# distribution's *base prefix* onto the .npz files it actually lives in, so the
# dashboard can tell "already encoded" from "needs encoding" without guessing at
# filenames the CLIs own. Torch-free (numpy only) — the rule against importing
# the heavy tier here still holds.
sys.path.insert(0, str(REPO))
from semantic_anarchy import dist_paths  # noqa: E402
# Also torch-free: the labels dataset + experiment identity. The record schema,
# the score range and the tail-weighted summary all live there so the dashboard,
# scripts/experiment_report.py and the tests can't disagree about what a label is.
from semantic_anarchy import labels as labelset  # noqa: E402
# Torch-free too: fitting a distribution to the latents of a picked set of
# images is pure numpy over sidecars that already exist, so the naming rules and
# the manifest format are shared with scripts/fit_selection.py rather than
# re-implemented here.
from semantic_anarchy import selection_fit as fitset  # noqa: E402
# Likewise torch-free: the house sd15 CFG negative, so the sidebar shows the
# real text (and any SA_SD15_NEGATIVE override) instead of a copy that drifts.
from semantic_anarchy.pipeline import default_sd15_negative  # noqa: E402
# The image format the renderers write (JPEG by default) plus the full set the
# gallery must still READ: everything mined before the switch is a PNG, and a
# suffix check that only knows the current format would hide it.
from semantic_anarchy.io_utils import IMAGE_EXTS, find_image  # noqa: E402

SD15_NEGATIVE = default_sd15_negative() or ""
# Folder of "good init images" -- when init injection is on, each generation
# starts img2img from a RANDOM one (entropy injection). Drop images in here.
INIT_DIR = Path(os.path.expanduser(os.environ.get("SA_INIT_DIR", str(REPO / "init_images"))))
INIT_EXTS = IMAGE_EXTS + (".bmp",)
FAVS_FILE = OUTPUTS / "favorites.json"   # persisted list of favorited image rel-paths

# rel-path -> (mtime, parsed .json sidecar). The gallery poll and the labeling
# queue both read a few fields out of every image's sidecar, and there are tens
# of thousands of them; this is what keeps that from being tens of thousands of
# json parses per request. Keyed on mtime so a rewritten sidecar is picked up.
_SIDECAR_CACHE: dict = {}


def sidecar_for(rel: str) -> dict:
    """The ``.json`` sidecar of an outputs-relative image, or ``{}``."""
    j = OUTPUTS / Path(rel).with_suffix(".json")
    try:
        mtime = j.stat().st_mtime
    except OSError:
        return {}
    hit = _SIDECAR_CACHE.get(rel)
    if hit is not None and hit[0] == mtime:
        return hit[1]
    try:
        data = json.loads(j.read_text())
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}
    _SIDECAR_CACHE[rel] = (mtime, data)
    return data


def _distance_for(rel: str) -> Optional[float]:
    return sidecar_for(rel).get("distance")


def init_images_count() -> int:
    """Total init images anywhere under INIT_DIR (recursive = the 'any folder' pool)."""
    if INIT_DIR.is_dir():
        return sum(1 for p in INIT_DIR.rglob("*") if p.suffix.lower() in INIT_EXTS)
    return 0


def init_folders() -> list:
    """Subfolders of INIT_DIR that contain images, with per-folder counts."""
    out = []
    if INIT_DIR.is_dir():
        # images sitting directly in the root (not in a subfolder)
        root_n = sum(1 for p in INIT_DIR.iterdir()
                     if p.is_file() and p.suffix.lower() in INIT_EXTS)
        if root_n:
            out.append({"name": "(root)", "path": "", "count": root_n})
        for d in sorted(p for p in INIT_DIR.iterdir() if p.is_dir()):
            n = sum(1 for p in d.rglob("*") if p.suffix.lower() in INIT_EXTS)
            if n:
                out.append({"name": d.name, "path": d.name, "count": n})
    return out


def resolve_init_dir(folder: Optional[str]) -> Path:
    """Map a requested folder name to a safe path under INIT_DIR.

    ``None``/``""``/``"__any__"`` -> INIT_DIR root (recursive = pick from any
    folder). Otherwise the named subfolder, sandboxed against traversal.
    """
    base = INIT_DIR.resolve()
    if not folder or folder in ("__any__", "(root)"):
        return base
    cand = (INIT_DIR / folder).resolve()
    if base not in cand.parents and cand != base:
        raise HTTPException(400, f"init folder outside init_images/: {folder}")
    if not cand.is_dir():
        raise HTTPException(400, f"init folder not found: {folder}")
    return cand
_favs_lock = threading.Lock()


def load_favs() -> set:
    try:
        return set(json.loads(FAVS_FILE.read_text()))
    except Exception:
        return set()


def save_favs(favs: set) -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    FAVS_FILE.write_text(json.dumps(sorted(favs), indent=0))
# Interpreter used to run the heavy scripts (a venv with torch+diffusers).
PYTHON = os.environ.get("SA_PYTHON") or os.sys.executable
# Flow-model jobs (flux2 / krea2) run in the dedicated flux venv (diffusers>=0.39).
FLUX_PYTHON = os.environ.get("SA_FLUX_PYTHON", str(REPO / ".venv-flux" / "bin" / "python"))


def python_for(backend: str) -> str:
    return FLUX_PYTHON if backend in ("flux2", "krea2") else PYTHON
# SD1.5 single-file checkpoint (downloaded per the README).
SD15_CKPT = os.path.expanduser(
    os.environ.get("SA_SD15_CKPT", "~/models/v1-5-pruned-emaonly.safetensors")
)
# SD 2.1 (768, v-pred) single-file checkpoint (mirror), also via --ckpt.
SD2_CKPT = os.path.expanduser(
    os.environ.get("SA_SD2_CKPT", "~/models/v2-1_768-ema-pruned.safetensors")
)
SINGLE_FILE_CKPT = {"sd15": SD15_CKPT, "sd2": SD2_CKPT}
# SDXL repos already cached in ~/.cache/huggingface (HF ids). base-1.0 is the
# DEFAULT — it takes CFG and actually expresses the sampled drift. turbo is a
# 1-step/no-CFG distilled model that mode-collapses to generic output, so it's
# a poor fit for this method and only offered as a fast preview.
SDXL_MODELS = {
    "sdxl-base-1.0": "stabilityai/stable-diffusion-xl-base-1.0",
    "sdxl-turbo": "stabilityai/sdxl-turbo",
}
SDXL_DEFAULT_MODEL = "sdxl-base-1.0"
# Per-model generation defaults applied when the user leaves steps/guidance
# blank. base needs real CFG + steps; turbo is 1-step/no-CFG by construction.
SDXL_MODEL_DEFAULTS = {
    "sdxl-base-1.0": {"steps": 30, "guidance": 7.0},
    "sdxl-turbo": {"steps": 1, "guidance": 0.0},
}

FLUX2_MODEL = os.environ.get("SA_FLUX2_MODEL", "black-forest-labs/FLUX.2-klein-4B")
KREA2_MODEL = os.environ.get("SA_KREA2_MODEL", "krea/Krea-2-Raw")

# --------------------------------------------------------------------------- #
# Hand-picked checkpoints (webui/model_config.json)
# --------------------------------------------------------------------------- #
# The env-var defaults above are the *fallback*. The model picker in the sidebar
# writes a per-backend absolute path here, and that wins whenever it is set —
# so the choice survives a restart without anyone editing run.sh. The file is
# machine-specific (everyone's weights live somewhere else), hence gitignored.
MODEL_CONFIG_FILE = Path(
    os.environ.get("SA_MODEL_CONFIG", str(REPO / "webui" / "model_config.json"))
)
_model_cfg_lock = threading.Lock()

CKPT_EXTS = (".safetensors", ".ckpt")
# What makes a folder a diffusers model rather than just a folder of files.
DIFFUSERS_MARKERS = ("model_index.json", "unet", "transformer")

# Per-backend fallback when nothing is hand-picked (env vars / cached HF ids).
MODEL_DEFAULTS = {
    "sd15": SD15_CKPT,
    "sd2": SD2_CKPT,
    "sdxl": SDXL_MODELS[SDXL_DEFAULT_MODEL],
    "flux2": FLUX2_MODEL,
    "krea2": KREA2_MODEL,
}


def load_model_config() -> dict:
    """``{backend: path}`` of hand-picked checkpoints (missing file -> ``{}``)."""
    try:
        raw = json.loads(MODEL_CONFIG_FILE.read_text())
    except Exception:
        return {}
    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, dict):
        return {}
    return {k: v for k, v in models.items() if k in BACKENDS and isinstance(v, str) and v}


def save_model_config(models: dict) -> None:
    MODEL_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODEL_CONFIG_FILE.write_text(json.dumps({"models": models}, indent=2) + "\n")


def path_kind(p: Path) -> Optional[str]:
    """``"ckpt"`` for a single-file checkpoint, ``"diffusers"`` for a model folder."""
    if p.is_file():
        return "ckpt" if p.suffix.lower() in CKPT_EXTS else None
    if p.is_dir():
        return "diffusers" if any((p / m).exists() for m in DIFFUSERS_MARKERS) else None
    return None


def validate_model_path(path: str) -> Path:
    """Resolve + sanity-check a hand-picked model path, or 400."""
    p = Path(os.path.expanduser(path.strip())).resolve()
    if not p.exists():
        raise HTTPException(400, f"path does not exist: {p}")
    kind = path_kind(p)
    if kind is None:
        if p.is_file():
            raise HTTPException(400, f"not a checkpoint (.safetensors/.ckpt): {p.name}")
        raise HTTPException(
            400, f"not a diffusers model folder (no model_index.json): {p}"
        )
    return p


def model_flags_for_path(path: str) -> list[str]:
    """A diffusers *folder* loads via from_pretrained (--model); a single-file
    .ckpt/.safetensors via from_single_file (--ckpt)."""
    return ["--model", path] if Path(path).is_dir() else ["--ckpt", path]


def selected_model(backend: str) -> Optional[str]:
    """The hand-picked path for a backend, if one is configured."""
    return load_model_config().get(backend)


def effective_model(backend: str, model_key: Optional[str] = None) -> str:
    """The checkpoint this backend will actually load.

    Sidebar pick > the sdxl repo dropdown > the env-var / HF default. This is
    the string a mined corpus is tagged with, so switching checkpoints switches
    which fit of that corpus is in play (rather than silently reusing the wrong
    one).
    """
    picked = selected_model(backend)
    if picked:
        return picked
    if backend == "sdxl" and model_key in SDXL_MODELS:
        return SDXL_MODELS[model_key]
    return MODEL_DEFAULTS[backend]


# --------------------------------------------------------------------------- #
# Base distribution (webui/dist_config.json)
# --------------------------------------------------------------------------- #
# Which fit `--dist` points at. Two built-ins (the repo's own mined corpus and
# the evolved ★ branch) plus two "point at a file on disk" kinds:
#
#   prompts — a .txt corpus. Its latents live BESIDE the .txt, tagged with the
#             checkpoint that encoded them, so one corpus can hold a separate
#             fit per model and switching checkpoints switches fits.
#   file    — an .npz someone already mined/evolved, picked directly.
#
# Persisted per backend so a restart comes back to the same distribution.
DIST_CONFIG_FILE = Path(
    os.environ.get("SA_DIST_CONFIG", str(REPO / "webui" / "dist_config.json"))
)
_dist_cfg_lock = threading.Lock()

BASE_DIST = "outputs/dist"              # the repo's mined corpus (prompts_1000)
EVOLVED_DIST = "outputs/dist_evolved"   # scripts/evolve_favorites.py's branch
DEFAULT_PROMPTS = "prompts_1000.txt"
DIST_KINDS = ("base", "evolved", "prompts", "file")
DIST_EXTS = (dist_paths.PROMPTS_EXT, ".npz")
DIST_LABELS = {"base": "base corpus", "evolved": "evolved ★ branch"}


def load_dist_config() -> dict:
    """``{backend: {"kind": ..., "path": ...}}`` (missing/garbled file -> ``{}``)."""
    try:
        raw = json.loads(DIST_CONFIG_FILE.read_text())
    except Exception:
        return {}
    dists = raw.get("dists") if isinstance(raw, dict) else None
    if not isinstance(dists, dict):
        return {}
    out = {}
    for b, row in dists.items():
        if b in BACKENDS and isinstance(row, dict) and row.get("kind") in DIST_KINDS:
            path = row.get("path")
            out[b] = {"kind": row["kind"], "path": path if isinstance(path, str) else None}
    return out


def save_dist_config(dists: dict) -> None:
    DIST_CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    DIST_CONFIG_FILE.write_text(json.dumps({"dists": dists}, indent=2) + "\n")


def resolve_dist_file(path: str, kind: Optional[str] = None) -> Path:
    """Sandbox + sanity-check a hand-picked corpus / distribution file.

    Same roots as the model browser: user input only ever reaches argv after
    passing through here.
    """
    p = Path(os.path.expanduser(str(path).strip())).resolve()
    if not any(p == r or r in p.parents for r in browse_roots()):
        raise HTTPException(403, f"outside the browsable roots: {p}")
    suffix = p.suffix.lower()
    want = {"prompts": (dist_paths.PROMPTS_EXT,), "file": (".npz",)}.get(kind, DIST_EXTS)
    if suffix not in want:
        raise HTTPException(
            400, f"expected {' or '.join(want)}, got {p.name}"
        )
    if not p.is_file():
        raise HTTPException(400, f"file not found: {p}")
    return p


def dist_kind_for(path: str) -> str:
    """Which kind a picked file is, by extension."""
    return "prompts" if Path(path).suffix.lower() == dist_paths.PROMPTS_EXT else "file"


def _abs_base(base: str) -> str:
    """Bases are repo-relative on the command line (jobs run with cwd=REPO);
    resolve them here so existence checks don't depend on the server's cwd."""
    p = Path(base)
    return str(p if p.is_absolute() else REPO / p)


def dist_base_for(backend: str, kind: str, path: Optional[str],
                  model_key: Optional[str] = None) -> str:
    """The ``--dist`` / ``--out`` base a (kind, path) choice resolves to."""
    if kind == "base":
        return BASE_DIST
    if kind == "evolved":
        return EVOLVED_DIST
    if not path:
        raise HTTPException(400, f"{kind}: no file selected")
    if kind == "prompts":
        return dist_paths.prompt_dist_base(path, effective_model(backend, model_key))
    return dist_paths.base_from_npz(path, backend)


def describe_dist(backend: str, kind: str, path: Optional[str],
                  model_key: Optional[str] = None) -> dict:
    """One distribution choice, told everything the picker needs to render it:
    where its files would be, whether they're there, and what encoded them."""
    if kind not in DIST_KINDS:
        raise HTTPException(400, f"unknown distribution kind {kind!r}")
    model = effective_model(backend, model_key)
    base = dist_base_for(backend, kind, path, model_key)
    files = dist_paths.dist_files(_abs_base(base), backend)
    ready = all(f.is_file() for f in files)
    # ".npz" in the sidebar's one-line label is noise — the name is the identity.
    label = DIST_LABELS.get(kind) or Path(path or base).name.removesuffix(".npz")
    # A fit made from picked images carries a manifest naming them. It is what
    # tells the picker to say "fitted from 42 images" rather than "42 prompts",
    # and to keep quiet about corpus-only corrections it was never going to have.
    man = fitset.read_manifest(_abs_base(base))
    return {
        "backend": backend,
        "kind": kind,
        "path": str(path) if path else None,
        "base": base,
        "label": label,
        "ready": ready,
        "files": [{"path": str(f), "exists": f.is_file()} for f in files],
        "meta": dist_paths.dist_meta(_abs_base(base), backend) if ready else None,
        "fit": {k: man.get(k) for k in
                ("name", "n_samples", "note", "created", "models")} if man else None,
        "model": {
            "path": model,
            "name": Path(model).name if Path(model).is_absolute() else model,
            "slug": dist_paths.model_slug(model),
        },
    }


def current_dist(backend: str, model_key: Optional[str] = None,
                 cfg: Optional[dict] = None) -> dict:
    """The persisted choice for a backend (the base corpus when nothing is set)."""
    cfg = load_dist_config() if cfg is None else cfg
    row = cfg.get(backend) or {"kind": "base", "path": None}
    try:
        return describe_dist(backend, row["kind"], row.get("path"), model_key)
    except HTTPException:
        # A selection whose file vanished shouldn't brick the dashboard — fall
        # back to the base corpus and let the picker show the choice again.
        return describe_dist(backend, "base", None, model_key)


def resolve_dist_base(backend: str, requested: str = "base",
                      model_key: Optional[str] = None) -> str:
    """Which distribution a run samples, as a ``--dist`` base.

    ``requested == "evolved"`` is the legacy dropdown's switch (the /legacy page
    still sends it); everything else follows the persisted picker choice.
    """
    if requested == "evolved":
        row = describe_dist(backend, "evolved", None, model_key)
        if not row["ready"]:
            raise HTTPException(
                400, f"no evolved branch for {backend} yet — run 🧪 Evolve ★ first"
            )
        return row["base"]
    row = current_dist(backend, model_key)
    if not row["ready"]:
        missing = ", ".join(f["path"] for f in row["files"] if not f["exists"])
        raise HTTPException(
            400,
            f"“{row['label']}” is not encoded for {row['model']['name']} yet "
            f"(missing {missing}) — open Base distribution in the sidebar and "
            f"hit “Encode prompt distribution”",
        )
    return row["base"]


# --------------------------------------------------------------------------- #
# Filesystem browsing (fallback picker) + the native OS dialog
# --------------------------------------------------------------------------- #
def browse_roots() -> list:
    """Roots the in-browser file browser may walk. Extra ones via SA_MODEL_ROOTS
    (``:``-separated) — the native dialog is not restricted this way, since that
    one is driven by whoever is sitting at the machine."""
    cands = [Path.home(), REPO, Path("/mnt"), Path("/media"), Path("/opt")]
    for extra in os.environ.get("SA_MODEL_ROOTS", "").split(os.pathsep):
        if extra.strip():
            cands.append(Path(os.path.expanduser(extra.strip())))
    out, seen = [], set()
    for c in cands:
        try:
            r = c.resolve()
        except OSError:
            continue
        if r.is_dir() and str(r) not in seen:
            seen.add(str(r))
            out.append(r)
    return out


def resolve_browse_path(path: Optional[str]) -> Path:
    """Sandbox a browse request to ``browse_roots()``."""
    roots = browse_roots()
    if not path:
        return roots[0]
    p = Path(os.path.expanduser(path)).resolve()
    for r in roots:
        if p == r or r in p.parents:
            break
    else:
        raise HTTPException(403, f"outside the browsable roots: {p}")
    if not p.is_dir():
        raise HTTPException(400, f"not a directory: {p}")
    return p


def _display_env() -> dict:
    """Environment for the native dialog. Falls back to SA_DISPLAY when the
    server was started from a headless shell but a desktop session exists."""
    env = dict(os.environ)
    if not env.get("DISPLAY") and not env.get("WAYLAND_DISPLAY"):
        if os.environ.get("SA_DISPLAY"):
            env["DISPLAY"] = os.environ["SA_DISPLAY"]
    return env


def native_picker_tool() -> Optional[str]:
    """Which OS file dialog we can drive on *this host*, or None."""
    import shutil
    import sys as _sys

    if _sys.platform == "darwin":
        return "osascript" if shutil.which("osascript") else None
    env = _display_env()
    if not (env.get("DISPLAY") or env.get("WAYLAND_DISPLAY")):
        return None
    for tool in ("zenity", "kdialog"):
        if shutil.which(tool):
            return tool
    return None


# One dialog at a time: a second one would sit invisible behind the first.
_picker_lock = threading.Lock()
PICKER_TIMEOUT = 300.0


def run_native_picker(mode: str, start: Optional[str]) -> Optional[str]:
    """Pop a real OS file dialog on the server's desktop; return the picked path
    (None = the user cancelled). ``mode`` is ``"file"`` or ``"folder"``."""
    tool = native_picker_tool()
    if tool is None:
        raise HTTPException(
            409, "no OS file dialog available on the server host "
                 "(no DISPLAY, or zenity/kdialog not installed)"
        )
    if not _picker_lock.acquire(blocking=False):
        raise HTTPException(409, "a file dialog is already open on the server")
    try:
        # `start` only ever reaches argv as a directory path we resolved
        # ourselves, and every call is a list (no shell).
        try:
            start_dir = str(resolve_browse_path(start)) if start else str(Path.home())
        except HTTPException:
            start_dir = str(Path.home())
        title = "Pick a checkpoint folder" if mode == "folder" else "Pick a .safetensors checkpoint"
        if tool == "zenity":
            argv = ["zenity", "--file-selection", f"--title={title}",
                    f"--filename={start_dir}/"]
            if mode == "folder":
                argv.append("--directory")
            else:
                argv += ["--file-filter=Checkpoints | *.safetensors *.ckpt",
                         "--file-filter=All files | *"]
        elif tool == "kdialog":
            argv = (["kdialog", "--title", title, "--getexistingdirectory", start_dir]
                    if mode == "folder"
                    else ["kdialog", "--title", title, "--getopenfilename",
                          start_dir, "*.safetensors *.ckpt|Checkpoints"])
        else:  # osascript (macOS)
            kind = "folder" if mode == "folder" else "file"
            argv = ["osascript", "-e",
                    f'POSIX path of (choose {kind} with prompt "{title}")']
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=PICKER_TIMEOUT, env=_display_env())
        except subprocess.TimeoutExpired:
            raise HTTPException(504, "the file dialog timed out (nobody picked anything)")
        except OSError as exc:
            raise HTTPException(500, f"could not launch {tool}: {exc}")
        out = proc.stdout.strip()
        if proc.returncode != 0 or not out:
            return None                       # cancelled / dismissed
        return out.splitlines()[0].strip()
    finally:
        _picker_lock.release()


# Allow-lists so user input can never become an arbitrary command.
BACKENDS = {"sd15", "sd2", "sdxl", "flux2", "krea2"}
SAMPLERS = {"diagonal", "pca", "blend", "hybrid", "split"}
LENGTH_MODES = {"off", "corpus", "fixed"}
NEG_MODES = {"text", "mean", "empty", "zeros"}
SCHEDULERS = {"default", "ddim", "euler", "euler_a", "dpm"}
INTERPS = {"lanczos", "bicubic", "bilinear", "nearest"}    # hires upscale resampler

# Gallery buckets keyed by filename prefix.
GALLERY_BUCKETS = [
    ("generated", "anarchy_"),
    ("temperature", "temperature_sweep"),
    ("sampler", "sampler_sweep"),
    ("marginals", "mined_marginals"),
]


# --------------------------------------------------------------------------- #
# Job model + single-worker queue
# --------------------------------------------------------------------------- #
@dataclass
class Job:
    id: int
    action: str
    label: str
    argv: list[str]
    status: str = "queued"           # queued | running | done | error | cancelled
    rc: Optional[int] = None
    started: Optional[float] = None
    ended: Optional[float] = None
    log: list[str] = field(default_factory=list)
    cancel_requested: bool = False   # set by cancel(); decides the final status
    _proc: Optional[subprocess.Popen] = None


class Runner:
    """Serialises jobs onto one worker thread (one GPU)."""

    def __init__(self) -> None:
        self._q: "queue.Queue[Job]" = queue.Queue()
        self._jobs: dict[int, Job] = {}
        self._seq = 0
        self._lock = threading.Lock()
        self._current: Optional[Job] = None
        threading.Thread(target=self._loop, daemon=True).start()

    def submit(self, action: str, label: str, argv: list[str]) -> Job:
        with self._lock:
            self._seq += 1
            job = Job(id=self._seq, action=action, label=label, argv=argv)
            self._jobs[job.id] = job
        self._q.put(job)
        return job

    def get(self, job_id: int) -> Optional[Job]:
        return self._jobs.get(job_id)

    def cancel(self, job_id: int) -> bool:
        """Stop one job. Only ever touches that job's *child process* — the
        server, this worker thread and the rest of the queue keep running."""
        job = self._jobs.get(job_id)
        if not job:
            return False
        if job.status == "queued":
            job.status = "cancelled"
            job.log.append("[webui] cancelled before it started")
            return True
        proc = job._proc
        if job.status == "running" and proc:
            job.cancel_requested = True
            job.log.append("[webui] cancel requested — SIGTERM")
            proc.terminate()
            threading.Thread(target=self._reap, args=(job, proc), daemon=True).start()
            return True
        return False

    @staticmethod
    def _reap(job: Job, proc: subprocess.Popen, grace: float = 15.0) -> None:
        """SIGTERM is only a request, and a process inside a CUDA call can take
        a while to notice. Escalate once, on this job's pid alone."""
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            job.log.append(f"[webui] still alive after {grace:.0f}s — SIGKILL")
            proc.kill()

    def snapshot(self) -> dict:
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.id, reverse=True)
        return {
            "running": self._current.id if self._current else None,
            "jobs": [
                {
                    "id": j.id,
                    "action": j.action,
                    "label": j.label,
                    "status": j.status,
                    "rc": j.rc,
                    "started": j.started,
                    "ended": j.ended,
                    "lines": len(j.log),
                    "cmd": " ".join(shlex.quote(a) for a in j.argv),
                }
                for j in jobs
            ],
        }

    def _loop(self) -> None:
        while True:
            job = self._q.get()
            if job.status == "cancelled":
                continue
            self._current = job
            job.status = "running"
            job.started = time.time()
            env = dict(os.environ, PYTHONUNBUFFERED="1")
            try:
                job._proc = subprocess.Popen(
                    job.argv, cwd=str(REPO), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
                for line in job._proc.stdout:          # live stream
                    job.log.append(line.rstrip("\n"))
                    if len(job.log) > 5000:
                        del job.log[:1000]
                job.rc = job._proc.wait()
                # A cancelled job exits non-zero (rc -15 / -9); that is not an
                # error, so don't paint the queue red for it.
                if job.cancel_requested:
                    job.status = "cancelled"
                else:
                    job.status = "done" if job.rc == 0 else "error"
            except Exception as exc:                    # noqa: BLE001
                job.log.append(f"[webui] launch failed: {exc!r}")
                job.status = "error"
                job.rc = -1
            finally:
                job.ended = time.time()
                job._proc = None
                self._current = None


RUNNER = Runner()


# --------------------------------------------------------------------------- #
# Build argv for each action from validated params
# --------------------------------------------------------------------------- #
class RunRequest(BaseModel):
    action: str                       # generate | temp_sweep | sampler_sweep | mine
    backend: str = "sd15"
    model: Optional[str] = None       # sdxl model key (SDXL_MODELS)
    sampler: str = "diagonal"
    temperature: Optional[float] = None
    n: Optional[int] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    guidance: Optional[float] = None
    coherence: Optional[float] = None
    components: Optional[int] = None
    truncation: Optional[float] = None
    neg_mode: Optional[str] = None
    negative: Optional[str] = None    # sd15/sd2 CFG negative TEXT (None = the
                                      # house default; "" = no negative text)
    temps: Optional[str] = None       # sweep: "0.5,1.0,1.5"
    seeds: Optional[str] = None       # sweep: "0,1,2"
    scheduler: Optional[str] = None   # default | ddim | euler | euler_a | dpm
    width: Optional[int] = None
    height: Optional[int] = None
    comp_lo: Optional[int] = None     # pca: first axis (skip dominant/standard ones)
    equalize: bool = False            # pca: express selected axes at equal strength
    dist: str = "base"                # "evolved" forces the ★ branch (the legacy
                                      # page's switch); anything else follows the
                                      # picker's persisted choice for this backend
    target_distance: Optional[float] = None  # shell sampling: pin the distance gauge
    min_distance: Optional[float] = None      # floor: never below this distance
    init: bool = False                # start from a random good init image
    init_mode: str = "img2img"        # img2img (latent) | embedding (IP-Adapter)
    init_strength: float = 0.7        # img2img denoise from the init (0.6-0.8)
    ip_scale: float = 0.7             # IP-Adapter image-embedding strength
    init_folder: Optional[str] = None # which subfolder to pick from ("" / "__any__" = any)
    experiment: Optional[str] = None  # tag every image of this batch (E01-length)
    hypothesis: Optional[str] = None  # one falsifiable sentence -> the manifest
    # The corpus-autopsy corrections. All default to the historical behaviour;
    # see semantic_anarchy/distribution.py for what each one measured.
    rho: Optional[float] = None       # diagonal row coherence in [0,1]
    length_mode: Optional[str] = None # off | corpus | fixed (content/pad split)
    length: Optional[int] = None      # tokens, for length_mode=fixed
    empirical_head: Optional[int] = None  # pca: leading axes drawn from the corpus CDF
    temp_on: Optional[float] = None   # sampler=split: on-manifold temperature
    temp_off: Optional[float] = None  # sampler=split: off-manifold temperature
    radius_band: bool = False         # per-sample target radius from the corpus band
    radius_scale: Optional[float] = None  # shift that band outward


def model_flags(backend: str, model_key: Optional[str] = None) -> list[str]:
    """--ckpt for single-file backends (sd15/sd2), --model for sdxl (cached HF id).

    A checkpoint hand-picked in the sidebar (webui/model_config.json) overrides
    all of that for its backend.
    """
    picked = selected_model(backend)
    if picked:
        if not Path(picked).exists():
            raise HTTPException(
                400, f"{backend}: selected checkpoint no longer exists: {picked}"
            )
        return model_flags_for_path(picked)
    if backend in SINGLE_FILE_CKPT:
        ckpt = SINGLE_FILE_CKPT[backend]
        p = Path(ckpt)
        if not p.exists():
            raise HTTPException(400, f"{backend} checkpoint not found: {ckpt}")
        # A diffusers *folder* loads via from_pretrained (--model); a single-file
        # .ckpt/.safetensors loads via from_single_file (--ckpt).
        return ["--model", ckpt] if p.is_dir() else ["--ckpt", ckpt]
    if backend == "flux2":
        return ["--model", os.environ.get("SA_FLUX2_MODEL",
                                          "black-forest-labs/FLUX.2-klein-4B")]
    if backend == "krea2":
        return ["--model", os.environ.get("SA_KREA2_MODEL", "krea/Krea-2-Raw")]
    key = model_key or SDXL_DEFAULT_MODEL
    if key not in SDXL_MODELS:
        raise HTTPException(400, f"unknown sdxl model {key!r}")
    return ["--model", SDXL_MODELS[key]]


def _model_flags(req: RunRequest) -> list[str]:
    return model_flags(req.backend, req.model)


def _common_sampler_flags(req: RunRequest) -> list[str]:
    a: list[str] = ["--sampler", req.sampler]
    if req.coherence is not None:
        a += ["--coherence", str(req.coherence)]
    if req.components is not None:
        a += ["--components", str(req.components)]
    if req.truncation is not None:
        a += ["--truncation", str(req.truncation)]
    if req.comp_lo:
        a += ["--comp-lo", str(int(req.comp_lo))]
    if req.equalize:
        a += ["--equalize"]
    # ---- the corpus-autopsy knobs ----------------------------------------
    # Each is omitted when it would be a no-op, so an untouched sidebar still
    # produces the exact argv it did before they existed.
    if req.rho:
        a += ["--rho", str(float(req.rho))]
    if req.length_mode and req.length_mode != "off":
        a += ["--length-mode", req.length_mode]
        if req.length_mode == "fixed" and req.length is not None:
            a += ["--length", str(int(req.length))]
    if req.empirical_head:
        a += ["--empirical-head", str(int(req.empirical_head))]
    if req.sampler == "split":
        if req.temp_on is not None:
            a += ["--temp-on", str(float(req.temp_on))]
        if req.temp_off is not None:
            a += ["--temp-off", str(float(req.temp_off))]
    return a


def _gen_flags(req: RunRequest) -> list[str]:
    # For SDXL, never let steps/guidance fall through to the script's turbo
    # family defaults (1 step / no CFG) — backfill from the chosen model so base
    # always runs with real CFG. sd15 keeps the script's own defaults.
    steps, guidance = req.steps, req.guidance
    if req.backend == "sdxl":
        d = SDXL_MODEL_DEFAULTS.get(req.model or SDXL_DEFAULT_MODEL, {})
        if steps is None:
            steps = d.get("steps")
        if guidance is None:
            guidance = d.get("guidance")
    a: list[str] = []
    if steps is not None:
        a += ["--steps", str(steps)]
    if guidance is not None:
        a += ["--guidance", str(guidance)]
    if req.neg_mode:
        a += ["--neg-mode", req.neg_mode]
    # None = never sent (the script keeps the house default); "" = an explicit
    # "no negative text", which the script turns back into the empty prompt.
    if req.negative is not None:
        a += ["--negative", _clean_prompt(req.negative)]
    if req.scheduler and req.scheduler != "default":
        a += ["--scheduler", req.scheduler]
    return a


#: Prompts per text-encoder forward pass. Mining loads no UNet, so this is the
#: only thing sizing the encode; 8 is comfortable everywhere including the
#: cpu-offloaded flow models.
ENCODE_BATCH = 8


def mine_argv(backend: str, prompts: str, out: str,
              components: Optional[int] = None,
              model_key: Optional[str] = None,
              batch_size: Optional[int] = None) -> list[str]:
    """The encode pass: corpus .txt -> fitted distribution at ``out``."""
    batch = ENCODE_BATCH if batch_size is None else max(1, min(64, int(batch_size)))
    argv = ["scripts/mine_distribution.py", "--backend", backend,
            *model_flags(backend, model_key),
            "--prompts", str(prompts), "--out", str(out),
            "--batch-size", str(batch)]
    comps = components
    if comps is None and backend in ("flux2", "krea2"):
        comps = 256   # Qwen embeddings are huge; full-rank PCA would be GBs
    if comps is not None:
        argv += ["--components", str(int(comps))]
    return argv


def build_argv(req: RunRequest) -> tuple[str, list[str]]:
    if req.backend not in BACKENDS:
        raise HTTPException(400, f"bad backend {req.backend!r}")
    if req.sampler not in SAMPLERS:
        raise HTTPException(400, f"bad sampler {req.sampler!r}")
    if req.neg_mode and req.neg_mode not in NEG_MODES:
        raise HTTPException(400, f"bad neg_mode {req.neg_mode!r}")
    if req.length_mode and req.length_mode not in LENGTH_MODES:
        raise HTTPException(400, f"bad length_mode {req.length_mode!r}")
    if req.length_mode == "fixed" and req.length is None:
        raise HTTPException(400, "length_mode 'fixed' needs a length")
    if req.scheduler and req.scheduler not in SCHEDULERS:
        raise HTTPException(400, f"bad scheduler {req.scheduler!r}")

    base = [python_for(req.backend), "-u"]
    model = _model_flags(req)

    if req.action == "mine":
        # Mine re-encodes whatever corpus is selected — the sidebar's Mine
        # button and the picker's "Encode prompt distribution" write the same
        # files. Only a non-corpus selection falls back to the repo's own
        # prompts_1000.txt. (Resolved BEFORE the readiness check below: mining
        # is what makes an unencoded corpus ready.)
        row = current_dist(req.backend, req.model)
        prompts, out = (
            (row["path"], row["base"]) if row["kind"] == "prompts" and row["path"]
            else (DEFAULT_PROMPTS, BASE_DIST)
        )
        argv = base + mine_argv(req.backend, prompts, out, req.components, req.model)
        return f"mine · {req.backend} · {Path(prompts).name}", argv

    # Which distribution to sample: whatever the Base-distribution picker
    # persisted for this backend (or the legacy dropdown's evolved ★ branch).
    dist_base = resolve_dist_base(req.backend, req.dist, req.model)
    common = ["--backend", req.backend, *model, "--dist", dist_base]

    if req.action == "generate":
        argv = base + ["scripts/generate.py", *common, *_common_sampler_flags(req),
                       *_gen_flags(req),
                       *_experiment_flags(req.experiment, req.hypothesis)]
        if req.n is not None:
            argv += ["--n", str(req.n)]
        if req.temperature is not None:
            argv += ["--temperature", str(req.temperature)]
        if req.target_distance is not None:
            argv += ["--target-distance", str(req.target_distance)]
        if req.min_distance is not None:
            argv += ["--min-distance", str(req.min_distance)]
        if req.radius_band:
            argv += ["--radius-band"]
            if req.radius_scale is not None:
                argv += ["--radius-scale", str(float(req.radius_scale))]
        if req.seed is not None:
            argv += ["--seed", str(req.seed)]
        if req.width:
            argv += ["--width", str(int(req.width))]
        if req.height:
            argv += ["--height", str(int(req.height))]
        # init-image injection (only if enabled AND images are present)
        if req.init and init_images_count() > 0:
            if req.init_mode not in ("img2img", "embedding"):
                raise HTTPException(400, f"bad init_mode {req.init_mode!r}")
            d = resolve_init_dir(req.init_folder)
            argv += ["--init-dir", str(d), "--init-mode", req.init_mode]
            if req.init_mode == "embedding":
                argv += ["--ip-scale", str(req.ip_scale)]
            else:
                argv += ["--init-strength", str(req.init_strength)]
        exp = labelset.clean_experiment_id(req.experiment)
        label = (f"generate · {req.backend} · {req.sampler} · "
                 f"T={req.temperature or 1.0} · n={req.n or 8}"
                 + (f" · {exp}" if exp else ""))
        return label, argv

    if req.action == "temp_sweep":
        argv = base + ["scripts/temperature_sweep.py", *common,
                       *_common_sampler_flags(req), *_gen_flags(req)]
        if req.temps:
            argv += ["--temps", _clean_csv(req.temps)]
        if req.seeds:
            argv += ["--seeds", _clean_csv(req.seeds)]
        label = f"temp-sweep · {req.backend} · {req.sampler}"
        return label, argv

    if req.action == "sampler_sweep":
        argv = base + ["scripts/sampler_sweep.py", *common, *_gen_flags(req)]
        if req.temperature is not None:
            argv += ["--temperature", str(req.temperature)]
        if req.coherence is not None:
            argv += ["--coherence", str(req.coherence)]
        if req.seeds:
            argv += ["--seeds", _clean_csv(req.seeds)]
        label = f"sampler-sweep · {req.backend}"
        return label, argv

    raise HTTPException(400, f"unknown action {req.action!r}")


#: Longest negative prompt the UI may send. CLIP truncates at 77 tokens anyway,
#: so this only exists to keep a runaway paste out of argv and the job log.
MAX_NEGATIVE_CHARS = 1000


def _experiment_flags(experiment: Optional[str],
                      hypothesis: Optional[str] = None) -> list[str]:
    """``--experiment``/``--hypothesis`` for the scripts that write sidecars.

    The id is slugged by the same rule the CLI and the dataset use, so what the
    dashboard shows, what argv carries and what a label records are one string.
    """
    exp = labelset.clean_experiment_id(experiment)
    if not exp:
        return []
    flags = ["--experiment", exp]
    if hypothesis and hypothesis.strip():
        flags += ["--hypothesis", _clean_prompt(hypothesis)]
    return flags


def _clean_prompt(s: str) -> str:
    """Sanitise free-form prompt text bound for argv.

    The negative prompt is the one knob that can't be allow-listed, so it gets
    the next-best thing: it travels as a single element of a list-form argv (no
    shell, no interpolation), control characters and newlines are flattened to
    spaces so it can't forge extra lines in the job log, and it is length-capped.
    """
    flat = " ".join(s.split())
    if len(flat) > MAX_NEGATIVE_CHARS:
        raise HTTPException(400, f"negative prompt too long (max {MAX_NEGATIVE_CHARS} chars)")
    return flat


def _clean_csv(s: str) -> str:
    """Keep only number-ish comma lists (defends the sweep flags)."""
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        try:
            float(p)
        except ValueError:
            raise HTTPException(400, f"bad number in list: {p!r}")
    return ",".join(parts)


# --------------------------------------------------------------------------- #
# App + routes
# --------------------------------------------------------------------------- #
app = FastAPI(title="Semantic Anarchy Explorer")

# Built React frontend (webui/frontend/dist). Mounted at "/" at the BOTTOM of
# this file, after every /api route exists, so the SPA only catches what's left.
FRONTEND_DIST = Path(__file__).resolve().parent / "frontend" / "dist"


@app.get("/legacy", response_class=HTMLResponse)
def legacy_index() -> str:
    """The original inline dashboard, kept reachable until parity is confirmed."""
    return INDEX_HTML


@app.get("/label")
def label_page() -> FileResponse:
    """The labeling page as its OWN tab — same bundle, no dashboard chrome.

    StaticFiles(html=True) only serves index.html for "/", so a second entry
    point needs a route of its own; the SPA reads `location.pathname` and mounts
    the standalone shell instead of the dashboard. Labeling is a different mode
    of attention from generating, and it wants the whole window.
    """
    index = FRONTEND_DIST / "index.html"
    if not index.is_file():
        raise HTTPException(404, "frontend not built — run `make build`")
    return FileResponse(index)


@app.post("/api/run")
def api_run(req: RunRequest) -> JSONResponse:
    label, argv = build_argv(req)
    job = RUNNER.submit(req.action, label, argv)
    return JSONResponse({"job_id": job.id, "label": label})


#: Upscale engines, in the order the UI offers them.
#: hires -- scripts/upscale.py: same model, same latents, last `strength` of the
#:          ORIGINAL schedule. The faithful default.
#: flux  -- scripts/refine_flux.py: klein reference-regeneration (different model).
#: sd    -- scripts/refine.py: general img2img, optionally tiled.
REFINE_ENGINES = {"hires", "flux", "sd"}

#: The hires pass replays conditioning, so only SD-family sidecars qualify.
HIRES_BACKENDS = {"sd15", "sd2", "sdxl"}


class RefineRequest(BaseModel):
    src: str                          # filename (or outputs-relative path) of a PNG
    scale: float = 2.0                # upscale factor (hires: snapped to 16px)
    steps: Optional[int] = None       # hires: unset = the original's own step count
    strength: float = 0.3             # hires: fraction of the original schedule re-run
    scheduler: Optional[str] = None   # default ddim in refine.py when unset
    tiled: bool = True                # tiled native-res detail pass (Ultimate-SD-Upscale style)
    overlap: int = 128
    engine: str = "hires"             # hires (same-latent) | flux (klein) | sd (img2img)
    prompt: Optional[str] = None      # flux engine: override the upscale instruction
    interp: str = "lanczos"           # hires: resampling filter for the enlarge


@app.post("/api/refine")
def api_refine(req: RefineRequest) -> JSONResponse:
    # Resolve + sandbox the source to an image under outputs/.
    src = _resolve_output_image(req.src)
    if req.engine not in REFINE_ENGINES:
        raise HTTPException(400, f"bad engine {req.engine!r}")
    if not (0.0 < req.scale <= 3.0):
        raise HTTPException(400, "scale must be in (0, 3]")
    if not (0.0 < req.strength <= 1.0):
        raise HTTPException(400, "strength must be in (0, 1]")

    if "anarchy_sdxl" in src.name:
        backend = "sdxl"
    elif "anarchy_sd2" in src.name:
        backend = "sd2"
    else:
        backend = "sd15"
    model = _model_flags(RunRequest(action="refine", backend=backend))
    if req.engine == "hires":
        # The conditioning may live one or more `refined_from` hops back (this is
        # also the pre-flight: an untraceable source 400s instead of burning a job).
        origin = _explorable_source(src)
        obackend = next((b for b in ("sdxl", "sd2") if f"anarchy_{b}_" in origin.name), "sd15")
        if obackend not in HIRES_BACKENDS:
            raise HTTPException(400, f"{obackend} latents can't be replayed; use the FLUX engine")
        if req.interp not in INTERPS:
            raise HTTPException(400, f"bad interp {req.interp!r}")
        model = _model_flags(RunRequest(action="refine", backend=obackend))
        argv = [PYTHON, "-u", "scripts/upscale.py", "--backend", obackend, *model,
                "--src", str(src), "--factor", str(req.scale),
                "--denoise", str(req.strength), "--interp", req.interp]
        if req.steps is not None:
            argv += ["--steps", str(int(req.steps))]
        label = f"upscale · {obackend} · {src.name} · x{req.scale} · d{req.strength}"
        job = RUNNER.submit("refine", label, argv)
        return JSONResponse({"job_id": job.id, "label": label})
    if req.engine == "flux":
        argv = [FLUX_PYTHON, "-u", "scripts/refine_flux.py", "--src", str(src),
                "--scale", str(req.scale)]
        if req.steps is not None:
            argv += ["--steps", str(int(req.steps))]
        if req.prompt:
            argv += ["--prompt", req.prompt]
        label = f"refine · flux-klein · {src.name} · x{req.scale}"
        job = RUNNER.submit("refine", label, argv)
        return JSONResponse({"job_id": job.id, "label": label})
    if req.scheduler and req.scheduler not in SCHEDULERS:
        raise HTTPException(400, f"bad scheduler {req.scheduler!r}")
    argv = [PYTHON, "-u", "scripts/refine.py", "--backend", backend, *model,
            "--src", str(src), "--scale", str(req.scale), "--strength", str(req.strength)]
    if req.steps is not None:
        argv += ["--steps", str(int(req.steps))]
    if req.scheduler:
        argv += ["--scheduler", req.scheduler]
    if req.tiled:
        argv += ["--tiled", "--overlap", str(int(req.overlap))]
    mode = "tiled" if req.tiled else "single"
    label = f"refine · {backend} · {src.name} · x{req.scale} · d{req.strength} · {mode}"
    job = RUNNER.submit("refine", label, argv)
    return JSONResponse({"job_id": job.id, "label": label})


class ExploreRequest(BaseModel):
    src: str                          # anchor image (outputs-relative or bare name)
    mode: str = "neighborhood"        # neighborhood | breed | walk
    b: Optional[str] = None           # second parent (breed)
    radius: float = 0.3
    mutate: float = 0.15
    direction: str = "outward"        # walk: outward | random | axis
    step: float = 0.15                # walk: per-frame step
    axis: Optional[int] = None        # walk: which principal axis
    n: int = 6
    steps: Optional[int] = None
    guidance: Optional[float] = None
    experiment: Optional[str] = None  # tag the children too (E11-grids)


def _resolve_output_image(rel: str) -> Path:
    """Resolve an outputs-relative (or bare) image name, sandboxed to outputs/.

    Any extension in :data:`IMAGE_EXTS` counts, and the *stem* is the identity:
    a persisted ``.png`` rel-path (favorites, timeline, an old browser tab) still
    resolves after the same render was re-made as ``.jpg``.
    """
    base = OUTPUTS.resolve()
    p = (base / Path(rel)).resolve()
    if not p.is_file():
        alt = (base / "generated" / Path(rel).name).resolve()
        p = alt if alt.is_file() else p
    hit = find_image(p)
    if hit is None or base not in hit.parents:
        raise HTTPException(404, f"image not found under outputs/: {rel}")
    return hit


def _explorable_source(p: Path) -> Path:
    """Resolve to an image that has conditioning (.npz). Upscaled/refined
    outputs have none -- follow their ``refined_from`` sidecar link back to the
    original (through chains of refinements) so 🧭/🚶/🧬 on an upscale act on
    the image it came from."""
    for _ in range(5):
        if p.with_suffix(".npz").is_file():
            return p
        j = p.with_suffix(".json")
        parent = None
        if j.is_file():
            try:
                parent = json.loads(j.read_text()).get("refined_from")
            except Exception:
                parent = None
        if not parent:
            break
        cand = p.parent / parent
        if not cand.is_file():
            break
        p = cand
    raise HTTPException(
        400, f"{p.name} has no conditioning sidecar (and no traceable original)")


@app.post("/api/explore")
def api_explore(req: ExploreRequest) -> JSONResponse:
    if req.mode not in ("neighborhood", "breed", "walk"):
        raise HTTPException(400, f"bad mode {req.mode!r}")
    if req.direction not in ("outward", "random", "axis"):
        raise HTTPException(400, f"bad direction {req.direction!r}")
    src = _explorable_source(_resolve_output_image(req.src))
    backend = "sd15"
    for b in ("flux2", "krea2", "sdxl", "sd2"):
        if f"anarchy_{b}_" in src.name:
            backend = b
            break
    model = _model_flags(RunRequest(action="explore", backend=backend))
    argv = [python_for(backend), "-u", "scripts/explore.py", "--backend", backend, *model,
            "--mode", req.mode, "--src", str(src),
            "--n", str(int(req.n)), "--scheduler", "ddim"]
    if req.mode == "breed":
        b = _explorable_source(_resolve_output_image(req.b or ""))
        if backend not in b.name:
            raise HTTPException(400, "breed parents must be from the same backend")
        argv += ["--b", str(b), "--mutate", str(req.mutate)]
        label = f"breed · {backend} · {src.name} × {b.name}"
    elif req.mode == "walk":
        argv += ["--direction", req.direction, "--step", str(req.step)]
        if req.axis is not None:
            argv += ["--axis", str(int(req.axis))]
        label = f"walk · {backend} · {src.name} · {req.direction} ×{req.n}"
    else:
        argv += ["--radius", str(req.radius)]
        label = f"explore · {backend} · {src.name} · r={req.radius}"
    # Backfill steps/guidance so SDXL never falls through to the turbo family
    # defaults (1 step / no CFG) -- children must render with real CFG.
    d = SDXL_MODEL_DEFAULTS.get(SDXL_DEFAULT_MODEL, {}) if backend == "sdxl" else {}
    steps = req.steps if req.steps is not None else d.get("steps")
    guidance = req.guidance if req.guidance is not None else d.get("guidance")
    if steps is not None:
        argv += ["--steps", str(int(steps))]
    if guidance is not None:
        argv += ["--guidance", str(guidance)]
    if backend == "sdxl":
        argv += ["--neg-mode", "mean"]
    argv += _experiment_flags(req.experiment)
    job = RUNNER.submit("explore", label, argv)
    return JSONResponse({"job_id": job.id, "label": label})


class EvolveRequest(BaseModel):
    backend: Optional[str] = None     # None = backend with most starred images
    n: int = 8
    temperature: float = 1.0
    base_blend: float = 0.25
    experiment: Optional[str] = None


def _most_starred_backend() -> str:
    favs = load_favs()
    counts = {b: sum(1 for r in favs if f"anarchy_{b}_" in Path(r).name)
              for b in BACKENDS}
    best = max(counts, key=counts.get)
    if counts[best] == 0:
        raise HTTPException(400, "no starred images to evolve from")
    return best


@app.post("/api/evolve")
def api_evolve(req: EvolveRequest) -> JSONResponse:
    backend = req.backend or _most_starred_backend()
    if backend not in BACKENDS:
        raise HTTPException(400, f"bad backend {backend!r}")
    model = _model_flags(RunRequest(action="evolve", backend=backend))
    # Branch off whatever corpus is selected, not always the repo's own — the
    # elites were sampled from that fit, and its PCA axes are what get grafted.
    argv = [python_for(backend), "-u", "scripts/evolve_favorites.py", "--backend", backend, *model,
            "--dist", resolve_dist_base(backend),
            "--n", str(int(req.n)), "--temperature", str(req.temperature),
            "--base-blend", str(req.base_blend), "--scheduler", "ddim"]
    d = SDXL_MODEL_DEFAULTS.get(SDXL_DEFAULT_MODEL, {}) if backend == "sdxl" else {}
    if d.get("steps") is not None:
        argv += ["--steps", str(d["steps"]), "--guidance", str(d["guidance"]),
                 "--neg-mode", "mean"]
    argv += _experiment_flags(req.experiment)
    label = f"evolve★ · {backend} · T={req.temperature}"
    job = RUNNER.submit("evolve", label, argv)
    return JSONResponse({"job_id": job.id, "label": label})


@app.post("/api/resonance")
def api_resonance() -> JSONResponse:
    """Queue the resonance engine: embed new images, novelty + taste model."""
    argv = [PYTHON, "-u", "scripts/resonance.py"]
    job = RUNNER.submit("resonance", "analyze · novelty + resonance", argv)
    return JSONResponse({"job_id": job.id})


class InvertRequest(BaseModel):
    src: str
    tokens: int = 12
    space: str = "clip"      # clip = image match (any backend); native = match
                             # the stored conditioning in the model's own encoder


@app.post("/api/invert")
def api_invert(req: InvertRequest) -> JSONResponse:
    """Queue PEZ hard-prompt inversion (arXiv:2302.03668): find the nearest
    TYPEABLE prompt to an image born without one. Ruler, not leash."""
    src = _resolve_output_image(req.src)
    if req.space == "native":
        src = _explorable_source(src)   # native needs the .npz; upscales redirect
    argv = [PYTHON, "-u", "scripts/invert_prompt.py", "--src", str(src),
            "--space", req.space,
            "--tokens", str(req.tokens), "--steps", "500", "--restarts", "3"]
    job = RUNNER.submit("invert", f"invert[{req.space}] · {src.name} · {req.tokens} tok", argv)
    return JSONResponse({"job_id": job.id})


class GenPromptRequest(BaseModel):
    src: str                 # image whose sidecar holds the discovered prompt
    which: str = "inverted"  # inverted (CLIP PEZ) | native


@app.post("/api/genprompt")
def api_genprompt(req: GenPromptRequest) -> JSONResponse:
    """Render what the discovered hard prompt actually produces, through the
    same backend and seed, so discovery and best-words-can-do can be compared."""
    src = _resolve_output_image(req.src)
    j = src.with_suffix(".json")
    meta = {}
    if j.is_file():
        try:
            meta = json.loads(j.read_text())
        except Exception:
            pass
    prompt = meta.get(f"{req.which}_prompt")
    # Upscales/refines record the REFINE engine as their model; walk back to
    # the generation ancestor for backend/model/params (prompt stays as clicked).
    gsrc, gmeta = src, meta
    for _ in range(5):
        if gmeta.get("kind") != "refine" or not gmeta.get("refined_from"):
            break
        cand = gsrc.parent / gmeta["refined_from"]
        cj = cand.with_suffix(".json")
        if not cand.is_file() or not cj.is_file():
            break
        gsrc = cand
        try:
            gmeta = json.loads(cj.read_text())
        except Exception:
            gmeta = {}
    meta = gmeta
    if not prompt:                       # e.g. native prompt saved on the ancestor
        prompt = meta.get(f"{req.which}_prompt") or meta.get("inverted_prompt")
    if not prompt:
        raise HTTPException(400, f"{src.name}: no discovered prompt yet -- run 🔤 first")
    m = re.match(r"anarchy_([a-z0-9]+)_", gsrc.name)
    backend = meta.get("backend") or (m.group(1) if m else "sd15")
    argv = [python_for(backend), "-u", "scripts/generate_prompted.py",
            "--backend", backend, "--prompt", prompt,
            "--parent", src.name, "--prompt-kind",
            ("pez" if req.which == "inverted" else "native")]
    model = meta.get("model")
    if model and model != "(default)":
        # single-file checkpoints need --ckpt (from_single_file), repos --model
        argv += (["--ckpt", model] if model.endswith((".safetensors", ".ckpt"))
                 else ["--model", model])
    for k, flag in (("steps", "--steps"), ("guidance", "--guidance"),
                    ("image_seed", "--seed"), ("height", "--height"),
                    ("width", "--width")):
        if meta.get(k) is not None:
            argv += [flag, str(meta[k])]
    if meta.get("scheduler") and meta["scheduler"] != "default":
        argv += ["--scheduler", meta["scheduler"]]
    job = RUNNER.submit("genprompt", f"from-prompt · {src.name}", argv)
    return JSONResponse({"job_id": job.id})


@app.get("/api/tasteband")
def api_tasteband() -> JSONResponse:
    """Distance stats of the starred images -- where your keepers live."""
    favs = load_favs()
    ds = []
    for rel in favs:
        j = OUTPUTS / Path(rel).with_suffix(".json")
        if j.is_file():
            try:
                d = json.loads(j.read_text()).get("distance")
                if d is not None:
                    ds.append(float(d))
            except Exception:
                pass
    if not ds:
        return JSONResponse({"count": 0})
    ds.sort()
    return JSONResponse({
        "count": len(ds),
        "mean": round(sum(ds) / len(ds), 2),
        "p25": round(ds[len(ds) // 4], 2),
        "p75": round(ds[(3 * len(ds)) // 4], 2),
    })


@app.post("/api/cancel/{job_id}")
def api_cancel(job_id: int) -> JSONResponse:
    return JSONResponse({"ok": RUNNER.cancel(job_id)})


@app.get("/api/state")
def api_state() -> JSONResponse:
    return JSONResponse(RUNNER.snapshot())


@app.get("/api/log/{job_id}", response_class=PlainTextResponse)
def api_log(job_id: int) -> str:
    job = RUNNER.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")
    return "\n".join(job.log)


# How often the SSE generator looks for freshly appended lines, and how long it
# waits before emitting a keep-alive comment on a silent job.
_SSE_TICK = 0.4
_SSE_PING_EVERY = 15.0


@app.get("/api/log/{job_id}/stream")
async def api_log_stream(job_id: int, request: Request) -> StreamingResponse:
    """Server-sent events: the job's log, pushed as it is appended.

    Replaces polling ``/api/log/{job_id}`` (which still works). Each ``lines``
    frame carries a JSON array of *new* lines and an ``id:`` equal to the number
    of lines sent so far, so a reconnecting EventSource resumes exactly where it
    left off via ``Last-Event-ID`` instead of replaying the whole transcript.

    The runner trims its buffer at 5000 lines; when that happens the cursor is
    past the end, and a ``reset`` frame tells the client to clear and re-sync.
    """
    job = RUNNER.get(job_id)
    if not job:
        raise HTTPException(404, "no such job")

    try:
        idx = int(request.headers.get("last-event-id") or 0)
    except ValueError:
        idx = 0

    async def gen():
        nonlocal idx
        last_ping = time.monotonic()
        while True:
            if await request.is_disconnected():
                return
            log = job.log
            n = len(log)
            if idx > n:                       # buffer was trimmed under us
                idx = 0
                yield "event: reset\ndata: {}\n\n"
                continue
            if n > idx:
                chunk = log[idx:n]
                idx = n
                yield f"id: {idx}\nevent: lines\ndata: {json.dumps(chunk)}\n\n"
                last_ping = time.monotonic()
                continue
            if job.status in ("done", "error", "cancelled"):
                payload = json.dumps({"status": job.status, "rc": job.rc})
                yield f"event: status\ndata: {payload}\n\n"
                return
            if time.monotonic() - last_ping > _SSE_PING_EVERY:
                last_ping = time.monotonic()
                yield ": ping\n\n"
            await asyncio.sleep(_SSE_TICK)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",      # don't let a reverse proxy buffer it
        },
    )


@app.get("/api/images")
def api_images() -> JSONResponse:
    favs = load_favs()
    def _load_json(name):
        try:
            return json.loads((OUTPUTS / name).read_text())
        except Exception:
            return {}
    scores = _load_json("scores.json")
    novelty = _load_json("novelty.json")
    resonance = _load_json("resonance.json")
    buckets: dict[str, list] = {k: [] for k, _ in GALLERY_BUCKETS}
    buckets["favorites"] = []
    buckets["top"] = []
    buckets["frontier"] = []
    by_rel: dict[str, dict] = {}
    if OUTPUTS.is_dir():
        # generate.py writes to outputs/generated/; sweeps + marginals to
        # outputs/ root. rglob covers both. Renders are JPEG now and PNG before
        # the switch, and the analysis plots are PNG forever -- scan them all and
        # let the bucket prefixes decide what shows.
        for p in (q for e in IMAGE_EXTS for q in OUTPUTS.rglob(f"*{e}")):
            name = p.name
            rel = p.relative_to(OUTPUTS).as_posix()
            for key, prefix in GALLERY_BUCKETS:
                if name.startswith(prefix):
                    st = p.stat()
                    item = {
                        "name": name,
                        "rel": rel,
                        "url": f"/img?path={rel}",
                        "mtime": st.st_mtime,
                        "size": st.st_size,
                        "fav": rel in favs,
                        "score": scores.get(rel),
                        "dist": _distance_for(rel),
                        "nov": novelty.get(rel),
                        "res": resonance.get(rel),
                    }
                    buckets[key].append(item)
                    by_rel[rel] = item
                    break
    # The favorites bucket pulls the matching items (skipping any deleted files).
    buckets["favorites"] = [by_rel[r] for r in favs if r in by_rel]
    # Top rated: every scored generated image, best first.
    scored = [it for it in buckets["generated"] if it["score"] is not None]
    buckets["top"] = sorted(scored, key=lambda d: d["score"], reverse=True)[:80]
    # 🎯 Frontier: Pareto front of novelty x resonance (fall back to the generic
    # aesthetic score until the taste model exists). "Nothing else in the
    # gallery is both newer AND more you."
    cand = [it for it in buckets["generated"]
            if it["nov"] is not None and (it["res"] or it["score"]) is not None]
    cand.sort(key=lambda d: d["nov"], reverse=True)
    # Peel successive Pareto layers: the strict front is razor-thin (often <10
    # of 10k), so keep taking "the front of what remains" until ~120 images.
    def _r(it):
        return it["res"] if it["res"] is not None else it["score"]
    front, pool = [], cand
    while pool and len(front) < 120:
        best, layer, rest = -1e9, [], []
        for it in pool:                       # pool stays novelty-sorted
            if _r(it) > best:
                layer.append(it)
                best = _r(it)
            else:
                rest.append(it)
        layer.sort(key=_r, reverse=True)
        front.extend(layer)
        pool = rest
    buckets["frontier"] = front[:120]
    for key in buckets:
        if key not in ("top", "frontier"):
            buckets[key].sort(key=lambda d: d["mtime"], reverse=True)
    return JSONResponse(buckets)


@app.post("/api/score")
def api_score() -> JSONResponse:
    """Queue an aesthetic-scoring pass over all generated images."""
    argv = [PYTHON, "-u", "scripts/score_images.py"]
    job = RUNNER.submit("score", "score · aesthetic (all images)", argv)
    return JSONResponse({"job_id": job.id})


class FavRequest(BaseModel):
    rel: str
    on: bool = True


@app.post("/api/favorite")
def api_favorite(req: FavRequest) -> JSONResponse:
    # Only allow favoriting images that actually live under outputs/.
    base = OUTPUTS.resolve()
    target = (base / req.rel).resolve()
    if base not in target.parents or target.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(404, "not an outputs image")
    rel = target.relative_to(base).as_posix()
    with _favs_lock:
        favs = load_favs()
        favs.add(rel) if req.on else favs.discard(rel)
        save_favs(favs)
    return JSONResponse({"ok": True, "fav": req.on, "count": len(favs)})


# --------------------------------------------------------------------------- #
# Labeling — the referee of the exploration loop
# --------------------------------------------------------------------------- #
# Labels are the ONE output of this project that cannot be regenerated, so they
# do not live under the gitignored outputs/: append-only JSONL at the repo root,
# committed to git. This file only orchestrates — the record schema, the score
# range and the summary maths all come from semantic_anarchy/labels.py.
LABELS_FILE = labelset.labels_file()   # repo/labels/labels.jsonl (SA_LABELS_FILE overrides)

#: How the queue may be narrowed. "unlabeled" is the working default;
#: "labeled" exists for auditing your own past calls, "all" for a blind re-pass.
LABEL_SCOPES = ("unlabeled", "all", "labeled")
#: Which gallery slice to pull from. Only generated/ carries per-image sidecars,
#: so that (and its starred subset) is everything that can be labeled.
LABEL_BUCKETS = ("generated", "favorites")
#: The queue selector's magic experiment value for "images with no id".
UNTAGGED = "__none__"
#: Fields of the sidecar the labeling page shows above the image. Small type,
#: collapsible — the label stays perceptual, not analytical.
LABEL_KNOB_KEYS = (
    "kind", "sampler", "temperature", "coherence", "components", "comp_lo",
    "equalize", "truncation", "steps", "guidance", "scheduler", "neg_mode",
    "target_distance", "min_distance", "mode", "radius", "step", "direction",
)

_labels_cache: tuple = (None, [])


def load_labels() -> list:
    """Every record, cached against the file's (mtime, size).

    Re-read only when the file actually changed, so a keypress-per-second
    labeling session doesn't re-parse the whole dataset on every request.
    """
    global _labels_cache
    try:
        st = LABELS_FILE.stat()
        key = (st.st_mtime, st.st_size)
    except OSError:
        key = None
    if _labels_cache[0] == key:
        return _labels_cache[1]
    recs = labelset.read_labels(LABELS_FILE)
    _labels_cache = (key, recs)
    return recs


def labeled_scores() -> dict:
    """rel -> the score that currently stands for it (latest record wins)."""
    return {rel: rec["score"]
            for rel, rec in labelset.latest_by_rel(load_labels()).items()}


#: The dimensions the queue can be sliced along, each exposed as a facet with
#: counts. Every one is derived from the image's own sidecar/pixels — no new
#: bookkeeping — and every one is an *equality* filter, so the UI is a row of
#: dropdowns and the server never parses user-supplied expressions.
LABEL_FACETS = ("experiment", "backend", "ckpt", "folder", "size", "kind", "sampler")
#: Sentinel for "this image has no value on that dimension" (untagged, no ckpt…).
UNSET = "__none__"
#: Kept for readability at the call sites that mean the experiment dimension.
UNTAGGED = UNSET

# rel -> (mtime, row). Deriving a row costs a sidecar parse and, for the images
# whose sidecar predates height/width, an image-header read; caching on
# mtime keeps a facet refresh over 10k images cheap.
_ROW_CACHE: dict = {}


def _label_index() -> list:
    """Every labelable image, with each dimension the queue can filter on.

    Anything named ``anarchy_*`` (any image extension) anywhere under
    ``outputs/`` — that prefix is
    exactly the set of images that carry a per-image ``.json`` sidecar, and a
    label on an image with no sidecar would record no knobs. ``--outdir`` can put
    those outside ``generated/``, hence the recursive walk and the ``folder``
    facet.
    """
    if not OUTPUTS.is_dir():
        return []
    rows = []
    for p in (q for e in IMAGE_EXTS for q in OUTPUTS.rglob(f"anarchy_*{e}")):
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        rel = p.relative_to(OUTPUTS).as_posix()
        hit = _ROW_CACHE.get(rel)
        if hit is not None and hit[0] == mtime:
            rows.append(hit[1])
            continue
        meta = sidecar_for(rel)
        # Sidecar first (free), pixels second: evolve-branch sidecars record no
        # size at all, and a refine's recorded size can predate its own upscale.
        w, h = meta.get("width"), meta.get("height")
        if not (w and h):
            size = _image_size(p)
            h, w = size if size else (None, None)
        backend = meta.get("backend")
        if not backend:
            for b in BACKENDS:                      # fall back to the filename
                if f"anarchy_{b}_" in p.name:
                    backend = b
                    break
        row = {
            "rel": rel,
            "url": f"/img?path={rel}",
            "mtime": mtime,
            "experiment": labelset.clean_experiment_id(meta.get("experiment")),
            "backend": backend,
            "ckpt": dist_paths.model_slug(meta["model"]) if meta.get("model") else None,
            "folder": Path(rel).parent.as_posix() or ".",
            "size": f"{w}x{h}" if w and h else None,
            "kind": meta.get("kind"),
            "sampler": meta.get("sampler"),
            "distance": meta.get("distance"),
            "image_seed": meta.get("image_seed"),
            # Does this image carry its own conditioning? One stat, cached with
            # the rest of the row — it is what decides whether the image can go
            # into a selection fit (an upscale carries none of its own).
            "latents": (OUTPUTS / rel).with_suffix(".npz").is_file(),
            "knobs": {k: meta[k] for k in LABEL_KNOB_KEYS
                      if k in meta and meta[k] is not None},
        }
        _ROW_CACHE[rel] = (mtime, row)
        rows.append(row)
    return rows


def _facet_ok(row: dict, dim: str, want: str) -> bool:
    """One equality filter. ``""`` = any, ``__none__`` = images missing a value."""
    if not want:
        return True
    have = row.get(dim)
    return (have is None or have == "") if want == UNSET else have == want


def _select(rows: list, filters: dict, since: Optional[float],
            until: Optional[float]) -> list:
    out = []
    for row in rows:
        if since is not None and row["mtime"] < since:
            continue
        if until is not None and row["mtime"] > until:
            continue
        if all(_facet_ok(row, dim, want) for dim, want in filters.items()):
            out.append(row)
    return out


def _shuffle_key(seed: int, rel: str) -> bytes:
    """A stable pseudo-random order.

    Hashing (seed, rel) rather than shuffling a list means the order of what is
    LEFT doesn't change as images drop out of the queue on being labeled — so
    re-fetching mid-session doesn't teleport you somewhere else in the batch.
    """
    return hashlib.blake2b(f"{seed}:{rel}".encode(), digest_size=8).digest()


@app.get("/api/label/facets")
def api_label_facets() -> JSONResponse:
    """What there is to choose from: every facet value with its counts.

    Computed over the WHOLE labelable set rather than the current selection, so
    an option can never vanish because of an earlier pick. How many images a
    given combination actually matches is answered live by the queue itself.
    """
    scores = labeled_scores()
    favs = load_favs()
    rows = _label_index()
    tally: dict = {dim: {} for dim in LABEL_FACETS}
    for row in rows:
        unlabeled = row["rel"] not in scores
        for dim in LABEL_FACETS:
            key = row.get(dim) or UNSET
            cell = tally[dim].setdefault(key, {"value": key, "count": 0, "unlabeled": 0})
            cell["count"] += 1
            cell["unlabeled"] += int(unlabeled)
    times = [r["mtime"] for r in rows]
    return JSONResponse({
        "total": len(rows),
        "unlabeled": sum(1 for r in rows if r["rel"] not in scores),
        "favorites": sum(1 for r in rows if r["rel"] in favs),
        "oldest": min(times) if times else None,
        "newest": max(times) if times else None,
        "facets": {
            dim: sorted(cells.values(),
                        key=lambda c: (c["value"] == UNSET, -c["count"]))
            for dim, cells in tally.items()
        },
    })


@app.get("/api/label/queue")
def api_label_queue(experiment: Optional[str] = None, scope: str = "unlabeled",
                    bucket: str = "generated", order: str = "shuffle",
                    seed: int = 0, limit: int = 500,
                    backend: Optional[str] = None, ckpt: Optional[str] = None,
                    folder: Optional[str] = None, size: Optional[str] = None,
                    kind: Optional[str] = None, sampler: Optional[str] = None,
                    since: Optional[float] = None,
                    until: Optional[float] = None) -> JSONResponse:
    """The images to label next, shuffled within the selection by default.

    Generation order correlates with everything (seed, position in a sweep), so
    labeling in it invites the eye to find a trend that isn't there. Every filter
    is an equality match on a value the image itself carries, plus the mtime
    window — nothing here interprets a user-supplied expression.
    """
    if scope not in LABEL_SCOPES:
        raise HTTPException(400, f"bad scope {scope!r}")
    if bucket not in LABEL_BUCKETS:
        raise HTTPException(400, f"bad bucket {bucket!r}")
    scores = labeled_scores()
    favs = load_favs()
    filters = {
        "experiment": (experiment or "").strip(),
        "backend": (backend or "").strip(),
        "ckpt": (ckpt or "").strip(),
        "folder": (folder or "").strip(),
        "size": (size or "").strip(),
        "kind": (kind or "").strip(),
        "sampler": (sampler or "").strip(),
    }

    rows = []
    for row in _select(_label_index(), filters, since, until):
        if bucket == "favorites" and row["rel"] not in favs:
            continue
        score = scores.get(row["rel"])
        if scope == "unlabeled" and score is not None:
            continue
        if scope == "labeled" and score is None:
            continue
        rows.append({**row, "score": score, "fav": row["rel"] in favs})

    total = len(rows)
    if order == "new":
        rows.sort(key=lambda r: r["mtime"], reverse=True)
    elif order == "old":
        rows.sort(key=lambda r: r["mtime"])
    else:
        rows.sort(key=lambda r: _shuffle_key(seed, r["rel"]))
    return JSONResponse({
        "queue": rows[:max(1, min(2000, limit))],
        "total": total,
        "labeled": sum(1 for r in rows if r["score"] is not None),
        "filters": {k: v or None for k, v in filters.items()},
        "since": since,
        "until": until,
        "scope": scope,
        "bucket": bucket,
        "order": order,
    })


class LabelRequest(BaseModel):
    rel: str
    score: int


@app.post("/api/label")
def api_label(req: LabelRequest) -> JSONResponse:
    """Append one label. Relabeling appends too — history is never rewritten."""
    base = OUTPUTS.resolve()
    target = (base / req.rel).resolve()
    if base not in target.parents or target.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(404, "not an outputs image")
    rel = target.relative_to(base).as_posix()
    try:
        rec = labelset.make_record(rel, req.score, sidecar_for(rel))
    except (TypeError, ValueError) as exc:
        raise HTTPException(400, str(exc))
    labelset.append_label(rec, LABELS_FILE)
    return JSONResponse({"ok": True, "rel": rel, "score": rec["score"],
                         "experiment": rec["experiment"],
                         "count": len(labeled_scores())})


@app.get("/api/labels")
def api_labels() -> JSONResponse:
    """The dataset's summary: overall, then one tail-weighted row per experiment."""
    records = load_labels()
    latest = list(labelset.latest_by_rel(records).values())
    groups = labelset.by_experiment(latest)
    rows = [{"id": exp, **labelset.summarize_records(recs)}
            for exp, recs in sorted(groups.items())]
    rows.sort(key=lambda r: (r["keeper_rate"] or 0, r["n"]), reverse=True)
    return JSONResponse({
        "count": len(latest),
        "records": len(records),
        "file": str(LABELS_FILE),
        "overall": labelset.summarize_records(latest),
        "experiments": rows,
    })


@app.get("/api/experiments")
def api_experiments() -> JSONResponse:
    """Every experiment id the gallery or the manifests know about.

    Manifest-only ids (a batch that was cancelled before it wrote an image) and
    sidecar-only ids (a run launched from the CLI without a manifest) both show
    up — the queue selector should never hide a batch that exists.
    """
    scores = labeled_scores()
    images: dict = {}
    for row in _label_index():
        exp = row.get("experiment") or ""
        cell = images.setdefault(exp, {"images": 0, "labeled": 0})
        cell["images"] += 1
        if row["rel"] in scores:
            cell["labeled"] += 1

    out = []
    manifests = {m["id"]: m for m in labelset.list_manifests(REPO)}
    for exp in sorted(set(images) | set(manifests)):
        if not exp and not images.get(exp, {}).get("images"):
            continue
        m = manifests.get(exp) or {}
        runs = m.get("runs") or []
        counts = images.get(exp, {"images": 0, "labeled": 0})
        out.append({
            "id": exp,
            "hypothesis": m.get("hypothesis"),
            "created": m.get("created"),
            "runs": len(runs),
            "seed_panel": any(r.get("seed_panel") for r in runs),
            **counts,
        })
    # Most work left to do first; untagged (id "") sorts last whatever it holds.
    out.sort(key=lambda r: (r["id"] == "", -(r["images"] - r["labeled"]), r["id"]))
    return JSONResponse(out)


@app.get("/api/meta")
def api_meta(path: str) -> JSONResponse:
    """Return the param sidecar (``<stem>.json``) for an image, or {}.

    Native inversions run against the ORIGINAL's conditioning, so refines
    inherit their ancestor's native_prompt for display (marked native_from).
    """
    base = OUTPUTS.resolve()
    target = (base / path).resolve()
    if base not in target.parents:
        raise HTTPException(404, "not found")
    j = target.with_suffix(".json")
    meta = {}
    if j.is_file():
        try:
            meta = json.loads(j.read_text())
        except Exception:
            meta = {}
    if "native_prompt" not in meta:
        p, m = target, dict(meta)
        for _ in range(5):
            if not m.get("refined_from"):
                break
            p = p.parent / m["refined_from"]
            pj = p.with_suffix(".json")
            if not pj.is_file():
                break
            try:
                m = json.loads(pj.read_text())
            except Exception:
                break
            if "native_prompt" in m:
                for k in ("native_prompt", "native_sim", "native_tokens", "native_vocab"):
                    if k in m:
                        meta[k] = m[k]
                meta["native_from"] = p.name
                break
    return JSONResponse(meta)


@app.get("/img")
def api_img(path: str) -> FileResponse:
    # Serve PNGs anywhere under outputs/ (incl. outputs/generated/), guarding
    # against path traversal by confirming the resolved file stays inside it.
    base = OUTPUTS.resolve()
    target = (base / path).resolve()
    if (base not in target.parents
            or target.suffix.lower() not in IMAGE_EXTS + (".mp4",)
            or not target.is_file()):
        raise HTTPException(404, "not found")
    return FileResponse(target)


def _wipe_candidates():
    """Non-starred images with aesthetic score < 5. Favorites and their whole
    ancestry (refined_from / parent chains) are protected -- deleting a
    favorite's original would break native inversion and exploration."""
    favs = set()
    try:
        favs = set(json.loads((OUTPUTS / "favorites.json").read_text()))
    except Exception:
        pass
    try:
        scores = json.loads((OUTPUTS / "scores.json").read_text())
    except Exception:
        scores = {}
    protected = set(favs)
    for rel in list(favs):                      # protect ancestry chains
        p = OUTPUTS / rel
        for _ in range(6):
            jj = p.with_suffix(".json")
            if not jj.is_file():
                break
            try:
                m = json.loads(jj.read_text())
            except Exception:
                break
            up = m.get("refined_from") or m.get("parent")
            if not up:
                break
            p = p.parent / up
            protected.add(str(p.relative_to(OUTPUTS)))
    out = []
    gen = OUTPUTS / "generated"
    for img in sorted(q for e in IMAGE_EXTS for q in gen.glob(f"anarchy_*{e}")):
        rel = str(img.relative_to(OUTPUTS))
        sc = scores.get(rel)
        if rel not in protected and sc is not None and sc < 5.0:
            out.append(rel)
    return out


@app.get("/api/wipe/preview")
def api_wipe_preview() -> JSONResponse:
    return JSONResponse({"count": len(_wipe_candidates())})


@app.post("/api/wipe")
def api_wipe() -> JSONResponse:
    doomed = _wipe_candidates()
    gone = set(doomed)
    n_files = 0
    for rel in doomed:
        base = OUTPUTS / rel
        for ext in IMAGE_EXTS + (".json", ".npz"):
            p = base.with_suffix(ext)
            if p.is_file():
                p.unlink()
                n_files += 1
    for cache in ("novelty.json", "resonance.json", "scores.json"):
        cp = OUTPUTS / cache
        if cp.is_file():
            try:
                d = json.loads(cp.read_text())
                cp.write_text(json.dumps(
                    {k: v for k, v in d.items() if k not in gone}))
            except Exception:
                pass
    emb = OUTPUTS / "clip_embeds.npz"
    if emb.is_file():
        try:
            import numpy as np
            z = np.load(emb)
            names, vecs = list(z["names"]), z["vecs"]
            keep = [i for i, nm in enumerate(names) if str(nm) not in gone]
            np.savez_compressed(emb, names=np.array([str(names[i]) for i in keep]),
                                vecs=vecs[keep])
        except Exception:
            pass
    return JSONResponse({"deleted": len(doomed), "files": n_files})


# --------------------------------------------------------------------------- #
# Films — latent travel through a timeline of keyframes
# --------------------------------------------------------------------------- #
# Backends whose sidecars morph_film.py can interpolate (FILM_TENSORS there).
FILM_BACKENDS = {"sd15", "sd2", "sdxl"}
FILM_INTERPS = {"slerp", "lerp"}
FILM_EASINGS = {"smooth", "smoother", "linear"}
FILM_REFINES = {"none", "flux"}
MAX_FILM_KEYS = 64
MAX_FILM_FRAMES = 3000            # a 3-minute film at 16fps; past this it's a typo


def _image_size(p: Path) -> Optional[tuple[int, int]]:
    """``(height, width)`` from the file header — PNG IHDR or JPEG SOF, no PIL.

    The rendered pixels are ground truth: older sidecars (evolve branches)
    recorded no height/width at all. Renders are JPEG now, so this walks the
    marker chain for those; both paths read a few dozen bytes, which is what
    keeps a facet refresh over 10k images cheap.
    """
    try:
        with p.open("rb") as f:
            head = f.read(24)
            if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
                w, h = struct.unpack(">II", head[16:24])
                return int(h), int(w)
            if head[:2] != b"\xff\xd8":                        # not a JPEG either
                return None
            f.seek(2)
            while True:
                b = f.read(1)
                if not b:
                    return None
                if b != b"\xff":                                # resync to a marker
                    continue
                marker = f.read(1)
                while marker == b"\xff":                        # fill bytes
                    marker = f.read(1)
                if not marker or marker == b"\xd9":
                    return None
                m = marker[0]
                if m in (0x01,) or 0xd0 <= m <= 0xd8:           # standalone markers
                    continue
                seg = f.read(2)
                if len(seg) < 2:
                    return None
                length = struct.unpack(">H", seg)[0]
                # SOF0..SOF15 carry the frame size; DHT/DAC/DNL are not SOFs.
                if 0xc0 <= m <= 0xcf and m not in (0xc4, 0xc8, 0xcc):
                    body = f.read(5)
                    if len(body) < 5:
                        return None
                    h, w = struct.unpack(">HH", body[1:5])
                    return int(h), int(w)
                f.seek(length - 2, 1)
    except Exception:                                          # noqa: BLE001
        return None


def _keyframe_row(raw: str) -> dict:
    """Everything the timeline needs to warn BEFORE a render is queued."""
    row = {"rel": raw, "source": None, "backend": None,
           "height": None, "width": None, "filmable": False, "error": None}
    try:
        src = _explorable_source(_resolve_output_image(raw))
    except HTTPException as exc:
        row["error"] = str(exc.detail)
        return row
    row["source"] = src.relative_to(OUTPUTS.resolve()).as_posix()
    backend = "sd15"
    for cand in ("flux2", "krea2", "sdxl", "sd2"):
        if f"anarchy_{cand}_" in src.name:
            backend = cand
            break
    row["backend"] = backend
    size = _image_size(src)
    if size:
        row["height"], row["width"] = size
    row["filmable"] = backend in FILM_BACKENDS
    if not row["filmable"]:
        row["error"] = f"{backend} keyframes can't be filmed yet"
    return row


class KeyframesRequest(BaseModel):
    images: list[str]


@app.post("/api/keyframes")
def api_keyframes(req: KeyframesRequest) -> JSONResponse:
    """Probe the timeline's keyframes: backend, real resolution, filmability.

    Read-only and cheap (a 24-byte header read per image) — the UI calls it on
    every timeline edit so mixed backends or mixed resolutions surface as a
    warning instead of as a surprise in the finished film.
    """
    return JSONResponse([_keyframe_row(r) for r in req.images[:MAX_FILM_KEYS]])


class FilmRequest(BaseModel):
    images: list[str]                 # ordered keyframes (outputs-relative)
    name: Optional[str] = None        # slugified; auto-named when blank
    height: Optional[int] = None      # film resolution; None = keyframe 1's own
    width: Optional[int] = None
    fps: int = 16
    frames_per: int = 24              # frames rendered per transition
    interp: str = "slerp"             # slerp | lerp
    easing: str = "smooth"            # smooth | smoother | linear
    loop: bool = False                # travel last -> first too
    refine: str = "none"              # none | flux (klein upscale of every frame)
    scale: float = 1.5                # flux refine scale
    fixed_noise: bool = False         # conditioning-only travel
    noise_window: float = 1.0         # centered fraction of a transition the noise moves in
    film_seed: int = 42
    steps: Optional[int] = None       # override the keyframes' own values
    guidance: Optional[float] = None


def _film_name(requested: Optional[str]) -> str:
    """Slugify the requested name (or invent one) and never reuse a folder."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "-", (requested or "").strip()).strip("-")[:40]
    if not stem:
        stem = f"travel_{time.strftime('%m%d_%H%M%S')}"
    root = OUTPUTS / "films"
    name, i = stem, 2
    while (root / name).exists():
        name = f"{stem}-{i}"
        i += 1
    return name


@app.post("/api/film")
def api_film(req: FilmRequest) -> JSONResponse:
    """Queue a latent-travel film through the timeline's keyframes.

    Every keyframe is resolved to the image that actually owns the conditioning
    (upscales redirect to their original), sandboxed under outputs/, and checked
    to come from the same backend — mixing them would interpolate tensors that
    don't share a space.
    """
    if req.interp not in FILM_INTERPS:
        raise HTTPException(400, f"bad interp {req.interp!r}")
    if req.easing not in FILM_EASINGS:
        raise HTTPException(400, f"bad easing {req.easing!r}")
    if req.refine not in FILM_REFINES:
        raise HTTPException(400, f"bad refine {req.refine!r}")
    if not (2 <= len(req.images) <= MAX_FILM_KEYS):
        raise HTTPException(400, f"a film needs 2–{MAX_FILM_KEYS} keyframes "
                                 f"(got {len(req.images)})")
    if not (1 <= req.fps <= 60):
        raise HTTPException(400, "fps must be in 1–60")
    if not (1 <= req.frames_per <= 480):
        raise HTTPException(400, "frames between keyframes must be in 1–480")
    if not (0.05 <= req.noise_window <= 1.0):
        raise HTTPException(400, "noise window must be in 0.05–1.0")
    if not (0.0 < req.scale <= 3.0):
        raise HTTPException(400, "scale must be in (0, 3]")
    for dim, val in (("height", req.height), ("width", req.width)):
        if val is not None and (not (64 <= val <= 2048) or val % 8):
            raise HTTPException(400, f"{dim} must be a multiple of 8 in 64–2048")
    segments = len(req.images) if req.loop else len(req.images) - 1
    frames = segments * req.frames_per + 1
    if frames > MAX_FILM_FRAMES:
        raise HTTPException(400, f"{frames} frames is over the {MAX_FILM_FRAMES} "
                                 f"cap — lower 'frames between keyframes'")

    rels, backends = [], set()
    for raw in req.images:
        src = _explorable_source(_resolve_output_image(raw))
        rels.append(src.relative_to(OUTPUTS.resolve()).as_posix())
        b = "sd15"
        for cand in ("flux2", "krea2", "sdxl", "sd2"):
            if f"anarchy_{cand}_" in src.name:
                b = cand
                break
        backends.add(b)
    if len(backends) > 1:
        raise HTTPException(400, f"keyframes mix backends ({', '.join(sorted(backends))})"
                                 " — their conditioning tensors aren't compatible")
    backend = backends.pop()
    if backend not in FILM_BACKENDS:
        raise HTTPException(400, f"{backend} images can't be filmed yet "
                                 f"(supported: {', '.join(sorted(FILM_BACKENDS))})")

    name = _film_name(req.name)
    python = FLUX_PYTHON if req.refine == "flux" else python_for(backend)
    argv = [python, "-u", "scripts/morph_film.py", "--name", name,
            "--fps", str(req.fps), "--frames-per", str(req.frames_per),
            "--interp", req.interp, "--easing", req.easing,
            "--refine", req.refine, "--noise-window", str(req.noise_window),
            "--film-seed", str(int(req.film_seed))]
    if req.loop:
        argv += ["--loop"]
    if req.fixed_noise:
        argv += ["--fixed-noise"]
    if req.refine == "flux":
        argv += ["--scale", str(req.scale)]
    if req.steps is not None:
        argv += ["--steps", str(int(req.steps))]
    if req.guidance is not None:
        argv += ["--guidance", str(req.guidance)]
    if req.height:
        argv += ["--height", str(int(req.height))]
    if req.width:
        argv += ["--width", str(int(req.width))]
    argv += ["--images", *rels]

    label = (f"film · {name} · {len(rels)} keys × {req.frames_per} · "
             f"{frames}f @ {req.fps}fps · {req.interp}")
    job = RUNNER.submit("film", label, argv)
    return JSONResponse({"job_id": job.id, "label": label, "name": name,
                         "frames": frames})


class FilmDeleteRequest(BaseModel):
    dir: str


@app.post("/api/films/delete")
def api_films_delete(req: FilmDeleteRequest) -> JSONResponse:
    if "/" in req.dir or ".." in req.dir or not req.dir:
        raise HTTPException(400, "bad film dir")
    d = (OUTPUTS / "films" / req.dir).resolve()
    if (OUTPUTS / "films").resolve() not in d.parents or not d.is_dir():
        raise HTTPException(404, "not found")
    import shutil
    shutil.rmtree(d)
    return JSONResponse({"deleted": req.dir})


@app.get("/api/films")
def api_films() -> JSONResponse:
    """List rendered morph films (outputs/films/<name>/<name>.mp4 + manifest)."""
    films = []
    root = OUTPUTS / "films"
    if root.is_dir():
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            for mp4 in sorted(d.glob("*.mp4")):
                info = {}
                mj = d / "film.json"
                if mj.is_file():
                    try:
                        info = json.loads(mj.read_text())
                    except Exception:
                        pass
                st = mp4.stat()
                films.append({
                    "name": mp4.stem, "dir": d.name,
                    "rel": str(mp4.relative_to(OUTPUTS)),
                    "mtime": st.st_mtime, "size": st.st_size,
                    "frames": info.get("frames"), "fps": info.get("fps"),
                    "keyframes": info.get("keyframes", []),
                    "refine": info.get("refine"),
                    "interp": info.get("interp"), "easing": info.get("easing"),
                    "loop": info.get("loop"), "backend": info.get("backend"),
                    "duration": info.get("duration"),
                })
    films.sort(key=lambda f: f["mtime"], reverse=True)
    return JSONResponse(films)


@app.get("/api/config")
def api_config() -> JSONResponse:
    return JSONResponse({
        "python": PYTHON,
        "sd15_ckpt": SD15_CKPT,
        "sd15_ckpt_exists": Path(SD15_CKPT).exists(),
        "sd2_ckpt": SD2_CKPT,
        "sd2_ckpt_exists": Path(SD2_CKPT).exists(),
        "sdxl_models": SDXL_MODELS,
        # Hand-picked checkpoints (webui/model_config.json) — these override the
        # env-var defaults above, so the sidebar hints have to know about them.
        "picked_models": load_model_config(),
        # The house sd15 CFG negative, so the sidebar can show the actual text
        # it will sample against instead of just naming the mode.
        "sd15_negative": SD15_NEGATIVE,
        "init_dir": str(INIT_DIR),
        "init_count": init_images_count(),
        "init_folders": init_folders(),
        "repo": str(REPO),
        # The fixed seed panel, so the sidebar can name the one flag pair that
        # makes a batch comparable to every other comparative batch.
        "seed_panel": {
            "seed": labelset.SEED_PANEL_SEED,
            "n": labelset.SEED_PANEL_N,
            "seeds": list(labelset.SEED_PANEL),
        },
        "labels_file": str(LABELS_FILE),
    })


# --------------------------------------------------------------------------- #
# Model picker
# --------------------------------------------------------------------------- #
def _model_row(backend: str, picked: dict) -> dict:
    sel = picked.get(backend)
    default = MODEL_DEFAULTS[backend]
    effective = sel or default
    p = Path(effective)
    # An HF repo id ("stabilityai/...") is neither a file nor a folder here —
    # it resolves out of the HF cache at load time, so "exists" is unknowable.
    is_local = sel is not None or p.is_absolute()
    return {
        "backend": backend,
        "selected": sel,
        "default": default,
        "effective": effective,
        "name": p.name if is_local else effective,
        "kind": path_kind(p) if is_local else "repo",
        "exists": p.exists() if is_local else None,
    }


@app.get("/api/model")
def api_model_get() -> JSONResponse:
    with _model_cfg_lock:
        picked = load_model_config()
    return JSONResponse({
        "backends": {b: _model_row(b, picked) for b in sorted(BACKENDS)},
        "config_file": str(MODEL_CONFIG_FILE),
        "native_picker": native_picker_tool(),
        "roots": [{"name": r.name or str(r), "path": str(r)} for r in browse_roots()],
    })


class ModelSelectRequest(BaseModel):
    backend: str
    path: Optional[str] = None       # None / "" clears back to the default


@app.post("/api/model")
def api_model_set(req: ModelSelectRequest) -> JSONResponse:
    if req.backend not in BACKENDS:
        raise HTTPException(400, f"unknown backend {req.backend!r}")
    with _model_cfg_lock:
        picked = load_model_config()
        if req.path and req.path.strip():
            picked[req.backend] = str(validate_model_path(req.path))
        else:
            picked.pop(req.backend, None)
        save_model_config(picked)
    return JSONResponse(_model_row(req.backend, picked))


# --------------------------------------------------------------------------- #
# Base-distribution picker
# --------------------------------------------------------------------------- #
@app.get("/api/dist")
def api_dist_get(backend: str = "sd15", model: Optional[str] = None) -> JSONResponse:
    """The distribution this backend currently samples from."""
    if backend not in BACKENDS:
        raise HTTPException(400, f"unknown backend {backend!r}")
    with _dist_cfg_lock:
        cfg = load_dist_config()
    return JSONResponse({
        **current_dist(backend, model, cfg),
        "config_file": str(DIST_CONFIG_FILE),
        "default_prompts": DEFAULT_PROMPTS,
    })


@app.get("/api/dist/probe")
def api_dist_probe(backend: str, path: str,
                   model: Optional[str] = None) -> JSONResponse:
    """What picking this file would mean: where its latents live, and whether
    they've been encoded with the checkpoint that's currently active."""
    if backend not in BACKENDS:
        raise HTTPException(400, f"unknown backend {backend!r}")
    kind = dist_kind_for(path)
    p = resolve_dist_file(path, kind)
    return JSONResponse(describe_dist(backend, kind, str(p), model))


class DistSelectRequest(BaseModel):
    backend: str
    kind: str = "base"               # base | evolved | prompts | file
    path: Optional[str] = None       # required for prompts / file
    model: Optional[str] = None      # sdxl repo key (tags which fit is meant)


@app.post("/api/dist")
def api_dist_set(req: DistSelectRequest) -> JSONResponse:
    """Persist the pick. Only a *ready* distribution can be selected — an
    unencoded corpus has to go through /api/dist/encode first."""
    if req.backend not in BACKENDS:
        raise HTTPException(400, f"unknown backend {req.backend!r}")
    if req.kind not in DIST_KINDS:
        raise HTTPException(400, f"unknown distribution kind {req.kind!r}")
    path = (
        str(resolve_dist_file(req.path or "", req.kind))
        if req.kind in ("prompts", "file") else None
    )
    row = describe_dist(req.backend, req.kind, path, req.model)
    if not row["ready"]:
        missing = ", ".join(f["path"] for f in row["files"] if not f["exists"])
        raise HTTPException(400, f"not encoded yet (missing {missing})")
    with _dist_cfg_lock:
        cfg = load_dist_config()
        cfg[req.backend] = {"kind": req.kind, "path": path}
        save_dist_config(cfg)
    return JSONResponse(row)


class DistEncodeRequest(BaseModel):
    backend: str
    path: str                        # the .txt corpus to encode
    model: Optional[str] = None
    components: Optional[int] = None
    batch_size: Optional[int] = None  # prompts per forward pass (default 8)


@app.post("/api/dist/encode")
def api_dist_encode(req: DistEncodeRequest) -> JSONResponse:
    """Queue the encode pass for a prompt corpus (the same mine job the sidebar
    runs), writing its latents beside the .txt under the active checkpoint."""
    if req.backend not in BACKENDS:
        raise HTTPException(400, f"unknown backend {req.backend!r}")
    p = resolve_dist_file(req.path, "prompts")
    try:
        n = sum(1 for line in p.read_text(errors="replace").splitlines()
                if line.strip() and not line.lstrip().startswith("#"))
    except OSError as exc:
        raise HTTPException(400, f"unreadable corpus: {exc}")
    if n == 0:
        raise HTTPException(400, f"no prompts in {p.name}")
    out = dist_base_for(req.backend, "prompts", str(p), req.model)
    argv = [python_for(req.backend), "-u",
            *mine_argv(req.backend, str(p), out, req.components, req.model,
                       req.batch_size)]
    label = f"encode · {req.backend} · {p.name} ({n} prompts)"
    job = RUNNER.submit("mine", label, argv)
    return JSONResponse({"job_id": job.id, "label": label, "out": out})


# --------------------------------------------------------------------------- #
# Selection fits — a distribution fitted to the latents of images you picked
# --------------------------------------------------------------------------- #
# The other half of the base-distribution story. A mined fit answers "what does
# this corpus of PROMPTS look like"; a selection fit answers "what do the images
# I KEPT look like", using the .npz conditioning every generated image already
# carries. No text encoder, no GPU — it is a stack-and-fit over files on disk,
# which is why this can be a couple of seconds rather than a mining run.
#
# Everything here is selection plumbing: which images are candidates, which the
# user picked, and handing that list to scripts/fit_selection.py. The fit itself
# (and the naming rules the picker then reads back) lives in
# semantic_anarchy/selection_fit.py.
FIT_DIR = REPO / fitset.FIT_DIR
#: Bound on one selection. Well past any hand-picked set; exists so a runaway
#: client can't ask the server to stack 100k latents into RAM.
MAX_FIT_IMAGES = 4000
#: How the candidate list may be ordered.
FIT_ORDERS = ("new", "old", "score", "distance")
#: Which images may be fitted, by label score.
FIT_SCORED = ("any", "labeled", "unlabeled")


def _fit_rows(filters: dict, since: Optional[float], until: Optional[float],
              starred: bool, scored: str, min_score: Optional[int],
              max_score: Optional[int], latents_only: bool = True) -> list:
    """Candidate images for a fit, with their label score and star.

    Deliberately the SAME index the labeling queue is built from
    (:func:`_label_index`), so "everything from experiment E07 rendered in the
    last 24h" means one thing in both places. What this adds on top is the label
    score (the point of labeling being to select with it) and the star.
    """
    scores = labeled_scores()
    favs = load_favs()
    out = []
    for row in _select(_label_index(), filters, since, until):
        if latents_only and not row.get("latents"):
            continue
        fav = row["rel"] in favs
        if starred and not fav:
            continue
        score = scores.get(row["rel"])
        if scored == "labeled" and score is None:
            continue
        if scored == "unlabeled" and score is not None:
            continue
        # A score threshold only ever admits scored images — "8 and up" cannot
        # sensibly include the ones nobody has judged.
        if min_score is not None and (score is None or score < min_score):
            continue
        if max_score is not None and (score is None or score > max_score):
            continue
        out.append({**row, "score": score, "fav": fav})
    return out


@app.get("/api/fit/candidates")
def api_fit_candidates(backend: Optional[str] = None, experiment: Optional[str] = None,
                       ckpt: Optional[str] = None, folder: Optional[str] = None,
                       size: Optional[str] = None, kind: Optional[str] = None,
                       sampler: Optional[str] = None,
                       since: Optional[float] = None, until: Optional[float] = None,
                       starred: bool = False, scored: str = "any",
                       min_score: Optional[int] = None,
                       max_score: Optional[int] = None,
                       order: str = "new", limit: int = 600) -> JSONResponse:
    """Which images match a filter, and how many — the `n` in "fit on n samples".

    ``total`` is the whole match; ``rows`` is capped so a 10k-image filter still
    renders. The count the button shows is over what the user actually selected,
    which the client tracks — this is what fills the pool to select *from*.
    """
    if order not in FIT_ORDERS:
        raise HTTPException(400, f"bad order {order!r}")
    if scored not in FIT_SCORED:
        raise HTTPException(400, f"bad scored {scored!r}")
    if backend is not None and backend and backend not in BACKENDS:
        raise HTTPException(400, f"unknown backend {backend!r}")
    filters = {
        "experiment": (experiment or "").strip(),
        "backend": (backend or "").strip(),
        "ckpt": (ckpt or "").strip(),
        "folder": (folder or "").strip(),
        "size": (size or "").strip(),
        "kind": (kind or "").strip(),
        "sampler": (sampler or "").strip(),
    }
    rows = _fit_rows(filters, since, until, starred, scored, min_score, max_score)
    if order == "old":
        rows.sort(key=lambda r: r["mtime"])
    elif order == "score":
        rows.sort(key=lambda r: (r["score"] if r["score"] is not None else -1,
                                 r["mtime"]), reverse=True)
    elif order == "distance":
        rows.sort(key=lambda r: (r["distance"] if r["distance"] is not None else -1),
                  reverse=True)
    else:
        rows.sort(key=lambda r: r["mtime"], reverse=True)
    capped = max(1, min(2000, limit))
    return JSONResponse({
        "total": len(rows),
        "shown": min(len(rows), capped),
        "rows": rows[:capped],
        "backends": sorted({r["backend"] for r in rows if r["backend"]}),
    })


def _fit_backend_of(rel: str) -> Optional[str]:
    """Which backend an image came from (sidecar first, filename second)."""
    b = sidecar_for(rel).get("backend")
    if b in BACKENDS:
        return b
    for name in BACKENDS:
        if f"anarchy_{name}_" in Path(rel).name:
            return name
    return None


def _fit_row(base: Path, backend: Optional[str]) -> dict:
    """One saved fit, told what the picker needs: is it there, and what is in it."""
    man = fitset.read_manifest(base) or {}
    b = backend or man.get("backend")
    files = [str(f) for f in (dist_paths.dist_files(base, b) if b else [])]
    meta = dist_paths.dist_meta(base, b) if b else None
    return {
        "name": man.get("name") or base.name,
        "base": str(base),
        "backend": man.get("backend"),
        "created": man.get("created"),
        "n_samples": man.get("n_samples"),
        "note": man.get("note"),
        "models": man.get("models") or [],
        "ready": bool(files) and all(Path(f).is_file() for f in files),
        "files": files,
        "meta": meta,
    }


@app.get("/api/fit/list")
def api_fit_list(backend: Optional[str] = None) -> JSONResponse:
    """Every saved selection fit, newest first (the picker's third section)."""
    if backend is not None and backend and backend not in BACKENDS:
        raise HTTPException(400, f"unknown backend {backend!r}")
    if not FIT_DIR.is_dir():
        return JSONResponse([])
    rows = [_fit_row(Path(str(m)[: -len(fitset.MANIFEST_SUFFIX)]), backend or None)
            for m in FIT_DIR.glob(f"*{fitset.MANIFEST_SUFFIX}")]
    rows.sort(key=lambda r: r.get("created") or 0, reverse=True)
    return JSONResponse(rows)


class FitRequest(BaseModel):
    name: str
    rels: list[str] = []              # outputs-relative image paths
    backend: Optional[str] = None     # None = inferred from the selection
    components: Optional[int] = None  # None/0 = the selection's full N-1 rank
    note: Optional[str] = None
    overwrite: bool = False


def _fit_target(name: str) -> Path:
    """Sandbox a fit name to the fits directory (no traversal, no absolutes)."""
    slug = fitset.slug_name(name)
    if not slug:
        raise HTTPException(400, "give the fit a name (letters, digits, - and _)")
    target = (FIT_DIR / slug).resolve()
    if FIT_DIR.resolve() not in target.parents:
        raise HTTPException(400, f"bad fit name {name!r}")
    return target


@app.post("/api/fit")
def api_fit(req: FitRequest) -> JSONResponse:
    """Fit a distribution to the latents of the picked images.

    The selection travels as a JSON file rather than argv — a few hundred paths
    is normal here — and every path is resolved through the same outputs/
    sandbox the rest of the API uses before it is written there.
    """
    if req.backend is not None and req.backend and req.backend not in BACKENDS:
        raise HTTPException(400, f"unknown backend {req.backend!r}")
    rels = list(dict.fromkeys(r for r in req.rels if r))   # de-dup, keep order
    if not rels:
        raise HTTPException(400, "nothing selected")
    if len(rels) > MAX_FIT_IMAGES:
        raise HTTPException(400, f"{len(rels)} images selected — the cap is "
                                 f"{MAX_FIT_IMAGES}")
    target = _fit_target(req.name)
    base = str(target.relative_to(REPO)) if REPO in target.parents else str(target)

    paths, backends = [], {}
    for rel in rels:
        p = _resolve_output_image(rel)
        paths.append(str(p))
        b = _fit_backend_of(p.relative_to(OUTPUTS.resolve()).as_posix())
        if b:
            backends[b] = backends.get(b, 0) + 1
    backend = req.backend or (max(backends, key=backends.get) if backends else None)
    if not backend:
        raise HTTPException(400, "could not tell which backend these images are "
                                 "from — pick one explicitly")
    if len(backends) > 1:
        # Conditioning shapes differ per backend, so the minority would be
        # dropped by the fit anyway. Say so here instead of in a job log.
        others = ", ".join(f"{k}×{v}" for k, v in backends.items() if k != backend)
        raise HTTPException(400, f"the selection mixes backends ({others} alongside "
                                 f"{backend}×{backends[backend]}) — a fit is per "
                                 f"backend, so filter to one first")
    if len(paths) < fitset.MIN_SAMPLES:
        raise HTTPException(400, f"{len(paths)} image(s) selected — a fit needs at "
                                 f"least {fitset.MIN_SAMPLES}")

    existing = dist_paths.dist_files(target, backend)
    if not req.overwrite and any(f.is_file() for f in existing):
        raise HTTPException(409, f"“{target.name}” already exists — rename it or "
                                 f"confirm the overwrite")

    FIT_DIR.mkdir(parents=True, exist_ok=True)
    sources = target.with_name(target.name + ".sources.json")
    sources.write_text(json.dumps(paths, indent=1))

    argv = [PYTHON, "-u", "scripts/fit_selection.py",
            "--backend", backend, "--name", target.name,
            "--dir", str(FIT_DIR.relative_to(REPO)),
            "--from-file", str(sources)]
    if req.components:
        argv += ["--components", str(int(req.components))]
    if req.note:
        argv += ["--note", _clean_prompt(req.note)]
    label = f"fit · {backend} · {target.name} ({len(paths)} images)"
    job = RUNNER.submit("fit", label, argv)
    return JSONResponse({
        "job_id": job.id, "label": label, "name": target.name,
        "base": base, "backend": backend, "n": len(paths),
        # What to hand /api/dist once the job lands, so the client can select
        # the new fit without knowing the naming stack.
        "file": str(dist_paths.dist_files(target, backend)[0]),
    })


class FitDeleteRequest(BaseModel):
    name: str


@app.post("/api/fit/delete")
def api_fit_delete(req: FitDeleteRequest) -> JSONResponse:
    """Delete a saved fit (its .npz set, meta, manifest and source list)."""
    target = _fit_target(req.name)
    removed = []
    # Which backend wrote it isn't recorded in the *filenames* the caller knows,
    # so sweep every backend's naming stack — a name only ever holds one fit.
    cands = [target.with_name(target.name + suffix)
             for suffix in (fitset.MANIFEST_SUFFIX, ".sources.json")]
    for b in BACKENDS:
        for f in dist_paths.dist_files(target, b):
            cands += [f, f.with_suffix(".meta.json")]
    for f in cands:
        try:
            if f.is_file():
                f.unlink()
                removed.append(str(f))
        except OSError as exc:
            raise HTTPException(500, f"could not delete {f.name}: {exc}")
    if not removed:
        raise HTTPException(404, f"no fit named {target.name!r}")
    return JSONResponse({"deleted": removed})


class BrowseRequest(BaseModel):
    mode: str = "file"               # file | folder
    start: Optional[str] = None


@app.post("/api/model/native")
def api_model_native(req: BrowseRequest) -> JSONResponse:
    """Open a real OS file dialog *on the machine running this server*. Useless
    from a remote tailnet device — the UI falls back to /api/fs there."""
    if req.mode not in ("file", "folder"):
        raise HTTPException(400, f"unknown mode {req.mode!r}")
    path = run_native_picker(req.mode, req.start)
    if path is None:
        return JSONResponse({"cancelled": True, "path": None})
    return JSONResponse({"cancelled": False, "path": str(validate_model_path(path))})


@app.get("/api/fs")
def api_fs(path: Optional[str] = None, pick: str = "model",
           backend: Optional[str] = None, model: Optional[str] = None) -> JSONResponse:
    """One directory listing for the in-browser pickers, sandboxed to
    browse_roots().

    ``pick=model`` (default) lists subfolders + single-file checkpoints;
    ``pick=dist`` lists subfolders + prompt corpora (.txt) and saved fits
    (.npz). In dist mode a ``backend`` marks each corpus ``ready`` when its
    latents for the active checkpoint already exist — so "needs encoding" is
    visible while browsing, before anything is picked.
    """
    if pick not in ("model", "dist"):
        raise HTTPException(400, f"unknown pick mode {pick!r}")
    if backend is not None and backend not in BACKENDS:
        raise HTTPException(400, f"unknown backend {backend!r}")
    here = resolve_browse_path(path)
    roots = browse_roots()
    entries = []
    try:
        children = sorted(here.iterdir(), key=lambda p: p.name.lower())
    except PermissionError:
        raise HTTPException(403, f"not readable: {here}")
    for c in children:
        if c.name.startswith("."):
            continue
        try:
            if c.is_dir():
                entries.append({"name": c.name, "path": str(c), "dir": True,
                                "kind": None if pick == "dist" else path_kind(c),
                                "size": None})
            elif pick == "dist" and c.suffix.lower() in DIST_EXTS:
                row = {"name": c.name, "path": str(c), "dir": False,
                       "kind": dist_kind_for(str(c)), "size": c.stat().st_size}
                if backend and row["kind"] == "prompts":
                    base = dist_base_for(backend, "prompts", str(c), model)
                    row["ready"] = dist_paths.dist_ready(_abs_base(base), backend)
                entries.append(row)
            elif pick == "model" and c.suffix.lower() in CKPT_EXTS:
                entries.append({"name": c.name, "path": str(c), "dir": False,
                                "kind": "ckpt", "size": c.stat().st_size})
        except OSError:
            continue
    entries.sort(key=lambda e: (not e["dir"], e["name"].lower()))
    at_root = any(here == r for r in roots)
    return JSONResponse({
        "path": str(here),
        "parent": None if at_root else str(here.parent),
        "kind": path_kind(here),
        "entries": entries,
        "roots": [{"name": r.name or str(r), "path": str(r)} for r in roots],
    })


# --------------------------------------------------------------------------- #
# Frontend (single inline page; vanilla JS, polls /api/state + /api/images)
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Semantic Anarchy — Explorer</title>
<style>
  :root { --bg:#0d0e12; --panel:#16181f; --panel2:#1d2029; --line:#2a2e3a;
          --ink:#e7e9ee; --dim:#9aa0ad; --accent:#e0533d; --ok:#4caf50;
          --run:#e0a13d; --err:#e0533d; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  header { padding:14px 20px; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:14px; }
  header h1 { font-size:17px; margin:0; letter-spacing:.3px; }
  header .sub { color:var(--dim); font-size:12px; }
  .layout { display:grid; grid-template-columns:320px 1fr; gap:0; min-height:calc(100vh - 52px); }
  .side { border-right:1px solid var(--line); padding:16px; overflow-y:auto;
          max-height:calc(100vh - 52px); position:sticky; top:0; }
  .main { padding:16px 20px; overflow-y:auto; max-height:calc(100vh - 52px); }
  h2 { font-size:12px; text-transform:uppercase; letter-spacing:.8px;
       color:var(--dim); margin:18px 0 8px; }
  label { display:block; font-size:12px; color:var(--dim); margin:10px 0 3px; }
  input, select { width:100%; background:var(--panel2); color:var(--ink);
                  border:1px solid var(--line); border-radius:7px; padding:7px 9px; font:inherit; }
  .row { display:grid; grid-template-columns:1fr 1fr; gap:8px; }
  .seg { display:flex; gap:6px; flex-wrap:wrap; }
  .seg button { flex:1; }
  button { background:var(--panel2); color:var(--ink); border:1px solid var(--line);
           border-radius:7px; padding:8px 10px; font:inherit; cursor:pointer; }
  button:hover { border-color:#3a4150; }
  button.sel { background:var(--accent); border-color:var(--accent); color:#fff; }
  button.run { background:var(--accent); border-color:var(--accent); color:#fff;
               width:100%; padding:11px; font-weight:600; margin-top:14px; }
  button.run:disabled { opacity:.5; cursor:not-allowed; }
  .hint { font-size:11px; color:var(--dim); margin-top:4px; }
  .warn { color:var(--err); }
  .jobs { display:flex; flex-direction:column; gap:6px; }
  .job { background:var(--panel); border:1px solid var(--line); border-radius:8px;
         padding:8px 10px; cursor:pointer; }
  .job.active { border-color:var(--accent); }
  .job .t { display:flex; justify-content:space-between; gap:8px; }
  .job .lbl { font-size:12px; }
  .badge { font-size:10px; padding:1px 7px; border-radius:20px; text-transform:uppercase;
           letter-spacing:.5px; white-space:nowrap; }
  .b-running{background:var(--run);color:#000;} .b-done{background:#2e7d32;color:#fff;}
  .b-error{background:var(--err);color:#fff;} .b-queued{background:#3a4150;color:#cfd3dc;}
  .b-cancelled{background:#555;color:#fff;}
  pre.log { background:#07080b; border:1px solid var(--line); border-radius:8px;
            padding:12px; height:240px; overflow:auto; font:12px/1.5 ui-monospace,Menlo,monospace;
            white-space:pre-wrap; color:#cfd3dc; margin:0; }
  .tabs { display:flex; gap:6px; margin:6px 0 12px; flex-wrap:wrap; }
  .tabs button.sel { background:var(--accent); border-color:var(--accent); }
  .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:12px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:10px;
          overflow:hidden; }
  .card img { width:100%; display:block; background:#000; cursor:zoom-in; }
  .card .meta { padding:6px 9px; font-size:11px; color:var(--dim);
                display:flex; justify-content:space-between; gap:6px;
                flex-wrap:wrap; row-gap:4px; }
  .card .meta .nm { flex:1 1 100%; }
  .card .meta .nm { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .sheet img { cursor:zoom-in; }
  .empty { color:var(--dim); padding:30px; text-align:center; }
  #lightbox { position:fixed; inset:0; background:rgba(0,0,0,.92); display:none;
              align-items:center; justify-content:center; z-index:50; padding:20px; }
  #lightbox img { max-width:96vw; max-height:94vh; }
  .statusline { font-size:12px; color:var(--dim); margin-top:6px; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }
  .dot.idle{background:#3a4150;} .dot.busy{background:var(--run);}
</style>
</head>
<body>
<header>
  <h1>Semantic&nbsp;Anarchy</h1>
  <span class="sub">promptless explorer · <span id="host"></span></span>
  <span class="statusline" style="margin-left:auto"><span id="busydot" class="dot idle"></span><span id="busytxt">idle</span></span>
</header>

<div class="layout">
  <!-- ------------------------------------------------ controls ---------- -->
  <aside class="side">
    <h2>Action</h2>
    <div class="seg" id="actions">
      <button data-act="generate" class="sel">Generate</button>
      <button data-act="temp_sweep">Temp sweep</button>
      <button data-act="sampler_sweep">Sampler sweep</button>
      <button data-act="mine">Mine</button>
    </div>

    <h2>Model</h2>
    <div class="seg" id="backends">
      <button data-be="sd15" class="sel">SD 1.5</button>
      <button data-be="sd2">SD 2.1</button>
      <button data-be="sdxl">SDXL</button>
      <button data-be="flux2">FLUX.2</button>
      <button data-be="krea2">Krea 2</button>
    </div>
    <div id="sdxlModelWrap" style="display:none">
      <label>SDXL checkpoint</label>
      <select id="model">
        <option value="sdxl-base-1.0">sdxl-base-1.0 (30 steps, CFG) — recommended</option>
        <option value="sdxl-turbo">sdxl-turbo (1 step, no CFG — fast preview, generic)</option>
      </select>
    </div>
    <div id="ckptHint" class="hint"></div>
    <label title="which learned distribution to sample from">Distribution</label>
    <select id="distSel">
      <option value="base">selected distribution (set in the React UI)</option>
      <option value="evolved">evolved ★ branch (from 🧪)</option>
    </select>

    <div class="genonly">
      <h2>Sampler</h2>
      <select id="sampler">
        <option value="diagonal">diagonal — independent coords (raw)</option>
        <option value="pca">pca — on the corpus manifold (T&gt;1 extrapolates)</option>
        <option value="blend">blend — interpolate diagonal/pca</option>
        <option value="hybrid">hybrid — SLERP two real concepts</option>
      </select>

      <div class="row">
        <div><label>Temperature</label><input id="temperature" type="number" step="0.1" placeholder="1.0"></div>
        <div class="coh"><label>Coherence λ</label><input id="coherence" type="number" step="0.1" placeholder="0.5"></div>
      </div>
      <label title="shell sampling: pin every sample's distance gauge to this ring (overrides how far temperature lands)">Target distance</label>
      <input id="target_distance" type="number" step="0.1" placeholder="off">
      <div class="hint" id="bandHint"></div>
      <label>Sampler / scheduler</label>
      <select id="scheduler">
        <option value="default">default scheduler</option>
        <option value="ddim">DDIM (smooth, for high-step renders)</option>
        <option value="euler">Euler</option>
        <option value="euler_a">Euler ancestral</option>
        <option value="dpm">DPM++ 2M</option>
      </select>
    </div>

    <div class="genimg">
      <div class="row">
        <div class="ngen"><label>Images (n)</label><input id="n" type="number" placeholder="8"></div>
        <div class="nseed"><label>Seed</label><input id="seed" type="number" placeholder="random"></div>
      </div>
      <div class="row">
        <div><label>Steps</label><input id="steps" type="number" placeholder="auto (try 50)"></div>
        <div><label>Guidance</label><input id="guidance" type="number" step="0.5" placeholder="auto"></div>
      </div>
      <label>Aspect ratio</label>
      <select id="aspect">
        <option value="">default (square)</option>
        <option value="1:1">square 1:1</option>
        <option value="3:2">landscape 3:2</option>
        <option value="2:3">portrait 2:3</option>
        <option value="4:3">landscape 4:3</option>
        <option value="3:4">portrait 3:4</option>
        <option value="16:9">wide 16:9</option>
        <option value="9:16">tall 9:16</option>
        <option value="21:9">cinematic 21:9</option>
      </select>
      <div class="hint" id="aspectHint"></div>
      <div class="row">
        <div><label>Width</label><input id="width" type="number" placeholder="auto"></div>
        <div><label>Height</label><input id="height" type="number" placeholder="auto"></div>
      </div>
      <div class="row">
        <div><label title="start from a random good init image">Init folder</label>
          <select id="initFolder"><option value="off">off</option></select></div>
        <div><label>Init mode</label>
          <select id="initMode">
            <option value="img2img">img2img (structure)</option>
            <option value="embedding">image-embedding (content)</option>
          </select></div>
      </div>
      <div class="row">
        <div><label>Strength / scale</label><input id="init_strength" type="number" step="0.05" placeholder="0.7"></div>
        <div></div>
      </div>
      <div class="hint" id="initHint"></div>
    </div>

    <div class="sweeponly" style="display:none">
      <div class="tempsweeponly">
        <label>Temperatures (csv)</label><input id="temps" placeholder="0.5,1.0,1.5,2.0">
      </div>
      <label>Seeds (csv)</label><input id="seeds" placeholder="0,1,2">
    </div>

    <details>
      <summary style="cursor:pointer;color:var(--dim);font-size:12px;margin-top:14px">Advanced</summary>
      <div class="row">
        <div><label>Components</label><input id="components" type="number" placeholder="all"></div>
        <div><label>Truncation σ</label><input id="truncation" type="number" step="0.5" placeholder="off"></div>
      </div>
      <div class="row">
        <div><label title="skip the dominant/standard PCA axes; higher = stranger subjects">Comp-lo (weird axis)</label>
          <input id="comp_lo" type="number" placeholder="0"></div>
        <div><label>Equalize</label>
          <select id="equalize"><option value="">off</option><option value="1">on (express minor axes)</option></select></div>
      </div>
      <div class="hint">For non-standard subjects: sampler <b>pca</b>, comp-lo ~40–200, equalize <b>on</b>, temp ~1.1–1.4.</div>
      <label>neg-mode</label>
      <select id="neg_mode">
        <option value="">auto</option><option value="text">text (house negative)</option>
        <option value="mean">mean</option>
        <option value="empty">empty</option><option value="zeros">zeros</option>
      </select>
      <label>negative prompt <span class="hint">(sd15/sd2)</span></label>
      <textarea id="negative" rows="4" spellcheck="false"
                style="width:100%;font:inherit;font-size:12px"></textarea>
      <div class="hint">Prefilled with the house SD1.5 negative — edit freely.
        Clear the box to fall back to it; use neg-mode <b>empty</b> for no
        negative text at all.</div>
    </details>

    <button class="run" id="runBtn">Run ▶</button>
    <div class="hint" id="runHint"></div>

    <h2>Jobs</h2>
    <div class="jobs" id="jobs"></div>
  </aside>

  <!-- ------------------------------------------------ main -------------- -->
  <main class="main">
    <h2 style="margin-top:0">Job log <span id="logTitle" style="color:var(--ink);text-transform:none;letter-spacing:0"></span></h2>
    <pre class="log" id="log">select or start a job…</pre>

    <h2>Gallery</h2>
    <div class="tabs" id="tabs">
      <button data-tab="generated" class="sel">Generated</button>
      <button data-tab="frontier" title="Pareto front of novelty × your resonance">🎯 Frontier</button>
      <button data-tab="top">🏆 Top rated</button>
      <button data-tab="favorites">★ Favorites</button>
      <button data-tab="temperature">Temp sweeps</button>
      <button data-tab="sampler">Sampler sweeps</button>
      <button data-tab="marginals">Marginals</button>
      <button data-tab="films">🎞 Films</button>
      <select id="sortBy" title="gallery order" style="margin-left:auto;width:150px">
        <option value="new">newest first</option>
        <option value="score">score ↓</option>
        <option value="dist">distance ↓</option>
        <option value="dist_asc">distance ↑</option>
        <option value="nov">novelty ↓</option>
        <option value="res">resonance ↓</option>
      </select>
      <button id="evolveBtn" title="refit a distribution branch around your ★ favorites and sample it">🧪 Evolve ★</button>
      <button id="analyzeBtn" title="embed new images, recompute novelty + retrain the taste model from your ★">🎯 Analyze</button>
      <button id="scoreBtn" title="aesthetic-score all images">Score all ▶</button>
      <button id="wipeBtn" title="delete all non-starred images scored below 5 (unscored images are kept)" style="color:#d98a8a">🧹 Wipe &lt;5</button>
    </div>
    <div id="refineBar" style="display:flex;gap:10px;align-items:flex-end;flex-wrap:wrap;
         margin:0 0 12px;padding:10px 12px;background:var(--panel);border:1px solid var(--line);border-radius:8px">
      <div><label style="margin-top:0">Engine</label>
        <select id="rfEngine" style="width:170px">
          <option value="hires" selected>Same-latent hires</option>
          <option value="flux">FLUX klein</option>
          <option value="sd">SD img2img</option></select></div>
      <div class="rfHires"><label style="margin-top:0"
           title="target = source × this, snapped to a multiple of 16 px">Upscale <b id="rfFactorOut">×2.00</b></label>
        <input id="rfFactor" type="range" min="1" max="3" step="0.05" value="2.0" style="width:150px"></div>
      <div class="rfHires"><label style="margin-top:0"
           title="fraction of the ORIGINAL schedule re-run on the enlarged image, with the same latents and seed">Denoise <b id="rfDenoiseOut">0.30 · last 30%</b></label>
        <input id="rfDenoise" type="range" min="0.05" max="1" step="0.05" value="0.3" style="width:165px"></div>
      <div class="rfOther"><label style="margin-top:0">Upscale ×</label>
        <select id="rfScale" style="width:90px"><option>1.25</option><option selected>1.5</option><option>2.0</option></select></div>
      <div class="rfOther"><label style="margin-top:0">Steps</label>
        <input id="rfSteps" type="number" placeholder="40" style="width:80px"></div>
      <div><label style="margin-top:0">SD mode</label>
        <select id="rfMode" style="width:140px"><option value="tiled" selected>Detail (tiled)</option>
          <option value="single">Standard (1 pass)</option></select></div>
      <div><label style="margin-top:0">Denoise</label>
        <input id="rfStrength" type="number" step="0.05" placeholder="0.45" style="width:90px"></div>
      <div><label style="margin-top:0">Scheduler</label>
        <select id="rfSched" style="width:110px"><option value="ddim" selected>DDIM</option>
          <option value="default">default</option><option value="dpm">DPM++ 2M</option></select></div>
      <div><label style="margin-top:0" title="FLUX engine: instruction given with the reference image">Flux prompt</label>
        <select id="rfPromptSel" style="width:170px">
          <option value="faithful" selected>faithful upscale</option>
          <option value="recreate">creative re-render</option>
          <option value="custom">custom…</option>
        </select></div>
      <div id="rfPromptCustomWrap" style="display:none"><label style="margin-top:0">Custom prompt</label>
        <input id="rfPrompt" placeholder="your instruction" style="width:220px"></div>
      <div style="border-left:1px solid var(--line);align-self:stretch"></div>
      <div><label style="margin-top:0" title="perturbation size for 🧭 Explore (fraction of corpus spread)">Explore radius</label>
        <input id="exRadius" type="number" step="0.05" placeholder="0.3" style="width:90px"></div>
      <div><label style="margin-top:0">Children</label>
        <input id="exN" type="number" placeholder="6" style="width:70px"></div>
      <div><label style="margin-top:0" title="🚶 walk: distance growth per frame (0.15 = +15% further out each step)">Walk step</label>
        <input id="wkStep" type="number" step="0.05" placeholder="0.15" style="width:80px"></div>
      <span class="hint" style="margin:0 0 6px" id="exploreHint"><b>🧭 Explore</b> samples around an image's latent point (hill-climb on taste); <b>🧬 Breed</b>: click on two images to SLERP-cross them.</span>
    </div>
    <div id="gallery"></div>
    <div id="galleryFoot" style="text-align:center;color:var(--dim);font-size:12px;padding:14px 0 30px"></div>
  </main>
</div>

<div id="lightbox">
  <button id="lbPrev" title="previous (←)" style="position:fixed;left:14px;top:50%;transform:translateY(-50%);
    z-index:60;font-size:26px;padding:14px 16px;background:rgba(22,24,31,.75);border:1px solid var(--line);
    border-radius:10px;color:#e7e9ee;cursor:pointer">‹</button>
  <button id="lbNext" title="next (→)" style="position:fixed;right:14px;top:50%;transform:translateY(-50%);
    z-index:60;font-size:26px;padding:14px 16px;background:rgba(22,24,31,.75);border:1px solid var(--line);
    border-radius:10px;color:#e7e9ee;cursor:pointer">›</button>
  <img id="lightimg">
  <div id="lightmeta" style="position:fixed;left:0;right:0;bottom:0;background:rgba(7,8,11,.92);
       border-top:1px solid var(--line);padding:10px 16px;font:12px/1.6 ui-monospace,Menlo,monospace;
       color:#cfd3dc;max-height:38vh;overflow:auto"></div>
</div>

<script>
const $ = s => document.querySelector(s);
let action = "generate", backend = "sd15", tab = "generated";
const PAGE = 200; let shown = PAGE;   // gallery pagination: render this many, +PAGE on scroll
let currentView = [], lightIdx = -1;  // lightbox ←/→ navigation over the current sort order
let selectedJob = null, images = {}, lastBusy = null;

document.getElementById("host").textContent = location.host;

// ---- segmented controls ----
function wireSeg(containerSel, attr, cb) {
  document.querySelectorAll(containerSel + " button").forEach(b => {
    b.onclick = () => {
      document.querySelectorAll(containerSel + " button").forEach(x => x.classList.remove("sel"));
      b.classList.add("sel"); cb(b.dataset[attr]);
    };
  });
}
wireSeg("#actions", "act", a => { action = a; syncForm(); });
wireSeg("#backends", "be", b => { backend = b; syncForm(); });
wireSeg("#tabs", "tab", t => { tab = t; shown = PAGE; $(".main").scrollTop = 0; renderGallery(); });

function syncForm() {
  const sweep = action === "temp_sweep" || action === "sampler_sweep";
  const gen = action === "generate";
  $("#sdxlModelWrap").style.display = backend === "sdxl" ? "" : "none";
  document.querySelectorAll(".genonly").forEach(e => e.style.display = (gen || sweep) ? "" : "none");
  document.querySelectorAll(".genimg").forEach(e => e.style.display = gen ? "" : "none");
  document.querySelectorAll(".sweeponly").forEach(e => e.style.display = sweep ? "" : "none");
  document.querySelectorAll(".tempsweeponly").forEach(e => e.style.display = action === "temp_sweep" ? "" : "none");
  document.querySelectorAll(".coh").forEach(e => e.style.display =
    ($("#sampler").value === "blend") ? "" : "none");
  // sampler_sweep ignores the sampler dropdown (it sweeps all three itself)
  $("#sampler").parentElement.querySelector("label")?.remove?.();
  $("#runBtn").textContent = action === "mine" ? "Mine ▶" : "Run ▶";
  syncModelDefaults();
  if (typeof applyAspect === "function") applyAspect();  // recompute dims for the backend
  refreshCkptHint();
  const c = window._cfg || {};
  const ih = $("#initHint");
  if (ih) ih.innerHTML = (c.init_count > 0)
    ? `${c.init_count} init image(s) across ${(c.init_folders||[]).length} folder(s) in <code>init_images/</code>`
    : `<span class="warn">no init images yet — drop folders/images in <code>${c.init_dir || 'init_images/'}</code></span>`;
}
function populateInitFolders() {
  const sel = $("#initFolder"); if (!sel) return;
  const c = window._cfg || {}; const folders = c.init_folders || [];
  const cur = sel.value;
  let html = `<option value="off">off</option>`;
  if (folders.length) html += `<option value="__any__">any folder (${c.init_count})</option>`;
  for (const f of folders) html += `<option value="${f.path}">${f.name} (${f.count})</option>`;
  sel.innerHTML = html;
  if ([...sel.options].some(o => o.value === cur)) sel.value = cur;  // keep selection
}
$("#sampler").onchange = syncForm;

// Aspect-ratio presets -> width/height sized for the current backend's native
// resolution (sd15 512, sd2 768, sdxl 1024), kept ~1:1 area and multiples of 64.
const NATIVE = {sd15: 512, sd2: 768, sdxl: 1024, flux2: 1024, krea2: 1024};
function applyAspect() {
  const v = $("#aspect").value;
  if (!v) { $("#width").value = ""; $("#height").value = ""; $("#aspectHint").textContent = ""; return; }
  const base = NATIVE[backend] || 1024, A = base * base;
  const [rw, rh] = v.split(":").map(Number), r = rw / rh;
  const round64 = x => Math.max(64, Math.round(x / 64) * 64);
  const w = round64(Math.sqrt(A * r)), h = round64(Math.sqrt(A / r));
  $("#width").value = w; $("#height").value = h;
  $("#aspectHint").textContent = `${w}×${h} (${backend} native ${base})`;
}
$("#aspect").onchange = applyAspect;

// SDXL: reflect the chosen model's step/guidance defaults in the placeholders
// so it's obvious base runs with CFG and turbo doesn't.
const MODEL_DEFAULTS = {"sdxl-base-1.0":{steps:30,guidance:7}, "sdxl-turbo":{steps:1,guidance:0}};
function syncModelDefaults() {
  if (backend !== "sdxl") { $("#steps").placeholder="auto"; $("#guidance").placeholder="auto"; return; }
  const d = MODEL_DEFAULTS[$("#model").value] || {};
  $("#steps").placeholder = "default " + d.steps;
  $("#guidance").placeholder = "default " + d.guidance;
}
$("#model").onchange = syncModelDefaults;

function refreshCkptHint() {
  const el = $("#ckptHint"); const c = window._cfg || {};
  if (backend === "sd15") {
    el.innerHTML = c.sd15_ckpt && !c.sd15_ckpt_exists
      ? `<span class="warn">SD1.5 checkpoint missing: ${c.sd15_ckpt}</span>`
      : `single-file ckpt → <code>--ckpt</code> (512²)`;
  } else if (backend === "sd2") {
    el.innerHTML = c.sd2_ckpt && !c.sd2_ckpt_exists
      ? `<span class="warn">SD2.1 checkpoint missing: ${c.sd2_ckpt}</span>`
      : `single-file 768 v-pred ckpt → <code>--ckpt</code> (768²)`;
  } else if (backend === "flux2") {
    el.textContent = "FLUX.2 klein (flow model, Qwen3 encoder) — mine first, then generate";
  } else if (backend === "krea2") {
    el.textContent = "Krea 2 Raw — use sampler diagonal (T 1.0–1.3) or blend λ0.6–0.7; pure pca looks washed (256-comp mine). Slow.";
  } else {
    el.textContent = "cached HF repo → --model (1024²)";
  }
}

// ---- run ----
function numOrNull(id) { const v = $("#"+id).value.trim(); return v === "" ? null : Number(v); }
$("#runBtn").onclick = async () => {
  const body = {
    action, backend,
    model: $("#model").value,
    sampler: $("#sampler").value,
    temperature: numOrNull("temperature"),
    n: numOrNull("n"), seed: numOrNull("seed"),
    steps: numOrNull("steps"), guidance: numOrNull("guidance"),
    coherence: numOrNull("coherence"), components: numOrNull("components"),
    truncation: numOrNull("truncation"),
    neg_mode: $("#neg_mode").value || null,
    // blank = don't pass --negative at all, i.e. keep the script's own default.
    negative: $("#negative").value.trim() || null,
    dist: $("#distSel").value,
    target_distance: numOrNull("target_distance"),
    temps: $("#temps").value.trim() || null,
    seeds: $("#seeds").value.trim() || null,
    scheduler: $("#scheduler").value || null,
    width: numOrNull("width"), height: numOrNull("height"),
    comp_lo: numOrNull("comp_lo"), equalize: $("#equalize").value === "1",
    init: $("#initFolder").value !== "off",
    init_folder: $("#initFolder").value === "off" ? null : $("#initFolder").value,
    init_mode: $("#initMode").value,
    init_strength: $("#init_strength").value.trim() === "" ? 0.7 : Number($("#init_strength").value),
    ip_scale: $("#init_strength").value.trim() === "" ? 0.7 : Number($("#init_strength").value),
  };
  $("#runHint").textContent = "submitting…";
  try {
    const r = await fetch("/api/run", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) { $("#runHint").innerHTML = `<span class="warn">${j.detail||"error"}</span>`; return; }
    selectedJob = j.job_id;
    $("#runHint").textContent = "queued: " + j.label;
    poll();
  } catch (e) { $("#runHint").innerHTML = `<span class="warn">${e}</span>`; }
};

// ---- polling ----
async function poll() {
  try {
    const st = await (await fetch("/api/state")).json();
    renderJobs(st);
    const busy = st.running !== null;
    $("#busydot").className = "dot " + (busy ? "busy" : "idle");
    $("#busytxt").textContent = busy ? "running job #" + st.running : "idle";
    if (selectedJob == null && st.jobs.length) selectedJob = st.jobs[0].id;
    if (selectedJob != null) loadLog(selectedJob);
    // refresh gallery whenever a job just finished, or periodically
    if (lastBusy === true && !busy) refreshImages();
    lastBusy = busy;
  } catch (e) {}
}
function renderJobs(st) {
  const box = $("#jobs"); box.innerHTML = "";
  st.jobs.slice(0, 12).forEach(j => {
    const d = document.createElement("div");
    d.className = "job" + (j.id === selectedJob ? " active" : "");
    d.innerHTML = `<div class="t"><span class="lbl">#${j.id} ${j.label}</span>
      <span class="badge b-${j.status}">${j.status}</span></div>`;
    d.onclick = () => { selectedJob = j.id; loadLog(j.id); renderJobs(st); };
    box.appendChild(d);
  });
}
async function loadLog(id) {
  try {
    const txt = await (await fetch("/api/log/" + id)).text();
    $("#logTitle").textContent = "#" + id;
    const el = $("#log");
    const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 30;
    el.textContent = txt || "(no output yet)";
    if (atBottom) el.scrollTop = el.scrollHeight;
  } catch (e) {}
}

// ---- gallery ----
async function refreshImages() {
  try { images = await (await fetch("/api/images")).json(); renderGallery(); } catch (e) {}
}
function fmtSize(b) { return b > 1e6 ? (b/1e6).toFixed(1)+"M" : Math.round(b/1e3)+"K"; }
function makeCard(im, sheet) {
  const c = document.createElement("div");
  c.className = "card";
  const src = im.url + "&t=" + Math.floor(im.mtime);
  const isAnarchy = (im.name || "").startsWith("anarchy_");
  const star = `<button class="fav" title="favorite" style="padding:2px 7px;font-size:12px;${im.fav?'color:#ffcf4d;border-color:#ffcf4d':''}">${im.fav?'★':'☆'}</button>`;
  const up = isAnarchy
    ? `<button class="up" title="upscale + more steps" style="padding:2px 8px;font-size:11px;white-space:nowrap">⤴ Upscale</button>` : "";
  const nav = isAnarchy
    ? `<button class="ex" title="explore around this (neighborhood)" style="padding:2px 8px;font-size:11px">🧭</button>` +
      `<button class="wk" title="walk outward toward the periphery from this point" style="padding:2px 8px;font-size:11px">🚶</button>` +
      `<button class="br" title="breed: click this then another image" style="padding:2px 8px;font-size:11px${pendingBreed===im.rel?';background:var(--accent);border-color:var(--accent)':''}">🧬</button>` : "";
  const sc = (im.score != null)
    ? `<span title="aesthetic score" style="color:#7fae7f">${im.score.toFixed(2)}</span>` : "";
  const dd = (im.dist != null)
    ? `<span title="distance from corpus center" style="color:#b48ee0">d${im.dist.toFixed(2)}</span>` : "";
  const nr = (tab === "frontier")
    ? `${im.nov != null ? `<span title="novelty (NN distance in gallery)" style="color:#e0a13d">n${im.nov.toFixed(2)}</span>` : ""}` +
      `${im.res != null ? `<span title="resonance P(you'd star it)" style="color:#e07ab8">r${im.res.toFixed(2)}</span>` : ""}` : "";
  c.innerHTML = `<img loading="lazy" src="${src}">
    <div class="meta"><span class="nm">${im.name}</span>${sc}${dd}${nr}${star}${nav}${up}<span>${fmtSize(im.size)}</span></div>`;
  c.querySelector("img").onclick = () => openLight(src, im.rel, im.score);
  const ub = c.querySelector(".up");
  if (ub) ub.onclick = () => refineImage(im.rel || im.name, ub);
  const ex = c.querySelector(".ex");
  if (ex) ex.onclick = () => exploreImage(im.rel, ex);
  const wk = c.querySelector(".wk");
  if (wk) wk.onclick = () => walkImage(im.rel, wk);
  const br = c.querySelector(".br");
  if (br) br.onclick = () => breedClick(im.rel);
  c.querySelector(".fav").onclick = () => toggleFav(im.rel, !im.fav);
  if (sheet) c.style.marginBottom = "16px";
  return c;
}
async function renderFilms(g) {
  let films = [];
  try { films = await (await fetch("/api/films")).json(); } catch(e){}
  if (!films.length) {
    g.innerHTML = `<div class="empty">no films yet — render one with scripts/morph_film.py</div>`;
    return;
  }
  g.innerHTML = films.map(f => `
    <div style="background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;max-width:820px;margin-bottom:14px">
      <div style="margin-bottom:8px"><span style="color:#e0a13d">🎞 ${f.name}</span>
        <span style="color:#9aa0ad;font-size:11px;margin-left:10px">${f.frames ?? "?"} frames · ${f.fps ?? "?"} fps · ${f.refine === "flux" ? "flux-refined" : "base"}</span>
        <button onclick="deleteFilm(this,'${f.dir}')" title="delete this film's whole folder (all its variants + frames)" style="float:right;color:#d98a8a;padding:2px 8px;font-size:11px">🗑 delete</button></div>
      <video controls loop style="width:100%;border-radius:8px;background:#000"
             src="/img?path=${encodeURIComponent(f.rel)}"></video>
      ${f.keyframes.length ? `<div style="color:#9aa0ad;font-size:11px;margin-top:6px">keyframes: ${f.keyframes.map(k => `<a href="#" style="color:#6fb3e0" onclick="openParent('${k.split("/").pop()}');return false;">${k.split("/").pop()}</a>`).join(" → ")}</div>` : ""}
    </div>`).join("");
}

async function deleteFilm(btn, dir){
  if (!confirm(`Delete film folder "${dir}" (all variants + frames)? This cannot be undone.`)) return;
  btn.disabled = true;
  await fetch("/api/films/delete", {method: "POST",
    headers: {"Content-Type": "application/json"}, body: JSON.stringify({dir: dir})});
  renderGallery();
}

function renderGallery() {
  const g = $("#gallery"); const list = images[tab] || [];
  if (tab === "films") {
    $("#refineBar").style.display = "none";
    g.className = "";
    $("#galleryFoot").textContent = "";
    renderFilms(g);
    return;
  }
  const upscalable = (tab === "generated" || tab === "favorites" || tab === "top" || tab === "frontier");
  $("#refineBar").style.display = upscalable ? "flex" : "none";
  if (!list.length) {
    const msg = tab === "favorites"
      ? "no favorites yet — click the ☆ on any image to add it."
      : tab === "top"
      ? "nothing scored yet — click “Score all ▶” to rank by aesthetic."
      : tab === "frontier"
      ? "no analysis yet — click “🎯 Analyze” to compute novelty + resonance."
      : `no ${tab} artifacts yet — run something.`;
    g.innerHTML = `<div class="empty">${msg}</div>`; $("#galleryFoot").textContent = ""; return;
  }
  const sheet = (tab === "temperature" || tab === "sampler" || tab === "marginals");
  g.className = sheet ? "sheet" : "grid";
  const prev = $(".main").scrollTop;            // preserve scroll across re-render
  if (shown > list.length) shown = Math.max(PAGE, Math.ceil(list.length / PAGE) * PAGE);
  // gallery ordering (newest is the server default)
  let view = list;
  const sb = $("#sortBy") ? $("#sortBy").value : "new";
  if (sb === "score") view = [...list].sort((a,b) => (b.score ?? -1) - (a.score ?? -1));
  else if (sb === "dist") view = [...list].sort((a,b) => (b.dist ?? -1) - (a.dist ?? -1));
  else if (sb === "dist_asc") view = [...list].sort((a,b) => (a.dist ?? 1e9) - (b.dist ?? 1e9));
  else if (sb === "nov") view = [...list].sort((a,b) => (b.nov ?? -1) - (a.nov ?? -1));
  else if (sb === "res") view = [...list].sort((a,b) => (b.res ?? -1) - (a.res ?? -1));
  currentView = view;
  g.innerHTML = "";
  const count = Math.min(shown, view.length);
  for (let i = 0; i < count; i++) g.appendChild(makeCard(view[i], sheet));
  $("#galleryFoot").textContent = count < list.length
    ? `showing ${count} of ${list.length} — scroll for more`
    : `${list.length} image(s)`;
  $(".main").scrollTop = prev;
}
// Infinite scroll: when near the bottom of the main pane, reveal the next page.
function maybeLoadMore() {
  const m = $(".main"); const list = images[tab] || [];
  if (shown >= list.length) return;
  if (m.scrollTop + m.clientHeight >= m.scrollHeight - 600) {
    shown += PAGE; renderGallery();
  }
}
let pendingBreed = null;   // rel of the first-clicked breed parent
async function exploreImage(rel, btn) {
  const body = {
    src: rel, mode: "neighborhood",
    radius: $("#exRadius").value.trim() === "" ? 0.3 : Number($("#exRadius").value),
    n: $("#exN").value.trim() === "" ? 6 : Number($("#exN").value),
  };
  if (btn) { btn.disabled = true; }
  try {
    const r = await fetch("/api/explore", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) alert(j.detail || "explore failed");
    else { selectedJob = j.job_id; poll(); }
  } catch(e){}
  if (btn) setTimeout(() => { btn.disabled = false; }, 3000);
}
async function walkImage(rel, btn) {
  const body = {
    src: rel, mode: "walk", direction: "outward",
    step: $("#wkStep").value.trim() === "" ? 0.15 : Number($("#wkStep").value),
    n: $("#exN").value.trim() === "" ? 6 : Number($("#exN").value),
  };
  if (btn) btn.disabled = true;
  try {
    const r = await fetch("/api/explore", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) alert(j.detail || "walk failed");
    else { selectedJob = j.job_id; poll(); }
  } catch(e){}
  if (btn) setTimeout(() => { btn.disabled = false; }, 3000);
}
async function breedClick(rel) {
  if (pendingBreed === rel) { pendingBreed = null; renderGallery(); return; }  // cancel
  if (pendingBreed === null) {
    pendingBreed = rel;
    $("#exploreHint").innerHTML = `🧬 parent A = <b>${rel.split("/").pop()}</b> — click 🧬 on a second image (same backend) to breed, or click it again to cancel.`;
    renderGallery(); return;
  }
  const body = {
    src: pendingBreed, b: rel, mode: "breed",
    mutate: 0.15,
    n: $("#exN").value.trim() === "" ? 6 : Number($("#exN").value),
  };
  pendingBreed = null;
  $("#exploreHint").innerHTML = `<b>🧭 Explore</b> samples around an image's latent point (hill-climb on taste); <b>🧬 Breed</b>: click on two images to SLERP-cross them.`;
  try {
    const r = await fetch("/api/explore", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) alert(j.detail || "breed failed");
    else { selectedJob = j.job_id; poll(); }
  } catch(e){}
  renderGallery();
}
async function toggleFav(rel, on) {
  try {
    await fetch("/api/favorite", {method:"POST", headers:{"Content-Type":"application/json"},
      body: JSON.stringify({rel, on})});
    await refreshImages();
  } catch (e) {}
}
// The creative preset = the original "Recreate" instruction (re-renders with
// license to reinterpret); faithful = the script's built-in default (null).
const RECREATE_PROMPT = "Recreate this exact image at higher resolution with maximum fine detail and texture fidelity. Keep the composition, colors, style and every element identical.";
function refinePrompt() {
  const sel = $("#rfPromptSel") ? $("#rfPromptSel").value : "faithful";
  if (sel === "recreate") return RECREATE_PROMPT;
  if (sel === "custom") return (($("#rfPrompt") || {}).value || "").trim() || null;
  return null;   // faithful -> script default
}
async function refineImage(name, btn) {
  // hires owns its own two knobs and reads steps/guidance/scheduler/seed off the
  // source image's sidecar; the other engines take the rest of the bar.
  const body = $("#rfEngine").value === "hires" ? {
    src: name, engine: "hires", tiled: false,
    scale: parseFloat($("#rfFactor").value),
    strength: parseFloat($("#rfDenoise").value),
  } : {
    src: name,
    scale: parseFloat($("#rfScale").value),
    steps: $("#rfSteps").value.trim() === "" ? null : Number($("#rfSteps").value),
    strength: $("#rfStrength").value.trim() === "" ? 0.45 : Number($("#rfStrength").value),
    scheduler: $("#rfSched").value || null,
    tiled: $("#rfMode").value === "tiled",
    engine: $("#rfEngine").value,
    prompt: refinePrompt(),
  };
  if (btn) { btn.disabled = true; btn.textContent = "queued…"; }
  try {
    const r = await fetch("/api/refine", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(body)});
    const j = await r.json();
    if (!r.ok) { if (btn){ btn.textContent="✗"; } alert(j.detail || "refine failed"); }
    else { selectedJob = j.job_id; if (btn){ btn.textContent="↑ #"+j.job_id; } poll(); }
  } catch (e) { if (btn) btn.textContent = "✗"; }
  setTimeout(() => { if (btn){ btn.disabled=false; btn.textContent="⤴ Upscale"; } }, 4000);
}
const PARAM_ORDER = ["kind","mode","parent","parent_b","distance","anchor_distance","radius","mutate",
  "direction","step","walk_frame","target_distance","elites","base_blend","dist",
  "backend","model","sampler","temperature","coherence",
  "components","comp_lo","equalize","truncation",
  "rho","length_mode","length","empirical_head","temp_on","temp_off",
  "radius_band","radius_scale","steps","guidance","scheduler","neg_mode","height","width","init_image","init_mode","init_strength","ip_scale",
  "batch_seed","image_seed","index","refined_from","cond_from","engine","factor","denoise",
  "denoise_steps","interp","scale","strength","cond_reused","src_size","out_size","seed"];
let currentRel = null, currentScore = null;
async function openLight(src, rel, score){
  currentRel = rel; currentScore = score;
  lightIdx = currentView.findIndex(x => x.rel === rel);
  $("#lightimg").src = src;
  $("#lightbox").style.display = "flex";
  const box = $("#lightmeta"); box.innerHTML = "loading params…";
  const scoreLine = (score != null) ? `<span style="color:#7fae7f;margin-left:10px">aesthetic ${score.toFixed(2)}</span>` : "";
  if (!rel) { box.textContent = ""; return; }
  try {
    const m = await (await fetch("/api/meta?path=" + encodeURIComponent(rel))).json();
    const genBtn = w => ` <button onclick="genFromPrompt(this,'${rel}','${w}')" style="padding:2px 8px;font-size:11px">🎨 generate from it</button>`;
    const nativeable = /anarchy_(sd15|sdxl)_/.test(rel);
    let inv = "";
    inv += m.inverted_prompt !== undefined
      ? `<div style="margin-top:6px;color:#d4b96a">🔤 \u201C${m.inverted_prompt}\u201D <span style="color:#9aa0ad">CLIP's eyes (PEZ, ${m.inverted_tokens} tok, sim ${m.inverted_sim})</span>${genBtn("inverted")}</div>`
      : `<div style="margin-top:6px;display:inline-block"><button id="invBtn" onclick="invertImage(this,'${rel}','clip')" style="padding:3px 10px;font-size:11px">🔤 reveal nearest prompt</button></div>`;
    if (m.native_prompt !== undefined)
      inv += `<div style="margin-top:4px;color:#8fd48a">🔡 \u201C${m.native_prompt}\u201D <span style="color:#9aa0ad">model's own encoder (native, cond-cos ${m.native_sim})</span>${genBtn("native")}</div>`;
    else if (nativeable)
      inv += `<span style="margin-left:8px"><button id="invBtnN" onclick="invertImage(this,'${rel}','native')" style="padding:3px 10px;font-size:11px">🔡 native prompt</button></span>`;
    const keys = Object.keys(m);
    if (!keys.length) { box.innerHTML = `<span style="color:#9aa0ad">${rel} — no params recorded (pre-dates param logging)</span>${scoreLine}` + inv; return; }
    const ordered = PARAM_ORDER.filter(k => k in m && m[k] !== null);
    const cli = buildCli(m);
    const LINKED = new Set(["parent", "parent_b", "refined_from", "cond_from"]);
    const render = k => {
      let v = m[k];
      if (LINKED.has(k) && typeof v === "string")
        v = `<a href="#" style="color:#6fb3e0" onclick="openParent('${v}');return false;">${v}</a>`;
      return `<span style="display:inline-block;margin-right:14px"><span style="color:#9aa0ad">${k}</span> ${v}</span>`;
    };
    box.innerHTML = `<div style="color:#e0a13d;margin-bottom:4px">${rel}${scoreLine}</div>` +
      ordered.map(render).join("") +
      (cli ? `<div style="margin-top:6px;color:#7fae7f">${cli}</div>` : "") + inv;
  } catch(e){ box.textContent = ""; }
}

function pollJob(jid, onDone){
  const t = setInterval(async () => {
    try {
      const st = await (await fetch("/api/state")).json();
      const j = (st.jobs || []).find(x => x.id === jid);
      if (j && ["done", "error", "cancelled"].includes(j.status)) {
        clearInterval(t); onDone(j.status);
      }
    } catch(e){}
  }, 4000);
}

async function invertImage(btn, rel, space){
  if (btn) { btn.disabled = true; btn.textContent = (space === "native" ? "🔡" : "🔤") + " inverting… (queued)"; }
  const r = await (await fetch("/api/invert", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({src: rel, space: space || "clip"})})).json();
  pollJob(r.job_id, () => {
    if (currentRel === rel)              // still looking at it -> refresh panel
      openLight($("#lightimg").src, rel, currentScore);
  });
}

async function genFromPrompt(btn, rel, which){
  if (btn) { btn.disabled = true; btn.textContent = "🎨 rendering… (queued)"; }
  const r = await (await fetch("/api/genprompt", {method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({src: rel, which: which})})).json();
  pollJob(r.job_id, (status) => {
    if (btn) btn.textContent = status === "done" ? "🎨 done — in gallery ✓" : "🎨 " + status;
    if (status === "done") refreshImages();   // the comparison appears beside its parent
  });
}

function openParent(name){
  const rel = "generated/" + name;
  openLight("/img?path=" + encodeURIComponent(rel), rel, null);
}
function buildCli(m){
  if (m.kind === "generate") {
    let s = `generate.py --backend ${m.backend} --sampler ${m.sampler} --temperature ${m.temperature} --seed ${m.batch_seed} --steps ${m.steps} --guidance ${m.guidance}`;
    if (m.scheduler && m.scheduler!=="default") s += ` --scheduler ${m.scheduler}`;
    if (m.coherence!=null) s += ` --coherence ${m.coherence}`;
    return s;
  }
  if (m.kind === "refine" && m.engine === "hires") return `upscale.py --src ${m.refined_from || "<orig>"} --factor ${m.factor} --denoise ${m.denoise} --interp ${m.interp}`;
  if (m.kind === "refine" && m.engine === "flux2-klein") return `refine_flux.py --src ${m.refined_from || "<orig>"} --scale ${m.scale} --steps ${m.steps}`;
  if (m.kind === "refine") return `refine.py --src <orig> --scale ${m.scale} --strength ${m.strength} --steps ${m.steps} --guidance ${m.guidance} --scheduler ${m.scheduler}`;
  if (m.kind === "explore") {
    let s = `explore.py --mode ${m.mode} --src ${m.parent} --seed ${m.batch_seed}`;
    if (m.mode === "breed") s += ` --b ${m.parent_b} --mutate ${m.mutate}`;
    else s += ` --radius ${m.radius}`;
    return s;
  }
  return "";
}
function lightNav(d){
  if (!currentView.length) return;
  let i = lightIdx < 0 ? (d > 0 ? 0 : currentView.length - 1) : lightIdx + d;
  if (i < 0 || i >= currentView.length) return;
  const im = currentView[i];
  openLight(im.url + "&t=" + Math.floor(im.mtime), im.rel, im.score);
}
$("#lbPrev").onclick = (e) => { e.stopPropagation(); lightNav(-1); };
$("#lbNext").onclick = (e) => { e.stopPropagation(); lightNav(1); };
document.addEventListener("keydown", (e) => {
  if ($("#lightbox").style.display !== "flex") return;
  if (e.key === "ArrowLeft") { e.preventDefault(); lightNav(-1); }
  else if (e.key === "ArrowRight") { e.preventDefault(); lightNav(1); }
  else if (e.key === "Escape") $("#lightbox").style.display = "none";
});
$("#lightbox").onclick = (e) => { if (e.target.id !== "lightmeta" && e.target.tagName !== "BUTTON") $("#lightbox").style.display = "none"; };
$("#lightmeta").onclick = (e) => e.stopPropagation();

$("#rfPromptSel").onchange = () => {
  $("#rfPromptCustomWrap").style.display = $("#rfPromptSel").value === "custom" ? "" : "none";
};
// Engine switch: each engine shows only the knobs it actually reads.
function rfSyncEngine() {
  const hires = $("#rfEngine").value === "hires";
  document.querySelectorAll(".rfHires").forEach(e => e.style.display = hires ? "" : "none");
  document.querySelectorAll(".rfOther").forEach(e => e.style.display = hires ? "none" : "");
}
$("#rfEngine").onchange = rfSyncEngine;
$("#rfFactor").oninput = () => {
  $("#rfFactorOut").textContent = "×" + Number($("#rfFactor").value).toFixed(2);
};
$("#rfDenoise").oninput = () => {
  const v = Number($("#rfDenoise").value);
  $("#rfDenoiseOut").textContent = v.toFixed(2) + " · last " + Math.round(v * 100) + "%";
};
rfSyncEngine();
$("#sortBy").onchange = () => { shown = PAGE; renderGallery(); };
$("#wipeBtn").onclick = async () => {
  const p = await (await fetch("/api/wipe/preview")).json();
  if (!p.count) { alert("nothing to wipe: no non-starred images scored below 5."); return; }
  if (!confirm(`Delete ${p.count} images (not starred, score < 5)?\nFavorites and their originals are protected. This cannot be undone.`)) return;
  $("#wipeBtn").disabled = true; $("#wipeBtn").textContent = "🧹 wiping…";
  const r = await (await fetch("/api/wipe", {method: "POST"})).json();
  $("#wipeBtn").disabled = false; $("#wipeBtn").textContent = "🧹 Wipe <5";
  alert(`deleted ${r.deleted} images (${r.files} files).`);
  refreshImages();
};

$("#evolveBtn").onclick = async () => {
  if (!confirm("Refit a distribution branch around your ★ favorites and sample 8 images from it?")) return;
  $("#evolveBtn").disabled = true;
  try {
    const r = await fetch("/api/evolve", {method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify({n: 8})});
    const j = await r.json();
    if (!r.ok) alert(j.detail || "evolve failed");
    else { selectedJob = j.job_id; poll(); }
  } catch(e){}
  setTimeout(() => { $("#evolveBtn").disabled = false; }, 4000);
};
async function refreshBand() {
  try {
    const b = await (await fetch("/api/tasteband")).json();
    const el = $("#bandHint"); if (!el) return;
    el.textContent = b.count
      ? `your ★ keepers: d≈${b.mean} (band ${b.p25}–${b.p75}, n=${b.count}) — try that as target`
      : "";
  } catch(e){}
}
$("#analyzeBtn").onclick = async () => {
  $("#analyzeBtn").disabled = true; $("#analyzeBtn").textContent = "analyzing…";
  try { const j = await (await fetch("/api/resonance", {method:"POST"})).json(); selectedJob = j.job_id; poll(); } catch(e){}
  setTimeout(()=>{ $("#analyzeBtn").disabled=false; $("#analyzeBtn").textContent="🎯 Analyze"; }, 4000);
};
$("#scoreBtn").onclick = async () => {
  $("#scoreBtn").disabled = true; $("#scoreBtn").textContent = "scoring…";
  try { const j = await (await fetch("/api/score", {method:"POST"})).json(); selectedJob = j.job_id; poll(); } catch(e){}
  setTimeout(()=>{ $("#scoreBtn").disabled=false; $("#scoreBtn").textContent="Score all ▶"; }, 4000);
};

// ---- boot ----
$(".main").addEventListener("scroll", maybeLoadMore, {passive: true});

(async () => {
  try { window._cfg = await (await fetch("/api/config")).json(); } catch(e){}
  // Show the real negative prompt in the box, not just its name — it's the one
  // string a "promptless" run still writes, so it should be readable and editable.
  if (!$("#negative").value) $("#negative").value = (window._cfg || {}).sd15_negative || "";
  populateInitFolders(); syncForm(); refreshImages(); poll(); refreshBand();
  setInterval(refreshBand, 20000);
  setInterval(poll, 1500);
  setInterval(() => { if (lastBusy) refreshImages(); }, 6000);
  // refresh init folders/counts so newly added folders show up without reload
  setInterval(async () => {
    try { window._cfg = await (await fetch("/api/config")).json(); populateInitFolders(); syncForm(); } catch(e){}
  }, 8000);
})();
</script>
</body>
</html>"""


# --------------------------------------------------------------------------- #
# Static frontend — mounted LAST so every /api route above wins the match.
# --------------------------------------------------------------------------- #
if (FRONTEND_DIST / "index.html").is_file():
    # html=True serves index.html for "/" (and for unknown sub-paths, which a
    # future client-side router would need).
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:
    # Not built yet -> keep the inline dashboard on "/" so the tool still works.
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("SA_HOST", "127.0.0.1")
    port = int(os.environ.get("SA_PORT", "8800"))
    built = (FRONTEND_DIST / "index.html").is_file()
    print(f"[webui] python (jobs) = {PYTHON}")
    print(f"[webui] frontend      = {'dist (react)' if built else 'inline (legacy)'}")
    print(f"[webui] serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")
