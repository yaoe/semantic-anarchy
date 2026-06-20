"""Stable Diffusion glue: encode prompts to embeddings, and decode embeddings to images.

Two halves of the trick:

1. ``encode_prompts`` runs the corpus of good prompts through CLIP to harvest the
   conditioning tensors we'll learn a distribution over. This is the *only* time
   the text encoder is used.

2. ``generate_from_embeddings`` feeds sampled conditioning tensors straight into
   the UNet via diffusers' ``prompt_embeds`` argument -- the text encoder is never
   touched. This is the "X out CLIPText" move from slide 9.

torch / diffusers are imported lazily so the rest of the package (distribution
mining math, plotting, tests) runs on a machine with neither installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


def _pick_device(requested: Optional[str] = None) -> str:
    import torch

    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _ensure_safetensors_ckpt(ckpt: str) -> str:
    """Return a ``.safetensors`` path usable by ``from_single_file``.

    ``.safetensors`` inputs pass straight through. A legacy ``.ckpt`` (a pickle,
    possibly carrying a ``pytorch_lightning`` global that torch 2.6+'s
    weights_only load rejects) is converted *once*: we load it with
    ``weights_only=False`` -- safe, it's a trusted local file the user pointed us
    at -- pull out the tensor ``state_dict``, and write a cached sibling
    ``.safetensors``. Subsequent runs reuse that cache.
    """
    import tempfile
    from pathlib import Path

    import torch
    from safetensors.torch import save_file

    src = Path(ckpt)
    if src.suffix == ".safetensors":
        return str(src)

    # Cache next to the ckpt if that dir is writable, else under the temp dir.
    cached = src.with_suffix(".safetensors")
    if not _dir_writable(cached.parent):
        cache_dir = Path(tempfile.gettempdir()) / "semantic_anarchy_converted"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached = cache_dir / (src.stem + ".safetensors")
    if cached.exists():
        print(f"[load] reusing converted ckpt -> {cached}")
        return str(cached)

    sd = _torch_load_lenient(src)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    tensors = {k: v.contiguous() for k, v in sd.items() if torch.is_tensor(v)}
    save_file(tensors, str(cached))
    print(f"[load] converted ckpt -> {cached}")
    return str(cached)


def _torch_load_lenient(src):
    """``torch.load`` a trusted legacy ckpt, tolerating missing pickled globals.

    Old webui/A1111 ``.ckpt`` files pickle a ``pytorch_lightning`` class in their
    top-level object. If that package isn't installed (or torch's weights_only
    default rejects it) the plain load raises ``ModuleNotFoundError`` /
    ``UnpicklingError``. We only want the tensors, so we retry with a custom
    unpickler that returns a harmless dummy for *any* global it can't import
    (e.g. ``pytorch_lightning.callbacks...ModelCheckpoint``). torch still rebuilds
    its storages, so the real tensors come through regardless.
    """
    import pickle

    import torch

    try:
        return torch.load(src, map_location="cpu", weights_only=False)
    except (ModuleNotFoundError, pickle.UnpicklingError, AttributeError):
        class _Tolerant(pickle.Unpickler):
            def find_class(self, mod, name):
                try:
                    return super().find_class(mod, name)
                except Exception:
                    # Unknown/uninstalled global -> a do-nothing placeholder.
                    return type("_Dummy", (), {"__setstate__": lambda self, s: None})

        # torch.load wants a pickle-module-like object; supply the full surface
        # (incl. __name__, which torch inspects to special-case dill).
        class _M:
            __name__ = "semantic_anarchy._tolerant_pickle"
            Unpickler = _Tolerant
            Pickler = pickle.Pickler
            load = staticmethod(pickle.load)
            loads = staticmethod(pickle.loads)
            dump = staticmethod(pickle.dump)
            dumps = staticmethod(pickle.dumps)
            HIGHEST_PROTOCOL = pickle.HIGHEST_PROTOCOL

        return torch.load(
            src, map_location="cpu", weights_only=False, pickle_module=_M()
        )


def _dir_writable(path) -> bool:
    import os

    return os.access(str(path), os.W_OK)


@dataclass
class SDModel:
    """Thin wrapper around a diffusers Stable Diffusion pipeline."""

    pipe: object
    device: str
    model_id: str
    _uncond: Optional[np.ndarray] = None  # cached empty-prompt embedding

    @classmethod
    def load(
        cls,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        dtype: str = "auto",
        ckpt: Optional[str] = None,
    ) -> "SDModel":
        """Load a Stable Diffusion pipeline.

        ``ckpt``:
            Path to a single-file checkpoint (``.ckpt`` / ``.safetensors``). When
            given, the pipeline is built via ``from_single_file`` -- so a local
            webui/AUTOMATIC1111 checkpoint works directly with no HF download.
            Otherwise we ``from_pretrained(model_id)`` (a repo id or diffusers folder).
        """
        import torch
        from diffusers import StableDiffusionPipeline

        device = _pick_device(device)
        if dtype == "auto":
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        if ckpt is not None:
            # Legacy ``.ckpt`` files are pickles that may stash a
            # ``pytorch_lightning`` global; under torch 2.6+ from_single_file's
            # weights_only=True load chokes on them. So for anything that isn't
            # already safetensors we convert it ourselves (trusted local file)
            # to a cached sibling .safetensors and load that.
            load_path = _ensure_safetensors_ckpt(ckpt)
            pipe = StableDiffusionPipeline.from_single_file(load_path, torch_dtype=torch_dtype)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        # We deliberately keep the safety checker off: these images have no prompt
        # and no semantics for it to reason about, and it only mangles the output.
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
        return cls(pipe=pipe, device=device, model_id=ckpt or model_id)

    @property
    def feature_shape(self) -> tuple[int, int]:
        """``(tokens, hidden)`` of the conditioning tensor, e.g. ``(77, 768)``."""
        tokens = self.pipe.tokenizer.model_max_length
        hidden = self.pipe.text_encoder.config.hidden_size
        return (tokens, hidden)

    # ----------------------------------------------------------- encoding ---
    def encode_prompts(
        self,
        prompts: Iterable[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Encode prompts -> conditioning embeddings ``(N, 77, hidden)`` as numpy."""
        import torch

        prompts = list(prompts)
        out = []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            embeds = self.pipe.encode_prompt(
                prompt=chunk,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            # diffusers returns (prompt_embeds, negative_prompt_embeds)
            prompt_embeds = embeds[0] if isinstance(embeds, tuple) else embeds
            out.append(prompt_embeds.detach().to(torch.float32).cpu().numpy())
        return np.concatenate(out, axis=0)

    def uncond_embedding(self) -> np.ndarray:
        """The empty-prompt embedding, used as the negative branch for CFG.

        Cached after the first call -- it never changes, and generation otherwise
        re-encodes the empty prompt on every batch.
        """
        if self._uncond is None:
            self._uncond = self.encode_prompts([""], batch_size=1)
        return self._uncond

    # ----------------------------------------------------------- decoding ---
    def generate_from_embeddings(
        self,
        embeddings: np.ndarray,
        negative_embedding: Optional[np.ndarray] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
    ) -> list:
        """Decode conditioning embeddings to PIL images -- no prompt involved.

        ``embeddings`` is ``(N, 77, hidden)``. Each is fed as ``prompt_embeds``;
        the text encoder is never called.
        """
        import torch

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 2:
            embeddings = embeddings[None]
        n = embeddings.shape[0]

        param_dtype = next(self.pipe.unet.parameters()).dtype
        cond = torch.from_numpy(embeddings).to(self.device, dtype=param_dtype)

        if negative_embedding is None:
            negative_embedding = self.uncond_embedding()
        neg = np.asarray(negative_embedding, dtype=np.float32)
        if neg.ndim == 2:
            neg = neg[None]
        neg = np.broadcast_to(neg, embeddings.shape).copy()
        neg = torch.from_numpy(neg).to(self.device, dtype=param_dtype)

        images = []
        for i in range(n):
            # Seed per image (seed + i) so each image is independently reproducible.
            generator = (
                torch.Generator(device=self.device).manual_seed(seed + i)
                if seed is not None
                else None
            )
            result = self.pipe(
                prompt_embeds=cond[i : i + 1],
                negative_prompt_embeds=neg[i : i + 1],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )
            images.append(result.images[0])
        return images


@dataclass
class SDXLModel:
    """Thin wrapper around a STOCK diffusers ``StableDiffusionXLPipeline``.

    SDXL conditions on TWO tensors instead of SD 1.x's one:

    * ``prompt_embeds`` -- the concatenated CLIP-L + CLIP-bigG token sequence,
      shape ``(N, 77, 2048)``.
    * ``pooled_prompt_embeds`` -- the bigG pooled vector, ``(N, 1280)``, which is
      added into the timestep embedding.

    Both are sampled from their own learned distribution and fed straight in;
    the text encoders are never called at generation time. No SAE, no hooks --
    just the stock pipeline's ``prompt_embeds`` / ``pooled_prompt_embeds`` args.
    """

    pipe: object
    device: str
    model_id: str
    _uncond: Optional[dict] = None  # cached empty-prompt (prompt_embeds, pooled)

    @classmethod
    def load(
        cls,
        model_id: str = "stabilityai/sdxl-turbo",
        device: Optional[str] = None,
        dtype: str = "auto",
        ckpt: Optional[str] = None,
    ) -> "SDXLModel":
        """Load a stock SDXL pipeline (``from_pretrained`` or ``from_single_file``)."""
        import torch
        from diffusers import StableDiffusionXLPipeline

        device = _pick_device(device)
        if dtype == "auto":
            torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        if ckpt is not None:
            load_path = _ensure_safetensors_ckpt(ckpt)
            pipe = StableDiffusionXLPipeline.from_single_file(load_path, torch_dtype=torch_dtype)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe=pipe, device=device, model_id=ckpt or model_id)

    @property
    def feature_shapes(self) -> dict[str, tuple[int, ...]]:
        """Named conditioning shapes: ``prompt_embeds`` and ``pooled``."""
        tokens = self.pipe.tokenizer.model_max_length
        # Combined CLIP-L (768) + CLIP-bigG (1280) hidden width = 2048; pooled = 1280.
        hidden = (self.pipe.text_encoder.config.hidden_size
                  + self.pipe.text_encoder_2.config.hidden_size)
        pooled = self.pipe.text_encoder_2.config.hidden_size
        return {"prompt_embeds": (tokens, hidden), "pooled": (pooled,)}

    # ----------------------------------------------------------- encoding ---
    def encode_prompts(self, prompts: Iterable[str], batch_size: int = 8) -> dict:
        """Encode prompts -> ``{"prompt_embeds": (N,77,2048), "pooled": (N,1280)}``.

        SDXL ``encode_prompt`` returns a 4-tuple
        ``(prompt_embeds, negative_prompt_embeds, pooled, negative_pooled)`` --
        we keep the positive pair.
        """
        import torch

        prompts = list(prompts)
        embeds, pooled = [], []
        for i in range(0, len(prompts), batch_size):
            chunk = prompts[i : i + batch_size]
            enc = self.pipe.encode_prompt(
                prompt=chunk,
                device=self.device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )
            # 4-tuple: (prompt_embeds, neg_embeds, pooled, neg_pooled)
            embeds.append(enc[0].detach().to(torch.float32).cpu().numpy())
            pooled.append(enc[2].detach().to(torch.float32).cpu().numpy())
        return {
            "prompt_embeds": np.concatenate(embeds, axis=0),   # (N, 77, 2048)
            "pooled": np.concatenate(pooled, axis=0),          # (N, 1280)
        }

    def uncond(self) -> dict:
        """The empty-prompt conditioning, cached, for the CFG negative branch."""
        if self._uncond is None:
            self._uncond = self.encode_prompts([""], batch_size=1)
        return self._uncond

    # ----------------------------------------------------------- decoding ---
    def generate_from_embeddings(
        self,
        embeddings: dict,
        negatives: Optional[dict] = None,
        num_inference_steps: int = 1,
        guidance_scale: float = 0.0,
        height: int = 1024,
        width: int = 1024,
        seed: Optional[int] = None,
    ) -> list:
        """Decode sampled SDXL conditioning tensors to images (no text encoder).

        ``embeddings`` = ``{"prompt_embeds": (N,77,2048), "pooled": (N,1280)}``.
        When ``guidance_scale > 1`` the CFG path runs and ``negatives`` (same dict
        shape) is supplied as ``negative_prompt_embeds`` / ``negative_pooled``;
        if ``negatives`` is None the empty-prompt encoding is used.
        """
        import torch

        pe = np.asarray(embeddings["prompt_embeds"], dtype=np.float32)
        po = np.asarray(embeddings["pooled"], dtype=np.float32)
        if pe.ndim == 2:
            pe = pe[None]
        if po.ndim == 1:
            po = po[None]
        n = pe.shape[0]

        param_dtype = next(self.pipe.unet.parameters()).dtype
        pe_t = torch.from_numpy(pe).to(self.device, dtype=param_dtype)
        po_t = torch.from_numpy(po).to(self.device, dtype=param_dtype)

        use_cfg = guidance_scale > 1.0
        if use_cfg:
            if negatives is None:
                negatives = self.uncond()
            npe = np.asarray(negatives["prompt_embeds"], dtype=np.float32)
            npo = np.asarray(negatives["pooled"], dtype=np.float32)
            if npe.ndim == 2:
                npe = npe[None]
            if npo.ndim == 1:
                npo = npo[None]
            # Broadcast a single negative across the batch.
            npe = np.broadcast_to(npe, pe.shape).copy()
            npo = np.broadcast_to(npo, po.shape).copy()
            npe_t = torch.from_numpy(npe).to(self.device, dtype=param_dtype)
            npo_t = torch.from_numpy(npo).to(self.device, dtype=param_dtype)

        images = []
        for i in range(n):
            generator = (
                torch.Generator(device="cpu").manual_seed(seed + i)
                if seed is not None else None
            )
            kwargs = dict(
                prompt_embeds=pe_t[i : i + 1],
                pooled_prompt_embeds=po_t[i : i + 1],
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )
            if use_cfg:
                kwargs["negative_prompt_embeds"] = npe_t[i : i + 1]
                kwargs["negative_pooled_prompt_embeds"] = npo_t[i : i + 1]
            result = self.pipe(**kwargs)
            images.append(result.images[0])
        return images
