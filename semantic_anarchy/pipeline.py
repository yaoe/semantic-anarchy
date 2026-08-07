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

import contextlib
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np


#: The house SD1.5 negative prompt, used as the CFG negative branch for every
#: sd15 sampling step (generate, hires-fix, film, explore).
#:
#: Not invented here — it is the string the Eden SD1.5 stack has always sampled
#: with. Of the 1,153 SD1.5-lineage run configs recorded across the sibling repos
#: (``cog/eden-sd-pipelines`` results for ``eden:eden-v1`` and
#: ``dreamlike-photoreal-2.0``), 1,140 used exactly this text, and it is
#: hardcoded at seven more sites in ``sd-lora-trainer`` / ``diffusion_trainer``.
#: Adopting it here keeps promptless samples in the same aesthetic basin as
#: everything else that has been rendered off these checkpoints.
#:
#: Override with ``SA_SD15_NEGATIVE``; set it empty to fall back to the
#: empty-prompt negative (equivalently, ``--neg-mode empty``).
SD15_NEGATIVE_PROMPT = (
    "nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs, "
    "disfigured, deformed body, blurry, blurred, watermark, text, grainy, "
    "signature, cut off, draft"
)


def default_sd15_negative() -> Optional[str]:
    """``SD15_NEGATIVE_PROMPT``, or the ``SA_SD15_NEGATIVE`` override.

    An override set to the empty string means "no negative text" — the CFG
    negative goes back to the empty-prompt encoding.
    """
    import os

    override = os.environ.get("SA_SD15_NEGATIVE")
    if override is None:
        return SD15_NEGATIVE_PROMPT
    return override.strip() or None


#: Components mining never touches. Passing them to ``from_pretrained`` /
#: ``from_single_file`` as ``None`` makes diffusers *skip loading them entirely*
#: — for a single-file SD1.5 checkpoint that turns a multi-second, multi-GB
#: pipeline load into ~0.4s and ~250MB of VRAM (the text encoder alone).
ENCODE_ONLY_SKIP = (
    "unet", "transformer", "vae", "safety_checker", "feature_extractor",
    "image_encoder",
)


def encode_only_kwargs(pipeline_class) -> dict:
    """``{component: None}`` for every skippable component this pipeline has.

    Filtered against the pipeline's own ``__init__`` signature, so a class that
    has no ``transformer`` (or no ``safety_checker``) never sees the kwarg.
    """
    import inspect

    params = inspect.signature(pipeline_class.__init__).parameters
    return {name: None for name in ENCODE_ONLY_SKIP if name in params}


def _load_encode_only(build, pipeline_class, encode_only: bool):
    """``build(skip_kwargs)`` a pipeline, encoder-only when asked.

    A partial pipeline is unusual enough that it's worth being defensive: if a
    diffusers/model combination refuses to construct one, say so and load the
    whole thing rather than failing the mine.
    """
    if not encode_only:
        return build({})
    try:
        pipe = build(encode_only_kwargs(pipeline_class))
        print("[pipeline] encoder-only load: no UNet/VAE (mining never denoises)")
        return pipe
    except Exception as exc:                                    # noqa: BLE001
        print(f"[pipeline] encoder-only load failed ({exc!r}); "
              f"loading the full pipeline instead")
        return build({})


@contextlib.contextmanager
def quiet_truncation_warnings():
    """Mute the per-batch "your input was truncated" warning while encoding.

    diffusers logs it -- with the whole dropped text inlined -- once per batch.
    Mining a corpus of long captions turns that into hundreds of multi-KB log
    lines that bury the progress bar (and blow the dashboard's log cap).
    Truncation at the encoder's token limit is inherent to the method, so
    :func:`report_truncation` states it ONCE with a count and this mutes the
    repeats.
    """
    import importlib

    restore = []
    for mod in ("diffusers.utils.logging", "transformers.utils.logging"):
        try:
            m = importlib.import_module(mod)
            restore.append((m, m.get_verbosity()))
            m.set_verbosity_error()
        except Exception:                                       # noqa: BLE001
            continue
    try:
        yield
    finally:
        for m, prev in restore:
            try:
                m.set_verbosity(prev)
            except Exception:                                   # noqa: BLE001
                pass


def report_truncation(tokenizer, prompts) -> int:
    """Print (once) how many prompts overflow the encoder's token limit.

    A corpus of long image captions is mostly *over* CLIP's 77 tokens; that's
    fine -- the tail is dropped and the fit is over what the encoder actually
    saw -- but it should be visible, not silent.
    """
    limit = getattr(tokenizer, "model_max_length", None)
    if not limit or limit > 1e6 or len(prompts) < 2:
        return 0
    with quiet_truncation_warnings():
        lengths = [len(ids) for ids in
                   tokenizer(list(prompts), truncation=False,
                             padding=False)["input_ids"]]
    over = sum(1 for n in lengths if n > limit)
    if over:
        print(f"[pipeline] {over}/{len(lengths)} prompts exceed the {limit}-token "
              f"limit and are truncated (longest {max(lengths)} tokens)")
    return over


def _encode_batches(prompts, batch_size: int, fn, on_batch=None) -> list:
    """Run ``fn(chunk)`` over ``prompts`` in batches, reporting progress.

    ``on_batch(done)`` receives the running count of *prompts* encoded (not
    batches), so callers can drive a progress bar without knowing the batching.
    """
    prompts = list(prompts)
    batch_size = max(1, int(batch_size))
    out = []
    with quiet_truncation_warnings():
        for i in range(0, len(prompts), batch_size):
            out.append(fn(prompts[i : i + batch_size]))
            if on_batch is not None:
                on_batch(min(i + batch_size, len(prompts)))
    return out


def _stream(images: list, index: int, image, on_image) -> None:
    """Record a finished image and hand it straight to the caller.

    Every ``generate_from_embeddings`` renders one image per pipe call, so the
    whole batch is only "done" when the last one lands. ``on_image(index,
    image)`` lets the caller write each image the *moment* it exists instead:
    the dashboard's gallery then shows image 1 of 8 after an eighth of the wall
    clock, and a cancelled batch keeps everything it had already rendered.
    """
    images.append(image)
    if on_image is not None:
        on_image(index, image)


def set_scheduler(pipe, name: Optional[str]) -> None:
    """Swap a pipeline's scheduler in place. ``ddim`` is the classic, smooth
    sampler the deck's nicer renders used; ``None``/``default`` leaves it alone."""
    if not name or name == "default":
        return
    import diffusers
    # Flow-matching pipelines (FLUX.2, Krea 2) have their own sigma schedule --
    # swapping in a DDPM-family scheduler would break them. Leave those alone.
    if "FlowMatch" in type(pipe.scheduler).__name__:
        return
    table = {
        "ddim": "DDIMScheduler",
        "euler": "EulerDiscreteScheduler",
        "euler_a": "EulerAncestralDiscreteScheduler",
        "dpm": "DPMSolverMultistepScheduler",
    }
    cls_name = table.get(name)
    if not cls_name:
        return
    cls = getattr(diffusers, cls_name)
    pipe.scheduler = cls.from_config(pipe.scheduler.config)


def _resample(name: Optional[str]):
    """PIL resampling filter by name. ``lanczos`` (the default) is the sharpest;
    ``bicubic`` is softer, ``nearest`` keeps hard pixel edges for the denoiser to
    re-interpret."""
    from PIL import Image

    table = {"lanczos": Image.LANCZOS, "bicubic": Image.BICUBIC,
             "nearest": Image.NEAREST, "bilinear": Image.BILINEAR}
    return table.get((name or "lanczos").lower(), Image.LANCZOS)


def _refine_target(image, scale: float, size=None) -> tuple[int, int]:
    """Target ``(w, h)`` for a refine pass: an explicit ``size`` wins (already
    snapped to a valid diffusion grid by ``upscale.target_size``), otherwise
    ``scale`` the source and round to 8."""
    if size is not None:
        return max(8, int(size[0])), max(8, int(size[1]))
    w, h = image.size
    return max(8, round(w * scale / 8) * 8), max(8, round(h * scale / 8) * 8)


def tiled_upscale(image, scale, tile, overlap, fn):
    """Ultimate-SD-Upscale-style detail pass.

    Lanczos-enlarge ``image`` to ``scale x``, then run ``fn`` (an img2img denoise
    that returns a same-size PIL image) over OVERLAPPING ``tile`` x ``tile`` tiles
    and feather-blend them back together. Because every tile is denoised at (near)
    the model's NATIVE resolution -- where it synthesizes the most coherent fine
    detail -- this adds real texture instead of the soft "zoom" a single
    off-native-resolution pass produces. ``overlap`` px of feathered blend hides
    the seams.
    """
    import numpy as np
    from PIL import Image

    w, h = image.size
    tw, th = max(tile, round(w * scale / 8) * 8), max(tile, round(h * scale / 8) * 8)
    base = image.convert("RGB").resize((tw, th), Image.LANCZOS)

    # 1-D feathered ramp (0.1..1) rising over `overlap` px from each tile edge.
    r = np.minimum(np.arange(tile), np.arange(tile)[::-1]).astype(np.float32)
    r = np.clip((r + 1.0) / max(1, overlap), 0.1, 1.0)
    mask = np.outer(r, r)[..., None]            # (tile, tile, 1)

    acc = np.zeros((th, tw, 3), np.float32)
    wsum = np.zeros((th, tw, 1), np.float32)
    step = max(1, tile - overlap)
    xs = sorted({min(x, tw - tile) for x in range(0, tw, step)} | {max(0, tw - tile)})
    ys = sorted({min(y, th - tile) for y in range(0, th, step)} | {max(0, th - tile)})
    for y in ys:
        for x in xs:
            crop = base.crop((x, y, x + tile, y + tile))
            out = np.asarray(fn(crop).convert("RGB"), np.float32)
            acc[y:y + tile, x:x + tile] += out * mask
            wsum[y:y + tile, x:x + tile] += mask
    res = (acc / np.maximum(wsum, 1e-6)).clip(0, 255).astype("uint8")
    return Image.fromarray(res)


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
    #: Text used for the CFG negative branch. ``None`` = the empty prompt.
    #: ``SD15Backend.load`` sets it to :data:`SD15_NEGATIVE_PROMPT`; sd2 leaves
    #: it None (its OpenCLIP-H encoder was never sampled with that string).
    negative_prompt: Optional[str] = None
    _uncond: Optional[np.ndarray] = None  # cached empty-prompt embedding
    _neg: Optional[np.ndarray] = None     # cached negative_prompt embedding

    @classmethod
    def load(
        cls,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        device: Optional[str] = None,
        dtype: str = "auto",
        ckpt: Optional[str] = None,
        single_file_config: Optional[str] = None,
        encode_only: bool = False,
    ) -> "SDModel":
        """Load a Stable Diffusion pipeline.

        ``encode_only``:
            Load ONLY the tokenizer + text encoder (see ``ENCODE_ONLY_SKIP``).
            Mining never denoises, so the UNet and VAE are dead weight — skipping
            them is the difference between seconds and ~0.4s, and between GBs of
            VRAM and ~250MB. Falls back to the full pipeline if a diffusers
            version won't build a partial one.
        ``ckpt``:
            Path to a single-file checkpoint (``.ckpt`` / ``.safetensors``). When
            given, the pipeline is built via ``from_single_file`` -- so a local
            webui/AUTOMATIC1111 checkpoint works directly with no HF download.
            Otherwise we ``from_pretrained(model_id)`` (a repo id or diffusers folder).
        ``single_file_config``:
            Optional diffusers repo id to read the *config* from when loading a
            single-file ckpt (``from_single_file(config=...)``). Needed for SD 2.x,
            whose canonical config repo is gated -- point this at a NON-gated
            diffusers mirror so the v-prediction scheduler / 1024-dim cross-attn
            config is picked up without hitting the gated repo.
        """
        import torch
        from diffusers import StableDiffusionPipeline

        device = _pick_device(device)
        if dtype == "auto":
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        def _build(skip: dict):
            if ckpt is not None:
                # Legacy ``.ckpt`` files are pickles that may stash a
                # ``pytorch_lightning`` global; under torch 2.6+ from_single_file's
                # weights_only=True load chokes on them. So for anything that isn't
                # already safetensors we convert it ourselves (trusted local file)
                # to a cached sibling .safetensors and load that.
                load_path = _ensure_safetensors_ckpt(ckpt)
                sf_kwargs = {"torch_dtype": torch_dtype, **skip}
                if single_file_config:
                    sf_kwargs["config"] = single_file_config
                return StableDiffusionPipeline.from_single_file(load_path, **sf_kwargs)
            return StableDiffusionPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype, **skip)

        pipe = _load_encode_only(_build, StableDiffusionPipeline, encode_only)
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
        batch_size: int = 8,
        on_batch=None,
    ) -> np.ndarray:
        """Encode prompts -> conditioning embeddings ``(N, 77, hidden)`` as numpy.

        ``on_batch(done)`` is called after each batch with the running prompt
        count, for progress reporting.
        """
        import torch

        prompts = list(prompts)
        report_truncation(self.pipe.tokenizer, prompts)

        def one(chunk):
            with torch.no_grad():
                embeds = self.pipe.encode_prompt(
                    prompt=chunk,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
            # diffusers returns (prompt_embeds, negative_prompt_embeds)
            prompt_embeds = embeds[0] if isinstance(embeds, tuple) else embeds
            return prompt_embeds.detach().to(torch.float32).cpu().numpy()

        return np.concatenate(_encode_batches(prompts, batch_size, one, on_batch), axis=0)

    def uncond_embedding(self) -> np.ndarray:
        """The empty-prompt embedding -- the corpus's geometric origin.

        This is the analysis anchor (``analysis.encode_corpus`` saves it as the
        ``uncond`` row) and the fallback negative when no negative text is set.
        Cached after the first call -- it never changes, and generation otherwise
        re-encodes the empty prompt on every batch.
        """
        if self._uncond is None:
            self._uncond = self.encode_prompts([""], batch_size=1)
        return self._uncond

    def negative_embedding(self) -> np.ndarray:
        """The CFG negative branch: :attr:`negative_prompt`, else the empty prompt.

        Encoding it is the ONLY text-encoder call generation makes, it happens
        once per process, and it is why ``negative_prompt`` must be set before
        the UNet is skipped -- an ``encode_only`` model never generates anyway.
        """
        if self.negative_prompt is None:
            return self.uncond_embedding()
        if self._neg is None:
            self._neg = self.encode_prompts([self.negative_prompt], batch_size=1)
        return self._neg

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
        init_images: Optional[list] = None,
        init_strength: float = 0.7,
        ip_images: Optional[list] = None,
        ip_scale: float = 0.7,
        on_image=None,
    ) -> list:
        """Decode conditioning embeddings to PIL images -- no prompt involved.

        ``embeddings`` is ``(N, 77, hidden)``. Each is fed as ``prompt_embeds``;
        the text encoder is never called. ``on_image(i, image)`` fires as each
        one finishes (see :func:`_stream`).

        Two ways to inject an init image (mutually exclusive):
        * ``init_images`` -- IMG2IMG: start the denoiser from the init's latent
          (its colors/shapes) at ``init_strength``.
        * ``ip_images`` -- IMAGE-EMBEDDING (IP-Adapter): run the init through a
          CLIP *image* encoder and inject that embedding via cross-attention at
          ``ip_scale`` (0=ignore, ~0.6-0.9 strong), while still starting from pure
          noise. The init steers *content/style*, not just structure.
        """
        import torch

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 2:
            embeddings = embeddings[None]
        n = embeddings.shape[0]

        param_dtype = next(self.pipe.unet.parameters()).dtype
        cond = torch.from_numpy(embeddings).to(self.device, dtype=param_dtype)

        if negative_embedding is None:
            negative_embedding = self.negative_embedding()
        neg = np.asarray(negative_embedding, dtype=np.float32)
        if neg.ndim == 2:
            neg = neg[None]
        neg = np.broadcast_to(neg, embeddings.shape).copy()
        neg = torch.from_numpy(neg).to(self.device, dtype=param_dtype)

        img2img = self._img2img_pipe() if init_images else None
        if ip_images:
            self._ensure_ip_adapter(ip_scale)

        images = []
        for i in range(n):
            # Seed per image (seed + i) so each image is independently reproducible.
            generator = (
                torch.Generator(device=self.device).manual_seed(seed + i)
                if seed is not None
                else None
            )
            if img2img is not None:
                init = init_images[i % len(init_images)].convert("RGB").resize((width, height))
                result = img2img(
                    prompt_embeds=cond[i : i + 1],
                    negative_prompt_embeds=neg[i : i + 1],
                    image=init, strength=init_strength,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, generator=generator,
                )
            else:
                kwargs = dict(
                    prompt_embeds=cond[i : i + 1],
                    negative_prompt_embeds=neg[i : i + 1],
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    height=height, width=width, generator=generator,
                )
                if ip_images:
                    kwargs["ip_adapter_image"] = ip_images[i % len(ip_images)].convert("RGB")
                result = self.pipe(**kwargs)
            _stream(images, i, result.images[0], on_image)
        return images

    def _img2img_pipe(self):
        """Lazily build (and cache) an img2img pipeline sharing these weights."""
        if getattr(self, "_img2img", None) is None:
            from diffusers import AutoPipelineForImage2Image
            self._img2img = AutoPipelineForImage2Image.from_pipe(self.pipe)
            try:
                self._img2img.enable_vae_tiling()
            except Exception:
                pass
        return self._img2img

    def _ensure_ip_adapter(self, scale: float):
        """Load the SD1.5 IP-Adapter (image-embedding) once and set its scale.

        Injects a CLIP image embedding of the init via cross-attention. SD2.1
        (1024-dim cross-attn) has no compatible IP-Adapter -> errors clearly."""
        hidden = self.pipe.text_encoder.config.hidden_size
        if hidden != 768:
            raise RuntimeError(
                f"IP-Adapter image-embedding not supported for this model "
                f"(cross-attn dim {hidden}); use SD1.5 or SDXL.")
        if not getattr(self, "_ip_loaded", False):
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", subfolder="models",
                weight_name="ip-adapter_sd15.safetensors")
            self._ip_loaded = True
        self.pipe.set_ip_adapter_scale(scale)

    # ----------------------------------------------------------- refine ---
    def refine_image(self, image, scale: float = 2.0, num_inference_steps: int = 50,
                     strength: float = 0.35, guidance_scale: float = 1.0,
                     seed: Optional[int] = None, cond=None, scheduler=None,
                     size=None, interp: str = "lanczos"):
        """Upscale + add denoising steps to an existing image (img2img hires-fix).

        The image is Lanczos-upscaled to ``scale x`` then run through an img2img
        pass for ``num_inference_steps`` (``strength`` fraction actually denoise).

        ``cond``:
            The ORIGINAL ``(77, hidden)`` conditioning that produced the image
            (saved as a sidecar at generation time). Reusing it -- with CFG
            (``guidance_scale > 1``, negative = :meth:`negative_embedding`) -- makes the pass
            *reinforce the same content* at higher resolution instead of drifting
            toward a generic unconditional refine. Falls back to the empty-prompt
            embedding when ``cond`` is None (older images without a sidecar).
        ``scheduler``: optional sampler swap (e.g. ``ddim``).
        ``size``: explicit ``(w, h)`` target, overriding ``scale`` -- how
            ``scripts/upscale.py`` pins a 16-px-aligned resolution.
        ``interp``: resampling filter for the enlarge (lanczos/bicubic/nearest).
        """
        import torch
        from diffusers import AutoPipelineForImage2Image

        if getattr(self, "_img2img", None) is None:
            self._img2img = AutoPipelineForImage2Image.from_pipe(self.pipe)
            try:
                self._img2img.enable_vae_tiling()
            except Exception:
                pass
        set_scheduler(self._img2img, scheduler)
        tw, th = _refine_target(image, scale, size)
        init = image.convert("RGB").resize((tw, th), _resample(interp))

        param_dtype = next(self.pipe.unet.parameters()).dtype
        src = self.uncond_embedding() if cond is None else cond
        pe = torch.from_numpy(np.asarray(src, dtype=np.float32)).to(
            self.device, dtype=param_dtype)
        if pe.ndim == 2:
            pe = pe[None]
        kwargs = dict(prompt_embeds=pe, image=init, strength=strength,
                      num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale)
        if guidance_scale > 1.0:
            neg = torch.from_numpy(np.asarray(self.negative_embedding(), dtype=np.float32)).to(
                self.device, dtype=param_dtype)
            kwargs["negative_prompt_embeds"] = neg
        if seed is not None:
            kwargs["generator"] = torch.Generator(device=self.device).manual_seed(seed)
        return self._img2img(**kwargs).images[0]


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
        encode_only: bool = False,
    ) -> "SDXLModel":
        """Load a stock SDXL pipeline (``from_pretrained`` or ``from_single_file``).

        ``encode_only`` loads just the two tokenizers + text encoders — mining
        never denoises, so the UNet and VAE are dead weight."""
        import torch
        from diffusers import StableDiffusionXLPipeline

        device = _pick_device(device)
        if dtype == "auto":
            torch_dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        else:
            torch_dtype = getattr(torch, dtype)

        def _build(skip: dict):
            if ckpt is not None:
                load_path = _ensure_safetensors_ckpt(ckpt)
                return StableDiffusionXLPipeline.from_single_file(
                    load_path, torch_dtype=torch_dtype, **skip)
            return StableDiffusionXLPipeline.from_pretrained(
                model_id, torch_dtype=torch_dtype, **skip)

        pipe = _load_encode_only(_build, StableDiffusionXLPipeline, encode_only)
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
    def encode_prompts(self, prompts: Iterable[str], batch_size: int = 8,
                       on_batch=None) -> dict:
        """Encode prompts -> ``{"prompt_embeds": (N,77,2048), "pooled": (N,1280)}``.

        SDXL ``encode_prompt`` returns a 4-tuple
        ``(prompt_embeds, negative_prompt_embeds, pooled, negative_pooled)`` --
        we keep the positive pair.
        """
        import torch

        prompts = list(prompts)
        report_truncation(self.pipe.tokenizer, prompts)

        def one(chunk):
            with torch.no_grad():
                enc = self.pipe.encode_prompt(
                    prompt=chunk,
                    device=self.device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=False,
                )
            # 4-tuple: (prompt_embeds, neg_embeds, pooled, neg_pooled)
            return (enc[0].detach().to(torch.float32).cpu().numpy(),
                    enc[2].detach().to(torch.float32).cpu().numpy())

        pairs = _encode_batches(prompts, batch_size, one, on_batch)
        embeds = [e for e, _ in pairs]
        pooled = [p for _, p in pairs]
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
        init_images: Optional[list] = None,
        init_strength: float = 0.7,
        ip_images: Optional[list] = None,
        ip_scale: float = 0.7,
        on_image=None,
    ) -> list:
        """Decode sampled SDXL conditioning tensors to images (no text encoder).

        ``on_image(i, image)`` fires as each one finishes (see :func:`_stream`).

        ``embeddings`` = ``{"prompt_embeds": (N,77,2048), "pooled": (N,1280)}``.
        When ``guidance_scale > 1`` the CFG path runs and ``negatives`` (same dict
        shape) is supplied as ``negative_prompt_embeds`` / ``negative_pooled``;
        if ``negatives`` is None the empty-prompt encoding is used.

        Init image injection (mutually exclusive):
        * ``init_images`` -- IMG2IMG from the init's latent at ``init_strength``.
        * ``ip_images`` -- IMAGE-EMBEDDING (IP-Adapter): CLIP image embedding of
          the init injected via cross-attention at ``ip_scale``; steers content/
          style while starting from pure noise.
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

        img2img = self._img2img_pipe() if init_images else None
        if ip_images:
            self._ensure_ip_adapter(ip_scale)

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
                generator=generator,
            )
            if use_cfg:
                kwargs["negative_prompt_embeds"] = npe_t[i : i + 1]
                kwargs["negative_pooled_prompt_embeds"] = npo_t[i : i + 1]
            if img2img is not None:
                kwargs["image"] = init_images[i % len(init_images)].convert("RGB").resize((width, height))
                kwargs["strength"] = init_strength
                result = img2img(**kwargs)
            else:
                kwargs["height"], kwargs["width"] = height, width
                if ip_images:
                    kwargs["ip_adapter_image"] = ip_images[i % len(ip_images)].convert("RGB")
                result = self.pipe(**kwargs)
            _stream(images, i, result.images[0], on_image)
        return images

    def _img2img_pipe(self):
        """Lazily build (and cache) an SDXL img2img pipeline sharing these weights."""
        if getattr(self, "_img2img", None) is None:
            from diffusers import AutoPipelineForImage2Image
            self._img2img = AutoPipelineForImage2Image.from_pipe(self.pipe)
            try:
                self._img2img.enable_vae_tiling()
            except Exception:
                pass
        return self._img2img

    def _ensure_ip_adapter(self, scale: float):
        """Load the SDXL IP-Adapter (ViT-H plus) once and set its image scale.

        Uses the cached ip-adapter-plus_sdxl_vit-h weights + the ViT-H image
        encoder (in the repo's models/image_encoder)."""
        if not getattr(self, "_ip_loaded", False):
            # image_encoder_folder WITH a slash is treated as repo-root-relative by
            # diffusers (the ViT-H encoder lives at models/image_encoder).
            self.pipe.load_ip_adapter(
                "h94/IP-Adapter", subfolder="sdxl_models",
                weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
                image_encoder_folder="models/image_encoder")
            self._ip_loaded = True
        self.pipe.set_ip_adapter_scale(scale)

    # ----------------------------------------------------------- refine ---
    def refine_image(self, image, scale: float = 1.5, num_inference_steps: int = 50,
                     strength: float = 0.35, guidance_scale: float = 1.0,
                     seed: Optional[int] = None, cond=None, scheduler=None,
                     size=None, interp: str = "lanczos"):
        """SDXL img2img upscale/refine (see SDModel.refine_image).

        ``cond`` is the original ``{"prompt_embeds": (77,2048), "pooled": (1280,)}``
        saved at generation time; reused (with CFG, negative = empty prompt) so the
        hires pass reinforces the same content. Falls back to the empty prompt when
        ``cond`` is None. VAE tiling keeps the high-res decode within memory.
        ``size``/``interp`` as in SDModel.refine_image.
        """
        import torch
        from diffusers import AutoPipelineForImage2Image

        if getattr(self, "_img2img", None) is None:
            self._img2img = AutoPipelineForImage2Image.from_pipe(self.pipe)
            try:
                self._img2img.enable_vae_tiling()
            except Exception:
                pass
        set_scheduler(self._img2img, scheduler)
        tw, th = _refine_target(image, scale, size)
        init = image.convert("RGB").resize((tw, th), _resample(interp))

        param_dtype = next(self.pipe.unet.parameters()).dtype
        src = self.uncond() if cond is None else cond

        def _t(a, want):
            # Pad to exactly `want` dims (prompt_embeds -> 3 (B,77,H); pooled -> 2
            # (B,H)). uncond tensors already carry a batch dim; sidecar ones don't.
            a = np.asarray(a, dtype=np.float32)
            while a.ndim < want:
                a = a[None]
            return torch.from_numpy(a).to(self.device, dtype=param_dtype)

        kwargs = dict(prompt_embeds=_t(src["prompt_embeds"], 3),
                      pooled_prompt_embeds=_t(src["pooled"], 2), image=init,
                      strength=strength, num_inference_steps=num_inference_steps,
                      guidance_scale=guidance_scale)
        if guidance_scale > 1.0:
            u = self.uncond()
            kwargs["negative_prompt_embeds"] = _t(u["prompt_embeds"], 3)
            kwargs["negative_pooled_prompt_embeds"] = _t(u["pooled"], 2)
        if seed is not None:
            kwargs["generator"] = torch.Generator(device="cpu").manual_seed(seed)
        return self._img2img(**kwargs).images[0]



def _flow_pipeline_kwargs(model_id: str, encode_only: bool = False) -> dict:
    """Loading kwargs for big flow models on a 24GB-VRAM / 30GB-RAM box.

    The 4B klein fits in bf16 with cpu-offload. Anything bigger (klein-9B,
    Krea 2) gets its transformer + LLM text encoder NF4-quantized so the whole
    pipeline lives on the GPU (~12-15GB) -- bnb weights can't be cpu-offloaded,
    and unquantized they don't fit either RAM or VRAM. Override with
    SA_FLOW_QUANT=off|nf4.

    ``encode_only`` mining skips the transformer entirely, so it must not be
    named in ``components_to_quantize`` (quantizing an unloaded component).
    """
    import os
    import torch

    mode = os.environ.get("SA_FLOW_QUANT", "auto")
    big = any(t in model_id for t in ("9B", "9b", "Krea", "krea"))
    if mode == "off" or (mode == "auto" and not big):
        return {"torch_dtype": torch.bfloat16, "_offload": True}
    from diffusers.quantizers import PipelineQuantizationConfig
    components = ["text_encoder"] if encode_only else ["transformer", "text_encoder"]
    return {
        "torch_dtype": torch.bfloat16,
        "_offload": False,
        "quantization_config": PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={"load_in_4bit": True,
                          "bnb_4bit_quant_type": "nf4",
                          "bnb_4bit_compute_dtype": torch.bfloat16},
            components_to_quantize=components,
        ),
    }


@dataclass
class Flux2Model:
    """FLUX.2 klein (4B/9B) behind the same duck-typed surface as SD/SDXL.

    Klein conditions on ONE tensor: the multi-layer Qwen3 hidden states that
    ``Flux2KleinPipeline.encode_prompt`` extracts (``text_encoder_out_layers``
    concatenated), fed back verbatim via ``prompt_embeds``. Distilled model:
    embedded guidance, no negative branch. VRAM: the 4B fits a 24GB card whole;
    the 9B (and its Qwen3-8B encoder) rides ``enable_model_cpu_offload``.
    """

    pipe: object
    device: str
    model_id: str

    #: mining pads/truncates prompts to this many tokens so every corpus row has
    #: one fixed feature shape (full 512 would make the PCA npz enormous).
    MAX_SEQ = 128

    @classmethod
    def load(cls, model_id: str = "black-forest-labs/FLUX.2-klein-4B",
             device: Optional[str] = None, dtype: str = "auto",
             ckpt: Optional[str] = None, encode_only: bool = False) -> "Flux2Model":
        import torch
        from diffusers import Flux2KleinPipeline

        device = _pick_device(device)
        kw = _flow_pipeline_kwargs(model_id, encode_only=encode_only)
        offload = kw.pop("_offload")
        pipe = _load_encode_only(
            lambda skip: Flux2KleinPipeline.from_pretrained(model_id, **kw, **skip),
            Flux2KleinPipeline, encode_only)
        if offload:
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe = pipe.to(device)
        else:
            pipe = pipe.to(device)   # NF4 components already live on GPU
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe=pipe, device=device, model_id=model_id)

    # ----------------------------------------------------------- encoding ---
    def encode_prompts(self, prompts, batch_size: int = 8, on_batch=None) -> np.ndarray:
        import torch

        def one(chunk):
            with torch.no_grad():
                pe = self.pipe.encode_prompt(
                    prompt=chunk, device=self.device,
                    num_images_per_prompt=1, prompt_embeds=None,
                    max_sequence_length=self.MAX_SEQ)
            if isinstance(pe, (tuple, list)):
                pe = pe[0]
            return pe.detach().to(torch.float32).cpu().numpy()

        return np.concatenate(_encode_batches(prompts, batch_size, one, on_batch), axis=0)

    # ----------------------------------------------------------- decoding ---
    def generate_from_embeddings(self, embeddings, negative_embedding=None,
                                 num_inference_steps: int = 28,
                                 guidance_scale: float = 4.0,
                                 height: int = 1024, width: int = 1024,
                                 seed: Optional[int] = None,
                                 init_images=None, init_strength: float = 0.7,
                                 ip_images=None, ip_scale: float = 0.7,
                                 on_image=None) -> list:
        import torch

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 2:
            embeddings = embeddings[None]
        n = embeddings.shape[0]
        pe = torch.from_numpy(embeddings).to(dtype=torch.bfloat16)

        images = []
        for i in range(n):
            generator = (torch.Generator(device="cpu").manual_seed(seed + i)
                         if seed is not None else None)
            kwargs = dict(prompt_embeds=pe[i:i + 1].to(self.device),
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale,
                          height=height, width=width, generator=generator)
            if init_images:
                # klein's native image conditioning (kontext-style reference)
                kwargs["image"] = [init_images[i % len(init_images)].convert("RGB")]
            _stream(images, i, self.pipe(**kwargs).images[0], on_image)
        return images


@dataclass
class Krea2Model:
    """Krea 2 (Raw/Turbo) behind the same duck-typed surface.

    Conditions on ONE tensor of multi-layer Qwen3-VL hidden states plus an
    attention mask; we mine at a fixed MAX_SEQ so shapes are uniform and feed an
    all-ones mask at generation (padding positions become live conditioning --
    anarchy by design). Real CFG: guidance_scale ~4.5 with an empty-prompt
    negative branch.
    """

    pipe: object
    device: str
    model_id: str
    _uncond: Optional[tuple] = None

    # Krea's encoder returns 12 stacked layers -> ~2560*12 dims per token; mine
    # short sequences and accumulate fp16 or a 30GB-RAM box drowns.
    MAX_SEQ = 64

    @classmethod
    def load(cls, model_id: str = "krea/Krea-2-Raw",
             device: Optional[str] = None, dtype: str = "auto",
             ckpt: Optional[str] = None, encode_only: bool = False) -> "Krea2Model":
        import torch
        from diffusers import Krea2Pipeline

        device = _pick_device(device)
        kw = _flow_pipeline_kwargs(model_id, encode_only=encode_only)
        offload = kw.pop("_offload")
        pipe = _load_encode_only(
            lambda skip: Krea2Pipeline.from_pretrained(model_id, **kw, **skip),
            Krea2Pipeline, encode_only)
        if offload:
            try:
                pipe.enable_model_cpu_offload()
            except Exception:
                pipe = pipe.to(device)
        else:
            pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)
        return cls(pipe=pipe, device=device, model_id=model_id)

    # ----------------------------------------------------------- encoding ---
    def encode_prompts(self, prompts, batch_size: int = 8, on_batch=None) -> np.ndarray:
        import torch

        def one(chunk):
            with torch.no_grad():
                enc = self.pipe.encode_prompt(
                    prompt=chunk, device=self.device,
                    num_images_per_prompt=1, prompt_embeds=None,
                    prompt_embeds_mask=None, max_sequence_length=self.MAX_SEQ)
            pe = enc[0] if isinstance(enc, (tuple, list)) else enc
            return pe.detach().to(torch.float16).cpu().numpy()

        return np.concatenate(_encode_batches(prompts, batch_size, one, on_batch), axis=0)

    def _uncond_embeds(self):
        import torch
        if self._uncond is None:
            with torch.no_grad():
                enc = self.pipe.encode_prompt(
                    prompt=[""], device=self.device, num_images_per_prompt=1,
                    prompt_embeds=None, prompt_embeds_mask=None,
                    max_sequence_length=self.MAX_SEQ)
            self._uncond = (enc[0], enc[1] if isinstance(enc, (tuple, list)) and len(enc) > 1 else None)
        return self._uncond

    # ----------------------------------------------------------- decoding ---
    def generate_from_embeddings(self, embeddings, negative_embedding=None,
                                 num_inference_steps: int = 28,
                                 guidance_scale: float = 4.5,
                                 height: int = 1024, width: int = 1024,
                                 seed: Optional[int] = None,
                                 init_images=None, init_strength: float = 0.7,
                                 ip_images=None, ip_scale: float = 0.7,
                                 on_image=None) -> list:
        import torch

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 2:
            embeddings = embeddings[None]
        n, s = embeddings.shape[0], embeddings.shape[1]
        pe = torch.from_numpy(embeddings).to(dtype=torch.bfloat16)
        mask = torch.ones((1, s), dtype=torch.bool)

        use_cfg = guidance_scale > 1.0
        if use_cfg:
            npe, nmask = self._uncond_embeds()

        images = []
        for i in range(n):
            generator = (torch.Generator(device="cpu").manual_seed(seed + i)
                         if seed is not None else None)
            kwargs = dict(prompt_embeds=pe[i:i + 1].to(self.device),
                          prompt_embeds_mask=mask.to(self.device),
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale,
                          height=height, width=width, generator=generator)
            if use_cfg:
                kwargs["negative_prompt_embeds"] = npe.to(self.device)
                if nmask is not None:
                    kwargs["negative_prompt_embeds_mask"] = nmask.to(self.device)
            _stream(images, i, self.pipe(**kwargs).images[0], on_image)
        return images
