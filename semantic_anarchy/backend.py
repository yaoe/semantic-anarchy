"""Backend abstraction -- run the SAME promptless drift method on SD1.5 or SDXL.

The whole "Semantic Anarchy" pipeline is model-agnostic once you describe a model
by its *named conditioning tensors*:

* SD1.5 conditions on ONE tensor:  ``{"embeds": (N, 77, 768)}``.
* SDXL conditions on TWO:          ``{"prompt_embeds": (N, 77, 2048),
                                      "pooled": (N, 1280)}``.

A :class:`Backend` exposes four verbs that every script drives generically:

* ``encode(prompts)``  -> ``{name: ndarray (N, *shape)}``       (mine corpus)
* ``fit(named)``       -> ``{name: EmbeddingDistribution}``      (one per tensor)
* ``sample(dists, ...)`` -> ``{name: ndarray (n, *shape)}``     (drift)
* ``generate(named, ...)`` -> ``list[PIL.Image]``               (decode)

The same ``--sampler/--temperature/--coherence/--components/--truncation`` knobs
flow straight through to :meth:`EmbeddingDistribution.sample`, so sd15 and sdxl
share *identical* drift/sweep/evolve logic. No SAE, no hooks -- stock pipelines.

torch / diffusers are imported lazily (inside ``load``), so ``fit``/``sample``
and the tests run with neither installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .distribution import EmbeddingDistribution


# Default generation knobs per model family (documented in the README / scripts).
BACKEND_DEFAULTS = {
    "sd15": {"steps": 30, "guidance": 7.5, "height": 512, "width": 512},
    # SD 2.1 (768, v-prediction): one text encoder like sd15 but OpenCLIP-H
    # (hidden 1024) and a native 768x768. Likes slightly higher guidance.
    "sd2": {"steps": 30, "guidance": 9.0, "height": 768, "width": 768},
    # SDXL family default to Turbo's 1-step/no-CFG; override for base SDXL with
    # --steps 30 --guidance 7 (see README "defaults" table).
    "sdxl": {"steps": 1, "guidance": 0.0, "height": 1024, "width": 1024},
    # FLUX.2 klein (distilled flow model, embedded guidance, Qwen3 encoder).
    "flux2": {"steps": 28, "guidance": 4.0, "height": 1024, "width": 1024},
    # Krea 2 Raw (undistilled flow model, real CFG, Qwen3-VL encoder).
    "krea2": {"steps": 28, "guidance": 4.5, "height": 1024, "width": 1024},
}


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interpolation between two (arbitrary-shape) embeddings.

    Interpolates direction on the great circle and magnitude linearly -- the
    standard "concept fusion" blend that stays on the shell the endpoints live
    on (plain lerp cuts through the low-magnitude interior). Falls back to lerp
    for (near-)parallel inputs.
    """
    fa, fb = a.reshape(-1), b.reshape(-1)
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    if na < 1e-8 or nb < 1e-8:
        return ((1 - t) * a + t * b)
    ua, ub = fa / na, fb / nb
    dot = float(np.clip(ua @ ub, -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-4:
        direction = (1 - t) * ua + t * ub
    else:
        so = np.sin(omega)
        direction = (np.sin((1 - t) * omega) / so) * ua + (np.sin(t * omega) / so) * ub
    mag = (1 - t) * na + t * nb
    return (mag * direction).reshape(a.shape)


class Backend:
    """Common interface; subclasses wrap a concrete SD pipeline.

    ``tensor_names`` lists the named conditioning tensors this backend mines and
    samples (one for sd15, two for sdxl). ``feature_shapes`` maps each to its
    per-sample shape (e.g. ``(77, 768)``). These come from the loaded model, but
    subclasses may expose them statically for torch-free tests via
    :meth:`describe`.
    """

    name: str = "base"
    tensor_names: tuple[str, ...] = ()

    # ---- distribution math (torch-free) ------------------------------------
    def fit(self, named, per_token: bool = True,
            n_components: Optional[int] = None) -> dict:
        """Fit one :class:`EmbeddingDistribution` per named tensor."""
        return {
            k: EmbeddingDistribution.fit(np.asarray(v), per_token=per_token,
                                         n_components=n_components)
            for k, v in named.items()
        }

    def sample(self, dists, n: int = 1, **kw) -> dict:
        """Sample every named tensor with the SAME sampler/temperature/etc.

        ``kw`` is forwarded verbatim to :meth:`EmbeddingDistribution.sample`
        (temperature, truncation, rng, sampler, coherence, components).
        """
        return {k: d.sample(n=n, **kw) for k, d in dists.items()}

    def perturb(self, dists, anchors, n: int = 6, radius: float = 0.3,
                rng=None) -> dict:
        """Sample ``n`` points around a per-tensor anchor dict (local search).

        ``anchors`` maps tensor name -> one embedding (e.g. loaded from a
        generated image's ``.npz`` sidecar). Every named tensor is perturbed
        with the SAME radius so the sample stays a coherent conditioning set.
        """
        rng = rng or np.random.default_rng()
        return {k: d.neighborhood(anchors[k], n=n, radius=radius, rng=rng)
                for k, d in dists.items()}

    def breed(self, dists, a, b, n: int = 6, mutate: float = 0.15,
              rng=None) -> dict:
        """Picbreeder-style children of two parents: SLERP blends + mutation.

        ``a``/``b`` are per-tensor anchor dicts. Children are spread across the
        interpolation interval (t in [0.15, 0.85]) with a small ``mutate``-radius
        perturbation on top so siblings differ even at the same t.
        """
        rng = rng or np.random.default_rng()
        ts = np.linspace(0.15, 0.85, n)
        out = {}
        for k, d in dists.items():
            kids = np.stack([_slerp(np.asarray(a[k], dtype=np.float64),
                                    np.asarray(b[k], dtype=np.float64), t)
                             for t in ts]).astype(np.float32)
            if mutate > 0:
                kids = kids + mutate * (
                    d._pca_dev(n, None, rng, None)
                    if d.pca_components is not None else
                    d._diagonal_dev(n, None, rng)
                ).astype(np.float32)
            out[k] = kids
        return out

    def walk(self, dists, anchors, steps: int = 6, step: float = 0.15,
             mode: str = "outward", rng=None, axis=None) -> dict:
        """Persistent-direction walk from a per-tensor anchor dict (see
        :meth:`EmbeddingDistribution.walk`)."""
        rng = rng or np.random.default_rng()
        return {k: d.walk(anchors[k], steps=steps, step=step, mode=mode,
                          rng=rng, axis=axis)
                for k, d in dists.items()}

    def floor_distance(self, dists, named, floor: float) -> dict:
        """Push any sample whose distance gauge is below ``floor`` out onto it.

        Keeps the direction; rescales the deviation (all tensors by the same
        factor so the conditioning stays coherent). The hunter's "never below
        d=1.0" guard -- inside the corpus core everything is bland.
        """
        names = list(named.keys())
        n = len(np.asarray(named[names[0]]))
        out = {k: np.asarray(v, dtype=np.float64).copy() for k, v in named.items()}
        for i in range(n):
            d = self.distance(dists, {k: out[k][i] for k in names})
            if d < floor:
                f = floor / max(d, 1e-6)
                for k in names:
                    out[k][i] = dists[k].mean + (out[k][i] - dists[k].mean) * f
        return {k: v.astype(np.float32) for k, v in out.items()}

    def retarget(self, dists, named, target: float) -> dict:
        """Pin every sampled tensor's distance gauge to ``target`` (shell sampling)."""
        return {k: dists[k].retarget(v, target) for k, v in named.items()}

    def distance(self, dists, named_one) -> float:
        """Mean RMS z-score distance of one per-tensor sample from the corpus center."""
        vals = [dists[k].distance(named_one[k]) for k in dists]
        return float(np.mean(vals))

    def _tensor_prefix(self, prefix: Path, name: str) -> Path:
        """Per-tensor save prefix. Use a ``__`` separator (not a dot) so the tensor
        name can't be mistaken for a path suffix by ``Path.with_suffix``."""
        return prefix.with_name(f"{prefix.name}__{name}")

    def save_dists(self, dists, prefix: str | Path) -> list[Path]:
        """Save each named distribution.

        Single-tensor backends (sd15) save the lone tensor at ``<prefix>``
        directly (keeping the original ``outputs/dist`` layout). Multi-tensor
        backends (sdxl) save to ``<prefix>__<name>`` siblings. Returns the
        ``.npz`` paths actually written.
        """
        prefix = Path(prefix)
        written = []
        if len(self.tensor_names) == 1:
            (name,) = self.tensor_names
            dists[name].save(prefix)
            written.append(Path(str(prefix) + ".npz") if prefix.suffix != ".npz" else prefix)
        else:
            for name in self.tensor_names:
                p = self._tensor_prefix(prefix, name)
                dists[name].save(p)
                written.append(Path(str(p) + ".npz") if p.suffix != ".npz" else p)
        return written

    def load_dists(self, prefix: str | Path) -> dict:
        """Inverse of :meth:`save_dists`."""
        prefix = Path(prefix)
        if len(self.tensor_names) == 1:
            (name,) = self.tensor_names
            return {name: EmbeddingDistribution.load(prefix)}
        return {
            name: EmbeddingDistribution.load(self._tensor_prefix(prefix, name))
            for name in self.tensor_names
        }

    # ---- model-touching verbs (subclasses implement) -----------------------
    def encode(self, prompts) -> dict:                       # pragma: no cover
        raise NotImplementedError

    def generate(self, named, **kw) -> list:                 # pragma: no cover
        raise NotImplementedError


class SD15Backend(Backend):
    """SD1.5: one conditioning tensor ``embeds`` of shape ``(77, 768)``."""

    name = "sd15"
    tensor_names = ("embeds",)

    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, model_id=None, ckpt=None, device=None):
        from .pipeline import SDModel
        model = SDModel.load(
            model_id=model_id or "runwayml/stable-diffusion-v1-5",
            device=device, ckpt=ckpt)
        return cls(model)

    def encode(self, prompts) -> dict:
        return {"embeds": self.model.encode_prompts(prompts)}

    def generate(self, named, guidance=7.5, steps=30, seed=None,
                 height=512, width=512, neg_mode="empty",
                 init_images=None, init_strength=0.7,
                 ip_images=None, ip_scale=0.7) -> list:
        # SD1.5 CFG negative is the empty-prompt encoding (SDModel handles it).
        return self.model.generate_from_embeddings(
            named["embeds"], num_inference_steps=steps, guidance_scale=guidance,
            height=height, width=width, seed=seed,
            init_images=init_images, init_strength=init_strength,
            ip_images=ip_images, ip_scale=ip_scale)


class SD2Backend(SD15Backend):
    """SD 2.1: one conditioning tensor ``embeds`` of shape ``(77, 1024)``.

    Architecturally identical to sd15 from this pipeline's view -- a single
    ``StableDiffusionPipeline`` whose text encoder happens to be OpenCLIP-H
    (hidden 1024). Only the default model and the namespacing name differ, so we
    inherit sd15's encode/generate verbatim.
    """

    name = "sd2"

    @classmethod
    def load(cls, model_id=None, ckpt=None, device=None):
        import os
        from .pipeline import SDModel
        # SD2's canonical config repo is gated; when loading a single-file ckpt
        # read the config from a NON-gated diffusers mirror (override via env).
        sf_config = (os.environ.get("SA_SD2_CONFIG", "philschmid/stable-diffusion-2-1")
                     if ckpt else None)
        model = SDModel.load(
            model_id=model_id or "stabilityai/stable-diffusion-2-1",
            device=device, ckpt=ckpt, single_file_config=sf_config)
        return cls(model)


class SDXLBackend(Backend):
    """SDXL: two tensors -- ``prompt_embeds`` (77,2048) and ``pooled`` (1280)."""

    name = "sdxl"
    tensor_names = ("prompt_embeds", "pooled")

    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, model_id=None, ckpt=None, device=None):
        from .pipeline import SDXLModel
        model = SDXLModel.load(
            model_id=model_id or "stabilityai/sdxl-turbo",
            device=device, ckpt=ckpt)
        return cls(model)

    def encode(self, prompts) -> dict:
        return self.model.encode_prompts(prompts)

    def generate(self, named, dists=None, guidance=0.0, steps=1, seed=None,
                 height=1024, width=1024, neg_mode="mean",
                 init_images=None, init_strength=0.7,
                 ip_images=None, ip_scale=0.7) -> list:
        negatives = self._negatives(named, dists, neg_mode) if guidance > 1.0 else None
        return self.model.generate_from_embeddings(
            named, negatives=negatives, num_inference_steps=steps,
            guidance_scale=guidance, height=height, width=width, seed=seed,
            init_images=init_images, init_strength=init_strength,
            ip_images=ip_images, ip_scale=ip_scale)

    def _negatives(self, named, dists, neg_mode) -> dict:
        """Build the CFG negative for both tensors.

        mean  -> the corpus mean (from the fitted distributions); CFG pushes away
                 from the average prompt toward the sampled/extrapolated point.
        empty -> the empty-prompt encoding.
        zeros -> zero tensors of the right shape.
        """
        if neg_mode == "mean" and dists is not None:
            return {k: dists[k].mean[None] for k in self.tensor_names}
        if neg_mode == "zeros":
            return {k: np.zeros((1,) + named[k].shape[1:], dtype=np.float32)
                    for k in self.tensor_names}
        return self.model.uncond()  # "empty"


class Flux2Backend(SD15Backend):
    """FLUX.2 klein: one conditioning tensor ``embeds`` of multi-layer Qwen3
    hidden states (mined at a fixed 128-token length). Inherits the single-
    tensor fit/sample/save plumbing from sd15."""

    name = "flux2"

    @classmethod
    def load(cls, model_id=None, ckpt=None, device=None):
        import os
        from .pipeline import Flux2Model
        model = Flux2Model.load(
            model_id=model_id or os.environ.get(
                "SA_FLUX2_MODEL", "black-forest-labs/FLUX.2-klein-4B"),
            device=device)
        return cls(model)

    def generate(self, named, guidance=4.0, steps=28, seed=None,
                 height=1024, width=1024, neg_mode="empty",
                 init_images=None, init_strength=0.7,
                 ip_images=None, ip_scale=0.7) -> list:
        return self.model.generate_from_embeddings(
            named["embeds"], num_inference_steps=steps, guidance_scale=guidance,
            height=height, width=width, seed=seed,
            init_images=init_images, init_strength=init_strength)


class Krea2Backend(SD15Backend):
    """Krea 2: one conditioning tensor of Qwen3-VL multi-layer hidden states."""

    name = "krea2"

    @classmethod
    def load(cls, model_id=None, ckpt=None, device=None):
        import os
        from .pipeline import Krea2Model
        model = Krea2Model.load(
            model_id=model_id or os.environ.get("SA_KREA2_MODEL", "krea/Krea-2-Raw"),
            device=device)
        return cls(model)

    def generate(self, named, guidance=4.5, steps=28, seed=None,
                 height=1024, width=1024, neg_mode="empty",
                 init_images=None, init_strength=0.7,
                 ip_images=None, ip_scale=0.7) -> list:
        return self.model.generate_from_embeddings(
            named["embeds"], num_inference_steps=steps, guidance_scale=guidance,
            height=height, width=width, seed=seed)


def make_backend(name: str, model_id=None, ckpt=None, device=None) -> Backend:
    """Load the backend named ``sd15`` or ``sdxl`` (touches torch/diffusers)."""
    if name == "sd15":
        return SD15Backend.load(model_id=model_id, ckpt=ckpt, device=device)
    if name == "sd2":
        return SD2Backend.load(model_id=model_id, ckpt=ckpt, device=device)
    if name == "sdxl":
        return SDXLBackend.load(model_id=model_id, ckpt=ckpt, device=device)
    if name == "flux2":
        return Flux2Backend.load(model_id=model_id, ckpt=ckpt, device=device)
    if name == "krea2":
        return Krea2Backend.load(model_id=model_id, ckpt=ckpt, device=device)
    raise ValueError(f"unknown backend {name!r}; choose sd15 | sd2 | sdxl | flux2 | krea2")


# Lightweight, torch-free handles for the distribution-only verbs (fit/sample/
# save/load) -- used by tests and any code that has named arrays already and
# doesn't need to load a model.
def dist_backend(name: str) -> Backend:
    """Return a model-less Backend instance exposing only the numpy verbs."""
    if name == "sd15":
        b = SD15Backend.__new__(SD15Backend)
    elif name == "sd2":
        b = SD2Backend.__new__(SD2Backend)
    elif name == "sdxl":
        b = SDXLBackend.__new__(SDXLBackend)
    elif name == "flux2":
        b = Flux2Backend.__new__(Flux2Backend)
    elif name == "krea2":
        b = Krea2Backend.__new__(Krea2Backend)
    else:
        raise ValueError(f"unknown backend {name!r}; choose sd15 | sd2 | sdxl | flux2 | krea2")
    return b
