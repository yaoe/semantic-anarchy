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
    # SDXL family default to Turbo's 1-step/no-CFG; override for base SDXL with
    # --steps 30 --guidance 7 (see README "defaults" table).
    "sdxl": {"steps": 1, "guidance": 0.0, "height": 1024, "width": 1024},
}


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
                 height=512, width=512, neg_mode="empty") -> list:
        # SD1.5 CFG negative is the empty-prompt encoding (SDModel handles it).
        return self.model.generate_from_embeddings(
            named["embeds"], num_inference_steps=steps, guidance_scale=guidance,
            height=height, width=width, seed=seed)


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
                 height=1024, width=1024, neg_mode="mean") -> list:
        negatives = self._negatives(named, dists, neg_mode) if guidance > 1.0 else None
        return self.model.generate_from_embeddings(
            named, negatives=negatives, num_inference_steps=steps,
            guidance_scale=guidance, height=height, width=width, seed=seed)

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


def make_backend(name: str, model_id=None, ckpt=None, device=None) -> Backend:
    """Load the backend named ``sd15`` or ``sdxl`` (touches torch/diffusers)."""
    if name == "sd15":
        return SD15Backend.load(model_id=model_id, ckpt=ckpt, device=device)
    if name == "sdxl":
        return SDXLBackend.load(model_id=model_id, ckpt=ckpt, device=device)
    raise ValueError(f"unknown backend {name!r}; choose sd15 | sdxl")


# Lightweight, torch-free handles for the distribution-only verbs (fit/sample/
# save/load) -- used by tests and any code that has named arrays already and
# doesn't need to load a model.
def dist_backend(name: str) -> Backend:
    """Return a model-less Backend instance exposing only the numpy verbs."""
    if name == "sd15":
        b = SD15Backend.__new__(SD15Backend)
    elif name == "sdxl":
        b = SDXLBackend.__new__(SDXLBackend)
    else:
        raise ValueError(f"unknown backend {name!r}; choose sd15 | sdxl")
    return b
