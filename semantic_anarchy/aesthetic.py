"""Aesthetic scoring -- "selection through aesthetic resonance" (slide 13).

Once we are sampling conditioning embeddings with no linguistic meaning, the only
way to steer is *taste*: decode a batch, score the images, keep what resonates.
The evolution loop (:mod:`semantic_anarchy.evolve`) consumes a :class:`Scorer`
to decide which embeddings survive into the next "branch".

Three scorers ship here, all behind the same ``score(images) -> np.ndarray``
interface:

* :class:`AestheticScorer` -- the real thing: CLIP image features fed through the
  LAION aesthetic-predictor linear head (the same one Eden used for selection).
* :class:`HumanScorer` -- you are the aesthetic oracle; it shows images and asks.
* :class:`RandomScorer` -- a dependency-free stand-in for dry runs and tests.

torch / transformers are imported lazily and *every* failure path degrades
gracefully -- a missing weight file or absent torch must never hard-crash the
package, only the real CLIP path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .clip_compat import image_features
from .io_utils import image_ext, save_image

# The LAION aesthetic predictor expects a 768-d CLIP ViT-L/14 image embedding.
_CLIP_MODEL_ID = "openai/clip-vit-large-patch14"
_CLIP_EMBED_DIM = 768

# Where we look for the LAION linear head by default. Get it from
#   https://github.com/christophschuhmann/improved-aesthetic-predictor
# (file ``sac+logos+ava1-l14-linearMSE.pth``) or the LAION-Aesthetics release.
_DEFAULT_WEIGHTS = Path("weights/sac+logos+ava1-l14-linearMSE.pth")


class Scorer:
    """Interface: turn a list of PIL images into a 1-D array of scores.

    Higher is "better" / more aesthetically resonant. Subclasses override
    :meth:`score`. Kept deliberately tiny so the evolution loop can be unit
    tested against a plain Python callable instead of a model.
    """

    name = "base"

    def score(self, images: Sequence) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def __call__(self, images: Sequence) -> np.ndarray:
        return self.score(images)


class RandomScorer(Scorer):
    """Deterministic-ish random scores -- a fallback for dry runs with no model.

    Useful to exercise the full sample -> score -> select -> refit plumbing
    without torch, CLIP weights, or even real images present.
    """

    name = "random"

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def score(self, images: Sequence) -> np.ndarray:
        return self.rng.uniform(0.0, 10.0, size=len(images)).astype(np.float32)


class HumanScorer(Scorer):
    """Interactive scorer -- you rate each shown image (slide 13, literally).

    "Mining human preferences" by hand. Each image is saved to ``display_dir``
    (so you can open it) and you type a 1-10 score at the prompt. Pressing enter
    accepts the ``default``. This is the ground-truth signal the automated
    aesthetic head is only ever approximating.
    """

    name = "human"

    def __init__(self, display_dir: str | Path = "outputs/rate", default: float = 5.0):
        self.display_dir = Path(display_dir)
        self.display_dir.mkdir(parents=True, exist_ok=True)
        self.default = float(default)

    def score(self, images: Sequence) -> np.ndarray:
        scores = []
        for i, img in enumerate(images):
            path = self.display_dir / f"rate_{i:03d}{image_ext()}"
            try:
                save_image(img, path)
            except Exception:
                path = None
            where = f" (saved {path})" if path else ""
            raw = input(f"Score image {i + 1}/{len(images)}{where} [0-10, enter={self.default}]: ")
            try:
                scores.append(float(raw))
            except ValueError:
                scores.append(self.default)
        return np.asarray(scores, dtype=np.float32)


class AestheticScorer(Scorer):
    """LAION aesthetic predictor on top of CLIP ViT-L/14 image features.

    Pipeline: image -> CLIP image encoder -> L2-normalized 768-d embedding ->
    linear MLP head -> scalar aesthetic score (the LAION-Aesthetics convention,
    roughly 1-10). Weights are loaded from ``weights_path`` if present.

    Degrades gracefully: construction never raises. If torch / transformers /
    the weight file are missing, :attr:`available` stays False and
    :meth:`score` falls back to a :class:`RandomScorer` (with a one-time
    warning) so an evolution run still completes instead of crashing.
    """

    name = "aesthetic"

    def __init__(
        self,
        weights_path: str | Path = _DEFAULT_WEIGHTS,
        device: Optional[str] = None,
        clip_model_id: str = _CLIP_MODEL_ID,
    ):
        self.weights_path = Path(weights_path)
        self.clip_model_id = clip_model_id
        self.available = False
        self._reason = ""
        self._fallback = RandomScorer()
        self._model = None
        self._clip = None
        self._processor = None
        self._device = device
        self._try_load()

    def _try_load(self) -> None:
        try:
            import torch  # noqa: F401
            from transformers import CLIPModel, CLIPProcessor
        except Exception as exc:  # torch / transformers absent
            self._reason = f"torch/transformers unavailable ({exc!r})"
            return
        if not self.weights_path.exists():
            self._reason = (
                f"aesthetic weights not found at {self.weights_path} -- "
                "download sac+logos+ava1-l14-linearMSE.pth from the "
                "improved-aesthetic-predictor repo"
            )
            return
        try:
            import torch

            if self._device is None:
                if torch.cuda.is_available():
                    self._device = "cuda"
                elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    self._device = "mps"
                else:
                    self._device = "cpu"
            self._clip = CLIPModel.from_pretrained(self.clip_model_id).to(self._device).eval()
            self._processor = CLIPProcessor.from_pretrained(self.clip_model_id)
            self._model = _build_aesthetic_head(_CLIP_EMBED_DIM)
            state = torch.load(self.weights_path, map_location="cpu")
            # Some releases wrap the MLP in a `layers` module -> keys are
            # "layers.0.weight" etc. Strip that prefix to match our bare Sequential.
            if any(k.startswith("layers.") for k in state):
                state = {k.replace("layers.", "", 1): v for k, v in state.items()}
            self._model.load_state_dict(state)
            self._model = self._model.to(self._device).eval()
            self.available = True
        except Exception as exc:  # corrupt weights, shape mismatch, OOM...
            self._reason = f"failed to initialise aesthetic head ({exc!r})"
            self.available = False

    def score(self, images: Sequence) -> np.ndarray:
        if not self.available:
            if not getattr(self, "_warned", False):
                print(f"[AestheticScorer] unavailable: {self._reason}. Falling back to random scores.")
                self._warned = True
            return self._fallback.score(images)

        import torch

        with torch.no_grad():
            inputs = self._processor(images=list(images), return_tensors="pt").to(self._device)
            feats = image_features(self._clip, inputs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            scores = self._model(feats.float()).squeeze(-1)
        return scores.detach().cpu().numpy().astype(np.float32)


def _build_aesthetic_head(input_dim: int):
    """The LAION improved-aesthetic-predictor MLP (matches its state_dict keys)."""
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, 1024),
        nn.Dropout(0.2),
        nn.Linear(1024, 128),
        nn.Dropout(0.2),
        nn.Linear(128, 64),
        nn.Dropout(0.1),
        nn.Linear(64, 16),
        nn.Linear(16, 1),
    )


def get_scorer(name: str = "random", **kwargs) -> Scorer:
    """Factory: ``"aesthetic"`` | ``"human"`` | ``"random"`` -> a :class:`Scorer`.

    Unknown names raise; ``"aesthetic"`` never raises here -- it returns a scorer
    that internally falls back to random if its model can't load.
    """
    name = name.lower()
    if name == "aesthetic":
        return AestheticScorer(**kwargs)
    if name == "human":
        return HumanScorer(**kwargs)
    if name == "random":
        return RandomScorer(**kwargs)
    raise ValueError(f"unknown scorer {name!r}; choose aesthetic | human | random")
