"""Learn and sample the semantic distribution of conditioning embeddings.

This is the statistical heart of "Semantic Anarchy". Stable Diffusion conditions
its UNet on a text-conditioning tensor ``c`` of shape ``(77, 768)`` (SD 1.x) that
normally comes from running a prompt through the CLIP text encoder.

The deck's claim: if you encode a corpus of "good prompts" and look at where their
``c`` tensors land, you can *learn that distribution* and then sample brand-new
``c`` directly -- never writing a prompt again. Slides 6-7 of the deck show the
model used: an **independent Gaussian per coordinate** (each subplot is one
coordinate's histogram with a fitted ``Mean`` / ``Std``).

Everything here is pure NumPy so it can be tested and demoed without torch / a GPU.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np


def _truncate(z: np.ndarray, truncation: Optional[float], rng) -> np.ndarray:
    """Resample then clip standard-normal coefficients to a typical set.

    Any coefficient beyond ``truncation`` sigma is redrawn a few times, then hard
    clipped -- the GAN "truncation trick", shared by every sampler so the knob
    means the same thing whether we draw per-coordinate or per-component.
    """
    if truncation is None:
        return z
    mask = np.abs(z) > truncation
    for _ in range(8):
        if not mask.any():
            break
        z[mask] = rng.standard_normal(int(mask.sum()))
        mask = np.abs(z) > truncation
    return np.clip(z, -truncation, truncation)


def _slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two flat vectors at fraction ``t``.

    Interpolates the *direction* on the unit hypersphere (so a fusion keeps the
    magnitude scale of the endpoints rather than collapsing toward zero like a
    plain lerp). Falls back to a plain lerp when the two vectors are nearly
    parallel (tiny angle) to avoid a divide-by-near-zero.
    """
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    if na < 1e-12 or nb < 1e-12:
        return (1 - t) * a64 + t * b64
    ua, ub = a64 / na, b64 / nb
    dot = float(np.clip(np.dot(ua, ub), -1.0, 1.0))
    theta = np.arccos(dot)
    if theta < 1e-4:  # nearly parallel -> lerp is numerically safe
        return (1 - t) * a64 + t * b64
    sin_theta = np.sin(theta)
    wa = np.sin((1 - t) * theta) / sin_theta
    wb = np.sin(t * theta) / sin_theta
    # Interpolate direction, then a lerp of the magnitudes for a natural scale.
    direction = wa * ua + wb * ub
    mag = (1 - t) * na + t * nb
    return direction * mag


@dataclass
class EmbeddingDistribution:
    """A per-coordinate Gaussian fit over conditioning embeddings.

    Attributes
    ----------
    mean, std:
        Arrays broadcastable to a single embedding's ``feature_shape`` (e.g.
        ``(77, 768)``). When ``per_token`` is False these collapse the token axis
        so every token position shares one ``(768,)`` Gaussian.
    feature_shape:
        Shape of a single embedding, e.g. ``(77, 768)``.
    per_token:
        Whether statistics are estimated independently per token position.
    n_samples:
        How many embeddings the fit was estimated from.
    pca_components, pca_std:
        Optional low-rank structure of the corpus cloud. ``pca_components`` is
        ``(k, D)`` (the top ``k`` right-singular vectors of the centered, flattened
        embeddings) and ``pca_std`` is ``(k,)`` (the per-component standard
        deviation). They let us sample *along the corpus manifold* rather than
        treating every coordinate as independent -- the ``"pca"`` / ``"blend"``
        samplers trade off this coherence against the per-coordinate "anarchy".
    total_var:
        Total variance of the centered corpus (``sum of all coordinate variances``),
        used to report what fraction the retained components capture.
    corpus_embeddings:
        Optional retained subset of the raw fitted embeddings, shape
        ``(M, *feature_shape)`` (capped at fit time). The ``"hybrid"`` sampler
        SLERPs between random pairs of these to fuse two real "concepts" into
        surprising new ones.
    """

    mean: np.ndarray
    std: np.ndarray
    feature_shape: tuple[int, ...]
    per_token: bool = True
    n_samples: int = 0
    pca_components: Optional[np.ndarray] = None
    pca_std: Optional[np.ndarray] = None
    total_var: float = 0.0
    corpus_embeddings: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit ---
    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        per_token: bool = True,
        std_floor: float = 1e-4,
        n_components: Optional[int] = None,
        max_corpus: int = 256,
    ) -> "EmbeddingDistribution":
        """Fit independent Gaussians (and low-rank PCA structure) to embeddings.

        Parameters
        ----------
        embeddings:
            Array of shape ``(N, *feature_shape)``, e.g. ``(1000, 77, 768)``.
        per_token:
            If True (default), estimate ``mean``/``std`` independently for every
            ``(token, feature)`` coordinate. If False, pool over the token axis
            so all 77 positions share one ``(768,)`` Gaussian -- a stronger prior
            that produces calmer, more "typical" samples.
        std_floor:
            Minimum std, guarding against degenerate (collapsed) coordinates.
        n_components:
            How many principal components of the (flattened, centered) corpus to
            keep. ``None`` keeps the maximum (``N-1``). These power the ``"pca"``
            and ``"blend"`` samplers; the per-coordinate ``mean``/``std`` above are
            unchanged.
        max_corpus:
            Cap on how many raw embeddings to retain as ``corpus_embeddings`` (for
            the ``"hybrid"`` SLERP sampler). If the corpus is larger, a random
            subset of this many rows is kept to bound the saved size.
        """
        embeddings = np.asarray(embeddings, dtype=np.float64)
        if embeddings.ndim < 2:
            raise ValueError("embeddings must be (N, *feature_shape)")
        feature_shape = embeddings.shape[1:]
        n = embeddings.shape[0]

        if per_token:
            mean = embeddings.mean(axis=0)
            std = embeddings.std(axis=0)
        else:
            # Pool across the leading "token" axis (axis 1) as well as samples.
            flat = embeddings.reshape(embeddings.shape[0], feature_shape[0], -1)
            mean_f = flat.mean(axis=(0, 1))  # (features,)
            std_f = flat.std(axis=(0, 1))
            mean = np.broadcast_to(mean_f, feature_shape).copy()
            std = np.broadcast_to(std_f, feature_shape).copy()

        std = np.maximum(std, std_floor)

        # ---- low-rank PCA over the flattened, centered embeddings (float64) ----
        pca_components = pca_std = None
        total_var = 0.0
        if n > 1:
            x_flat = embeddings.reshape(n, -1)
            x_centered = x_flat - mean.reshape(1, -1)
            # Total variance with the same (N-1) normalisation the SVD uses, so
            # pca_variance_fraction is properly bounded in [0, 1].
            total_var = float((x_centered ** 2).sum() / (n - 1))
            # Economy SVD: U (N,r) S (r,) Vt (r,D), r = min(N,D).
            _, s, vt = np.linalg.svd(x_centered, full_matrices=False)
            k = n - 1 if n_components is None else min(int(n_components), n - 1)
            k = max(1, k)
            pca_components = vt[:k].astype(np.float32)            # (k, D)
            pca_std = (s[:k] / np.sqrt(n - 1)).astype(np.float32)  # (k,)

        # ---- retain a bounded subset of raw embeddings for the hybrid sampler --
        if n <= max_corpus:
            corpus = embeddings
        else:
            idx = np.random.default_rng(0).choice(n, size=max_corpus, replace=False)
            corpus = embeddings[idx]
        corpus_embeddings = corpus.astype(np.float32)

        return cls(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            feature_shape=tuple(int(d) for d in feature_shape),
            per_token=per_token,
            n_samples=int(n),
            pca_components=pca_components,
            pca_std=pca_std,
            total_var=total_var,
            corpus_embeddings=corpus_embeddings,
        )

    # --------------------------------------------------------------- sample ---
    def sample(
        self,
        n: int = 1,
        temperature: float = 1.0,
        truncation: Optional[float] = None,
        rng: Optional[np.random.Generator] = None,
        sampler: str = "diagonal",
        coherence: float = 0.5,
        components: Optional[int] = None,
    ) -> np.ndarray:
        """Draw ``n`` fresh conditioning embeddings from the learned distribution.

        Parameters
        ----------
        temperature:
            Scales the *whole* deviation from the mean. ``1.0`` reproduces the
            corpus spread; ``>1`` pushes into wilder, less-typical territory (more
            anarchy); ``<1`` stays near the bland center of the prompt cloud.
        truncation:
            If set, clip every Gaussian coefficient to this many sigma (a "typical
            set" / truncation trick borrowed from GANs) to avoid rare blow-outs.
        sampler:
            Where on the anarchy<->coherence axis to draw from:

            * ``"diagonal"`` (default) -- independent per-coordinate Gaussians, the
              original "pile of independent Gaussians" from the deck. Maximal
              anarchy: ignores correlations between coordinates.
            * ``"pca"`` -- draw within the low-rank corpus subspace, so samples
              stay *on the manifold* the real prompts live on (most coherent).
            * ``"blend"`` -- interpolate the two *covariances* exactly via
              ``coherence`` (``lambda``): ``Cov = lambda * Cov_pca + (1-lambda) *
              diag``. ``lambda=1`` -> pure pca, ``lambda=0`` -> pure diagonal.
            * ``"hybrid"`` -- pick two random *real* corpus embeddings and SLERP
              between them (plus a little noise): a concept fusion that lands
              between two things the prompts actually meant. Needs
              ``corpus_embeddings`` (retained at fit time).
        coherence:
            The blend weight ``lambda`` in ``[0, 1]`` for ``sampler="blend"``.
        components:
            For ``"pca"``/``"blend"``, use only the top ``components`` principal
            axes (``None`` = all retained at fit time).
        """
        rng = rng or np.random.default_rng()

        if sampler == "hybrid":
            # Concept fusion via SLERP of two real corpus embeddings -- returns
            # full samples directly (the mean is already inside them).
            return self._hybrid(n, temperature, rng)

        if sampler == "diagonal":
            dev = self._diagonal_dev(n, truncation, rng)
        elif sampler == "pca":
            dev = self._pca_dev(n, truncation, rng, components)
        elif sampler == "blend":
            lam = float(np.clip(coherence, 0.0, 1.0))
            # Endpoints are exactly the pure samplers (and consume rng identically),
            # so blend(0) == diagonal and blend(1) == pca to the bit.
            if lam <= 0.0:
                dev = self._diagonal_dev(n, truncation, rng)
            elif lam >= 1.0:
                dev = self._pca_dev(n, truncation, rng, components)
            else:
                # sqrt-weighting blends the covariances (not the samples) linearly:
                # Cov = lambda*Cov_pca + (1-lambda)*Cov_diag.
                pca = self._pca_dev(n, truncation, rng, components)
                diag = self._diagonal_dev(n, truncation, rng)
                dev = np.sqrt(lam) * pca + np.sqrt(1.0 - lam) * diag
        else:
            raise ValueError(
                f"unknown sampler {sampler!r}; choose diagonal | pca | blend | hybrid"
            )

        samples = self.mean[None] + temperature * dev
        return samples.astype(np.float32)

    def _hybrid(self, n, temperature, rng) -> np.ndarray:
        """SLERP between two random real corpus embeddings -> concept fusion."""
        if self.corpus_embeddings is None or len(self.corpus_embeddings) < 2:
            raise ValueError(
                "this distribution has no corpus_embeddings; re-mine to enable the "
                "'hybrid' sampler (fit retains them by default)"
            )
        corpus = self.corpus_embeddings.astype(np.float64)
        m = corpus.shape[0]
        flat = corpus.reshape(m, -1)
        out = np.empty((n, flat.shape[1]), dtype=np.float64)
        for i in range(n):
            ai, bi = rng.choice(m, size=2, replace=False)
            a, b = flat[ai], flat[bi]
            t = rng.uniform(0.3, 0.7)
            out[i] = _slerp(a, b, t)
        # A little gaussian jitter (scaled by temperature & per-coord std) keeps
        # fusions from being exact corpus chords without leaving the neighbourhood.
        noise = rng.standard_normal((n, flat.shape[1])) * self.std.reshape(1, -1)
        out = out + 0.15 * temperature * noise
        return out.reshape(n, *self.feature_shape).astype(np.float32)

    def _diagonal_dev(self, n, truncation, rng) -> np.ndarray:
        """Per-coordinate deviation ``std (.) z`` with optional truncation."""
        z = rng.standard_normal((n, *self.feature_shape))
        z = _truncate(z, truncation, rng)
        return self.std[None].astype(np.float64) * z

    def _pca_dev(self, n, truncation, rng, components) -> np.ndarray:
        """Deviation drawn within the low-rank PCA subspace, reshaped to features."""
        if self.pca_components is None or self.pca_std is None:
            raise ValueError(
                "this distribution was saved without PCA; re-mine to enable the "
                "'pca'/'blend' samplers (fit(..., n_components=...))"
            )
        comps = self.pca_components.astype(np.float64)  # (k, D)
        pstd = self.pca_std.astype(np.float64)          # (k,)
        k = comps.shape[0] if components is None else min(int(components), comps.shape[0])
        comps, pstd = comps[:k], pstd[:k]
        a = rng.standard_normal((n, k))                 # (n, k) gaussian coeffs
        a = _truncate(a, truncation, rng)
        flat = (a * pstd[None]) @ comps                 # (n, D) on the manifold
        return flat.reshape(n, *self.feature_shape)

    def interpolate(self, a: np.ndarray, b: np.ndarray, steps: int = 8) -> np.ndarray:
        """Linear walk between two embeddings (a latent "tween" for animation)."""
        ts = np.linspace(0.0, 1.0, steps)[:, None, None] if len(self.feature_shape) == 2 \
            else np.linspace(0.0, 1.0, steps).reshape((steps,) + (1,) * len(self.feature_shape))
        return ((1 - ts) * a[None] + ts * b[None]).astype(np.float32)

    def project(self, embedding: np.ndarray, truncation: float = 3.0) -> np.ndarray:
        """'Creation through projection': pull an arbitrary tensor onto the typical set.

        Whitens against the learned Gaussian, clips to ``truncation`` sigma, then
        un-whitens -- so any tensor (noise, a doodle, an out-of-domain embedding)
        is projected into the region the model treats as plausible conditioning.
        """
        z = (embedding - self.mean) / self.std
        z = np.clip(z, -truncation, truncation)
        return (self.mean + z * self.std).astype(np.float32)

    # ------------------------------------------------------------ evolution ---
    def refit_from_elites(
        self,
        elites: np.ndarray,
        base_blend: float = 0.25,
        std_floor_frac: float = 0.1,
    ) -> "EmbeddingDistribution":
        """Form a new distribution centered on preferred ("elite") embeddings.

        This is the "evolutionary branch" step: after selecting samples by
        aesthetic resonance, refit the Gaussian to them. We blend the elite
        spread with the base spread and floor it, so branches specialize without
        collapsing to a single point.
        """
        elites = np.asarray(elites, dtype=np.float64)
        mean = elites.mean(axis=0)
        std = elites.std(axis=0)
        std = base_blend * self.std + (1 - base_blend) * std
        std = np.maximum(std, std_floor_frac * self.std)
        return EmbeddingDistribution(
            mean=mean.astype(np.float32),
            std=std.astype(np.float32),
            feature_shape=self.feature_shape,
            per_token=self.per_token,
            n_samples=int(elites.shape[0]),
        )

    # --------------------------------------------------------------- io -------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"mean": self.mean, "std": self.std}
        if self.pca_components is not None:
            arrays["pca_components"] = self.pca_components
            arrays["pca_std"] = self.pca_std
        if self.corpus_embeddings is not None:
            arrays["corpus_embeddings"] = self.corpus_embeddings
        np.savez_compressed(path, **arrays)
        meta = {
            "feature_shape": list(self.feature_shape),
            "per_token": self.per_token,
            "n_samples": self.n_samples,
            "total_var": self.total_var,
            "has_pca": self.pca_components is not None,
            "has_corpus": self.corpus_embeddings is not None,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "EmbeddingDistribution":
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        data = np.load(path)
        meta = json.loads(path.with_suffix(".meta.json").read_text())
        return cls(
            mean=data["mean"],
            std=data["std"],
            feature_shape=tuple(meta["feature_shape"]),
            per_token=meta["per_token"],
            n_samples=meta["n_samples"],
            pca_components=data["pca_components"] if "pca_components" in data else None,
            pca_std=data["pca_std"] if "pca_std" in data else None,
            total_var=float(meta.get("total_var", 0.0)),
            corpus_embeddings=(
                data["corpus_embeddings"] if "corpus_embeddings" in data else None
            ),
        )

    def pca_variance_fraction(self) -> float:
        """Fraction of the corpus's total variance the retained components hold."""
        if self.pca_std is None or self.total_var <= 0:
            return 0.0
        return float((self.pca_std.astype(np.float64) ** 2).sum() / self.total_var)

    def summary(self) -> dict:
        skip = ("mean", "std", "pca_components", "pca_std", "corpus_embeddings")
        return {
            **{k: v for k, v in asdict(self).items() if k not in skip},
            "mean_of_means": float(self.mean.mean()),
            "mean_of_stds": float(self.std.mean()),
            "n_components": None if self.pca_std is None else int(self.pca_std.shape[0]),
            "pca_variance_fraction": round(self.pca_variance_fraction(), 4),
            "corpus_rows": None if self.corpus_embeddings is None else int(len(self.corpus_embeddings)),
        }
