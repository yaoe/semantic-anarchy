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
        # Huge feature dims (flow-model Qwen conditioning: ~1M coords) cannot
        # afford float64 copies or a direct SVD on a 30GB-RAM box -- stay in
        # float32 and use the Gram trick for PCA there.
        big = int(np.prod(np.asarray(embeddings).shape[1:])) > 200_000
        embeddings = np.asarray(embeddings, dtype=np.float32 if big else np.float64)
        if embeddings.ndim < 2:
            raise ValueError("embeddings must be (N, *feature_shape)")
        feature_shape = embeddings.shape[1:]
        n = embeddings.shape[0]
        if big:
            max_corpus = min(max_corpus, 32)   # bound the saved npz size

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
            if big:
                total_var = float(np.einsum('ij,ij->', x_centered, x_centered,
                                            dtype=np.float64) / (n - 1))
            else:
                total_var = float((x_centered ** 2).sum() / (n - 1))
            k = n - 1 if n_components is None else min(int(n_components), n - 1)
            k = max(1, k)
            if big:
                # Gram trick: eigendecompose the (N,N) Gram matrix instead of
                # SVD-ing the (N,D) matrix -- same top-k right singular vectors,
                # a few MB of workspace instead of tens of GB.
                g = (x_centered @ x_centered.T).astype(np.float64)   # (N, N)
                evals, evecs = np.linalg.eigh(g)                     # ascending
                order = np.argsort(evals)[::-1][:k]
                s = np.sqrt(np.maximum(evals[order], 1e-12))         # singular values
                u = evecs[:, order].astype(np.float32)               # (N, k)
                vt_k = (u.T @ x_centered) / s[:, None].astype(np.float32)  # (k, D)
                pca_components = vt_k.astype(np.float32)
                pca_std = (s / np.sqrt(n - 1)).astype(np.float32)
            else:
                # Economy SVD: U (N,r) S (r,) Vt (r,D), r = min(N,D).
                _, s, vt = np.linalg.svd(x_centered, full_matrices=False)
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
        comp_lo: int = 0,
        equalize: bool = False,
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
            For ``"pca"``/``"blend"``, use only ``components`` principal axes
            starting at ``comp_lo`` (``None`` = all from ``comp_lo`` on).
        comp_lo:
            First principal axis to sample. ``0`` = the dominant (most generic,
            "tasteful concept-art") directions. Set it higher to SKIP those and
            ride the *idiosyncratic* mid/minor axes -> stranger but still on the
            manifold (the "weird-but-coherent" knob).
        equalize:
            Give every selected axis the same (RMS) magnitude instead of its
            natural variance. Without this, minor axes barely register; with it,
            their oddities are expressed at full strength.
        """
        rng = rng or np.random.default_rng()

        if sampler == "hybrid":
            # Concept fusion via SLERP of two real corpus embeddings -- returns
            # full samples directly (the mean is already inside them).
            return self._hybrid(n, temperature, rng)

        if sampler == "diagonal":
            dev = self._diagonal_dev(n, truncation, rng)
        elif sampler == "pca":
            dev = self._pca_dev(n, truncation, rng, components, comp_lo, equalize)
        elif sampler == "blend":
            lam = float(np.clip(coherence, 0.0, 1.0))
            # Endpoints are exactly the pure samplers (and consume rng identically),
            # so blend(0) == diagonal and blend(1) == pca to the bit.
            if lam <= 0.0:
                dev = self._diagonal_dev(n, truncation, rng)
            elif lam >= 1.0:
                dev = self._pca_dev(n, truncation, rng, components, comp_lo, equalize)
            else:
                # sqrt-weighting blends the covariances (not the samples) linearly:
                # Cov = lambda*Cov_pca + (1-lambda)*Cov_diag.
                pca = self._pca_dev(n, truncation, rng, components, comp_lo, equalize)
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

    def _pca_dev(self, n, truncation, rng, components, comp_lo=0, equalize=False) -> np.ndarray:
        """Deviation drawn within the low-rank PCA subspace, reshaped to features.

        ``comp_lo``/``components`` select an axis BAND ``[comp_lo : comp_lo+components]``
        instead of always the top axes -- letting you ride the idiosyncratic
        mid/minor directions. ``equalize`` gives each selected axis the RMS
        magnitude of the full spectrum (so minor axes are actually expressed)
        rather than its own (tiny) natural variance.
        """
        if self.pca_components is None or self.pca_std is None:
            raise ValueError(
                "this distribution was saved without PCA; re-mine to enable the "
                "'pca'/'blend' samplers (fit(..., n_components=...))"
            )
        comps = self.pca_components.astype(np.float64)  # (k, D)
        pstd = self.pca_std.astype(np.float64)          # (k,)
        total = comps.shape[0]
        lo = max(0, min(int(comp_lo), total - 1))
        hi = total if components is None else min(lo + int(components), total)
        comps_b, pstd_b = comps[lo:hi], pstd[lo:hi]
        a = rng.standard_normal((n, hi - lo))           # (n, m) gaussian coeffs
        a = _truncate(a, truncation, rng)
        if equalize:
            scale = np.full(hi - lo, float(np.sqrt(np.mean(pstd ** 2))))  # RMS of full spectrum
        else:
            scale = pstd_b
        flat = (a * scale[None]) @ comps_b              # (n, D) on the manifold
        return flat.reshape(n, *self.feature_shape)

    def distance(self, embedding: np.ndarray) -> float:
        """RMS z-score distance of one embedding from the corpus center.

        Whitened by the fitted per-coordinate Gaussian, so the scale is
        human-readable: a typical corpus sample sits near 1.0, a temperature-T
        diagonal sample near T. This is the "how far from the promptable
        center" gauge used to calibrate the periphery.
        """
        # Floor the per-coordinate sigma: near-constant coordinates (structural
        # CLIP dims with ~zero corpus variance) would otherwise explode the
        # z-score for any anchor not minted from this exact fit.
        std = np.maximum(self.std, 0.1 * float(np.mean(self.std)))
        z = (np.asarray(embedding, dtype=np.float64) - self.mean) / std
        return float(np.sqrt(np.mean(z * z)))

    def neighborhood(
        self,
        anchor: np.ndarray,
        n: int = 6,
        radius: float = 0.3,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Sample ``n`` points in a small ball around ``anchor`` (local search).

        The perturbation is drawn along the corpus's PCA axes (falling back to
        the diagonal Gaussian if PCA wasn't fitted) and scaled by ``radius``:
        a fraction of the corpus's own spread, so ``0.3`` means "step 30% of a
        typical corpus deviation away from the anchor". This is the
        "explore around this image" primitive -- hill-climbing on taste.
        """
        rng = rng or np.random.default_rng()
        anchor = np.asarray(anchor, dtype=np.float32)
        if self.pca_components is not None and self.pca_std is not None:
            dev = self._pca_dev(n, None, rng, None)
        else:
            dev = self._diagonal_dev(n, None, rng)
        return (anchor[None] + radius * dev).astype(np.float32)

    def walk(
        self,
        anchor: np.ndarray,
        steps: int = 6,
        step: float = 0.15,
        mode: str = "outward",
        rng: Optional[np.random.Generator] = None,
        axis: Optional[int] = None,
    ) -> np.ndarray:
        """March from ``anchor`` in ONE persistent direction; return the strip.

        Modes:
        * ``outward`` -- along the anchor's own ray from the corpus center, i.e.
          straight toward the periphery. Each step multiplies the deviation by
          ``(1 + step)``, so the distance gauge grows ~``step`` per frame --
          watching legibility strain as you leave the promptable region.
        * ``axis`` -- along principal axis ``axis`` (a *meaningful* direction).
        * ``random`` -- a persistent random on-manifold direction.
        """
        anchor = np.asarray(anchor, dtype=np.float64)
        if mode == "outward":
            dev = anchor - self.mean
            out = [self.mean + dev * (1.0 + step * (i + 1)) for i in range(steps)]
            return np.stack(out).astype(np.float32)
        if mode == "axis" and self.pca_components is not None:
            k = int(axis or 0) % self.pca_components.shape[0]
            d = (self.pca_std[k] * self.pca_components[k]).reshape(self.feature_shape)
        else:
            rng = rng or np.random.default_rng()
            d = (self._pca_dev(1, None, rng, None)[0]
                 if self.pca_components is not None else
                 self._diagonal_dev(1, None, rng)[0])
        return np.stack([anchor + step * (i + 1) * d
                         for i in range(steps)]).astype(np.float32)

    def retarget(self, samples: np.ndarray, target: float) -> np.ndarray:
        """Rescale each sample's deviation so its distance gauge hits ``target``.

        Shell sampling: keep the sampled *direction*, pin the *radius* -- so you
        draw from the ring where your keepers live instead of blindly scaling
        temperature.
        """
        samples = np.asarray(samples, dtype=np.float64)
        out = []
        for x in samples:
            d = self.distance(x)
            dev = x - self.mean
            out.append(self.mean + dev * (target / max(d, 1e-6)))
        return np.stack(out).astype(np.float32)

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
