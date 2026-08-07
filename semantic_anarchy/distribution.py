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

The corpus autopsy (``analysis.py`` -> ``distribution_report.html``) measured four
places where that model is provably wrong about the corpus, and each has a
correction here. They are all **opt-in knobs, never replacements**: the "broken"
behaviours are textures worth keeping, so every default reproduces the original
sampler bit-for-bit.

* **length conditioning** -- CLIP pads to 77 with EOS, so at any middle position
  the corpus is two populations (content vs padding) and the fitted Gaussian
  peaks in the gap between them. ``fit(..., lengths=)`` estimates ``(mu, sigma)``
  twice per position and ``sample(..., lengths=)`` draws each sample conditional
  on where its EOS falls. Prompt length was PC1/PC2 in disguise (|r| 0.69/0.64),
  so this doubles as the corpus's single biggest semantic dial.
* **row coherence (rho)** -- the corpus's 77 token rows agree at ~0.65; diagonal
  samples agree at 0.00. ``sample(..., rho=)`` restores any coherence in [0, 1]
  with the marginals left exact.
* **empirical PCA head + radius band** -- PC1 is bimodal, so an N(0,1) draw on it
  lands in an empty gap; and the corpus is a *band* of radii (spread 0.03) while
  every sampler is a 9x tighter spike. ``empirical_head=`` draws the leading
  coefficients from their own CDFs, ``sample_radii()`` + ``retarget()`` draw the
  target radius from the corpus's own.
* **on/off-manifold split** -- ``sampler="split"`` gives the PCA-subspace
  projection of a diagonal deviation and its orthogonal remainder separate
  temperatures, which is the diagnostic for where the interesting weirdness lives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

#: How many leading PCA axes keep an empirical (sorted-coefficient) CDF at fit
#: time, for ``sample(empirical_head=...)``. Costs ``N`` floats per axis -- the
#: gap-in-PC1 problem is a property of the first two or three axes, so a handful
#: is plenty and the rest stay Gaussian.
HEAD_AXES = 8

#: Minimum members a side of the EOS boundary needs before that position gets
#: its own conditional ``(mu, sigma)``; below it the position falls back to the
#: unconditional fit (positions past every prompt's EOS are always-padding, and
#: position 0 is always BOS).
MIN_LENGTH_GROUP = 20


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


def _floor_std(std: np.ndarray) -> np.ndarray:
    """The sigma the distance gauge whitens by: floored at 10% of the mean.

    Near-constant coordinates (structural CLIP dims with ~zero corpus variance)
    would otherwise explode the z-score of any anchor not minted from this exact
    fit. Shared by :meth:`EmbeddingDistribution.distance` and the corpus-radius
    band computed at fit time, so the two agree by construction.
    """
    std = np.asarray(std, dtype=np.float64)
    return np.maximum(std, 0.1 * float(np.mean(std)))


def _length_stats(embeddings, lengths, mean, std, std_floor,
                  min_group: int = MIN_LENGTH_GROUP):
    """Per-position ``(mu, sigma)`` on each side of the EOS boundary.

    CLIP pads to 77 with EOS, so position ``t`` of the corpus is a MIXTURE: the
    prompts long enough to still be writing content there, and the prompts that
    ran out and are padding. The report measured that binary explaining up to 89%
    of a coordinate's variance -- which means the single fitted Gaussian peaks in
    the gap between two lobes and samples land where no prompt ever does.

    A position whose smaller side has fewer than ``min_group`` members (position
    0 is always BOS; the tail is always padding) keeps the unconditional fit on
    both sides, so nothing collapses onto a handful of rows.
    """
    n, t_dim = embeddings.shape[0], embeddings.shape[1]
    mean_c, std_c = np.array(mean, dtype=np.float64), np.array(std, dtype=np.float64)
    mean_p, std_p = mean_c.copy(), std_c.copy()
    for t in range(t_dim):
        content = lengths > t          # this prompt is still writing at t
        n_c = int(content.sum())
        if n_c < min_group or (n - n_c) < min_group:
            continue                   # one side too thin -> keep the pooled fit
        rows = embeddings[:, t]        # (N, *rest)
        mean_c[t] = rows[content].mean(axis=0)
        std_c[t] = rows[content].std(axis=0)
        mean_p[t] = rows[~content].mean(axis=0)
        std_p[t] = rows[~content].std(axis=0)
    return (mean_c, np.maximum(std_c, std_floor),
            mean_p, np.maximum(std_p, std_floor))


def _deflate(dev: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Remove each row's component along the matching row of ``direction``.

    One Gram-Schmidt step, per sample. Used so a subspace deviation stops
    carrying the very shift the length split is about to impose explicitly.
    """
    n = dev.shape[0]
    f = dev.reshape(n, -1)
    d = np.asarray(direction, dtype=np.float64).reshape(n, -1)
    nrm = np.einsum("ij,ij->i", d, d)
    coef = np.einsum("ij,ij->i", f, d) / np.maximum(nrm, 1e-12)
    return (f - coef[:, None] * d).reshape(dev.shape)


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
    mean_content, std_content, mean_pad, std_pad:
        The *length-conditional* fit: per-position statistics estimated
        separately over the prompts whose content reaches this position and the
        prompts already padding here. Same shape as ``mean``. Present only when
        ``fit`` was handed ``lengths`` (see :meth:`has_length_stats`).
    lengths:
        The corpus's own EOS positions, ``(N,)`` -- the length histogram
        ``draw_lengths("corpus")`` bootstraps from.
    pca_head:
        ``(N, HEAD_AXES)`` sorted, unit-variance coefficients of the corpus along
        the leading PCA axes: the empirical CDFs ``empirical_head=`` samples from
        instead of N(0,1). PC1 is bimodal, so this is where the Gaussian is worst.
    corpus_distance:
        ``(N,)`` -- each corpus embedding's own :meth:`distance` gauge. The
        radius *band* the corpus actually occupies, which :meth:`sample_radii`
        draws targets from.
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
    # ---- the corpus-autopsy corrections (all optional, all opt-in at sample
    # time; a distribution mined before they existed simply has None here) -----
    mean_content: Optional[np.ndarray] = None
    std_content: Optional[np.ndarray] = None
    mean_pad: Optional[np.ndarray] = None
    std_pad: Optional[np.ndarray] = None
    lengths: Optional[np.ndarray] = None
    pca_head: Optional[np.ndarray] = None
    corpus_distance: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ fit ---
    @classmethod
    def fit(
        cls,
        embeddings: np.ndarray,
        per_token: bool = True,
        std_floor: float = 1e-4,
        n_components: Optional[int] = None,
        max_corpus: int = 256,
        lengths: Optional[np.ndarray] = None,
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
            keep. ``None`` keeps the maximum (``N-1``) -- but each axis costs a
            full ``D`` floats on disk, and the tail of the spectrum is noise, so
            callers that mine real corpora pass a cap (see
            ``scripts/mine_distribution.py``'s ``MAX_COMPONENTS``). These power
            the ``"pca"`` and ``"blend"`` samplers; the per-coordinate
            ``mean``/``std`` above are unchanged.
        max_corpus:
            Cap on how many raw embeddings to retain as ``corpus_embeddings`` (for
            the ``"hybrid"`` SLERP sampler). If the corpus is larger, a random
            subset of this many rows is kept to bound the saved size.
        lengths:
            Optional ``(N,)`` content length per prompt -- the EOS position in the
            token sequence. Given these, the fit ALSO estimates every position's
            ``(mu, sigma)`` twice, once over the prompts still writing content
            there and once over the prompts already padding, which is what
            ``sample(lengths=...)`` conditions on. Ignored for tensors with no
            token axis (sdxl's ``pooled``) and for the very wide flow-model
            conditioning, where the extra four full-size arrays don't pay.
        """
        # Huge feature dims (flow-model Qwen conditioning: ~1M coords) cannot
        # afford float64 copies or a direct SVD on a 30GB-RAM box -- stay in
        # float32 there. (Which PCA *algorithm* to use is a separate question,
        # decided by the sample/feature ratio below -- don't conflate the two.)
        wide = int(np.prod(np.asarray(embeddings).shape[1:])) > 200_000
        embeddings = np.asarray(embeddings, dtype=np.float32 if wide else np.float64)
        if embeddings.ndim < 2:
            raise ValueError("embeddings must be (N, *feature_shape)")
        feature_shape = embeddings.shape[1:]
        n = embeddings.shape[0]
        if wide:
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

        # ---- low-rank PCA over the flattened, centered embeddings -------------
        pca_components = pca_std = pca_head = corpus_distance = None
        total_var = 0.0
        if n > 1:
            d_flat = int(np.prod(feature_shape))
            x_centered = embeddings.reshape(n, -1) - mean.reshape(1, -1)
            # Total variance with the same (N-1) normalisation the SVD uses, so
            # pca_variance_fraction is properly bounded in [0, 1]. einsum sums
            # in place -- `(x ** 2).sum()` would materialise a second full-size
            # copy (2GB for a 4k-prompt sd15 corpus).
            total_var = float(np.einsum("ij,ij->", x_centered, x_centered,
                                        dtype=np.float64) / (n - 1))
            k = n - 1 if n_components is None else min(int(n_components), n - 1)
            k = max(1, k)
            if n <= d_flat:
                # Gram trick. X = U S Vt  =>  X Xt = U S^2 Ut, and Vt = S^-1 Ut X,
                # so the (N,N) eigendecomposition yields the same top-k right
                # singular vectors as an (N,D) SVD. Every real corpus has fewer
                # samples than coordinates (4k prompts vs 59k coords for sd15),
                # and in that regime this is ~16x faster than LAPACK's economy
                # SVD -- and the only tractable option at all for the flow
                # models' ~1M-coordinate conditioning.
                g = np.asarray(x_centered @ x_centered.T, dtype=np.float64)  # (N, N)
                evals, evecs = np.linalg.eigh(g)                     # ascending
                order = np.argsort(evals)[::-1][:k]
                s = np.sqrt(np.maximum(evals[order], 1e-12))         # singular values
                u = evecs[:, order].astype(x_centered.dtype)         # (N, k)
                vt_k = (u.T @ x_centered) / s[:, None].astype(x_centered.dtype)
                pca_components = vt_k.astype(np.float32)             # (k, D)
                pca_std = (s / np.sqrt(n - 1)).astype(np.float32)    # (k,)
                # The corpus's own coefficient along each leading axis, in the
                # SAME unit-variance units _pca_dev draws in (score / pca_std,
                # which for the Gram trick is just U * sqrt(n-1)). Sorted, so
                # sampling it is a searchsorted-free inverse-CDF lookup.
                pca_head = np.sort(
                    u[:, :HEAD_AXES].astype(np.float64) * np.sqrt(n - 1), axis=0
                ).astype(np.float32)
            else:
                # More samples than coordinates: the Gram matrix would be the
                # bigger object, so SVD the data directly.
                # Economy SVD: U (N,r) S (r,) Vt (r,D), r = min(N,D).
                u, s, vt = np.linalg.svd(x_centered, full_matrices=False)
                pca_components = vt[:k].astype(np.float32)            # (k, D)
                pca_std = (s[:k] / np.sqrt(n - 1)).astype(np.float32)  # (k,)
                pca_head = np.sort(
                    u[:, :HEAD_AXES].astype(np.float64) * np.sqrt(n - 1), axis=0
                ).astype(np.float32)

            # ---- the radius BAND the corpus occupies -------------------------
            # Every sample's own distance() gauge, so sample_radii() can draw a
            # target radius from the real spread instead of a 9x-tighter spike.
            # Chunked: a whole whitened copy of a 4k-prompt sd15 corpus is 2GB.
            inv = (1.0 / _floor_std(std)).reshape(1, -1)
            corpus_distance = np.empty(n, dtype=np.float64)
            for i0 in range(0, n, 256):
                blk = x_centered[i0:i0 + 256] * inv
                corpus_distance[i0:i0 + 256] = np.sqrt(np.mean(blk * blk, axis=1))
            corpus_distance = corpus_distance.astype(np.float32)

        # ---- length-conditional per-position statistics (the past-EOS split) --
        mean_content = std_content = mean_pad = std_pad = None
        keep_lengths = None
        if lengths is not None and len(feature_shape) >= 2 and not wide:
            keep_lengths = np.clip(np.asarray(lengths, dtype=np.int32).reshape(-1),
                                   0, feature_shape[0])
            if keep_lengths.shape[0] != n:
                raise ValueError(
                    f"lengths has {keep_lengths.shape[0]} entries for {n} embeddings")
            mean_content, std_content, mean_pad, std_pad = _length_stats(
                embeddings, keep_lengths, mean, std, std_floor)

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
            mean_content=None if mean_content is None else mean_content.astype(np.float32),
            std_content=None if std_content is None else std_content.astype(np.float32),
            mean_pad=None if mean_pad is None else mean_pad.astype(np.float32),
            std_pad=None if std_pad is None else std_pad.astype(np.float32),
            lengths=keep_lengths,
            pca_head=pca_head,
            corpus_distance=corpus_distance,
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
        rho: float = 0.0,
        lengths: Optional[np.ndarray] = None,
        empirical_head: int = 0,
        temp_on: float = 1.0,
        temp_off: float = 1.0,
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
            * ``"split"`` -- a diagonal deviation decomposed into its PCA-subspace
              projection ("on-manifold") and the orthogonal remainder
              ("off-manifold"), each scaled by its own temperature. The
              diagnostic sampler: ``temp_on=1, temp_off=0`` is the diagonal
              collapsed onto the corpus subspace, ``temp_on=0, temp_off=1`` is
              pure off-manifold noise, and the corners answer where the good
              weirdness actually lives.
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
            their oddities are expressed at full strength. With ``components``
            left unset the band now STOPS at :meth:`noise_floor_axes` -- equalising
            into the noise tail spends the entropy budget on axes that carry no
            corpus structure.
        rho:
            Row coherence in ``[0, 1]`` for the diagonal draw:
            ``dev_t = sqrt(rho)*u + sqrt(1-rho)*v_t`` with ``u`` shared across
            token positions and ``v_t`` fresh per position. Per-coordinate
            marginals stay exact; the correlation between two token rows lands at
            ``rho`` by construction. ``0`` (default) is the historical static,
            ``~0.65`` is corpus-like, ``1`` smears ONE deviation through the whole
            sentence. Ignored for tensors with no token axis.
        lengths:
            ``(n,)`` content lengths (EOS positions), typically from
            :meth:`draw_lengths`. Each sample is then drawn from the conditional
            fit on its own side of the boundary: content statistics before its
            EOS, padding statistics after. Requires :meth:`has_length_stats`;
            silently ignored otherwise (sdxl's ``pooled`` has no token axis).
        empirical_head:
            Draw the first ``empirical_head`` PCA coefficients from the corpus's
            own CDF (:attr:`pca_head`) rather than N(0,1). PC1 is bimodal, so a
            Gaussian puts its densest mass in a region no prompt occupies.
            Only affects ``"pca"``/``"blend"``, and only the axes actually
            selected by ``comp_lo``/``components``.
        temp_on, temp_off:
            ``sampler="split"`` only: separate multipliers for the on-manifold
            (PCA-subspace) and off-manifold (orthogonal) halves of the deviation.
            Both are multiplied by ``temperature`` on top.
        """
        rng = rng or np.random.default_rng()

        if sampler == "hybrid":
            # Concept fusion via SLERP of two real corpus embeddings -- returns
            # full samples directly (the mean is already inside them, so neither
            # the length split nor rho has anything to act on).
            return self._hybrid(n, temperature, rng)

        # ---- length conditioning: which (mu, sigma) field this batch draws from
        # `content` is the (n, T) "still writing" mask, handed to the diagonal
        # draw so rho shares its deviation within the content span only.
        mu, scale, content = self.mean[None].astype(np.float64), None, None
        if lengths is not None and self.has_length_stats:
            mu, sigma, content = self._length_fields(lengths, n)
            scale = sigma / np.maximum(self.std[None].astype(np.float64), 1e-12)

        # `subspace` is what the deviation ACTUALLY came from, not what was asked
        # for -- blend(0) is a diagonal draw and must keep being treated as one,
        # or the "bit-exact at the endpoints" guarantee dies under length
        # conditioning.
        subspace = True
        if sampler == "diagonal":
            dev = self._diagonal_dev(n, truncation, rng, rho, content)
            subspace = False
        elif sampler == "pca":
            dev = self._pca_dev(n, truncation, rng, components, comp_lo, equalize,
                                empirical_head)
        elif sampler == "split":
            dev = self._split_dev(n, truncation, rng, temp_on, temp_off,
                                  components, comp_lo, rho, content)
        elif sampler == "blend":
            lam = float(np.clip(coherence, 0.0, 1.0))
            # Endpoints are exactly the pure samplers (and consume rng identically),
            # so blend(0) == diagonal and blend(1) == pca to the bit.
            if lam <= 0.0:
                dev = self._diagonal_dev(n, truncation, rng, rho, content)
                subspace = False
            elif lam >= 1.0:
                dev = self._pca_dev(n, truncation, rng, components, comp_lo, equalize,
                                    empirical_head)
            else:
                # sqrt-weighting blends the covariances (not the samples) linearly:
                # Cov = lambda*Cov_pca + (1-lambda)*Cov_diag.
                pca = self._pca_dev(n, truncation, rng, components, comp_lo, equalize,
                                    empirical_head)
                diag = self._diagonal_dev(n, truncation, rng, rho, content)
                dev = np.sqrt(lam) * pca + np.sqrt(1.0 - lam) * diag
        else:
            raise ValueError(
                f"unknown sampler {sampler!r}; choose "
                f"diagonal | pca | blend | hybrid | split"
            )

        if scale is not None:
            # Re-scale the deviation from the marginal sigma it was drawn against
            # to this sample's conditional one. For "diagonal" that is exact --
            # std * z * (sigma_L/std) IS the conditional Gaussian. The subspace
            # samplers additionally need the length direction taken back out:
            # length is PC1/PC2 in disguise, so a PCA deviation already carries a
            # length shift and adding the conditional mean on top would count it
            # twice. Projecting out the shift we are about to impose fixes that.
            dev = dev * scale
            if subspace:
                dev = _deflate(dev, mu - self.mean[None].astype(np.float64))

        samples = mu + temperature * dev
        return samples.astype(np.float32)

    # ------------------------------------------------- the length dimension ---
    @property
    def has_length_stats(self) -> bool:
        """Whether this fit carries the content/padding split (E01)."""
        return self.mean_content is not None and self.std_pad is not None

    def draw_lengths(self, n: int, rng=None, mode: str = "corpus",
                     length: Optional[int] = None) -> np.ndarray:
        """``(n,)`` content lengths to condition a batch on.

        ``"corpus"`` bootstraps the corpus's own length histogram (so a batch has
        the mix of long and short prompts the corpus does); ``"fixed"`` pins every
        sample to ``length`` -- "sample me a 60-token image", the single biggest
        semantic dial the corpus owns.
        """
        t_dim = self.feature_shape[0] if self.feature_shape else 0
        if mode == "fixed":
            if length is None:
                raise ValueError("length_mode='fixed' needs an explicit length")
            return np.full(n, int(np.clip(length, 1, t_dim)), dtype=np.int32)
        if mode != "corpus":
            raise ValueError(f"unknown length mode {mode!r}; choose corpus | fixed")
        if self.lengths is None or len(self.lengths) == 0:
            raise ValueError(
                "this distribution has no corpus length histogram; re-mine to "
                "enable length conditioning (mine_distribution.py records it)")
        rng = rng or np.random.default_rng()
        pool = np.asarray(self.lengths, dtype=np.int32)
        return pool[rng.integers(0, len(pool), size=n)]

    def _length_fields(self, lengths, n: int):
        """Broadcast the conditional fit to ``(n, *feature_shape)`` mu/sigma."""
        t_dim = self.feature_shape[0]
        lens = np.asarray(lengths, dtype=np.int64).reshape(-1)
        if lens.shape[0] == 1 and n > 1:
            lens = np.repeat(lens, n)
        if lens.shape[0] != n:
            raise ValueError(f"lengths has {lens.shape[0]} entries for n={n}")
        lens = np.clip(lens, 0, t_dim)
        content = np.arange(t_dim)[None, :] < lens[:, None]        # (n, T)
        m = content.reshape(n, t_dim, *((1,) * (len(self.feature_shape) - 1)))
        mu = np.where(m, self.mean_content[None].astype(np.float64),
                      self.mean_pad[None].astype(np.float64))
        sigma = np.where(m, self.std_content[None].astype(np.float64),
                         self.std_pad[None].astype(np.float64))
        return mu, sigma, content

    # ----------------------------------------------------- the radius band ---
    def noise_floor_axes(self, margin: float = 1.0) -> Optional[int]:
        """How many PCA axes rise above a shuffled-coordinate null.

        Shuffling each coordinate independently across samples destroys every
        correlation while preserving the per-coordinate variances, and the
        resulting Gram spectrum is Marchenko-Pastur with ratio ``n/D``: its top
        edge sits at ``total_var * (1 + sqrt(n/D))^2 / (n-1)``. Axes below that
        are indistinguishable from stored noise -- sampling them spends the
        entropy budget on nothing.

        Cheap and closed-form (no second PCA), and mildly conservative because MP
        assumes equal coordinate variances: on the 4,144-prompt sd15 corpus it
        says 379 where the measured shuffle null says 416.
        """
        if self.pca_std is None or self.total_var <= 0 or self.n_samples < 3:
            return None
        d_flat = int(np.prod(self.feature_shape))
        if d_flat <= 0:
            return None
        edge = (self.total_var * (1.0 + np.sqrt(self.n_samples / d_flat)) ** 2
                / (self.n_samples - 1))
        return int((self.pca_std.astype(np.float64) ** 2 > margin * edge).sum())

    def sample_radii(self, n: int, rng=None, scale: float = 1.0) -> np.ndarray:
        """``(n,)`` target distances bootstrapped from the corpus's own band.

        The corpus is a *band* of radii (sd15: mean 0.99, spread 0.031, range
        0.89-1.11) while every sampler produces a ~9x tighter spike. Feed these to
        :meth:`retarget` and each sample lands somewhere a real prompt could have.
        ``scale`` shifts the whole band outward (the extrapolation knob).
        """
        if self.corpus_distance is None or len(self.corpus_distance) == 0:
            raise ValueError(
                "this distribution has no corpus radius band; re-mine to enable "
                "--radius-band")
        rng = rng or np.random.default_rng()
        pool = np.asarray(self.corpus_distance, dtype=np.float64)
        return pool[rng.integers(0, len(pool), size=n)] * float(scale)

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

    def _diagonal_dev(self, n, truncation, rng, rho: float = 0.0,
                      content=None) -> np.ndarray:
        """Per-coordinate deviation ``std (.) z``, optionally row-coherent.

        With ``rho > 0`` the standard normal becomes
        ``sqrt(rho)*u + sqrt(1-rho)*v_t``: ``u`` is drawn once per sample and
        shared across every token position, ``v_t`` is fresh per position. Each
        coordinate's marginal is still exactly N(0,1) (``rho + (1-rho) = 1``),
        while the correlation between any two token rows is ``rho`` by
        construction -- the corpus sits at ~0.65, the historical diagonal at 0.00.

        ``content`` (the ``(n, T)`` pre-EOS mask, when length conditioning is on)
        confines the shared ``u`` to the content span: padding positions are not
        part of the sentence, so smearing the sentence's deviation into them just
        blurs the boundary the length split exists to sharpen.
        """
        z = rng.standard_normal((n, *self.feature_shape))
        z = _truncate(z, truncation, rng)
        rho = float(np.clip(rho, 0.0, 1.0))
        if rho > 0.0 and len(self.feature_shape) >= 2:
            t_dim = self.feature_shape[0]
            u = rng.standard_normal((n, 1, *self.feature_shape[1:]))
            u = _truncate(u, truncation, rng)
            mixed = np.sqrt(rho) * u + np.sqrt(1.0 - rho) * z
            if content is None:
                z = mixed
            else:
                m = np.asarray(content).reshape(
                    n, t_dim, *((1,) * (len(self.feature_shape) - 1)))
                z = np.where(m, mixed, z)
        return self.std[None].astype(np.float64) * z

    def _split_dev(self, n, truncation, rng, temp_on=1.0, temp_off=1.0,
                   components=None, comp_lo=0, rho=0.0, content=None) -> np.ndarray:
        """Diagonal deviation, on-manifold and off-manifold halves scaled apart.

        The retained PCA components are an orthonormal basis of the corpus
        subspace, so ``dev_on = (dev . V^T) V`` is the part of a diagonal draw the
        corpus could have produced and ``dev - dev_on`` is the part it never does.
        Giving the two their own temperature turns "is the good weirdness on the
        manifold, off it, or between?" into four batches.

        Note the "manifold" here is only what the fit RETAINED (512 axes by
        default, not the full N-1 rank), so a low-rank mine widens what counts as
        off-manifold. Same caveat as ``--components``.
        """
        if self.pca_components is None:
            raise ValueError(
                "this distribution was saved without PCA; re-mine to enable the "
                "'split' sampler")
        dev = self._diagonal_dev(n, truncation, rng, rho, content)
        comps = self.pca_components
        total = comps.shape[0]
        lo = max(0, min(int(comp_lo), total - 1))
        hi = total if components is None else min(lo + int(components), total)
        basis = np.ascontiguousarray(comps[lo:hi], dtype=np.float32)   # (m, D)
        flat = dev.reshape(n, -1).astype(np.float32)
        on = (flat @ basis.T) @ basis
        off = flat - on
        out = float(temp_on) * on + float(temp_off) * off
        return out.astype(np.float64).reshape(n, *self.feature_shape)

    def _empirical_coeff(self, axis: int, n: int, rng) -> np.ndarray:
        """``n`` draws from the corpus's own CDF along PCA ``axis``.

        ``pca_head`` is already sorted, so an inverse-CDF draw is a linearly
        interpolated lookup at a uniform position -- no searchsorted, no kernel.
        """
        col = self.pca_head[:, axis].astype(np.float64)
        m = col.shape[0]
        pos = rng.random(n) * (m - 1)
        i0 = np.floor(pos).astype(np.int64)
        frac = pos - i0
        i1 = np.minimum(i0 + 1, m - 1)
        return col[i0] * (1.0 - frac) + col[i1] * frac

    def _pca_dev(self, n, truncation, rng, components, comp_lo=0, equalize=False,
                 empirical_head=0) -> np.ndarray:
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
        if components is not None:
            hi = min(lo + int(components), total)
        else:
            hi = total
            if equalize:
                # Equalising means "express every selected axis at full strength",
                # and past the shuffle-null the axes carry no corpus structure --
                # so an unbounded equalised band spends most of its budget
                # amplifying stored noise. Cap it (still overridable: pass an
                # explicit `components` to reach further).
                floor_k = self.noise_floor_axes()
                if floor_k:
                    hi = max(lo + 1, min(total, floor_k))
        comps_b, pstd_b = comps[lo:hi], pstd[lo:hi]
        a = rng.standard_normal((n, hi - lo))           # (n, m) gaussian coeffs
        a = _truncate(a, truncation, rng)
        # The leading axes are the ones the Gaussian gets worst (PC1 is two lobes
        # with a hole where N(0,1) is densest), so optionally draw those from the
        # corpus's own CDF instead. Selected axes only -- comp_lo can skip past.
        head = 0 if self.pca_head is None else min(int(empirical_head),
                                                   self.pca_head.shape[1])
        for axis in range(lo, min(hi, head)):
            a[:, axis - lo] = self._empirical_coeff(axis, n, rng)
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
        z = (np.asarray(embedding, dtype=np.float64) - self.mean) / _floor_std(self.std)
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

    def retarget(self, samples: np.ndarray, target) -> np.ndarray:
        """Rescale each sample's deviation so its distance gauge hits ``target``.

        Shell sampling: keep the sampled *direction*, pin the *radius* -- so you
        draw from the ring where your keepers live instead of blindly scaling
        temperature. ``target`` is a scalar (one shell) or a per-sample array
        (a *band*, e.g. from :meth:`sample_radii`).
        """
        samples = np.asarray(samples, dtype=np.float64)
        targets = np.broadcast_to(
            np.asarray(target, dtype=np.float64).reshape(-1), (len(samples),))
        out = []
        for x, t in zip(samples, targets):
            d = self.distance(x)
            dev = x - self.mean
            out.append(self.mean + dev * (t / max(d, 1e-6)))
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
        # The corpus-autopsy extras. Together they add ~4 feature rows plus a few
        # N-vectors -- negligible beside the PCA basis, and each one is what makes
        # a correction available at sample time rather than needing a re-mine.
        for key in ("mean_content", "std_content", "mean_pad", "std_pad",
                    "lengths", "pca_head", "corpus_distance"):
            val = getattr(self, key)
            if val is not None:
                arrays[key] = val
        np.savez_compressed(path, **arrays)
        meta = {
            "feature_shape": list(self.feature_shape),
            "per_token": self.per_token,
            "n_samples": self.n_samples,
            "total_var": self.total_var,
            "has_pca": self.pca_components is not None,
            "has_corpus": self.corpus_embeddings is not None,
            "has_length_stats": self.has_length_stats,
            "has_radius_band": self.corpus_distance is not None,
            "noise_floor_axes": self.noise_floor_axes(),
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
            # Absent from every .npz mined before the corrections existed, which
            # is exactly what `has_length_stats` / `sample_radii` report on.
            **{k: (data[k] if k in data else None)
               for k in ("mean_content", "std_content", "mean_pad", "std_pad",
                         "lengths", "pca_head", "corpus_distance")},
        )

    def pca_variance_fraction(self) -> float:
        """Fraction of the corpus's total variance the retained components hold."""
        if self.pca_std is None or self.total_var <= 0:
            return 0.0
        return float((self.pca_std.astype(np.float64) ** 2).sum() / self.total_var)

    def summary(self) -> dict:
        skip = ("mean", "std", "pca_components", "pca_std", "corpus_embeddings",
                "mean_content", "std_content", "mean_pad", "std_pad", "lengths",
                "pca_head", "corpus_distance")
        lens = self.lengths
        return {
            **{k: v for k, v in asdict(self).items() if k not in skip},
            "mean_of_means": float(self.mean.mean()),
            "mean_of_stds": float(self.std.mean()),
            "n_components": None if self.pca_std is None else int(self.pca_std.shape[0]),
            "pca_variance_fraction": round(self.pca_variance_fraction(), 4),
            "corpus_rows": None if self.corpus_embeddings is None else int(len(self.corpus_embeddings)),
            "noise_floor_axes": self.noise_floor_axes(),
            "length_stats": self.has_length_stats,
            "length_median": None if lens is None else int(np.median(lens)),
            "length_range": None if lens is None else [int(lens.min()), int(lens.max())],
            "radius_band": (
                None if self.corpus_distance is None else
                [round(float(self.corpus_distance.mean()), 4),
                 round(float(self.corpus_distance.std()), 4)]
            ),
        }
