"""Deep distributional statistics of a conditioning corpus (torch-free).

``EmbeddingDistribution`` fits the model this project *samples* from; this module
asks the prior question -- **what does the corpus actually look like?** -- with
enough resolution to design a better sampler:

* per-coordinate moments beyond (mu, sigma): skew, excess kurtosis, the empirical
  quantile envelope, and asymmetric half-sigmas (sigma+ / sigma-);
* where the variance lives (token position x channel), i.e. which coordinates are
  structurally frozen and which carry the signal;
* how non-Gaussian each marginal is, and how far the diagonal independence
  assumption is from the truth (correlation structure, PCA spectrum vs a
  shuffled null, radius concentration);
* how much of that variance the UNet can even *read*, via the cross-attention
  ``to_k``/``to_v`` column norms -- the only place conditioning enters the model.

Pure NumPy (SciPy optional, only for normality p-values). No torch: the corpus
arrives as a cached ``.npz`` produced once by :func:`encode_corpus`, which is the
single function here that touches the GPU tier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# CLIP ViT-L/14 (SD1.5) sequence layout.
BOS = 0                 # always the same token id -> structurally frozen
SEQ = 77


# --------------------------------------------------------------- caching ---
def encode_corpus(prompts_path: Path, out_path: Path, backend: str = "sd15",
                  ckpt: Optional[str] = None, model: Optional[str] = None) -> Path:
    """Encode a prompt file ONCE and cache everything the analysis needs.

    This is the only torch-touching function in the module; it is called at most
    once per corpus and everything downstream reads the ``.npz`` it writes:
    ``embeds`` (N, 77, H), ``token_ids`` (N, 77), ``eos_pos`` (N,) and the
    empty-prompt ``uncond`` row (the geometric origin every sample is measured
    against -- NOT the CFG negative, which for sd15 is the house negative prompt,
    ``pipeline.SD15_NEGATIVE_PROMPT``).
    """
    from .backend import make_backend

    prompts = [ln.strip() for ln in Path(prompts_path).read_text().splitlines()
               if ln.strip() and not ln.lstrip().startswith("#")]
    be = make_backend(backend, model_id=model, ckpt=ckpt)
    emb = np.asarray(be.encode(prompts)[be.tensor_names[0]], dtype=np.float32)

    tok = be.model.pipe.tokenizer
    ids = np.asarray(tok(prompts, padding="max_length", max_length=SEQ,
                         truncation=True).input_ids, dtype=np.int32)
    eos_pos = np.argmax(ids == tok.eos_token_id, axis=1).astype(np.int32)
    uncond = np.asarray(be.model.uncond_embedding(), dtype=np.float32)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, embeds=emb, token_ids=ids, eos_pos=eos_pos,
                        uncond=uncond)
    out_path.with_suffix(".prompts.json").write_text(json.dumps(prompts))
    return out_path


def cross_attn_sensitivity(ckpt: Path) -> dict:
    """Per-channel sensitivity of the UNet to each conditioning channel.

    Conditioning reaches SD's UNet through exactly two projections per
    cross-attention block: ``attn2.to_k`` and ``attn2.to_v`` (shape
    ``(inner, hidden)``). Column ``c`` of those matrices is the *only* path
    channel ``c`` has into the image, so its L2 norm is a direct, weight-derived
    sensitivity -- no gradients, no decoding.

    Returns ``{"to_k": (H,), "to_v": (H,), "per_layer_k": (L, H), ...}`` where
    each vector is the RMS-normalised column norm averaged over the L blocks
    (RMS so blocks of different ``inner`` width are comparable).
    """
    from safetensors import safe_open

    per_k, per_v, names = [], [], []
    with safe_open(str(ckpt), framework="numpy") as f:
        keys = sorted(k for k in f.keys()
                      if "attn2" in k and (k.endswith("to_k.weight")
                                           or k.endswith("to_v.weight")))
        for key in keys:
            w = np.asarray(f.get_tensor(key), dtype=np.float32)   # (inner, hidden)
            # RMS over output rows: comparable across blocks of different width.
            col = np.sqrt((w ** 2).mean(axis=0))
            (per_k if key.endswith("to_k.weight") else per_v).append(col)
            if key.endswith("to_k.weight"):
                names.append(key.rsplit(".attn2", 1)[0].replace(
                    "model.diffusion_model.", ""))
    per_k, per_v = np.asarray(per_k), np.asarray(per_v)
    return {"to_k": per_k.mean(axis=0), "to_v": per_v.mean(axis=0),
            "per_layer_k": per_k, "per_layer_v": per_v, "layers": names}


def cross_attn_weights(ckpt: Path) -> dict:
    """The raw ``attn2.to_k``/``to_v`` matrices, keyed by block name.

    Needed for the "readable variance" test: how much of a *sampled* tensor's
    variance survives the projection into key/value space.
    """
    from safetensors import safe_open

    out = {}
    with safe_open(str(ckpt), framework="numpy") as f:
        for key in sorted(f.keys()):
            if "attn2" in key and (key.endswith("to_k.weight")
                                   or key.endswith("to_v.weight")):
                block = key.rsplit(".attn2.", 1)[0].replace(
                    "model.diffusion_model.", "")
                kind = "to_k" if key.endswith("to_k.weight") else "to_v"
                out[f"{block}|{kind}"] = np.asarray(f.get_tensor(key),
                                                    dtype=np.float32)
    return out


# ------------------------------------------------------------- the stats ---
@dataclass
class CorpusStats:
    """Everything the figures need, computed once from the cached embeddings."""

    X: np.ndarray                    # (N, T, H) float32 raw conditioning
    prompts: list = field(default_factory=list)
    eos_pos: Optional[np.ndarray] = None
    token_ids: Optional[np.ndarray] = None
    uncond: Optional[np.ndarray] = None

    # filled by :meth:`compute`
    mean: np.ndarray = None
    std: np.ndarray = None
    skew: np.ndarray = None
    kurt: np.ndarray = None          # EXCESS kurtosis (0 == Gaussian)
    q: dict = field(default_factory=dict)      # percentile -> (T, H)
    sig_pos: np.ndarray = None       # RMS of positive residuals
    sig_neg: np.ndarray = None       # RMS of negative residuals
    pca_sval: np.ndarray = None      # (k,) singular values / sqrt(N-1) == pca_std
    pca_scores: np.ndarray = None    # (N, k)
    pca_comp: Optional[np.ndarray] = None      # (k, D) -- optional, big
    null_sval: np.ndarray = None     # shuffled-coordinate null spectrum
    total_var: float = 0.0

    # ---------------------------------------------------------------------
    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def T(self) -> int:
        return self.X.shape[1]

    @property
    def H(self) -> int:
        return self.X.shape[2]

    @property
    def D(self) -> int:
        return self.T * self.H

    def flat(self) -> np.ndarray:
        return self.X.reshape(self.n, -1)

    def centered(self) -> np.ndarray:
        """(N, D) float32 deviations from the corpus mean."""
        return self.flat() - self.mean.reshape(1, -1)

    # ---------------------------------------------------------------------
    @classmethod
    def load(cls, cache: Path, keep_components: int = 0,
             stats_cache: Optional[Path] = None, seed: int = 0) -> "CorpusStats":
        """Load the cached corpus and compute (or reload) every statistic."""
        cache = Path(cache)
        z = np.load(cache)
        pj = cache.with_suffix(".prompts.json")
        self = cls(
            X=np.asarray(z["embeds"], dtype=np.float32),
            prompts=json.loads(pj.read_text()) if pj.exists() else [],
            eos_pos=z["eos_pos"] if "eos_pos" in z else None,
            token_ids=z["token_ids"] if "token_ids" in z else None,
            uncond=z["uncond"][0] if "uncond" in z else None,
        )
        if stats_cache and Path(stats_cache).exists():
            self._reload(Path(stats_cache))
        else:
            self.compute(keep_components=keep_components, seed=seed)
            if stats_cache:
                self._save(Path(stats_cache))
        return self

    def _save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(mean=self.mean, std=self.std, skew=self.skew,
                       kurt=self.kurt, sig_pos=self.sig_pos, sig_neg=self.sig_neg,
                       pca_sval=self.pca_sval, pca_scores=self.pca_scores,
                       null_sval=self.null_sval,
                       total_var=np.float64(self.total_var))
        payload.update({f"q{int(k)}": v for k, v in self.q.items()})
        if self.pca_comp is not None:
            payload["pca_comp"] = self.pca_comp
        np.savez_compressed(path, **payload)

    def _reload(self, path: Path) -> None:
        z = np.load(path)
        for k in ("mean", "std", "skew", "kurt", "sig_pos", "sig_neg",
                  "pca_sval", "pca_scores", "null_sval", "pca_comp"):
            if k in z:
                setattr(self, k, z[k])
        self.total_var = float(z["total_var"])
        self.q = {int(k[1:]): z[k] for k in z.files if k.startswith("q")}

    # ---------------------------------------------------------------------
    def compute(self, keep_components: int = 0, seed: int = 0) -> None:
        """Moments, quantile envelope, half-sigmas, PCA and its shuffled null."""
        x = self.X.astype(np.float64)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0)

        dev = x - self.mean[None]
        s = np.maximum(self.std, 1e-12)
        z = dev / s[None]
        self.skew = (z ** 3).mean(axis=0)
        self.kurt = (z ** 4).mean(axis=0) - 3.0

        # Empirical envelope: the old implementation clamped to k * [q05, q95].
        for p in (1, 5, 25, 50, 75, 95, 99):
            self.q[p] = np.percentile(x, p, axis=0)
        self.q[0] = x.min(axis=0)
        self.q[100] = x.max(axis=0)

        # Asymmetric spread: RMS of the residuals on each side of the mean.
        pos, neg = np.where(dev > 0, dev, np.nan), np.where(dev < 0, dev, np.nan)
        with np.errstate(invalid="ignore"):
            self.sig_pos = np.sqrt(np.nanmean(pos ** 2, axis=0))
            self.sig_neg = np.sqrt(np.nanmean(neg ** 2, axis=0))
        self.sig_pos = np.nan_to_num(self.sig_pos)
        self.sig_neg = np.nan_to_num(self.sig_neg)

        # ---- PCA via the Gram trick (N=1000 << D=59136) -------------------
        xc = dev.reshape(self.n, -1)
        self.total_var = float((xc ** 2).sum() / (self.n - 1))
        sval, scores, comp = _gram_pca(xc, keep_components)
        self.pca_sval, self.pca_scores, self.pca_comp = sval, scores, comp

        # ---- the null: shuffle every coordinate independently -------------
        # Kills all cross-coordinate correlation, preserves each marginal
        # exactly. Its spectrum is the noise floor a "real" component must beat.
        rng = np.random.default_rng(seed)
        xs = xc.copy()
        for j in range(xs.shape[1]):
            rng.shuffle(xs[:, j])
        xs -= xs.mean(axis=0, keepdims=True)
        self.null_sval, _, _ = _gram_pca(xs, 0)

    # ------------------------------------------------------------ helpers ---
    def token_region(self) -> np.ndarray:
        """Per-position label: 0 = BOS, 1 = content, 2 = post-EOS padding.

        A position is "content" if it is before the EOS of at least half the
        prompts -- i.e. the region where most prompts still carry real words.
        """
        reg = np.full(self.T, 2, dtype=np.int8)
        med = int(np.median(self.eos_pos)) if self.eos_pos is not None else 20
        reg[1:med] = 1
        reg[BOS] = 0
        return reg

    def coord_var(self) -> np.ndarray:
        return self.std ** 2

    def live_mask(self, frac: float = 1e-3) -> np.ndarray:
        """Coordinates whose sigma exceeds ``frac`` of the largest sigma."""
        return self.std > frac * self.std.max()


def _gram_pca(xc: np.ndarray, keep_components: int = 0):
    """Top-k PCA of a centered (N, D) matrix through its (N, N) Gram matrix.

    Returns ``(pca_std (k,), scores (N, k), components (k, D) or None)``.
    ``pca_std`` matches ``EmbeddingDistribution.pca_std`` (singular values over
    sqrt(N-1)) so the two are directly comparable.
    """
    n = xc.shape[0]
    g = xc @ xc.T
    evals, evecs = np.linalg.eigh(g)
    order = np.argsort(evals)[::-1][: n - 1]
    evals = np.maximum(evals[order], 0.0)
    u = evecs[:, order]
    sv = np.sqrt(evals)
    scores = u * sv[None]                       # (N, k) projections onto axes
    comp = None
    if keep_components:
        k = min(keep_components, len(sv))
        nz = np.maximum(sv[:k], 1e-12)
        comp = ((u[:, :k].T @ xc) / nz[:, None]).astype(np.float32)
    return (sv / np.sqrt(n - 1)).astype(np.float64), scores.astype(np.float64), comp


# ------------------------------------------------------- derived measures ---
def effective_dimension(var: np.ndarray) -> float:
    """Participation ratio of a variance spectrum: ``(sum v)^2 / sum v^2``.

    The number of directions that "matter": 1 for a single dominant axis, k for
    k equal axes. Reported alongside the raw component count because 999
    components does not mean 999 degrees of freedom.
    """
    var = np.asarray(var, dtype=np.float64)
    return float(var.sum() ** 2 / np.maximum((var ** 2).sum(), 1e-30))


def jarque_bera(n: int, skew: np.ndarray, kurt: np.ndarray) -> np.ndarray:
    """Jarque-Bera statistic per coordinate, from moments already computed.

    ``JB = n/6 * (S^2 + K^2/4)``, ~chi2(2) under normality, so >13.82 rejects at
    p<0.001. Vectorised over all 59,136 coordinates -- no per-coordinate test loop.
    """
    return (n / 6.0) * (skew ** 2 + (kurt ** 2) / 4.0)


def bimodality_coefficient(skew: np.ndarray, kurt: np.ndarray, n: int) -> np.ndarray:
    """Sarle's bimodality coefficient; > 5/9 suggests two modes."""
    m2 = kurt  # excess
    num = skew ** 2 + 1.0
    den = m2 + 3.0 * ((n - 1) ** 2) / ((n - 2) * (n - 3))
    return num / np.maximum(den, 1e-12)


def readable_variance(cov: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Variance the UNet actually sees, under the true vs the diagonal model.

    ``cov`` is a (H, H) channel covariance at one token position and ``w`` a
    cross-attention projection (inner, H). Returns
    ``(trace(W cov W^T), trace(W diag(cov) W^T))`` -- the second is what the
    ``diagonal`` sampler delivers. Their ratio is the sharpest single number for
    "how wrong is coordinate independence", measured where it matters: in the
    only space the model reads conditioning through.
    """
    true = float(np.einsum("ij,jk,ik->", w, cov, w, optimize=True))
    diag = float(((w ** 2) * np.diag(cov)[None]).sum())
    return true, diag


def central_mass(z: np.ndarray, half_width: float = 0.5) -> np.ndarray:
    """Fraction of each coordinate's samples within +-``half_width`` sigma.

    A Gaussian gives 0.3829 at 0.5 sigma. A coordinate that comes in far BELOW
    that has a hole where the fitted Gaussian puts its peak -- the signature of a
    two-lobed marginal, and precisely where the diagonal sampler wastes its
    draws. ``z`` is the standardized (N, ...) array.
    """
    return (np.abs(z) < half_width).mean(axis=0)


GAUSSIAN_CENTRAL_MASS = 0.38292492


def pad_gating(X: np.ndarray, eos_pos: np.ndarray, min_group: int = 20
               ) -> np.ndarray:
    """Variance explained at each coordinate by "is this position past EOS?".

    CLIP pads to 77 with EOS, so for any position in the middle of the prompt-
    length distribution the corpus is a MIXTURE of two populations: prompts
    whose content still reaches this far, and prompts that are already padding.
    That latent binary is not in any fitted model here. This returns the
    between-group variance fraction (eta^2) per (position, channel).
    """
    n, t_len, h = X.shape
    out = np.zeros((t_len, h))
    for t in range(t_len):
        m = eos_pos <= t
        if m.sum() < min_group or (~m).sum() < min_group:
            continue
        a, b = X[m, t, :], X[~m, t, :]
        pa = m.mean()
        gm = pa * a.mean(0) + (1 - pa) * b.mean(0)
        between = pa * (a.mean(0) - gm) ** 2 + (1 - pa) * (b.mean(0) - gm) ** 2
        out[t] = between / np.maximum(X[:, t, :].var(0), 1e-12)
    return out


def block_participation(X: np.ndarray, lo: int, hi: int) -> float:
    """Effective dimensionality of one slice of token positions.

    ``(hi - lo) * H`` coordinates go in; the participation ratio that comes out
    says how many independent directions that block really has. The gap between
    the two is how much of the "anarchy" budget the diagonal sampler spends on
    directions the corpus never uses.
    """
    b = X[:, lo:hi, :].reshape(len(X), -1).astype(np.float64)
    b -= b.mean(0, keepdims=True)
    ev = np.maximum(np.linalg.eigvalsh(b @ b.T)[::-1], 0.0)
    return effective_dimension(ev)


def readable_variance_ratio(X: np.ndarray, mean: np.ndarray, weights: dict,
                            positions) -> dict:
    """true / diagonal readable variance, per cross-attention matrix.

    For each token position the conditioning row is projected by ``to_k``/``to_v``
    independently, so this asks: of the variance the diagonal model *claims* to
    deliver into key/value space, how much does the real corpus have? A ratio
    near 1 means the failure of coordinate independence is NOT one of magnitude
    -- it is structural, and lives across token positions rather than within one.
    """
    out = {}
    for t in positions:
        d = X[:, t, :].astype(np.float64) - mean[t]
        cov = (d.T @ d) / (len(X) - 1)
        diagv = np.diag(cov)
        ratios = {}
        for name, w in weights.items():
            w = w.astype(np.float64)
            true = float(((w @ cov) * w).sum())
            diag = float(((w ** 2) * diagv[None]).sum())
            ratios[name] = true / max(diag, 1e-30)
        out[t] = ratios
    return out


def pairwise_cosine(x: np.ndarray, rng=None, m: int = 4000) -> np.ndarray:
    """Cosine similarities of ``m`` random row pairs of a (N, D) matrix."""
    rng = rng or np.random.default_rng(0)
    n = x.shape[0]
    i = rng.integers(0, n, m)
    j = rng.integers(0, n, m)
    keep = i != j
    i, j = i[keep], j[keep]
    a = x[i] / np.maximum(np.linalg.norm(x[i], axis=1, keepdims=True), 1e-12)
    b = x[j] / np.maximum(np.linalg.norm(x[j], axis=1, keepdims=True), 1e-12)
    return (a * b).sum(axis=1)


def gmm_bic(scores: np.ndarray, ks=(1, 2, 3, 4, 5, 6, 8, 10), seed: int = 0):
    """Diagonal-covariance GMM BIC over PCA scores -- is the corpus multimodal?

    Answers the open question from the report: the shipped model is a SINGLE
    Gaussian; the original description said "mixture". Fitted on the leading PCA
    scores (where any real cluster structure lives) with a diagonal covariance,
    so the parameter count stays honest at N=1000.
    """
    try:
        from sklearn.mixture import GaussianMixture
    except Exception:
        return None
    out = []
    for k in ks:
        gm = GaussianMixture(k, covariance_type="diag", random_state=seed,
                             n_init=3, max_iter=300).fit(scores)
        out.append((k, float(gm.bic(scores))))
    return out
