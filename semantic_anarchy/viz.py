"""Reproduce the per-dimension distribution plots from slides 6-7 of the deck.

Each subplot shows one coordinate of the conditioning embedding: a histogram of
its values across the prompt corpus, with the fitted Gaussian (``Mean`` / ``Std``)
overlaid in dashed red -- exactly the figure the talk uses to argue "this thing is
just a pile of independent Gaussians you can sample from."
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def _gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    sigma = max(sigma, 1e-6)
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def plot_marginals(
    embeddings: np.ndarray,
    coords: Optional[Sequence[tuple[int, ...]]] = None,
    n: int = 4,
    bins: int = 60,
    style: str = "hist",
    xlim: tuple[float, float] = (-4, 4),
    out_path: str | Path = "outputs/marginals.png",
    rng: Optional[np.random.Generator] = None,
):
    """Plot ``n`` coordinate marginals with their fitted Gaussians.

    Parameters
    ----------
    embeddings:
        ``(N, *feature_shape)`` stack of conditioning embeddings.
    coords:
        Specific coordinates to plot (tuples indexing ``feature_shape``). If None,
        ``n`` random coordinates are chosen.
    style:
        ``"hist"`` reproduces slide 7 (dense histogram, many samples); ``"rug"``
        reproduces slide 6 (sparse vertical ticks for the few-sample regime).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    embeddings = np.asarray(embeddings, dtype=np.float64)
    feature_shape = embeddings.shape[1:]
    rng = rng or np.random.default_rng(0)

    if coords is None:
        coords = [tuple(rng.integers(0, d) for d in feature_shape) for _ in range(n)]
    n = len(coords)

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.2), squeeze=False)
    xs = np.linspace(xlim[0], xlim[1], 400)

    for ax, coord in zip(axes[0], coords):
        idx = (slice(None),) + tuple(coord)
        values = embeddings[idx]
        mu, sigma = float(values.mean()), float(values.std())

        if style == "rug":
            for v in values:
                ax.axvline(v, color="tab:blue", alpha=0.7, linewidth=1.2)
            ax.set_ylim(0, 0.5)
        else:
            ax.hist(values, bins=bins, range=xlim, density=True,
                    color="tab:blue", alpha=0.85)

        ax.plot(xs, _gaussian(xs, mu, sigma), "r--", linewidth=1.6)
        ax.set_title(f"Mean: {mu:.2f}, Std: {sigma:.2f}")
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 0.5)

    axes[0][0].set_ylabel("Density")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, facecolor="white")
    plt.close(fig)
    return out_path
