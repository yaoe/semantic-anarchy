#!/usr/bin/env python3
"""The verifiable demo -- the whole Semantic Anarchy idea, no torch / no SD.

It never imports torch or diffusers. Instead it manufactures *synthetic*
"prompt embeddings" whose per-coordinate marginals look like the deck's varied
``Mean`` / ``Std`` subplots, then runs the real machinery on them:

1. Fit an :class:`EmbeddingDistribution` to the synthetic corpus.
2. Sample fresh embeddings from it (the "promptless generation" step, minus the
   actual SD decode).
3. Run a toy evolution with an analytic scorer -- selection by "resonance".
4. Emit BOTH slide-6 (sparse "rug", few samples) and slide-7 (dense histogram,
   many samples) marginal plots into ``outputs/``.

Run it::

    python scripts/demo_no_sd.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running both as ``python scripts/demo_no_sd.py`` and ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from semantic_anarchy import EmbeddingDistribution, evolve_distribution, plot_marginals


def synthetic_corpus(
    n: int,
    tokens: int = 77,
    hidden: int = 64,
    seed: int = 0,
) -> np.ndarray:
    """Make ``n`` fake conditioning embeddings with deck-like varied marginals.

    Real CLIP conditioning coordinates have a spread of means and stds -- some
    coordinates are sharp and centered, others wide and offset. We reproduce that
    by giving every ``(token, feature)`` coordinate its own random Gaussian and
    drawing a corpus from a *mixture* (two latent "prompt clusters"), so the
    marginals aren't all identical standard normals.
    """
    rng = np.random.default_rng(seed)
    shape = (tokens, hidden)

    # Per-coordinate target statistics, varied like the slides.
    mu_a = rng.normal(0.0, 0.8, size=shape)
    sd_a = rng.uniform(0.3, 1.4, size=shape)
    mu_b = rng.normal(0.0, 1.1, size=shape)
    sd_b = rng.uniform(0.2, 1.0, size=shape)

    # Two clusters -> a believable mixture, then concatenate.
    half = n // 2
    a = mu_a[None] + sd_a[None] * rng.standard_normal((half, *shape))
    b = mu_b[None] + sd_b[None] * rng.standard_normal((n - half, *shape))
    corpus = np.concatenate([a, b], axis=0)
    rng.shuffle(corpus)
    return corpus.astype(np.float32)


def analytic_objective(target: np.ndarray):
    """A pure-function 'aesthetic' scorer: closeness to a hidden target tensor.

    Stands in for "aesthetic resonance" -- the evolution loop should discover the
    target and drive the population's mean toward it, raising the mean score.
    """

    def _score(population: np.ndarray) -> np.ndarray:
        diff = population.reshape(population.shape[0], -1) - target.reshape(-1)[None]
        return (-np.sqrt((diff ** 2).mean(axis=1))).astype(np.float32)

    return _score


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000, help="corpus size (dense regime)")
    parser.add_argument("--tokens", type=int, default=77)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--pop-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=Path, default=Path("outputs"))
    args = parser.parse_args(argv)

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 70)
    print("SEMANTIC ANARCHY -- no-SD demo (synthetic prompt embeddings)")
    print("=" * 70)

    # 1. Harvest a synthetic corpus and fit the distribution -------------------
    corpus = synthetic_corpus(args.n, args.tokens, args.hidden, seed=args.seed)
    print(f"\n[1] synthetic corpus: {corpus.shape}  (N, tokens, hidden)")
    dist = EmbeddingDistribution.fit(corpus, per_token=True)
    print(f"    fitted distribution: {dist.summary()}")

    # Pick a few coordinates with genuinely different Mean/Std to plot.
    coords = _varied_coords(corpus, k=4, rng=rng)

    # 2. Slide-7 (dense histogram, many samples) -------------------------------
    hist_path = plot_marginals(
        corpus, coords=coords, style="hist",
        out_path=args.outdir / "marginals_hist_slide7.png", rng=rng,
    )
    print(f"\n[2] slide-7 dense-histogram marginals -> {hist_path}")

    # 3. Slide-6 (sparse rug, few samples) -------------------------------------
    few = corpus[:24]
    rug_path = plot_marginals(
        few, coords=coords, style="rug",
        out_path=args.outdir / "marginals_rug_slide6.png", rng=rng,
    )
    print(f"[3] slide-6 sparse-rug marginals (24 samples) -> {rug_path}")

    # 4. Promptless sampling ----------------------------------------------------
    samples = dist.sample(8, temperature=1.0, truncation=3.0, rng=rng)
    print(f"\n[4] sampled {samples.shape[0]} fresh embeddings directly from the "
          f"distribution (shape {samples.shape[1:]}); in the full pipeline these "
          "decode straight through the UNet -- no prompt, no text encoder.")

    # 5. Toy evolution toward a hidden 'aesthetic' target ----------------------
    target = dist.sample(1, temperature=1.5, rng=np.random.default_rng(99))[0]
    scorer = analytic_objective(target)
    print(f"\n[5] evolving toward a hidden aesthetic target "
          f"({args.generations} generations, pop={args.pop_size}):")
    evolved, history = evolve_distribution(
        dist, scorer,
        generations=args.generations, pop_size=args.pop_size,
        elite_frac=0.2, temperature=1.0, rng=rng, verbose=True,
    )

    # Evolved branch marginals, to show the distribution has moved.
    evolved_corpus = evolved.sample(args.n, rng=rng)
    branch_path = plot_marginals(
        evolved_corpus, coords=coords, style="hist",
        out_path=args.outdir / "marginals_evolved_branch.png", rng=rng,
    )

    start, end = history.elite_mean_score[0], history.elite_mean_score[-1]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("-" * 70)
    print(f"corpus                : {corpus.shape[0]} embeddings of shape "
          f"{corpus.shape[1:]}")
    print(f"distribution mean/std : mean_of_means={dist.summary()['mean_of_means']:+.3f}, "
          f"mean_of_stds={dist.summary()['mean_of_stds']:.3f}")
    print(f"evolution elite score : {start:+.3f} -> {end:+.3f}  "
          f"({'IMPROVED' if history.improved() else 'no improvement'})")
    print("plots written         :")
    for p in (hist_path, rug_path, branch_path):
        print(f"    - {p}")
    print("=" * 70)
    print("\nNo torch, no diffusers, no GPU were used. To decode embeddings into")
    print("actual images, install the full tier and run scripts/generate.py.")
    return 0


def _varied_coords(corpus: np.ndarray, k: int, rng) -> list[tuple[int, int]]:
    """Choose ``k`` coordinates whose means/stds differ, for a deck-like figure."""
    flat = corpus.reshape(corpus.shape[0], -1)
    stds = flat.std(axis=0)
    # Spread picks across the std range so subplots show varied 'Std:' values.
    order = np.argsort(stds)
    picks = order[np.linspace(0, len(order) - 1, k).astype(int)]
    tokens, hidden = corpus.shape[1], corpus.shape[2]
    return [(int(p // hidden), int(p % hidden)) for p in picks]


if __name__ == "__main__":
    raise SystemExit(main())
