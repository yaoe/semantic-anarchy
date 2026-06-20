"""Evolutionary preference loop -- growing "branches in the visual tree".

Slides 16 & 44: once you can sample conditioning embeddings and score them, you
can *evolve* the distribution toward what resonates. This is a cross-entropy
method (CEM) on the NumPy embeddings:

    for each generation:
        population = dist.sample(pop_size)      # draw candidate embeddings
        scores     = scorer(population)         # aesthetic resonance
        elites     = top-k by score             # selection
        dist       = dist.refit_from_elites(..) # the new evolutionary branch

The search itself touches no torch -- ``scorer`` is any callable
``embeddings -> scores``. For real images that callable wraps SDModel + an
image :class:`~semantic_anarchy.aesthetic.Scorer` (see :func:`make_image_scorer`),
but the selection logic is fully exercised by a pure analytic objective in the
tests and the no-SD demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from .distribution import EmbeddingDistribution

# An embedding scorer maps a population ``(P, *feature_shape)`` to scores ``(P,)``.
EmbeddingScorer = Callable[[np.ndarray], np.ndarray]


@dataclass
class EvolutionHistory:
    """Per-generation record of the evolutionary run."""

    mean_score: list[float] = field(default_factory=list)
    max_score: list[float] = field(default_factory=list)
    elite_mean_score: list[float] = field(default_factory=list)

    def log(self, scores: np.ndarray, elite_scores: np.ndarray) -> None:
        self.mean_score.append(float(scores.mean()))
        self.max_score.append(float(scores.max()))
        self.elite_mean_score.append(float(elite_scores.mean()))

    def improved(self) -> bool:
        """Did the elite-mean score end higher than it started?"""
        return len(self.elite_mean_score) >= 2 and (
            self.elite_mean_score[-1] >= self.elite_mean_score[0]
        )


def evolve_distribution(
    dist: EmbeddingDistribution,
    scorer: EmbeddingScorer,
    generations: int = 10,
    pop_size: int = 64,
    elite_frac: float = 0.25,
    temperature: float = 1.0,
    truncation: Optional[float] = None,
    base_blend: float = 0.25,
    rng: Optional[np.random.Generator] = None,
    verbose: bool = True,
) -> tuple[EmbeddingDistribution, EvolutionHistory]:
    """Run CEM on the embedding distribution, returning the evolved branch.

    Parameters
    ----------
    dist:
        Starting distribution (typically the corpus-mined one).
    scorer:
        Callable ``population -> scores`` (higher is better). Pure-function on
        embeddings, so the loop is GPU-free; image scoring is injected via this.
    generations, pop_size, elite_frac:
        Standard CEM knobs. ``elite_frac`` of each population seeds the next
        branch via :meth:`EmbeddingDistribution.refit_from_elites`.
    temperature, truncation:
        Passed through to :meth:`EmbeddingDistribution.sample` -- how wild the
        candidates each generation are.

    Returns
    -------
    (evolved_dist, history)
    """
    rng = rng or np.random.default_rng()
    n_elite = max(1, int(round(pop_size * elite_frac)))
    history = EvolutionHistory()

    for gen in range(generations):
        population = dist.sample(
            pop_size, temperature=temperature, truncation=truncation, rng=rng
        )
        scores = np.asarray(scorer(population), dtype=np.float64)
        if scores.shape[0] != population.shape[0]:
            raise ValueError("scorer returned wrong number of scores")

        elite_idx = np.argsort(scores)[-n_elite:]
        elites = population[elite_idx]
        elite_scores = scores[elite_idx]
        history.log(scores, elite_scores)

        dist = dist.refit_from_elites(elites, base_blend=base_blend)

        if verbose:
            print(
                f"gen {gen + 1:>3}/{generations}  "
                f"mean={scores.mean():7.3f}  max={scores.max():7.3f}  "
                f"elite_mean={elite_scores.mean():7.3f}"
            )

    return dist, history


def make_image_scorer(
    sd_model,
    image_scorer,
    negative_embedding: Optional[np.ndarray] = None,
    **gen_kwargs,
) -> EmbeddingScorer:
    """Wrap an SDModel + image :class:`Scorer` into an embedding scorer.

    The returned callable decodes each embedding to an image and scores it --
    this is the bridge that lets :func:`evolve_distribution` (which only knows
    about embeddings) drive a full render-and-judge loop. Kept out of the core
    search so the search stays torch-free and unit-testable.
    """

    def _score(population: np.ndarray) -> np.ndarray:
        images = sd_model.generate_from_embeddings(
            population, negative_embedding=negative_embedding, **gen_kwargs
        )
        return np.asarray(image_scorer.score(images), dtype=np.float32)

    return _score
