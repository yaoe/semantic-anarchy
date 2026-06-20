"""Semantic Anarchy -- chaos by design, promptless Stable Diffusion.

A small, faithful implementation of the early Eden.art / Abraham "Semantic
Anarchy: Chaos by design" method. The core move (slide 9 of the deck): **X out
the CLIP text encoder.** "There is no prompt."

The pipeline instead:

1. Encodes a corpus of good prompts once, harvesting their conditioning
   embeddings ``c`` (``(77, 768)`` for SD 1.x) -- :mod:`semantic_anarchy.pipeline`.
2. Learns that cloud's distribution as an **independent Gaussian per coordinate**
   (slides 6-7) -- :class:`~semantic_anarchy.distribution.EmbeddingDistribution`.
3. Samples brand-new ``c`` straight from that distribution and decodes them,
   bypassing the text encoder entirely -- pure AI aesthetic, no linguistics.
4. Keeps what scores well ("selection through aesthetic resonance", slide 13) --
   :mod:`semantic_anarchy.aesthetic` -- and evolves the distribution into
   personalized "evolutionary branches" (slides 16, 44) --
   :mod:`semantic_anarchy.evolve`.

Everything that doesn't strictly need a GPU is pure NumPy, so the statistics,
plots, evolution loop and tests run with no torch / diffusers installed. The
``demo_no_sd`` script exercises the whole idea end to end on synthetic embeddings.
"""

from __future__ import annotations

from .distribution import EmbeddingDistribution
from .backend import (
    Backend,
    SD15Backend,
    SDXLBackend,
    make_backend,
    dist_backend,
    BACKEND_DEFAULTS,
)
from .viz import plot_marginals
from .aesthetic import (
    Scorer,
    AestheticScorer,
    HumanScorer,
    RandomScorer,
    get_scorer,
)
from .evolve import EvolutionHistory, evolve_distribution

__version__ = "0.1.0"

__all__ = [
    "EmbeddingDistribution",
    "Backend",
    "SD15Backend",
    "SDXLBackend",
    "make_backend",
    "dist_backend",
    "BACKEND_DEFAULTS",
    "plot_marginals",
    "Scorer",
    "AestheticScorer",
    "HumanScorer",
    "RandomScorer",
    "get_scorer",
    "EvolutionHistory",
    "evolve_distribution",
    "__version__",
]
