"""Fit a distribution to the latents of a hand-picked set of images.

The other way to get an :class:`~semantic_anarchy.distribution.EmbeddingDistribution`
is to *mine* one: run a corpus of prompts through the text encoder and fit the
result (``scripts/mine_distribution.py``). This module does the same fit over a
corpus you assembled by eye — every image you starred, everything you scored 8+,
everything from one experiment — using the ``.npz`` conditioning sidecar each
generated image already carries. No text encoder, no GPU, no torch: the latents
are already on disk, so this is a stack-and-fit.

It replaces the old ``evolve_favorites.py`` move for the "sample more like these"
job, and the difference is the whole point. That one kept the *corpus* PCA basis
and merely re-centred the per-coordinate Gaussian on the elite mean, so the
``pca`` sampler went on drawing corpus-sized deviations around a centre the basis
knows nothing about — which lands off-manifold and looks it. Here the selection
is fitted from scratch: its own mean, its own std, and its own PCA subspace, so a
``pca`` draw is a combination of *the latents you picked* and nothing else.

The result is an ordinary fit saved under the ordinary naming rules
(:mod:`semantic_anarchy.dist_paths`), so it is selectable as a base distribution
and every sampler, temperature and correction works on it unchanged.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .backend import dist_backend
from .distribution import EmbeddingDistribution
from .dist_paths import backend_prefix, dist_files, dist_ready
from .upscale import conditioning_source

#: Where dashboard-made fits live. Inside ``outputs/`` (regenerable — the images
#: and their sidecars are the real source), one base prefix per fit.
FIT_DIR = Path("outputs/dist_fits")

#: Suffix of the manifest written beside every fit: which images went into it.
#: Sits on the *base* prefix, so it names the whole multi-tensor set at once.
MANIFEST_SUFFIX = ".fit.json"

#: Same PCA cap as mining, for the same reason (each axis is a full feature row
#: on disk). A hand-picked selection is almost always far below it.
MAX_COMPONENTS = 512

#: Below this the fit is degenerate — PCA needs at least two points to have a
#: direction at all, and a two-point "subspace" is a line segment.
MIN_SAMPLES = 3


def slug_name(raw: str) -> str:
    """A filename-safe fit name (same rule as ``dist_paths.model_slug``)."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(raw).strip()).strip("-._")
    return slug[:64]


def fit_base(name: str, directory: str | Path = FIT_DIR) -> str:
    """The ``--dist`` base a named fit is saved to (pre-backend-namespacing)."""
    slug = slug_name(name)
    if not slug:
        raise ValueError("fit name is empty after slugging")
    return str(Path(directory) / slug)


def manifest_path(base: str | Path) -> Path:
    """Where a fit's source manifest lives, given its base prefix."""
    return Path(str(base) + MANIFEST_SUFFIX)


def latents_for(image: str | Path) -> Optional[Path]:
    """The ``.npz`` holding this image's conditioning, or ``None``.

    Follows the ``refined_from`` chain, so an upscale contributes the latents of
    the original it was made from rather than dropping out of the selection.
    """
    try:
        src, _meta = conditioning_source(Path(image))
    except FileNotFoundError:
        return None
    npz = src.with_suffix(".npz")
    return npz if npz.is_file() else None


def stack_latents(
    npz_paths: Iterable[str | Path],
    tensor_names: Iterable[str],
) -> tuple[dict[str, np.ndarray], list[Path], list[tuple[Path, str]]]:
    """Stack per-image conditioning sidecars into one ``(N, *shape)`` array each.

    Returns ``(stacked, used, skipped)``. A sidecar missing a tensor the backend
    needs, or carrying a different feature shape than the first one accepted
    (a 77×768 sd15 latent among sdxl's 77×2048), is *skipped with a reason*
    rather than failing the whole fit — a hand-assembled selection is exactly
    where one stray image is likely to turn up.
    """
    names = list(tensor_names)
    cols: dict[str, list[np.ndarray]] = {n: [] for n in names}
    shapes: dict[str, tuple] = {}
    used: list[Path] = []
    skipped: list[tuple[Path, str]] = []

    for path in npz_paths:
        p = Path(path)
        try:
            with np.load(p) as data:
                arrays = {n: np.asarray(data[n]) for n in names if n in data.files}
        except (OSError, ValueError) as exc:
            skipped.append((p, f"unreadable ({exc.__class__.__name__})"))
            continue
        missing = [n for n in names if n not in arrays]
        if missing:
            skipped.append((p, f"no {'/'.join(missing)} tensor"))
            continue
        bad = next((n for n in names
                    if n in shapes and arrays[n].shape != shapes[n]), None)
        if bad is not None:
            skipped.append((p, f"{bad} is {arrays[bad].shape}, expected {shapes[bad]}"))
            continue
        for n in names:
            shapes.setdefault(n, arrays[n].shape)
            cols[n].append(arrays[n])
        used.append(p)

    stacked = {n: np.stack(v) for n, v in cols.items() if v}
    return stacked, used, skipped


def fit_latents(
    stacked: dict[str, np.ndarray],
    n_components: Optional[int] = None,
    max_corpus: int = 256,
) -> dict[str, EmbeddingDistribution]:
    """Fit one distribution per named tensor over the stacked selection.

    ``n_components=None`` keeps ``min(N-1, MAX_COMPONENTS)`` axes — for a
    selection that is almost always the full rank, which is what makes the
    ``pca`` sampler draw inside the span of the picked latents.
    """
    out = {}
    for name, arr in stacked.items():
        n = arr.shape[0]
        if n < 2:
            raise ValueError(f"{name}: need at least 2 samples to fit, got {n}")
        k = min(n - 1, MAX_COMPONENTS) if not n_components else int(n_components)
        out[name] = EmbeddingDistribution.fit(
            arr, n_components=k, max_corpus=max_corpus)
    return out


def write_manifest(
    base: str | Path,
    backend: str,
    sources: Iterable[str],
    *,
    name: Optional[str] = None,
    note: Optional[str] = None,
    models: Optional[Iterable[str]] = None,
    skipped: Optional[Iterable[tuple[str, str]]] = None,
    extra: Optional[dict] = None,
) -> Path:
    """Record which images this fit was built from, beside the fit itself.

    Not needed to *sample* it — the ``.npz`` is self-contained — but it is the
    only record of what "my keepers, week 3" actually meant, and it is what lets
    the picker list saved fits with a size and a date.
    """
    sources = [str(s) for s in sources]
    payload = {
        "kind": "selection",
        "name": name or Path(str(base)).name,
        "backend": backend,
        "created": time.time(),
        "n_samples": len(sources),
        "sources": sources,
        "models": sorted(set(models or [])),
        "note": note or None,
        "skipped": [{"path": str(p), "reason": r} for p, r in (skipped or [])],
        **(extra or {}),
    }
    path = manifest_path(base)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def read_manifest(base: str | Path) -> Optional[dict]:
    """A fit's manifest, or ``None`` when it has none (hand-made .npz)."""
    try:
        data = json.loads(manifest_path(base).read_text())
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def list_fits(directory: str | Path = FIT_DIR,
              backend: Optional[str] = None) -> list[dict]:
    """Every saved selection fit in ``directory``, newest first.

    Keyed on the manifest rather than the ``.npz``, because one base can expand
    to several tensor files and only the manifest names the *set*. ``backend``
    (when given) fills in whether that backend's files are actually on disk.
    """
    d = Path(directory)
    if not d.is_dir():
        return []
    rows = []
    for m in d.glob(f"*{MANIFEST_SUFFIX}"):
        base = str(m)[: -len(MANIFEST_SUFFIX)]
        data = read_manifest(base) or {}
        b = backend or data.get("backend")
        row = {
            "name": data.get("name") or Path(base).name,
            "base": base,
            "backend": data.get("backend"),
            "created": data.get("created"),
            "n_samples": data.get("n_samples"),
            "note": data.get("note"),
            "models": data.get("models") or [],
            "ready": bool(b) and dist_ready(base, b),
            "files": [str(f) for f in (dist_files(base, b) if b else [])],
        }
        rows.append(row)
    rows.sort(key=lambda r: r.get("created") or 0, reverse=True)
    return rows


def save_fit(
    dists: dict[str, EmbeddingDistribution],
    base: str | Path,
    backend: str,
) -> list[Path]:
    """Write the fitted tensors under the standard naming stack.

    ``dist_backend`` is the model-less instance — no torch, no weights — so the
    backend-namespacing and the per-tensor suffixes come out identical to a
    mined fit's without a pipeline ever being constructed.
    """
    return dist_backend(backend).save_dists(dists, backend_prefix(base, backend))
