"""Small output helpers -- keep generated artifacts non-destructive.

Re-running a generation into the same ``--outdir`` should never silently clobber
an earlier batch. :func:`unique_path` is the one rule every file write goes
through: if the target exists, it appends ``_1``, ``_2``, ... before the suffix
until it finds a free name.
"""

from __future__ import annotations

from pathlib import Path


def unique_path(path: str | Path) -> Path:
    """Return ``path``, or the first ``stem_N.suffix`` variant that doesn't exist.

    ``outputs/sweep.png`` -> ``outputs/sweep.png`` (free) or ``outputs/sweep_1.png``,
    ``outputs/sweep_2.png``, ... Parent directories are not created here.
    """
    path = Path(path)
    if not path.exists():
        return path
    stem, suffix, parent = path.stem, path.suffix, path.parent
    i = 1
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1
