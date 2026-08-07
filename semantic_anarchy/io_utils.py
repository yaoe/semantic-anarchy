"""Small output helpers -- keep generated artifacts non-destructive.

Re-running a generation into the same ``--outdir`` should never silently clobber
an earlier batch. :func:`unique_path` is the one rule every file write goes
through: if the target exists, it appends ``_1``, ``_2``, ... before the suffix
until it finds a free name.

This module also owns the **image format** every renderer writes. Generated
frames are stored as high-quality JPEG (4:4:4, no chroma subsampling) rather
than PNG: a 1024x1024 diffusion output costs ~1.5MB as PNG and ~250KB here,
which is the difference between a gallery that fits on disk and one that
doesn't. ``SA_IMAGE_FORMAT=png`` restores lossless output, ``SA_JPEG_QUALITY``
tunes the tradeoff. Everything that *reads* the gallery must accept
:data:`IMAGE_EXTS`, not just the current format -- images mined before the
switch are still PNGs and stay explorable.
"""

from __future__ import annotations

import os
from pathlib import Path

#: Every extension the gallery/scan/resolve paths recognise as a rendered image.
#: Order matters only for :func:`find_image`, which prefers the current format.
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")

#: Written next to every image, keyed on its stem (params + conditioning).
SIDECAR_EXTS = (".json", ".npz")

#: Format used when nothing overrides it.
DEFAULT_IMAGE_FORMAT = "jpg"

#: Quality for JPEG writes. 95 with subsampling off is visually lossless on
#: diffusion output while still ~6x smaller than PNG.
DEFAULT_JPEG_QUALITY = 95


def image_ext() -> str:
    """The suffix new renders are written with (``.jpg`` unless overridden)."""
    raw = os.environ.get("SA_IMAGE_FORMAT", DEFAULT_IMAGE_FORMAT).strip().lower()
    ext = "." + raw.lstrip(".")
    if ext == ".jpeg":
        ext = ".jpg"
    return ext if ext in IMAGE_EXTS else "." + DEFAULT_IMAGE_FORMAT


def jpeg_quality() -> int:
    """JPEG quality for new renders (``SA_JPEG_QUALITY``, default 95)."""
    try:
        q = int(os.environ.get("SA_JPEG_QUALITY", DEFAULT_JPEG_QUALITY))
    except ValueError:
        return DEFAULT_JPEG_QUALITY
    return max(1, min(100, q))


def with_image_ext(path: str | Path) -> Path:
    """``path`` with its suffix swapped for the configured image format."""
    return Path(path).with_suffix(image_ext())


def save_image(img, path: str | Path, quality: int | None = None) -> Path:
    """Write a PIL image, choosing encoder settings from ``path``'s suffix.

    JPEG is written at :func:`jpeg_quality` with ``subsampling=0`` -- chroma
    subsampling is what makes JPEG smear saturated edges, and diffusion output
    is full of them, so it is the one setting not worth the bytes it saves.
    """
    path = Path(path)
    if path.suffix.lower() in (".jpg", ".jpeg"):
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        kw = dict(quality=quality or jpeg_quality(), subsampling=0)
        try:
            img.save(path, "JPEG", optimize=True, **kw)
        except OSError:
            # Pillow's optimize pass buffers a whole scan and raises "broken
            # data stream" when a high-entropy image overruns MAXBLOCK. The
            # plain encoder always works and costs a few percent of size.
            img.save(path, "JPEG", **kw)
    elif path.suffix.lower() == ".webp":
        img.save(path, "WEBP", quality=quality or jpeg_quality(), method=4)
    else:
        img.save(path)
    return path


def find_image(path: str | Path) -> Path | None:
    """The existing image sharing ``path``'s stem, whatever its extension.

    Lets a ``.jpg``-named request resolve to a PNG rendered before the format
    switch (and vice versa). Returns ``None`` if nothing is there.
    """
    p = Path(path)
    if p.suffix.lower() in IMAGE_EXTS and p.is_file():
        return p
    for ext in (image_ext(),) + IMAGE_EXTS:
        cand = p.with_suffix(ext)
        if cand.is_file():
            return cand
    return None


def unique_path(path: str | Path) -> Path:
    """Return ``path``, or the first ``stem_N.suffix`` variant that doesn't exist.

    ``outputs/sweep.jpg`` -> ``outputs/sweep.jpg`` (free) or ``outputs/sweep_1.jpg``,
    ``outputs/sweep_2.jpg``, ... Parent directories are not created here.
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


def unique_image_path(path: str | Path) -> Path:
    """:func:`unique_path` for images, where the *stem* is the identity.

    A name is free only when no image of any extension and no ``.json``/``.npz``
    sidecar already claims that stem. Plain ``unique_path`` isn't enough once
    the format changed: ``anarchy_sd15_7_000.jpg`` looks free next to an older
    ``anarchy_sd15_7_000.png``, and writing it would overwrite that image's
    sidecars -- stripping a still-present render of its conditioning.
    """
    path = Path(path)
    parent, suffix = path.parent, path.suffix
    stem, i = path.stem, 0
    while True:
        cand = parent / f"{stem}{suffix}"
        taken = any((parent / f"{stem}{e}").exists()
                    for e in IMAGE_EXTS + SIDECAR_EXTS)
        if not taken:
            return cand
        i += 1
        stem = f"{path.stem}_{i}"
