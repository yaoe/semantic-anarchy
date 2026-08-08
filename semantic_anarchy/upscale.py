"""Hires-fix arithmetic: target resolution, denoise budget, conditioning lineage.

The numbers that decide the *shape* of an upscale job, kept torch-free (see the
two-tier rule in CLAUDE.md) so they are unit-testable without a GPU. The GPU half
lives in ``SDModel.refine_image`` / ``SDXLModel.refine_image``; the driver is
``scripts/upscale.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

#: SD VAEs downsample by 8, and the UNet halves the latent three more times, so
#: off-grid sizes get silently padded. 16 keeps both halves aligned and matches
#: the rounding refine_flux.py already uses.
LATENT_MULTIPLE = 16


def round_to(value: float, multiple: int = LATENT_MULTIPLE) -> int:
    """Nearest ``multiple``, never below one ``multiple``."""
    if multiple < 1:
        raise ValueError("multiple must be >= 1")
    return max(multiple, int(round(value / multiple)) * multiple)


def target_size(width: int, height: int, factor: float,
                multiple: int = LATENT_MULTIPLE,
                max_side: Optional[int] = None) -> tuple[int, int]:
    """``(w, h) * factor``, snapped to ``multiple`` and capped at ``max_side``.

    The cap is applied to the *unrounded* size and scales both axes together, so
    a clamped result keeps the source aspect ratio (up to the final snap).
    """
    if factor <= 0:
        raise ValueError("factor must be > 0")
    w, h = width * factor, height * factor
    if max_side and max(w, h) > max_side:
        shrink = max_side / max(w, h)
        w, h = w * shrink, h * shrink
    return round_to(w, multiple), round_to(h, multiple)


def denoise_steps(steps: int, denoise: float) -> int:
    """How many of ``steps`` an img2img pass at ``strength=denoise`` actually runs.

    Mirrors diffusers' ``get_timesteps``: it starts at index
    ``steps - int(steps * strength)`` and runs the rest, i.e. exactly the LAST
    ``denoise`` fraction of the original schedule. Same ``int()`` truncation here
    so the log and the sidecar report the true count, not a rounded guess.
    """
    return min(int(steps * denoise), steps)


def clamp_denoise(steps: int, denoise: float) -> float:
    """``denoise``, nudged up if it would buy zero steps (and never above 1.0).

    ``int(steps * strength) == 0`` leaves diffusers with an empty timestep list,
    so a 0.02 denoise on a 20-step original has to become "one step" rather than
    "no pass at all".
    """
    denoise = min(float(denoise), 1.0)
    if steps >= 1 and denoise_steps(steps, denoise) < 1:
        return min(1.0, 1.0 / steps + 1e-6)
    return denoise


def conditioning_source(png: Path, max_hops: int = 5) -> tuple[Path, dict]:
    """Walk ``refined_from`` back to the ancestor that still owns a ``.npz``.

    Upscaled outputs deliberately carry no conditioning sidecar of their own, so
    upscaling an upscale (2x then 2x again) has to trace the chain to find the
    conditioning that produced the original. Returns ``(png, meta)`` for that
    ancestor — its ``.json`` is also where the original steps/guidance/scheduler
    live. Raises ``FileNotFoundError`` when the chain runs dry.
    """
    p = Path(png)
    for _ in range(max_hops):
        if p.with_suffix(".npz").is_file():
            meta = {}
            j = p.with_suffix(".json")
            if j.is_file():
                try:
                    meta = json.loads(j.read_text())
                except ValueError:
                    meta = {}
            return p, meta
        parent = None
        j = p.with_suffix(".json")
        if j.is_file():
            try:
                parent = json.loads(j.read_text()).get("refined_from")
            except ValueError:
                parent = None
        if not parent:
            break
        cand = p.parent / parent
        if not cand.is_file():
            break
        p = cand
    raise FileNotFoundError(
        f"{Path(png).name} has no conditioning sidecar (.npz) and no traceable original")
