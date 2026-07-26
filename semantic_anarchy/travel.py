"""Latent travel -- the trajectory math behind keyframe interpolation films.

A "film" is a walk through a list of keyframes (images you already made, each
carrying its exact conditioning in a ``.npz`` sidecar). This module answers the
only two questions that walk raises, and answers them in pure NumPy so the whole
thing is testable without torch:

1. **Where is frame i?** :func:`frame_plan` turns ``(n_keys, frames_per)`` into
   an ordered list of ``(segment, t)`` pairs -- one per frame, with the eased
   ``t`` already applied, and the final frame pinned exactly on the last
   keyframe.
2. **What lives at t?** :func:`interpolate` blends two conditioning tensors,
   either spherically (``slerp`` -- stays on the shell the endpoints live on) or
   straight (``lerp`` -- cuts through the low-magnitude interior).

Easing is deliberately separate from the interpolation mode: ``smooth``
(smoothstep) makes the film *rest* on each keyframe and accelerate through the
middle, which is what makes a morph read as intentional rather than as a
constant-speed slide. ``linear`` keeps constant speed for loops meant to be
seamless in motion.

:func:`noise_t` handles the one asymmetry: the initial *noise* latent decides
composition, so it usually wants to travel over a NARROWER window than the
conditioning does (composition locked at both ends, drifting only through the
middle of the transition).
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

#: Interpolation modes accepted by :func:`interpolate` (UI + CLI allow-list).
INTERPOLATIONS = ("slerp", "lerp")

#: Easing curves accepted by :func:`ease` (UI + CLI allow-list).
EASINGS = ("smooth", "smoother", "linear")


# --------------------------------------------------------------------------- #
# blending two points
# --------------------------------------------------------------------------- #
def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Straight linear blend. Cheap, and it dips through the interior: the
    midpoint of two high-magnitude embeddings has a *lower* norm than either
    end, which for conditioning tensors reads as a washed-out middle."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    return ((1.0 - t) * a64 + t * b64).astype(np.float32)


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical interp: direction along the great circle, magnitude linear.

    Keeps the blend on the shell the endpoints live on, so the middle of a
    transition is as "loud" as its ends. Falls back to lerp for (near-)parallel
    or degenerate inputs.
    """
    fa = np.asarray(a, dtype=np.float64).reshape(-1)
    fb = np.asarray(b, dtype=np.float64).reshape(-1)
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    if na < 1e-8 or nb < 1e-8:
        return lerp(a, b, t)
    ua, ub = fa / na, fb / nb
    dot = float(np.clip(ua @ ub, -1.0, 1.0))
    omega = np.arccos(dot)
    if omega < 1e-4:
        direction = (1.0 - t) * ua + t * ub
    else:
        so = np.sin(omega)
        direction = (np.sin((1.0 - t) * omega) / so) * ua \
            + (np.sin(t * omega) / so) * ub
    mag = (1.0 - t) * na + t * nb
    return (mag * direction).reshape(np.asarray(a).shape).astype(np.float32)


def interpolate(a: np.ndarray, b: np.ndarray, t: float,
                mode: str = "slerp") -> np.ndarray:
    """Blend ``a`` -> ``b`` at ``t`` using the named mode (see :data:`INTERPOLATIONS`)."""
    if mode == "lerp":
        return lerp(a, b, t)
    if mode == "slerp":
        return slerp(a, b, t)
    raise ValueError(f"unknown interpolation {mode!r}; choose {'|'.join(INTERPOLATIONS)}")


def interpolate_named(a: dict, b: dict, t: float, mode: str = "slerp") -> dict:
    """Interpolate every named conditioning tensor with the SAME t and mode.

    Multi-tensor backends (sdxl's ``prompt_embeds`` + ``pooled``) must travel
    together or the conditioning set stops being coherent mid-transition.
    """
    return {k: interpolate(a[k], b[k], t, mode) for k in a}


# --------------------------------------------------------------------------- #
# pacing
# --------------------------------------------------------------------------- #
def ease(t: float, mode: str = "smooth") -> float:
    """Reshape a linear ``t`` in [0,1] into the film's pacing curve."""
    t = float(min(1.0, max(0.0, t)))
    if mode == "linear":
        return t
    if mode == "smooth":                       # smoothstep: zero 1st derivative
        return t * t * (3.0 - 2.0 * t)
    if mode == "smoother":                     # smootherstep: zero 1st AND 2nd
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    raise ValueError(f"unknown easing {mode!r}; choose {'|'.join(EASINGS)}")


def frame_plan(n_keys: int, frames_per: int,
               easing: str = "smooth") -> list[tuple[int, float]]:
    """Schedule every frame of a film through ``n_keys`` keyframes.

    Returns ``[(segment, t), ...]`` where ``segment`` indexes the keyframe pair
    ``(keys[segment], keys[segment + 1])`` and ``t`` is already eased. Each
    transition contributes ``frames_per`` frames starting AT its left keyframe
    (t=0) and stopping just short of its right one, so no keyframe is rendered
    twice; one final frame pins the very end at t=1.

    Total frames = ``(n_keys - 1) * frames_per + 1``.

    A looping film is expressed by passing a key list whose last entry repeats
    the first (see :func:`loop_keys`) -- nothing here needs to know.
    """
    if n_keys < 2:
        raise ValueError("a film needs at least 2 keyframes")
    if frames_per < 1:
        raise ValueError("frames_per must be >= 1")
    plan = [(s, ease(i / frames_per, easing))
            for s in range(n_keys - 1)
            for i in range(frames_per)]
    plan.append((n_keys - 2, 1.0))
    return plan


def loop_keys(keys: list) -> list:
    """Append the first keyframe so the walk closes back on itself."""
    return list(keys) + [keys[0]] if keys else []


def total_frames(n_keys: int, frames_per: int, loop: bool = False) -> int:
    """Frame count of the film ``frame_plan`` would schedule (UI estimate)."""
    segs = (n_keys if loop else n_keys - 1)
    return segs * frames_per + 1


def noise_t(t: float, window: float = 1.0, easing: str = "smooth") -> float:
    """Remap a transition's ``t`` onto a narrower, centered window for the NOISE.

    The conditioning says *what* the frame is; the init latent says *where
    everything sits*. Travelling both at once means the composition slides for
    the whole transition. ``window=0.4`` instead holds the composition still at
    both ends and drifts only through the middle 40% -- content morphs, camera
    doesn't. ``window=1.0`` travels alongside the conditioning.
    """
    w = max(1e-6, min(1.0, window))
    return ease((t - (0.5 - w / 2.0)) / w, easing)


def trajectory(keys: Iterable[dict], frames_per: int, interp: str = "slerp",
               easing: str = "smooth", loop: bool = False) -> list[dict]:
    """The whole film as conditioning dicts -- one per frame.

    ``keys`` are per-tensor conditioning dicts (``{"embeds": (77,768)}`` for
    sd15, ``{"prompt_embeds": ..., "pooled": ...}`` for sdxl). Convenience for
    tests and for anything that wants the trajectory without a GPU; the renderer
    itself walks :func:`frame_plan` so it can interleave the noise latents.
    """
    ks = loop_keys(list(keys)) if loop else list(keys)
    return [interpolate_named(ks[s], ks[s + 1], t, interp)
            for s, t in frame_plan(len(ks), frames_per, easing)]
