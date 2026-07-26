"""Torch-free tests for the latent-travel trajectory math (films)."""

import numpy as np
import pytest

from semantic_anarchy.travel import (
    EASINGS,
    INTERPOLATIONS,
    ease,
    frame_plan,
    interpolate,
    interpolate_named,
    loop_keys,
    noise_t,
    total_frames,
    trajectory,
)


def _pair(shape=(77, 16), seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape), rng.standard_normal(shape)


def test_interpolate_hits_endpoints():
    """t=0 and t=1 ARE the keyframes -- frame 0 / frame N reproduce exactly."""
    a, b = _pair()
    for mode in INTERPOLATIONS:
        np.testing.assert_allclose(interpolate(a, b, 0.0, mode), a, atol=1e-5)
        np.testing.assert_allclose(interpolate(a, b, 1.0, mode), b, atol=1e-5)


def test_slerp_keeps_magnitude_where_lerp_dips():
    """The reason slerp is the default: lerp's midpoint is washed out."""
    a, b = _pair()
    norm = lambda x: float(np.linalg.norm(np.asarray(x).reshape(-1)))  # noqa: E731
    ends = (norm(a) + norm(b)) / 2
    assert abs(norm(interpolate(a, b, 0.5, "slerp")) - ends) < 1e-3
    assert norm(interpolate(a, b, 0.5, "lerp")) < ends * 0.95


def test_interpolate_named_moves_every_tensor_together():
    a = {"prompt_embeds": np.ones((77, 8)), "pooled": np.ones(4)}
    b = {"prompt_embeds": np.full((77, 8), 3.0), "pooled": np.full(4, 3.0)}
    mid = interpolate_named(a, b, 0.5, "lerp")
    assert set(mid) == {"prompt_embeds", "pooled"}
    assert mid["prompt_embeds"].shape == (77, 8) and mid["pooled"].shape == (4,)
    for v in mid.values():
        np.testing.assert_allclose(v, 2.0, atol=1e-6)


def test_unknown_modes_are_rejected():
    a, b = _pair()
    with pytest.raises(ValueError):
        interpolate(a, b, 0.5, "cubic")
    with pytest.raises(ValueError):
        ease(0.5, "bounce")


def test_easings_are_monotone_and_pinned():
    for mode in EASINGS:
        assert ease(0.0, mode) == 0.0 and ease(1.0, mode) == 1.0
        ts = np.linspace(0, 1, 21)
        vals = [ease(float(t), mode) for t in ts]
        assert all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
    # smoothstep rests at the keyframes: early t moves slower than linear.
    assert ease(0.1, "smooth") < 0.1
    assert ease(0.1, "smoother") < ease(0.1, "smooth")
    assert ease(0.5, "smooth") == pytest.approx(0.5)


def test_frame_plan_covers_every_transition_once():
    plan = frame_plan(3, frames_per=4)
    assert len(plan) == total_frames(3, 4) == 9
    assert plan[0] == (0, 0.0)          # starts ON keyframe 0
    assert plan[4] == (1, 0.0)          # ... and ON keyframe 1, rendered once
    assert plan[-1] == (1, 1.0)         # ... and ends ON the last keyframe
    # t rises within a transition and never repeats a keyframe mid-film
    seg0 = [t for s, t in plan if s == 0]
    assert all(seg0[i] < seg0[i + 1] for i in range(len(seg0) - 1))
    assert seg0[-1] < 1.0


def test_frame_plan_guards():
    with pytest.raises(ValueError):
        frame_plan(1, 10)
    with pytest.raises(ValueError):
        frame_plan(3, 0)


def test_loop_closes_the_walk():
    keys = ["a", "b", "c"]
    assert loop_keys(keys) == ["a", "b", "c", "a"]
    assert total_frames(3, 6, loop=True) == 19    # 3 transitions + the end frame
    assert total_frames(3, 6, loop=False) == 13   # 2 transitions + the end frame


def test_noise_window_locks_the_ends():
    """A narrow window pins composition at both ends of a transition."""
    assert noise_t(0.0, 0.4) == 0.0
    assert noise_t(0.25, 0.4) == 0.0        # still locked on the left keyframe
    assert noise_t(0.5, 0.4) == pytest.approx(0.5)
    assert noise_t(0.75, 0.4) == 1.0        # already arrived at the right one
    assert noise_t(1.0, 0.4) == 1.0
    # window=1.0 tracks the conditioning's own eased t
    assert noise_t(0.3, 1.0) == pytest.approx(ease(0.3, "smooth"))


def test_trajectory_starts_and_ends_on_the_keyframes():
    a, b = _pair()
    keys = [{"embeds": a}, {"embeds": b}]
    frames = trajectory(keys, frames_per=8)
    assert len(frames) == 9
    np.testing.assert_allclose(frames[0]["embeds"], a, atol=1e-5)
    np.testing.assert_allclose(frames[-1]["embeds"], b, atol=1e-5)
    # a looped 2-key film returns home
    looped = trajectory(keys, frames_per=8, loop=True)
    assert len(looped) == 17
    np.testing.assert_allclose(looped[-1]["embeds"], a, atol=1e-5)
