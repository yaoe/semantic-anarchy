"""Torch-free tests for the navigation primitives (distance/neighborhood/breed)."""

import numpy as np

from semantic_anarchy import dist_backend
from semantic_anarchy.backend import _slerp


def _fit(b, n=40, shape=(77, 32)):
    named = {"embeds": np.random.default_rng(0).normal(
        1.0, 2.0, size=(n, *shape)).astype("float32")}
    return b.fit(named, per_token=True, n_components=10), named


def test_distance_scale():
    """Corpus samples sit near 1.0; scaled deviations scale the distance."""
    b = dist_backend("sd15")
    dists, named = _fit(b)
    d = dists["embeds"]
    corpus_d = np.mean([d.distance(e) for e in named["embeds"][:10]])
    assert 0.7 < corpus_d < 1.3
    assert d.distance(d.mean) < 0.05                      # the center is ~0
    far = d.mean + 3.0 * d.std * np.random.default_rng(1).standard_normal(d.feature_shape)
    assert d.distance(far) > 2.0


def test_neighborhood_stays_local():
    """Children orbit the anchor; smaller radius -> tighter orbit."""
    b = dist_backend("sd15")
    dists, named = _fit(b)
    d = dists["embeds"]
    anchor = named["embeds"][0]
    rng = np.random.default_rng(0)
    near = d.neighborhood(anchor, n=8, radius=0.1, rng=np.random.default_rng(1))
    wide = d.neighborhood(anchor, n=8, radius=0.6, rng=np.random.default_rng(1))
    assert near.shape == (8, *d.feature_shape)
    dn = np.linalg.norm((near - anchor).reshape(8, -1), axis=1).mean()
    dw = np.linalg.norm((wide - anchor).reshape(8, -1), axis=1).mean()
    assert dn < dw
    # radius scales the deviation linearly (same rng draws)
    assert abs(dw / dn - 6.0) < 0.01


def test_breed_children_between_parents():
    """SLERP children land between the parents and mutate=0 is deterministic."""
    b = dist_backend("sd15")
    dists, named = _fit(b)
    a, c = named["embeds"][0], named["embeds"][1]
    kids = b.breed(dists, {"embeds": a}, {"embeds": c}, n=5, mutate=0.0)
    k = kids["embeds"]
    assert k.shape == (5, *dists["embeds"].feature_shape)
    # each child is closer to BOTH parents than the parents are to each other
    pd = np.linalg.norm((a - c).reshape(-1))
    for child in k:
        assert np.linalg.norm((child - a).reshape(-1)) < pd
        assert np.linalg.norm((child - c).reshape(-1)) < pd


def test_walk_outward_grows_distance():
    """Outward walk raises the distance gauge ~step per frame, monotonically."""
    b = dist_backend("sd15")
    dists, named = _fit(b)
    d = dists["embeds"]
    anchor = named["embeds"][0]
    strip = d.walk(anchor, steps=5, step=0.2, mode="outward")
    ds = [d.distance(x) for x in strip]
    assert all(ds[i] < ds[i + 1] for i in range(len(ds) - 1))
    a0 = d.distance(anchor)
    assert abs(ds[0] / a0 - 1.2) < 0.05                    # first frame ≈ +20%


def test_retarget_pins_distance():
    b = dist_backend("sd15")
    dists, _ = _fit(b)
    d = dists["embeds"]
    s = d.sample(n=6, temperature=2.5, rng=np.random.default_rng(0))
    pinned = d.retarget(s, target=1.4)
    for x in pinned:
        assert abs(d.distance(x) - 1.4) < 0.02


def test_slerp_endpoints_and_magnitude():
    rng = np.random.default_rng(0)
    a, c = rng.standard_normal((77, 16)), rng.standard_normal((77, 16))
    np.testing.assert_allclose(_slerp(a, c, 0.0), a, atol=1e-6)
    np.testing.assert_allclose(_slerp(a, c, 1.0), c, atol=1e-6)
    mid = _slerp(a, c, 0.5)
    na, nc, nm = (np.linalg.norm(x.reshape(-1)) for x in (a, c, mid))
    assert abs(nm - (na + nc) / 2) < 1e-6                 # magnitude lerps
