"""Weirdness: one reach scale across distributions, so temperature can't compound."""
import numpy as np
import pytest

from semantic_anarchy.distribution import EmbeddingDistribution


def _corpus(n=400, shape=(6, 8), seed=0):
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((5, int(np.prod(shape))))
    x = rng.standard_normal((n, 5)) @ basis + 0.3 * rng.standard_normal((n, int(np.prod(shape))))
    return x.reshape(n, *shape).astype(np.float32)


@pytest.fixture(scope="module")
def root():
    return EmbeddingDistribution.fit(_corpus())


@pytest.mark.parametrize("sampler", ["diagonal", "pca", "blend", "split"])
def test_solved_temperature_lands_on_the_asked_weirdness(root, sampler):
    t, info = root.temperature_for(1.7, sampler=sampler, n_probe=256)
    x = root.sample(2000, temperature=t, sampler=sampler, rng=np.random.default_rng(1))
    w = np.sqrt(np.mean([root.weirdness(s) ** 2 for s in x]))
    assert w == pytest.approx(1.7, rel=0.03)
    assert not info["clamped"]


def test_a_child_fit_does_not_compound(root):
    """The bug: images drawn at T=1.7, refitted, then sampled at T=1.7 again."""
    kids = root.sample(60, temperature=1.7, rng=np.random.default_rng(2))
    child = EmbeddingDistribution.fit(kids)
    plain = child.sample(500, temperature=1.7, rng=np.random.default_rng(3))
    assert np.mean([root.weirdness(s) for s in plain]) > 2.2       # the overshoot

    child.set_reference(root)
    t, _ = child.temperature_for(1.7, sampler="diagonal", n_probe=256)
    fixed = child.sample(500, temperature=t, rng=np.random.default_rng(3))
    assert np.sqrt(np.mean([child.weirdness(s) ** 2 for s in fixed])) == pytest.approx(1.7, rel=0.05)
    assert t < 1.2                                                  # ~1, not 1.7


def test_reference_survives_save_and_load(root, tmp_path):
    child = EmbeddingDistribution.fit(root.sample(20, temperature=1.5, rng=np.random.default_rng(4)))
    child.set_reference(root)
    child.save(tmp_path / "c")
    back = EmbeddingDistribution.load(tmp_path / "c")
    np.testing.assert_allclose(back.ref_mean, root.mean)
    x = root.sample(1, rng=np.random.default_rng(5))[0]
    assert back.weirdness(x) == pytest.approx(root.weirdness(x), rel=1e-5)


def test_inheriting_from_a_child_keeps_the_root(root):
    a = EmbeddingDistribution.fit(root.sample(30, temperature=1.5, rng=np.random.default_rng(6)))
    a.set_reference(root)
    b = EmbeddingDistribution.fit(a.sample(30, rng=np.random.default_rng(7)))
    b.set_reference(a)
    np.testing.assert_allclose(b.ref_mean, root.mean)


def test_below_the_centre_clamps_to_zero(root):
    far = EmbeddingDistribution.fit(root.sample(50, temperature=0.2, rng=np.random.default_rng(8)) + 3.0)
    far.set_reference(root)
    t, info = far.temperature_for(0.5)
    assert t == 0.0 and info["clamped"]


def test_the_probe_does_not_touch_the_callers_rng(root):
    rng = np.random.default_rng(9)
    state = rng.bit_generator.state
    root.temperature_for(1.5, sampler="pca", rng=rng)
    assert rng.bit_generator.state == state


def test_hybrid_is_refused(root):
    with pytest.raises(ValueError):
        root.temperature_for(1.5, sampler="hybrid")


def test_refit_from_elites_inherits_the_frame(root):
    elites = root.sample(10, temperature=1.7, rng=np.random.default_rng(10))
    child = root.refit_from_elites(elites)
    np.testing.assert_allclose(child.ref_mean, root.mean)
