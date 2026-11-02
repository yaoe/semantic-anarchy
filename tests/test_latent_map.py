"""Latent-map projection math (torch-free)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from semantic_anarchy import EmbeddingDistribution
from semantic_anarchy.latent_map import (
    align, corpus_indices, explained, find_regions,
    neighbours_kept, off_manifold, project, region_faces, sketch, umap_3d,
    unproject, warm_start,
)

REPO = Path(__file__).resolve().parent.parent


def _dist(n=60, shape=(6, 10), seed=0):
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((4, int(np.prod(shape))))
    x = (rng.standard_normal((n, 4)) * [5, 3, 2, 1]) @ basis
    x = x.reshape(n, *shape) + rng.standard_normal((n, *shape)) * 0.05 + 2.0
    return EmbeddingDistribution.fit(x, max_corpus=16), x


def test_project_unproject_roundtrip_in_span():
    d, _ = _dist()
    z = np.array([1.5, -0.7, 0.3])
    x = unproject(d, z)
    assert x.shape == d.feature_shape
    np.testing.assert_allclose(project(d, x, 3)[0], z, atol=1e-4)


def test_mean_projects_to_origin_and_is_on_manifold():
    d, _ = _dist()
    np.testing.assert_allclose(project(d, d.mean, 4)[0], 0, atol=1e-4)
    assert off_manifold(d, unproject(d, [2.0, 1.0]))[0] < 1e-6


def test_corpus_points_have_unit_scale_on_leading_axes():
    d, x = _dist()
    z = project(d, x, 3)
    np.testing.assert_allclose(z.std(axis=0, ddof=1), 1.0, rtol=1e-3)


def test_diagonal_samples_are_off_manifold():
    d, _ = _dist(n=20, shape=(10, 20))   # 19 axes in a 200-dim space
    s = d.sample(n=8, sampler="diagonal", rng=np.random.default_rng(1))
    assert off_manifold(d, s).mean() > 0.8


def test_explained_share_bounded():
    d, _ = _dist()
    share = explained(d, 5)
    assert np.all(share >= 0) and share.sum() <= 1 + 1e-9
    assert share[0] > share[1] > share[2]


def test_sketch_ignores_nudges_on_constant_coordinates():
    """A coordinate the corpus never varies sits at the fit's sigma floor; a tiny
    nudge there (what a hybrid SLERP does to CLIP's start token) must not dwarf
    a real move, or one image owns the whole layout."""
    d, x = _dist()
    x = x.copy()
    x[:, 0, :] = 1.0                                # one constant token row
    d = EmbeddingDistribution.fit(x, max_corpus=16)
    nudged = d.mean.copy()
    nudged[0, :] += 1e-3                            # off the floor, imperceptibly
    real = unproject(d, [2.0])                      # a 2-sigma move along PC1
    assert np.linalg.norm(sketch(d, nudged, k=64)) < 0.1 * np.linalg.norm(sketch(d, real, k=64))


def test_sketch_is_deterministic_and_keeps_distances_across_slabs(monkeypatch):
    """Same input -> same sketch every call, and generating the matrix in many
    small slabs still preserves distances (Johnson-Lindenstrauss)."""
    import semantic_anarchy.latent_map as lm
    d, x = _dist()
    monkeypatch.setattr(lm, "SKETCH_BLOCK", 7)          # 60 features -> 9 slabs
    a = sketch(d, x[:6], k=512)
    np.testing.assert_array_equal(a, sketch(d, x[:6], k=512))
    std = np.maximum(d.std, 0.1 * float(np.mean(d.std))).reshape(-1)
    z = (x[:6].reshape(6, -1) - d.mean.reshape(-1)) / std
    ratio = np.linalg.norm(a[0] - a[1]) / np.linalg.norm(z[0] - z[1])
    assert 0.8 < ratio < 1.2


def test_corpus_indices_replays_fit_subset():
    d, x = _dist(n=60)
    idx = corpus_indices(d.n_samples, len(d.corpus_embeddings))
    np.testing.assert_allclose(d.corpus_embeddings, x[idx].astype(np.float32))


def test_script_end_to_end(tmp_path):
    d, x = _dist(n=30, shape=(4, 8))
    outputs = tmp_path / "outputs"
    (outputs / "generated").mkdir(parents=True)
    d.save(outputs / "dist")
    prompts = tmp_path / "prompts.txt"
    prompts.write_text("\n".join(f"prompt {i}" for i in range(30)))
    rng = np.random.default_rng(0)
    for i, kind in enumerate(["generate", "generate", "explore"]):
        stem = outputs / "generated" / f"anarchy_sd15_1_{i:03d}"
        np.savez(stem.with_suffix(".npz"), embeds=d.sample(1, rng=rng)[0])
        # the stem is the identity: renders are JPEG now, the old gallery is PNG
        stem.with_suffix(".jpg" if i else ".png").write_bytes(b"")
        meta = {"kind": kind, "backend": "sd15"}
        if kind == "explore":
            meta["parent"] = "anarchy_sd15_1_000.png"
        stem.with_suffix(".json").write_text(json.dumps(meta))
    out = tmp_path / "map.json"
    subprocess.run([sys.executable, str(REPO / "scripts" / "latent_map.py"),
                    "--outputs", str(outputs), "--prompts", str(prompts),
                    "--out", str(out)], check=True)
    doc = json.loads(out.read_text())
    assert len(doc["nodes"]) == 3 and len(doc["nodes"][0]["c"]) == 3
    # zero-byte fixtures: no CLIP cache, unreadable thumbnails, and neither stops the map
    assert doc["nodes"][0]["l"] is None and doc["frames"]["look"]["alike"] is None
    assert doc["unembedded"] == 0              # unreadable: no embedding is ever coming
    assert doc["edges"] == [{"a": 0, "b": 2, "kind": "parent"}]
    assert doc["corpus"][0]["label"].startswith("prompt ")
    assert len(doc["axes"]["corpus"]["extremes"]) == 3


def test_align_undoes_a_rotation_without_reflecting():
    rng = np.random.default_rng(0)
    target = rng.standard_normal((40, 3))
    th = 0.7
    rot = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    y = target @ rot * 2.5 + 4.0
    np.testing.assert_allclose(align(y, target, np.ones(40, dtype=bool)), target, atol=1e-8)
    # a mirror image can't be undone by a rotation, so it must not be "fixed"
    mirrored = target * [1, 1, -1]
    assert np.abs(align(mirrored, target, np.ones(40, dtype=bool)) - target).max() > 0.1


def test_warm_start_places_new_images_among_their_neighbours():
    x = np.array([[0.0, 0], [0.1, 0], [10, 10], [10.1, 10], [0.05, 0]])
    prev = np.array([[1.0, 1, 1], [1.2, 1, 1], [-5, -5, -5], [-5.2, -5, -5], [0, 0, 0]])
    known = np.array([True, True, True, True, False])
    init = warm_start(x, prev, known, "euclidean", k=2)
    np.testing.assert_allclose(init[4], [1.1, 1, 1])
    np.testing.assert_array_equal(init[:4], prev[:4])


def test_regions_find_clusters_and_keep_their_ids():
    pytest.importorskip("sklearn")
    rng = np.random.default_rng(0)
    y = np.concatenate([rng.normal(c, 0.1, (60, 3)) for c in ([0, 0, 0], [5, 0, 0], [0, 5, 0])])
    first = find_regions(y)
    assert len(set(first) - {-1}) == 3
    # labels from an earlier build survive a rebuild, whatever order clusters come in
    prior = np.where(first >= 0, first + 7, -1)
    again = find_regions(y, prior)
    assert set(again) - {-1} == set(prior) - {-1}
    assert (again[first >= 0] == prior[first >= 0]).mean() > 0.95
    faces = region_faces(y, first, k=2)
    assert all(len(v) == 2 and all(first[i] == c for i in v) for c, v in faces.items())


def test_regions_cap_a_cluster_that_swallowed_the_gallery(monkeypatch):
    sklearn_cluster = pytest.importorskip("sklearn.cluster")
    calls = []

    class Fake:
        def __init__(self, **kw):
            calls.append(kw)

        def fit_predict(self, y):
            if "max_cluster_size" in calls[-1]:
                return np.arange(len(y)) % 5
            return np.zeros(len(y), dtype=int)

    monkeypatch.setattr(sklearn_cluster, "HDBSCAN", Fake)
    labels = find_regions(np.zeros((120, 3)))
    assert calls[-1]["max_cluster_size"] == 20 and len(set(labels)) == 5


def test_umap_continues_from_the_previous_layout():
    pytest.importorskip("umap")
    rng = np.random.default_rng(0)
    x = np.concatenate([rng.normal(c, 0.2, (40, 8)) for c in (0.0, 4.0, -4.0)])
    first = umap_3d(x, "euclidean", 10, 0.1)
    assert first.shape == (120, 3)
    assert neighbours_kept(x, first, "euclidean") > 0.8
    known = np.ones(120, dtype=bool)
    known[-5:] = False
    again = umap_3d(x, "euclidean", 10, 0.1, prev=first, known=known)
    moved = np.linalg.norm(again[:-5] - first[:-5], axis=1)
    assert np.median(moved) < 0.5 * np.median(np.linalg.norm(first, axis=1))


def test_umap_handles_a_gallery_too_small_to_lay_out():
    assert umap_3d(np.zeros((3, 8)), "euclidean", 10, 0.1).shape == (3, 3)
