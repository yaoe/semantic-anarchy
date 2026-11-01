"""Latent-map projection math (torch-free)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from semantic_anarchy import EmbeddingDistribution
from semantic_anarchy.latent_map import (
    corpus_indices, explained, explored_frame, off_manifold, project, sketch,
    sketch_matrix, unproject,
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


def test_explored_frame_unit_std_and_stable_sign():
    rng = np.random.default_rng(3)
    pts = rng.standard_normal((40, 12)) * np.linspace(4, 0.5, 12)
    pts[-4:, 0] += 10                         # newest images drift one way
    c, share = explored_frame(pts, 3)
    np.testing.assert_allclose(c.std(axis=0, ddof=1), 1.0, rtol=1e-6)
    assert c[-4:, 0].mean() > 0
    assert share[0] >= share[1] >= share[2]


def test_sketch_ignores_nudges_on_constant_coordinates():
    """A coordinate the corpus never varies sits at the fit's sigma floor; a tiny
    nudge there (what a hybrid SLERP does to CLIP's start token) must not dwarf
    a real move, or one image owns the whole explored frame."""
    d, x = _dist()
    x = x.copy()
    x[:, 0, :] = 1.0                                # one constant token row
    d = EmbeddingDistribution.fit(x, max_corpus=16)
    r = sketch_matrix(d, k=64)
    nudged = d.mean.copy()
    nudged[0, :] += 1e-3                            # off the floor, imperceptibly
    real = unproject(d, [2.0])                      # a 2-sigma move along PC1
    assert np.linalg.norm(sketch(d, nudged, r)) < 0.1 * np.linalg.norm(sketch(d, real, r))


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
        stem.with_suffix(".png").write_bytes(b"")
        meta = {"kind": kind, "backend": "sd15"}
        if kind == "explore":
            meta["parent"] = "anarchy_sd15_1_000.png"
        stem.with_suffix(".json").write_text(json.dumps(meta))
    out = tmp_path / "map.json"
    subprocess.run([sys.executable, str(REPO / "scripts" / "latent_map.py"),
                    "--outputs", str(outputs), "--prompts", str(prompts),
                    "--dims", "3", "--out", str(out)], check=True)
    doc = json.loads(out.read_text())
    assert len(doc["nodes"]) == 3 and len(doc["nodes"][0]["c"]) == 3
    assert doc["edges"] == [{"a": 0, "b": 2, "kind": "parent"}]
    assert doc["corpus"][0]["label"].startswith("prompt ")
    assert len(doc["axes"]["corpus"]["extremes"]) == 3
