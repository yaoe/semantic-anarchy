"""Torch-free tests for member-anchored collection sampling.

What matters: ``spread=0`` returns the anchor's exact conditioning, spread
moves a child away from its anchor monotonically, every child names an anchor
whose render settings it inherits, and the member loader resolves the same
sidecars ``selection_fit`` does.
"""

import itertools
import json

import numpy as np
import pytest

from semantic_anarchy import collection_sampler as cs
from semantic_anarchy.collection_sampler import (
    Collection,
    group_by_checkpoint,
    load_members,
    read_image_list,
    sample,
)

SHAPE = (77, 16)


def _write_image(dirpath, stem, embeds, *, meta=None, ext=".jpg"):
    img = dirpath / f"{stem}{ext}"
    img.write_bytes(b"not really a jpeg")
    if embeds is not None:
        np.savez(dirpath / f"{stem}.npz", embeds=np.asarray(embeds, dtype=np.float32))
    (dirpath / f"{stem}.json").write_text(json.dumps(meta or {"backend": "sd15"}))
    return img


def _members(n=6, seed=0, shape=SHAPE):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(shape).astype(np.float32)
    base[0] = 3.0                     # a constant BOS row, like CLIP's
    out = []
    for i in range(n):
        x = base + rng.standard_normal(shape).astype(np.float32)
        x[0] = base[0]
        out.append(cs.Member(image=cs.Path(f"m{i}.jpg"), source=cs.Path(f"m{i}.jpg"),
                             latents={"embeds": x},
                             meta={"model": f"/ck/{'a' if i % 2 else 'b'}.safetensors",
                                   "steps": 30, "guidance": 7.5, "scheduler": "euler",
                                   "neg_mode": "empty"},
                             size=(512, 768)))
    return out


# ------------------------------------------------------------------ loading --
def test_load_members_follows_refine_chain_and_dedupes(tmp_path):
    e = np.ones(SHAPE)
    a = _write_image(tmp_path, "a", e, meta={"model": "/ck/x.safetensors", "width": 640,
                                             "height": 512})
    up = _write_image(tmp_path, "a_up", None, meta={"kind": "refine", "refined_from": "a.jpg"})
    _write_image(tmp_path, "b", 2 * e)
    members, skipped = load_members([a, up, tmp_path / "b.jpg", tmp_path / "nope.jpg"],
                                    read_size=False)
    assert [m.source.name for m in members] == ["a.jpg", "b.jpg"]
    assert members[0].render_settings()["width"] == 640
    reasons = dict(skipped)
    assert "duplicate" in reasons[str(up)]
    assert "no conditioning" in reasons[str(tmp_path / "nope.jpg")]


def test_load_members_skips_mismatched_shapes(tmp_path):
    a = _write_image(tmp_path, "a", np.zeros(SHAPE))
    b = _write_image(tmp_path, "b", np.zeros((77, 32)))
    members, skipped = load_members([a, b], read_size=False)
    assert len(members) == 1 and "expected" in skipped[0][1]


def test_read_image_list_accepts_json_and_text(tmp_path):
    j = tmp_path / "sel.json"
    j.write_text(json.dumps(["x.jpg", "y.jpg"]))
    t = tmp_path / "sel.txt"
    t.write_text("x.jpg\n\ny.jpg\n")
    d = tmp_path / "sel2.json"
    d.write_text(json.dumps({"images": ["x.jpg"]}))
    assert read_image_list(j) == read_image_list(t) == ["x.jpg", "y.jpg"]
    assert read_image_list(d) == ["x.jpg"]


def test_group_by_checkpoint_keeps_first_seen_order():
    members = _members(4)
    coll = Collection(members)
    kids = sample(coll, "member", 0.0, 6, rng=np.random.default_rng(0))
    settings = [members[k.anchor].render_settings() for k in kids]
    groups = group_by_checkpoint(settings)
    assert sorted(i for v in groups.values() for i in v) == list(range(6))
    assert list(groups) == list(dict.fromkeys(s["model"] for s in settings))
    for ck, idx in groups.items():
        assert all(members[kids[i].anchor].meta["model"] == ck for i in idx)


# ----------------------------------------------------------------- sampling --
@pytest.mark.parametrize("method", ["member", "slerp", "jitter", "crossover"])
def test_spread_zero_is_the_anchor_itself(method):
    members = _members(5)
    coll = Collection(members)
    kids = sample(coll, method, 0.0, 5, rng=np.random.default_rng(1))
    for k in kids:
        np.testing.assert_allclose(k.latents["embeds"],
                                   members[k.anchor].latents["embeds"], atol=1e-5)


def test_anchors_are_balanced_before_repeating():
    coll = Collection(_members(5))
    kids = sample(coll, "slerp", 1.0, 10, rng=np.random.default_rng(2))
    a = [k.anchor for k in kids]
    assert sorted(a[:5]) == list(range(5)) and sorted(a[5:]) == list(range(5))
    assert all(k.partner is not None and k.partner != k.anchor for k in kids)


@pytest.mark.parametrize("method", ["slerp", "jitter", "crossover"])
def test_more_spread_moves_further_from_the_anchor(method):
    members = _members(8, seed=3)
    coll = Collection(members)

    def mean_dist(s):
        kids = sample(coll, method, s, 64, rng=np.random.default_rng(4))
        return np.mean([np.linalg.norm(k.latents["embeds"] - members[k.anchor].latents["embeds"])
                        for k in kids])

    d = [mean_dist(s) for s in (0.25, 0.5, 1.0, 2.0)]
    assert all(x < y for x, y in itertools.pairwise(d)), d


@pytest.mark.parametrize("method", ["slerp", "jitter", "crossover"])
def test_bos_row_is_never_touched(method):
    members = _members(6)
    coll = Collection(members)
    for k in sample(coll, method, 3.0, 12, rng=np.random.default_rng(5)):
        np.testing.assert_allclose(k.latents["embeds"][0], members[0].latents["embeds"][0],
                                   atol=1e-5)


def test_slerp_and_jitter_keep_a_members_distance_from_the_centre():
    """The whole point of slerp over lerp: a child stays as far from the
    collection centre as its members (a lerp midpoint of near-orthogonal
    members shrinks by ~1/sqrt(2) toward the bland mean)."""
    members = _members(8, seed=6)
    coll = Collection(members)
    r = lambda x: np.linalg.norm(x - coll.mean["embeds"])
    member_r = np.mean([r(m.latents["embeds"]) for m in members])
    for method in ("slerp", "jitter"):
        kids = sample(coll, method, 1.0, 32, rng=np.random.default_rng(7))
        kid_r = np.mean([r(k.latents["embeds"]) for k in kids])
        assert abs(kid_r / member_r - 1) < 0.1, (method, kid_r, member_r)


def test_crossover_rows_come_from_members():
    members = _members(4, seed=8)
    coll = Collection(members)
    rows = {tuple(np.round(m.latents["embeds"][r], 4)) for m in members for r in range(77)}
    for k in sample(coll, "crossover", 1.0, 8, rng=np.random.default_rng(9)):
        for r in range(77):
            assert tuple(np.round(k.latents["embeds"][r], 4)) in rows


def test_single_member_collection_still_samples():
    coll = Collection(_members(1))
    kids = sample(coll, "jitter", 1.0, 3, rng=np.random.default_rng(10))
    assert len(kids) == 3 and all(k.anchor == 0 for k in kids)
    assert all(np.isfinite(k.latents["embeds"]).all() for k in kids)


def test_auto_and_unknown_methods():
    coll = Collection(_members(3))
    assert sample(coll, "auto", 1.0, 2, rng=np.random.default_rng(0))[0].method == cs.AUTO_METHOD
    with pytest.raises(ValueError):
        sample(coll, "gaussian", 1.0, 2)


def test_seed_mode_resolution():
    assert cs.resolve_seed_mode("anchor", "member", 0.0) == "fresh"
    assert cs.resolve_seed_mode("anchor", "slerp", 1.0) == "anchor"
    assert cs.resolve_seed_mode("auto", "slerp", 0.5) == "anchor"
    assert cs.resolve_seed_mode("auto", "slerp", 1.5) == "fresh"
    assert cs.resolve_seed_mode("auto", "member", 0.0) == "fresh"


def test_slerp_never_lands_on_the_anchor():
    """Anchor-seeded children must be variations, not re-renders."""
    coll = Collection(_members(6))
    kids = sample(coll, "slerp", 0.5, 40, rng=np.random.default_rng(11))
    ts = [k.t for k in kids]
    assert min(ts) >= cs.SLERP_T_FLOOR * 0.25 - 1e-9 and max(ts) <= 0.25 + 1e-9


def test_module_is_torch_free():
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(cs))
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name.split(".")[0] for n in top for a in n.names} | {
        (n.module or "").split(".")[0] for n in top if isinstance(n, ast.ImportFrom)}
    assert not names & {"torch", "diffusers", "transformers", "PIL"}
