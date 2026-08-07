"""Torch-free tests for the labels dataset + experiment identity (labels.py).

The dataset is the one non-regenerable output of the project, so the rules that
protect it get tests: append-only, latest-wins, malformed lines survivable, and
a record that stays complete after its PNG is gone.
"""

import json

import pytest

from semantic_anarchy.labels import (
    KEEPER_MIN, SCORE_MAX, SEED_PANEL, SEED_PANEL_N, SEED_PANEL_SEED,
    append_label, by_experiment, clean_experiment_id, clamp_score, latest_by_rel,
    list_manifests, make_record, percentile, read_labels, read_manifest,
    summarize, summarize_records, used_seed_panel, write_manifest,
)

SIDECAR = {
    "kind": "generate", "backend": "sd15", "model": "/models/juggernaut_reborn.safetensors",
    "sampler": "pca", "temperature": 1.4, "coherence": None, "components": 512,
    "steps": 30, "guidance": 7.5, "scheduler": "ddim", "neg_mode": "text",
    "distance": 1.37, "image_seed": 1003, "batch_seed": 1000, "index": 3,
    "dist": "outputs/dist", "experiment": "E01 · length",
}


# ------------------------------------------------------------ experiment ids --
@pytest.mark.parametrize("raw,want", [
    ("E00-baseline", "E00-baseline"),
    ("E07 · negatives", "E07-negatives"),
    ("  spaces  ", "spaces"),
    ("../../etc/passwd", "etc-passwd"),
    ("", None),
    (None, None),
    ("---", None),
])
def test_clean_experiment_id(raw, want):
    """Ids reach filenames and argv, so they are slugged once, here."""
    assert clean_experiment_id(raw) == want


def test_experiment_id_is_length_capped():
    assert len(clean_experiment_id("E" * 200)) <= 48


def test_seed_panel_is_one_flag_pair():
    """generate.py seeds image i with batch_seed + i, so the panel IS --seed
    1000 --n 16 -- if that ever stops holding, comparisons stop being paired."""
    assert SEED_PANEL == tuple(range(SEED_PANEL_SEED, SEED_PANEL_SEED + SEED_PANEL_N))
    assert used_seed_panel(SEED_PANEL_SEED, SEED_PANEL_N)
    assert used_seed_panel(SEED_PANEL_SEED, 32)          # a superset still pairs
    assert not used_seed_panel(SEED_PANEL_SEED, 8)       # a truncated panel doesn't
    assert not used_seed_panel(None, 16)
    assert not used_seed_panel(7, 16)


# ---------------------------------------------------------------- records ----
def test_record_snapshots_the_sidecar():
    rec = make_record("generated/anarchy_sd15_1000_003.png", 7, SIDECAR)
    assert rec["score"] == 7
    assert rec["experiment"] == "E01-length"        # slugged on the way in
    assert rec["ckpt_slug"] == "juggernaut_reborn"  # the .safetensors stem
    assert rec["distance"] == 1.37
    assert rec["image_seed"] == 1003
    assert rec["knobs"]["sampler"] == "pca"
    assert rec["knobs"]["temperature"] == 1.4
    # None-valued sidecar keys are dropped rather than stored as nulls.
    assert "coherence" not in rec["knobs"]
    # ...and nothing outside KNOB_KEYS leaks in.
    assert "index" not in rec["knobs"]


def test_record_survives_a_missing_sidecar():
    rec = make_record("generated/x.png", 3, None)
    assert rec["score"] == 3 and rec["knobs"] == {}
    assert rec["backend"] is None and rec["ckpt_slug"] is None


@pytest.mark.parametrize("bad", [-1, 10, 99])
def test_score_range_is_enforced(bad):
    with pytest.raises(ValueError):
        clamp_score(bad)


def test_scores_may_be_strings_from_the_wire():
    assert clamp_score("7") == 7


# ------------------------------------------------------------- the dataset ---
def test_append_only_and_latest_wins(tmp_path):
    f = tmp_path / "labels.jsonl"
    append_label(make_record("a.png", 3, SIDECAR), f)
    append_label(make_record("b.png", 8, SIDECAR), f)
    append_label(make_record("a.png", 9, SIDECAR), f)   # relabel

    recs = read_labels(f)
    assert len(recs) == 3, "history is never rewritten"
    latest = latest_by_rel(recs)
    assert latest["a.png"]["score"] == 9
    assert latest["b.png"]["score"] == 8
    # Every line is independently parseable JSON (the format the report reads).
    for line in f.read_text().splitlines():
        json.loads(line)


def test_malformed_lines_are_skipped_not_fatal(tmp_path):
    f = tmp_path / "labels.jsonl"
    append_label(make_record("a.png", 3), f)
    with f.open("a") as fh:
        fh.write("{not json\n\n")
        fh.write('{"rel": "no-score.png"}\n')
    append_label(make_record("b.png", 5), f)
    assert [r["rel"] for r in read_labels(f)] == ["a.png", "b.png"]


def test_read_labels_of_a_missing_file_is_empty(tmp_path):
    assert read_labels(tmp_path / "nope.jsonl") == []


def test_by_experiment_buckets_untagged_under_empty_string():
    recs = [
        make_record("a.png", 5, {**SIDECAR, "experiment": "E00"}),
        make_record("b.png", 5, {**SIDECAR, "experiment": "E00"}),
        make_record("c.png", 5, {}),
    ]
    groups = by_experiment(recs)
    assert len(groups["E00"]) == 2 and len(groups[""]) == 1


# ------------------------------------------------------------- statistics ----
def test_percentile_interpolates():
    assert percentile([], 90) is None
    assert percentile([5], 90) == 5.0
    assert percentile([0, 10], 50) == 5.0
    assert percentile([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 90) == pytest.approx(8.1)


def test_summary_is_tail_weighted():
    """The whole point of the report card: forty 3s + ten 9s must NOT look like
    fifty 6s. The mean says they're close; keeper-rate and P90 separate them."""
    spiky = summarize([3] * 40 + [9] * 10)
    flat = summarize([6] * 50)
    assert spiky["mean"] == pytest.approx(4.2)
    assert flat["mean"] == 6.0                    # the mean prefers the flat one
    assert spiky["keeper_rate"] == pytest.approx(0.2)
    assert flat["keeper_rate"] == 0.0             # ...and the tail metrics don't
    assert spiky["p90"] == 9.0 and flat["p90"] == 6.0


def test_summary_histogram_covers_every_score():
    s = summarize([0, 0, 5, 9])
    assert len(s["hist"]) == SCORE_MAX + 1
    assert s["hist"][0] == 2 and s["hist"][5] == 1 and s["hist"][9] == 1
    assert sum(s["hist"]) == s["n"] == 4


def test_summary_of_nothing_is_not_a_crash():
    s = summarize([])
    assert s["n"] == 0 and s["mean"] is None and s["keeper_rate"] is None
    assert sum(s["hist"]) == 0


def test_keeper_threshold_is_inclusive():
    assert summarize([KEEPER_MIN])["keeper_rate"] == 1.0
    assert summarize([KEEPER_MIN - 1])["keeper_rate"] == 0.0


def test_summarize_records_reads_the_score_field():
    assert summarize_records([make_record("a.png", 9)])["max"] == 9.0


# -------------------------------------------------------------- manifests ----
def test_manifest_accumulates_runs_and_keeps_the_hypothesis(tmp_path):
    write_manifest("E02-rho", {"argv": ["a"], "hypothesis": "rho reads as intent"},
                   root=tmp_path)
    write_manifest("E02-rho", {"argv": ["b"]}, root=tmp_path)

    doc = read_manifest("E02-rho", root=tmp_path)
    assert len(doc["runs"]) == 2, "an experiment is usually several batches"
    assert doc["hypothesis"] == "rho reads as intent", "a later batch can't erase it"
    assert [r["argv"] for r in doc["runs"]] == [["a"], ["b"]]
    assert [d["id"] for d in list_manifests(root=tmp_path)] == ["E02-rho"]


def test_manifest_needs_a_real_id(tmp_path):
    assert write_manifest("", {}, root=tmp_path) is None
    assert list_manifests(root=tmp_path) == []


def test_reading_an_absent_manifest_is_none(tmp_path):
    assert read_manifest("nope", root=tmp_path) is None
