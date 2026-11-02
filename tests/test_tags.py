"""The tag dataset: normalisation, latest-wins, and the derived vocabulary."""

from __future__ import annotations

import json

import pytest

from semantic_anarchy import tags as tagset


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def test_clean_tag_collapses_whitespace_and_trims():
    assert tagset.clean_tag("  wet   chrome ") == "wet chrome"


def test_clean_tag_keeps_unicode_and_emoji():
    assert tagset.clean_tag("🦴 bone cathedral") == "🦴 bone cathedral"


def test_clean_tag_strips_control_characters():
    assert tagset.clean_tag("wet\x00chr\x1fome") == "wetchrome"


def test_clean_tag_rejects_empty():
    for raw in (None, "", "   ", "\x00"):
        assert tagset.clean_tag(raw) is None


def test_clean_tag_truncates_to_max_len():
    tag = tagset.clean_tag("x" * (tagset.MAX_TAG_LEN + 20))
    assert len(tag) == tagset.MAX_TAG_LEN


def test_clean_tags_dedups_case_insensitively_keeping_first_spelling():
    assert tagset.clean_tags(["Wet Chrome", "wet chrome", "WET CHROME"]) == ["Wet Chrome"]


def test_clean_tags_preserves_assignment_order():
    assert tagset.clean_tags(["b", "a", "c"]) == ["b", "a", "c"]


def test_clean_tags_drops_empties_and_caps_count():
    assert tagset.clean_tags(["a", "", None, "  ", "b"]) == ["a", "b"]
    many = [f"t{i}" for i in range(tagset.MAX_TAGS + 10)]
    assert len(tagset.clean_tags(many)) == tagset.MAX_TAGS


# --------------------------------------------------------------------------- #
# The append-only file
# --------------------------------------------------------------------------- #
def test_append_and_read_round_trip(tmp_path):
    p = tmp_path / "tags.jsonl"
    tagset.append_record(tagset.make_record("a.png", ["one", "two"]), p)
    tagset.append_record(tagset.make_record("b.png", ["one"]), p)
    recs = tagset.read_records(p)
    assert [r["rel"] for r in recs] == ["a.png", "b.png"]
    assert recs[0]["tags"] == ["one", "two"]


def test_read_records_skips_malformed_lines(tmp_path):
    p = tmp_path / "tags.jsonl"
    p.write_text(
        json.dumps({"rel": "a.png", "ts": 1.0, "tags": ["x"]}) + "\n"
        "{not json\n"
        "\n"
        + json.dumps({"rel": "b.png"}) + "\n"           # no tags list
        + json.dumps({"tags": ["y"]}) + "\n"            # no rel
        + json.dumps({"rel": "c.png", "ts": 2.0, "tags": []}) + "\n",
        encoding="utf-8",
    )
    assert [r["rel"] for r in tagset.read_records(p)] == ["a.png", "c.png"]


def test_read_records_missing_file_is_empty(tmp_path):
    assert tagset.read_records(tmp_path / "nope.jsonl") == []


def test_tags_file_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SA_TAGS_FILE", str(tmp_path / "elsewhere.jsonl"))
    assert tagset.tags_file() == tmp_path / "elsewhere.jsonl"


# --------------------------------------------------------------------------- #
# Latest-wins
# --------------------------------------------------------------------------- #
def test_latest_record_wins():
    recs = [
        tagset.make_record("a.png", ["first"], ts=1.0),
        tagset.make_record("a.png", ["second", "third"], ts=2.0),
    ]
    assert tagset.latest_by_rel(recs) == {"a.png": ["second", "third"]}


def test_untagging_removes_the_image_entirely():
    recs = [
        tagset.make_record("a.png", ["gone"], ts=1.0),
        tagset.make_record("a.png", [], ts=2.0),
    ]
    assert tagset.latest_by_rel(recs) == {}


# --------------------------------------------------------------------------- #
# The derived vocabulary
# --------------------------------------------------------------------------- #
def test_vocabulary_counts_images_not_records():
    recs = [
        tagset.make_record("a.png", ["sky"], ts=1.0),
        tagset.make_record("a.png", ["sky"], ts=2.0),      # relabelled, same tag
        tagset.make_record("b.png", ["sky", "bone"], ts=3.0),
    ]
    assert tagset.vocabulary(recs) == [
        {"tag": "sky", "count": 2},
        {"tag": "bone", "count": 1},
    ]


def test_vocabulary_sorts_by_count_then_name():
    recs = [
        tagset.make_record("a.png", ["zeta", "alpha"], ts=1.0),
        tagset.make_record("b.png", ["alpha"], ts=2.0),
    ]
    assert [v["tag"] for v in tagset.vocabulary(recs)] == ["alpha", "zeta"]


def test_vocabulary_retires_a_tag_nobody_wears():
    recs = [
        tagset.make_record("a.png", ["temporary"], ts=1.0),
        tagset.make_record("a.png", ["kept"], ts=2.0),
    ]
    assert [v["tag"] for v in tagset.vocabulary(recs)] == ["kept"]


# --------------------------------------------------------------------------- #
# Membership
# --------------------------------------------------------------------------- #
def test_tagged_with_is_case_insensitive():
    tag_map = {"a.png": ["Wet Chrome"], "b.png": ["bone"]}
    assert tagset.tagged_with(tag_map, "wet chrome") == {"a.png"}
    assert tagset.tagged_with(tag_map, "  WET   CHROME  ") == {"a.png"}


def test_tagged_with_empty_matches_nothing():
    assert tagset.tagged_with({"a.png": ["x"]}, "") == set()


@pytest.mark.parametrize("want,expected", [("x", {"a.png", "b.png"}), ("y", {"b.png"})])
def test_tagged_with_multi(want, expected):
    tag_map = {"a.png": ["x"], "b.png": ["x", "y"]}
    assert tagset.tagged_with(tag_map, want) == expected
