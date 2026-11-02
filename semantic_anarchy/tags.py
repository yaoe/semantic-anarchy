"""Aesthetic tags — which *universe* an image belongs to.

The sibling of :mod:`.labels`, and the other half of a judgement. A score says
how GOOD an image is; a tag says what KIND it is. The two are independent on
purpose: a 9 in one aesthetic universe and a 9 in another are both keepers, but
averaging them, fitting a distribution across them, or reading one report card
over both describes an image that does not exist. Tagging is what keeps those
explorations separate while they share one gallery.

**The dataset** — ``labels/tags.jsonl`` at the repo root, append-only, one JSON
object per line, **git-tracked** for exactly the reason ``labels.jsonl`` is: a
render is free, a human deciding "this belongs with those" is not.

Each record carries the **full tag set** for one image at one moment::

    {"rel": "generated/anarchy_sd15_600040_000.png", "ts": 1786.0,
     "tags": ["bone cathedral", "wet chrome"]}

Storing the whole set rather than an add/remove delta means assigning and
un-assigning are the same operation, the latest record always stands on its own,
and :func:`latest_by_rel` is a plain last-one-wins scan. An image whose newest
record is ``{"tags": []}`` is deliberately untagged, which is different from an
image that was never tagged at all — both read as "no tags", but the history
keeps the distinction.

The **vocabulary is derived, never declared**: a tag exists because an image
wears it. Creating one is assigning it, and the last image to drop a tag retires
it. That is why there is no registry file to keep in sync.
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from .labels import repo_root

#: Longest a single tag may be. Tags are chips in a strip, not descriptions.
MAX_TAG_LEN = 32
#: Most tags one image may carry. A cap keeps a fat-fingered client from writing
#: an unbounded row; nothing in the UI approaches it.
MAX_TAGS = 16

#: Control characters are the only thing rejected outright — unicode and emoji
#: are welcome, because "🦴 bone cathedral" is a perfectly good universe name.
_CONTROL = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")

_append_lock = threading.Lock()


def tags_file(root: Optional[Path] = None) -> Path:
    """The append-only tag dataset. ``SA_TAGS_FILE`` relocates it (tests, forks)."""
    env = os.environ.get("SA_TAGS_FILE")
    if env and root is None:
        return Path(os.path.expanduser(env))
    return (root or repo_root()) / "labels" / "tags.jsonl"


def clean_tag(raw) -> Optional[str]:
    """Normalise one tag, or ``None`` if nothing survives.

    Whitespace is collapsed and the ends trimmed so ``" wet  chrome "`` and
    ``"wet chrome"`` are the same universe rather than two. Case is *preserved*
    (the tag is shown back to a human) but compared case-insensitively by
    :func:`clean_tags`, so ``"Wet Chrome"`` never splits off from ``"wet chrome"``.
    """
    if raw is None:
        return None
    text = _WHITESPACE.sub(" ", _CONTROL.sub("", str(raw))).strip()
    return text[:MAX_TAG_LEN].strip() or None


def clean_tags(raw: Iterable) -> list[str]:
    """Normalise a whole set: clean each, drop empties, de-dup, cap the count.

    Order is preserved (first spelling of a tag wins) rather than sorted, so the
    record reads the way the person assigned it.
    """
    out: list[str] = []
    seen: set[str] = set()
    for item in raw or ():
        tag = clean_tag(item)
        if tag is None:
            continue
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
        if len(out) >= MAX_TAGS:
            break
    return out


def make_record(rel: str, tags: Iterable, ts: Optional[float] = None) -> dict:
    """One tagging event: the complete set this image wears from now on."""
    return {
        "rel": str(rel),
        "ts": time.time() if ts is None else float(ts),
        "tags": clean_tags(tags),
    }


def append_record(rec: dict, path: Optional[Path] = None) -> Path:
    """Append one record. Never rewrites, never reorders — history is the file."""
    p = path or tags_file()
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, ensure_ascii=False)
    with _append_lock:
        # Same reasoning as labels.append_label: O_APPEND + one write() per short
        # line means two taggers (or the dashboard and a script) cannot interleave.
        with p.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return p


def read_records(path: Optional[Path] = None) -> list[dict]:
    """Every record, in file order. Malformed lines are skipped, not fatal."""
    p = path or tags_file()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if isinstance(rec, dict) and "rel" in rec and isinstance(rec.get("tags"), list):
            out.append(rec)
    return out


def latest_by_rel(records: Iterable[dict]) -> dict[str, list[str]]:
    """rel -> the tag set that currently stands (newest record wins).

    Images whose newest record is empty are dropped: callers want "what is
    tagged", and an explicit un-tag should read the same as never tagged.
    """
    latest: dict[str, list[str]] = {}
    for rec in records:
        latest[rec["rel"]] = clean_tags(rec.get("tags"))
    return {rel: tags for rel, tags in latest.items() if tags}


def vocabulary(records: Iterable[dict]) -> list[dict]:
    """Every tag in use, with how many images wear it — commonest first.

    Derived from the standing sets, so a tag that was assigned and then removed
    everywhere disappears on its own.
    """
    counts: dict[str, int] = {}
    spelling: dict[str, str] = {}
    for tags in latest_by_rel(records).values():
        for tag in tags:
            key = tag.casefold()
            counts[key] = counts.get(key, 0) + 1
            spelling.setdefault(key, tag)
    return [{"tag": spelling[k], "count": c}
            for k, c in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]


def tagged_with(tag_map: dict[str, list[str]], want: str) -> set[str]:
    """Every rel wearing ``want`` (case-insensitive) — the membership filter."""
    key = (clean_tag(want) or "").casefold()
    if not key:
        return set()
    return {rel for rel, tags in tag_map.items()
            if any(t.casefold() == key for t in tags)}
