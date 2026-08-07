#!/usr/bin/env python
"""Merge a directory of scraped prompt .txt files into ONE clean corpus file.

    python scripts/process_prompt_dir.py --dir prompts_filtered --out xander_prompts_to_encode.txt

What it does, in order:

  1. load    every *.txt in --dir (``# `` headers and blank lines skipped, same
             rule the repo's own loaders use)
  2. repair  strip the junk that comes with scraped files: python-list
             scaffolding (``name = [`` / quoted+comma lines / ``]``), literal
             ``\\uXXXX`` escapes, trailing CLI flags (``--w 768 --s 20``),
             bullet/number prefixes, smart quotes, spaced/doubled commas,
             whitespace, and the leading ``##`` on WordPiece caption fragments
             (the fragment itself is KEPT -- cut-ups make good prompts -- but a
             line starting with ``#`` would be invisible to the loaders)
  3. TOK     LoRA trigger tokens carry no meaning for a text encoder.  Clauses
             like "in the style of TOK" are deleted; a bare ``TOK`` that is the
             subject becomes ``a person`` (--tok-mode to change)
  4. filter  min chars / min words / ascii fraction / URL-and-code junk
  5. dedup   exact (on a canonical key), then near-duplicates
  6. spread  two caps that keep any one source from dominating the fit:
             --cluster-cap  max prompts per topical cluster (tf-idf cosine), for
                            files that are N rewrites of one theme
             --phrase-cap   max prompts sharing any one 5-word phrase, for
                            slot-filler template corpora, whose subjects really
                            are all different -- only the syntax repeats, and
                            the topical cap cannot see that
  7. write   deterministically shuffled, with a ``#`` provenance header

Near-duplicate detection runs on the prompt **as the text encoder will see it**
-- truncated to the CLIP 77-token window.  This matters: several source files
hold long prompts that are word-identical for the first 60 words and only drift
in the tail the encoder never reads.  Comparing full text would keep them all.

The metric is word-3-shingle Jaccard (blocked through an inverted index, so it
is near-linear rather than O(n^2)), confirmed on borderline pairs with
difflib's edit-distance ratio.  Shingles are the right primitive here because
prompt duplicates are usually re-orderings and small substitutions, which raw
Levenshtein on long strings scores as further apart than they are.

Pure stdlib; ``transformers`` is used only if importable, for exact CLIP token
counts in the report (word-based approximation otherwise).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CLIP_MAX_TOKENS = 77          # incl. BOS + EOS -> 75 content tokens
CLIP_CONTENT_TOKENS = 75

# ---------------------------------------------------------------- repair ----

_LIST_OPEN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*\[\s*$")
_LIST_CLOSE = re.compile(r"^[\]\)],?\s*$")
_BULLET = re.compile(r"^\s*(?:[-*•]|\d{1,3}[.)])\s+")
_CLI_FLAGS = re.compile(r"\s*--[A-Za-z_][A-Za-z0-9_-]*(?:\s+[^\s-][^\s]*)?")
_UNICODE_ESC = re.compile(r"\\u([0-9a-fA-F]{4})")
_WS = re.compile(r"\s+")
_URLISH = re.compile(r"https?://|www\.|\.(?:png|jpe?g|safetensors|ckpt|json|csv)\b", re.I)

_PUNCT_MAP = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": " - ", "…": "...", " ": " ",
}

# "in the style of TOK", "in TOK style", "a photo of TOK" -> handled below
_TOK_STYLE_CLAUSE = re.compile(
    r"[,;]?\s*(?:in|with)\s+the\s+style\s+of\s+TOK\b|[,;]?\s*in\s+TOK\s+style\b", re.I)
_TOK_BARE = re.compile(r"\bTOK\b")
# "a painting of a beautiful young TOK" -> "... a beautiful young a person".
# Drop the article the substitution duplicated, but only across adjectives --
# a preposition in the gap means the article belongs to a different noun
# ("a photo of a person" must survive intact).
_GAP = r"(?:(?!of\b|in\b|on\b|at\b|with\b|for\b|and\b|or\b|by\b|from\b|to\b|as\b)[a-z]+\s+){0,3}?"
_DOUBLE_ARTICLE = re.compile(
    rf"\b(a|an|the|my|his|her|their|our|your)\s+({_GAP})a person\b", re.I)
# "In the style of TOK: a large tree" loses its clause and is left holding the
# colon, so the cleanup has to run for every prompt, not just TOK ones.
_DANGLING_PUNCT = re.compile(r"\s*([,;:])\s*(?=[,;:])|^[\s,;:.\-#]+|[\s,;:\-]+$")
# Caption dumps leak WordPiece continuation tokens ("##ng a green path", "##raffes
# are sitting in an area of water").  These are KEPT -- as cut-up fragments they
# make good prompts -- but the leading '#' must go: the repo's own loaders skip
# '#' lines, so the corpus file would otherwise hold fewer prompts than its own
# header claims, silently.
# A source header is "# " + text; a WordPiece fragment is "##ng" with no space.
# Only the former is a comment -- that distinction is what lets the fragments
# through the loader at all.
_COMMENT = re.compile(r"#(\s|$)")
# Lexica-style scrapes are full of " ,tag" and ",,": harmless to the tokenizer
# but they fragment the exact-dedup key, so normalise before deduping.
_SPACED_COMMA = re.compile(r"\s+([,;:])")
_REPEAT_COMMA = re.compile(r"([,;:])\1+")


def unescape(text: str) -> str:
    """Turn literal ``\\uXXXX`` / ``\\n`` back into real characters."""
    text = _UNICODE_ESC.sub(lambda m: chr(int(m.group(1), 16)), text)
    return text.replace("\\n", " ").replace('\\"', '"').replace("\\'", "'")


def repair(raw: str) -> str | None:
    """Normalise one scraped line into a prompt, or None if it is scaffolding."""
    line = raw.strip()
    if not line or _COMMENT.match(line):
        return None
    if _LIST_OPEN.match(line) or _LIST_CLOSE.match(line):
        return None

    line = unescape(line)

    # python string literal: "prompt", / 'prompt',
    line = re.sub(r",\s*$", "", line)
    if len(line) > 1 and line[0] == line[-1] and line[0] in "\"'":
        line = line[1:-1]

    line = _BULLET.sub("", line)
    for bad, good in _PUNCT_MAP.items():
        line = line.replace(bad, good)
    line = unicodedata.normalize("NFKC", line)
    line = _CLI_FLAGS.sub("", line)
    line = _SPACED_COMMA.sub(r"\1", line)
    line = _REPEAT_COMMA.sub(r"\1", line)
    line = _WS.sub(" ", line).strip()
    return tidy_punct(line) or None


def tidy_punct(text: str) -> str:
    """Drop stray leading/trailing/doubled punctuation, to a fixed point."""
    while True:
        cleaned = _DANGLING_PUNCT.sub(lambda m: m.group(1) or "", text).strip()
        if cleaned == text:
            return text
        text = cleaned


def strip_tok(text: str, mode: str) -> str | None:
    """Neutralise LoRA trigger tokens.  Returns None to drop the prompt."""
    if "TOK" not in text:
        return text
    if mode == "keep":
        return text
    if mode == "drop":
        return None
    text = _TOK_STYLE_CLAUSE.sub("", text)
    text = _TOK_BARE.sub("a person", text)
    text = _DOUBLE_ARTICLE.sub(r"\1 \2person", text)
    return tidy_punct(_WS.sub(" ", text)) or None


# --------------------------------------------------------------- filters ----

def ascii_fraction(text: str) -> float:
    if not text:
        return 0.0
    return sum(c.isascii() for c in text) / len(text)


def keep(text: str, args) -> str | None:
    """Return a rejection reason, or None if the prompt passes."""
    if len(text) < args.min_chars:
        return "too_short"
    if args.max_chars and len(text) > args.max_chars:
        return "too_long"
    if len(text.split()) < args.min_words:
        return "too_few_words"
    if ascii_fraction(text) < args.min_ascii:
        return "non_latin"
    if _URLISH.search(text):
        return "url_or_filename"
    if not any(c.isalpha() for c in text):
        return "no_letters"
    return None


# ------------------------------------------------------------ similarity ----

_WORD = re.compile(r"[a-z0-9']+")

# Function words carry no topic.  Dropping them makes the unigram Jaccard used
# for topical clustering measure subject matter rather than sentence style.
STOPWORDS = frozenset("""
a an the of in on at to for with and or but is are was were be been being it its
this that these those there here as by from into onto over under above below near
his her their our your my he she they we you i it's very more most some any all
""".split())


def canon(text: str) -> str:
    """Aggressive key for *exact* dedup: case/punctuation/spacing insensitive."""
    return " ".join(_WORD.findall(text.lower()))


def shingles(text: str, n: int = 3) -> frozenset[str]:
    """Word n-grams; falls back to char 4-grams for very short prompts.

    Used for *near*-duplicate detection, where word order and exact phrasing
    are the signal.
    """
    words = _WORD.findall(text.lower())
    if len(words) < n + 1:
        s = "".join(words)
        return frozenset(s[i:i + 4] for i in range(max(1, len(s) - 3)))
    return frozenset(" ".join(words[i:i + n]) for i in range(len(words) - n + 1))


def topic_bag(text: str) -> frozenset[str]:
    """Content-word bag, the basis for *topical* clustering.

    n-gram Jaccard is blind to paraphrase: two LLM rewrites of "an ethereal
    being of woven light above a serene valley" share almost no 3-grams but
    nearly the same content words.  Several source files are exactly that --
    hundreds of rewrites of one theme -- so the cluster cap has to work in bag
    space or it does nothing.
    """
    return frozenset(w for w in _WORD.findall(text.lower())
                     if w not in STOPWORDS and len(w) > 2)


def tfidf_vectors(bags: list[frozenset[str]]) -> tuple[list[dict[str, float]], Counter]:
    """L2-normalised binary-tf * idf vectors, one per prompt.

    Plain Jaccard on bags punishes length (a 60-word rewrite and a 12-word
    prompt on the same subject cannot score high) and treats "light" as worth
    as much as "mycorrhizal".  Cosine on idf weights fixes both, which is what
    makes the cluster cap actually find the big thematic clumps.
    """
    n = len(bags)
    df: Counter = Counter()
    for bag in bags:
        df.update(bag)
    import math
    idf = {w: math.log(n / d) for w, d in df.items()}
    vecs = []
    for bag in bags:
        v = {w: idf[w] for w in bag if idf[w] > 0}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs.append({w: x / norm for w, x in v.items()})
    return vecs, df


def content_5grams(text: str) -> set[str]:
    """5-word phrases carrying at least two content words.

    The content-word floor keeps ordinary English connective tissue ("in the
    middle of the") out of the phrase cap, so only real template syntax counts.
    """
    w = _WORD.findall(text.lower())
    out = set()
    for i in range(len(w) - 4):
        gram = w[i:i + 5]
        if sum(1 for x in gram if x not in STOPWORDS and len(x) > 2) >= 2:
            out.add(" ".join(gram))
    return out


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(w * b[t] for t, w in a.items() if t in b)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def containment(a: frozenset, b: frozenset) -> float:
    """Overlap coefficient: how much of the *smaller* set the larger covers.

    Jaccard cannot see prefix duplicates -- a clean 20-word prompt and a
    scrape blob that begins with those same 20 words score far apart on it
    purely because of the length gap.  This catches them.
    """
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def ratio(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


class ShingleIndex:
    """Inverted index over shingles, used to find candidate near-duplicates.

    Very common shingles ("in the style of") are skipped as lookup keys once
    their document frequency crosses `df_cap` -- they generate huge candidate
    lists and no signal.  Each item is registered under at most `max_keys` of
    its rarest shingles, which keeps the whole pass near-linear.
    """

    def __init__(self, df_cap: int, max_keys: int = 24):
        self.buckets: dict[str, list[int]] = defaultdict(list)
        self.df: Counter = Counter()
        self.df_cap = df_cap
        self.max_keys = max_keys

    def candidates(self, sh: frozenset[str]) -> set[int]:
        out: set[int] = set()
        for key in sorted(sh, key=lambda s: self.df[s])[: self.max_keys]:
            if self.df[key] <= self.df_cap:
                out.update(self.buckets[key])
        return out

    def add(self, idx: int, sh: frozenset[str]) -> None:
        for key in sh:
            self.df[key] += 1
        for key in sorted(sh, key=lambda s: self.df[s])[: self.max_keys]:
            if self.df[key] <= self.df_cap:
                self.buckets[key].append(idx)


# ------------------------------------------------------------- tokenizer ----

def make_token_counter(model: str):
    """Real CLIP token counts if transformers is importable, else ~1.35*words."""
    try:
        from transformers import CLIPTokenizerFast, logging as hf_logging
        hf_logging.set_verbosity_error()          # the >77 warning is the point here
        tok = CLIPTokenizerFast.from_pretrained(model)

        def count(texts: list[str]) -> list[int]:
            out: list[int] = []
            for i in range(0, len(texts), 512):
                enc = tok(texts[i:i + 512], truncation=False)["input_ids"]
                out.extend(len(e) for e in enc)
            return out

        return count, True
    except Exception as exc:                                   # offline / no deps
        print(f"  (CLIP tokenizer unavailable -> approximating: {exc})", file=sys.stderr)

        def count(texts: list[str]) -> list[int]:
            return [int(len(_WORD.findall(t.lower())) * 1.35) + 2 for t in texts]

        return count, False


def clip_head(text: str, content_tokens: int, tokenizer_fn) -> str:
    """The part of `text` the encoder actually reads (approximate word cut)."""
    words = text.split()
    if len(words) <= content_tokens * 0.6:        # certainly fits
        return text
    return " ".join(words[: int(content_tokens / 1.35) + 1])


# ------------------------------------------------------------------ main ----

def interleave(groups: dict[str, list], seed: int) -> list:
    """Round-robin across source files (each internally shuffled).

    This is what makes the cluster cap fair: when 8 slots are available for a
    topic, they go to 8 different source files rather than the first 8 lines of
    whichever file happened to load first.
    """
    rng = random.Random(seed)
    pools = []
    for name in sorted(groups):
        items = list(groups[name])
        rng.shuffle(items)
        pools.append(items)
    out = []
    for i in range(max((len(p) for p in pools), default=0)):
        for p in pools:
            if i < len(p):
                out.append(p[i])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default="prompts_filtered", help="directory of *.txt prompt files")
    ap.add_argument("--out", default="xander_prompts_to_encode.txt")
    ap.add_argument("--report", default=None, help="also write the stats as JSON here")

    ap.add_argument("--min-chars", type=int, default=10)
    ap.add_argument("--max-chars", type=int, default=1000,
                    help="0 = no cap; the default only trims scrape blobs -- 4x the\n"
                         "CLIP window, so nothing the encoder reads is lost")
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--min-ascii", type=float, default=0.85,
                    help="drop lines with more non-ASCII than this (CJK junk)")
    ap.add_argument("--tok-mode", choices=("replace", "drop", "keep"), default="replace",
                    help="what to do with LoRA trigger tokens 'TOK'")

    ap.add_argument("--near", type=float, default=0.50, help="Jaccard >= this => near-duplicate")
    ap.add_argument("--near-ratio", type=float, default=0.85,
                    help="edit-distance ratio >= this (with Jaccard >= --near/2) => near-duplicate")
    ap.add_argument("--near-contain", type=float, default=0.80,
                    help="overlap coefficient >= this => near-duplicate (catches prefix/superset pairs)")
    ap.add_argument("--loose", type=float, default=0.55,
                    help="tf-idf cosine >= this => same topical cluster")
    ap.add_argument("--loose-min-overlap", type=int, default=4,
                    help="also require this many shared content words (guards short prompts)")
    ap.add_argument("--cluster-cap", type=int, default=8,
                    help="max prompts kept per topical cluster (0 = unlimited)")
    ap.add_argument("--phrase-cap", type=int, default=40,
                    help="max prompts sharing any one 5-word phrase (0 = off). "
                         "Blunts slot-filler template corpora, which the topical "
                         "cluster cap cannot see -- their subjects really are all different, "
                         "only the syntax repeats")
    ap.add_argument("--df-cap-frac", type=float, default=0.02,
                    help="skip shingles appearing in more than this fraction of the corpus")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    args = ap.parse_args()

    src_dir = Path(args.dir)
    if not src_dir.is_dir():
        print(f"no such directory: {src_dir}", file=sys.stderr)
        return 1

    stats: dict = {"dir": str(src_dir), "stages": {}, "rejected": Counter(), "per_source": {}}

    # -- 1/2/3/4: load, repair, filter -------------------------------------
    by_source: dict[str, list[str]] = {}
    raw_total = 0
    for path in sorted(src_dir.glob("*.txt")):
        kept: list[str] = []
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            text = repair(raw)
            if text is None:
                continue
            raw_total += 1
            text = strip_tok(text, args.tok_mode)
            if text is None:
                stats["rejected"]["tok_dropped"] += 1
                continue
            reason = keep(text, args)
            if reason:
                stats["rejected"][reason] += 1
                continue
            kept.append(text)
        if kept:
            by_source[path.name] = kept
        stats["per_source"][path.name] = {"raw": len(kept)}

    stats["stages"]["loaded"] = raw_total
    stats["stages"]["after_filters"] = sum(len(v) for v in by_source.values())
    print(f"loaded   {raw_total:>6} lines from {len(by_source)} files")
    print(f"filtered {stats['stages']['after_filters']:>6} pass hard filters "
          f"({', '.join(f'{k}={v}' for k, v in stats['rejected'].most_common())})")

    # -- 5a: exact dedup ----------------------------------------------------
    seen: dict[str, str] = {}
    ordered: list[tuple[str, str]] = []          # (source, text)
    groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for source, texts in by_source.items():
        for text in texts:
            key = canon(text)
            if key in seen:
                stats["rejected"]["exact_duplicate"] += 1
                continue
            seen[key] = text
            groups[source].append((source, text))
    ordered = interleave(groups, args.seed)
    stats["stages"]["after_exact_dedup"] = len(ordered)
    print(f"exact    {len(ordered):>6} unique "
          f"(-{stats['rejected']['exact_duplicate']} exact duplicates)")

    # -- 5b/6: near-dup + topical spread, on the CLIP-visible head ----------
    token_count, real_tokenizer = make_token_counter(args.clip_model)
    heads = [clip_head(t, CLIP_CONTENT_TOKENS, token_count) for _, t in ordered]
    shs = [shingles(h) for h in heads]
    bags = [topic_bag(h) for h in heads]
    vecs, topic_df = tfidf_vectors(bags)
    # index topical candidates on each prompt's rarest content words only
    rare = [frozenset(sorted(b, key=lambda w: topic_df[w])[:16]) for b in bags]

    df_cap = max(8, int(len(ordered) * args.df_cap_frac))
    index = ShingleIndex(df_cap)                 # near-dup space (3-grams)
    topic_index = ShingleIndex(max(32, int(len(ordered) * 0.05)), max_keys=12)
    kept_idx: list[int] = []
    cluster_of: dict[int, int] = {}
    cluster_size: Counter = Counter()
    n_near = n_capped = 0
    near_examples: list[tuple[str, str]] = []
    cap_examples: list[tuple[str, str]] = []

    for i, sh in enumerate(shs):
        best_dup, best_dup_j = None, 0.0
        for j in index.candidates(sh):
            jac = jaccard(sh, shs[j])
            if (jac >= args.near
                    or containment(sh, shs[j]) >= args.near_contain
                    or (jac >= args.near / 2
                        and ratio(heads[i], heads[j]) >= args.near_ratio)):
                if jac > best_dup_j:
                    best_dup, best_dup_j = j, jac

        best_loose, best_loose_j = None, 0.0
        if best_dup is None:
            for j in topic_index.candidates(rare[i]):
                if len(bags[i] & bags[j]) < args.loose_min_overlap:
                    continue
                sim = cosine(vecs[i], vecs[j])
                if sim >= args.loose and sim > best_loose_j:
                    best_loose, best_loose_j = j, sim

        if best_dup is not None:
            n_near += 1
            if len(near_examples) < 8:
                near_examples.append((ordered[best_dup][1][:110], ordered[i][1][:110]))
            continue
        cid = cluster_of.get(best_loose, i) if best_loose is not None else i
        if args.cluster_cap and cluster_size[cid] >= args.cluster_cap:
            n_capped += 1
            if len(cap_examples) < 8:
                cap_examples.append((ordered[cid][1][:110], ordered[i][1][:110]))
            continue
        cluster_of[i] = cid
        cluster_size[cid] += 1
        kept_idx.append(i)
        index.add(i, sh)
        topic_index.add(i, rare[i])

    stats["rejected"]["near_duplicate"] = n_near
    stats["rejected"]["cluster_cap"] = n_capped
    stats["stages"]["after_near_dedup"] = len(ordered) - n_near
    stats["stages"]["final"] = len(kept_idx)
    print(f"near     {len(ordered) - n_near:>6} after near-dup removal (-{n_near})")
    print(f"spread   {len(kept_idx):>6} after cluster cap {args.cluster_cap} (-{n_capped})")

    # -- 6b: template / phrase cap -----------------------------------------
    n_phrase = 0
    phrase_examples: list[str] = []
    if args.phrase_cap:
        used: Counter = Counter()
        survivors = []
        for i in kept_idx:
            grams = content_5grams(heads[i])
            if grams and max(used[g] for g in grams) >= args.phrase_cap:
                n_phrase += 1
                if len(phrase_examples) < 4:
                    phrase_examples.append(ordered[i][1][:110])
                continue
            used.update(grams)
            survivors.append(i)
        kept_idx = survivors
        stats["rejected"]["phrase_cap"] = n_phrase
        stats["stages"]["final"] = len(kept_idx)
        print(f"phrase   {len(kept_idx):>6} after phrase cap {args.phrase_cap} (-{n_phrase})")

    final = [ordered[i] for i in kept_idx]
    random.Random(args.seed + 1).shuffle(final)

    # -- 7: write -----------------------------------------------------------
    out_path = Path(args.out)
    src_counts = Counter(s for s, _ in final)
    header = [
        f"# {out_path.name} -- {len(final)} prompts",
        f"# built by scripts/process_prompt_dir.py from {src_dir}/ "
        f"({len(by_source)} files, {raw_total} raw lines)",
        f"# filters: min_chars={args.min_chars} min_words={args.min_words} "
        f"min_ascii={args.min_ascii} tok_mode={args.tok_mode}",
        f"# dedup: exact + jaccard>={args.near} (edit-ratio>={args.near_ratio}) on the "
        f"CLIP-77 head; cluster cap {args.cluster_cap} at jaccard>={args.loose}",
        f"# seed={args.seed}",
    ]
    out_path.write_text("\n".join(header + [t for _, t in final]) + "\n", encoding="utf-8")
    print(f"\nwrote    {out_path} ({len(final)} prompts)")

    # -- analysis -----------------------------------------------------------
    texts = [t for _, t in final]
    tokens = token_count(texts)
    chars = [len(t) for t in texts]
    over = sum(1 for n in tokens if n > CLIP_MAX_TOKENS)

    def pct(vals, p):
        s = sorted(vals)
        return s[min(len(s) - 1, int(len(s) * p / 100))]

    stats["per_source"] = {k: {"raw": v["raw"], "unique": len(groups.get(k, ())),
                               "final": src_counts.get(k, 0)}
                           for k, v in stats["per_source"].items()}
    stats["tokens"] = {"exact": real_tokenizer, "over_77": over,
                       "p50": pct(tokens, 50), "p90": pct(tokens, 90), "max": max(tokens)}
    stats["chars"] = {"p10": pct(chars, 10), "p50": pct(chars, 50),
                      "p90": pct(chars, 90), "max": max(chars)}
    stats["rejected"] = dict(stats["rejected"])

    print("\n--- corpus analysis " + "-" * 40)
    print(f"prompts            {len(texts)}")
    print(f"chars              p10={pct(chars,10)}  p50={pct(chars,50)}  "
          f"p90={pct(chars,90)}  max={max(chars)}")
    print(f"CLIP tokens{'' if real_tokenizer else ' (approx)'}        "
          f"p10={pct(tokens,10)}  p50={pct(tokens,50)}  p90={pct(tokens,90)}  max={max(tokens)}")
    print(f"over the 77 window {over} ({100*over/len(texts):.1f}%) -- their tails are never encoded")
    print(f"short (<=8 tokens) {sum(1 for n in tokens if n <= 8)} "
          f"-- mostly padding, wide EOS spread")

    vocab = Counter(w for t in texts for w in _WORD.findall(t.lower()))
    print(f"vocabulary         {len(vocab)} distinct words, "
          f"{sum(vocab.values())} total")
    print(f"top words          {', '.join(w for w, _ in vocab.most_common(12))}")

    openers = Counter(" ".join(_WORD.findall(t.lower())[:3]) for t in texts)
    print(f"top openers        {', '.join(f'{o!r}x{c}' for o, c in openers.most_common(6))}")

    # repeated long phrases are the fingerprint of a template generator: the
    # prompts are all distinct, but their *syntax* is one mould, and a fitted
    # distribution will happily learn that mould as its dominant axis.
    phrases: Counter = Counter()
    for t in texts:
        w = _WORD.findall(t.lower())
        phrases.update(set(" ".join(w[i:i + 5]) for i in range(len(w) - 4)))
    hot = [(p, c) for p, c in phrases.most_common(8)]
    print("repeated 5-grams   " + "; ".join(f"{p!r} x{c}" for p, c in hot[:4]))
    if len(hot) > 4:
        print("                   " + "; ".join(f"{p!r} x{c}" for p, c in hot[4:8]))
    stats["repeated_5grams"] = dict(hot)

    segs = [t.count(",") + 1 for t in texts]
    tagged = sum(1 for s in segs if s >= 5)
    print(f"comma segments     p50={pct(segs,50)}  p90={pct(segs,90)};  "
          f"{tagged} ({100*tagged/len(texts):.0f}%) are tag-style (>=5 segments)")

    big = sorted(cluster_size.items(), key=lambda kv: -kv[1])[:5]
    print(f"topical clusters   {len(cluster_size)} "
          f"({sum(1 for v in cluster_size.values() if v == 1)} singletons, "
          f"largest {', '.join(str(v) for _, v in big)})")
    print("largest cluster heads:")
    for cid, n in big[:3]:
        print(f"  [{n:>2}] {ordered[cid][1][:100]}")

    print("\nper-source contribution (final / unique / after-filters):")
    for name in sorted(stats["per_source"], key=lambda n: -src_counts.get(n, 0)):
        s = stats["per_source"][name]
        raw, uniq, fin = s["raw"], s["unique"], s["final"]
        if raw:
            print(f"  {fin:>5} / {uniq:<5} / {raw:<5} {100*fin/raw:>5.1f}%  {name}")

    if near_examples:
        print("\nsample near-duplicate kills (kept | dropped):")
        for a, b in near_examples[:4]:
            print(f"  K {a}\n  D {b}\n")
    if cap_examples:
        print("sample cluster-cap kills (cluster head | dropped):")
        for a, b in cap_examples[:3]:
            print(f"  K {a}\n  D {b}\n")

    if args.report:
        Path(args.report).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"report -> {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
