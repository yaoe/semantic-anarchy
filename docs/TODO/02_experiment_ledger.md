# 02 · The experiment ledger

The project's memory between sessions. **Append-only.** Newest entries at the
bottom of §Entries; the lexicon at the top grows as things earn names.

One entry per experiment, 3–6 lines. Written the same day the batch is labeled,
while the surprise is still fresh — a result you postpone writing up is a result
you will re-run by accident in three weeks.

Protocol for filling one in lives in
[01_exploration_plan.md §8](01_exploration_plan.md#8--standing-experiment-protocol-copy-into-each-ledger-entry).
Mechanics of tagging, labeling and reporting a batch:
[reference/labeling.md](../reference/labeling.md).

---

## Lexicon

Named discoveries — a direction, a region, a recipe that reliably produces
something. A name here is a word the project can use afterwards ("run it
anti-605ish at T 1.4"), which is the whole point of naming.

| name | what it is | how to get it | first seen |
|---|---|---|---|
| _(empty — nothing has earned a name yet)_ | | | |

---

## Entries

<!--
Template — copy, don't edit in place:

### E##-slug · <one-line title>
- **Hypothesis** — one sentence, falsifiable by eye.
- **Ran** — what varied, what was frozen, seed panel y/n, n images, checkpoint.
- **Numbers** — keeper-rate, P90, median; novelty; near-dupes. From
  `python scripts/experiment_report.py E##-slug`.
- **Learned** — the sentence you would tell someone who ran nothing.
- **Lexicon** — anything that earned a name (also add it to the table above).
- **Suggests next** — the follow-up, scheduled before the enthusiasm decays.
-->

### E-replay · why explore children didn't look like their parent (2026-11-08, agent)
- **Hypothesis** — the `.npz` replays fine; the render settings around it don't.
- **Ran** — 10 random recent sd15 images × {same seed, 3 fresh seeds, ±0.3 perturbation, the dashboard's old explore}, then a one-factor ablation on 4 of them. No labels (agent-judged).
- **Numbers** — same-seed replay MAE 0–1.8/255. Old explore: 0/10 seeds kept their look, all drifted to one shared saturated look.
- **Learned** — checkpoint ≫ size ≈ CFG negative ≫ scheduler. Children must replay the parent's settings → [reference/render-replay.md](../reference/render-replay.md).
- **Suggests next** — label a batch of explore children with `--stay-local` vs fresh noise; which reads as "more of this"?

### E-universe · sampling inside a hand-curated universe (2026-11-08, agent)
- **Hypothesis** — a fitted Gaussian can't make "more of" a curated set; member-anchored sampling can.
- **Ran** — 5 pseudo-universes (7–11 members) + the 53 ★ favourites; 9 strategies × 8 renders each; eye count + CLIP proxies. Pool of 500 renders kept out of the gallery.
- **Numbers** — eye in-look out of 40: fit T1.7 sidebar 0, fit T1 anchor-settings 17, slerp 0.5 anchor-seed 32. CLIP nn-member share .39 → .78 (members .83).
- **Learned** — members are independent directions; a fit draw is near none of them. The noise seed carries as much look as the conditioning (CLIP 0.67 fresh vs 0.99 same seed). → [reference/collection-sampling.md](../reference/collection-sampling.md)
- **Suggests next** — the human label: run 🪐 on a real tagged universe at spread 0.3 / 0.5 / 1.0 / 1.5 with one experiment id each and score them; does the keeper-rate fall off with spread, or do the wide ones find new pockets?
