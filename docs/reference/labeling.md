# The labeling loop — experiment identity, the dataset, the report card

The referee of the exploration sprint. A batch is only an experiment once it can
be tagged, labeled and measured; this is the machinery that makes each of those
one step.

```
  generate --experiment E01-length   →  every .json sidecar carries the id
                                        outputs/experiments/E01-length.json
  /label  (0–9, keyboard, own tab)   →  labels/labels.jsonl  (git-tracked)
  scripts/experiment_report.py       →  experiment_report.html
  docs/TODO/02_experiment_ledger.md  →  3–6 lines of what we learned
```

The compass behind all of it: *the final judge is subjective aesthetic
resonance.* Metrics exist to compress labeling effort into reusable direction,
never to overrule the eye. (It was on the page as a banner at first and came
back off — a sentence you have read four hundred times is screen space the image
should have.)

## Experiment identity

`--experiment <id>` on `generate.py` / `explore.py` / `evolve_favorites.py` (the
scripts that write per-image sidecars — a contact sheet has no sidecar to put an
id into, so the sweeps don't take the flag). `--hypothesis "<one sentence>"` goes
with it.

The id is slugged once, by `labels.clean_experiment_id`, wherever it enters —
CLI, dashboard, dataset — so `"E07 · negatives"` is `E07-negatives` everywhere.
Two things happen with it:

- it lands in every image's `.json` sidecar, which is how it reaches a label;
- `cli_args.record_experiment` writes/extends `outputs/experiments/<id>.json`
  (argv, backend, checkpoint, dist, seed, whether the seed panel was used).
  Re-running an id **appends a run** and keeps the first hypothesis — an
  experiment is usually several batches, and a later batch shouldn't erase why
  the first one ran. The manifest is scaffolding, hence `outputs/`; the labels
  carry the id independently of it.

The manifest is written *after* the batch seed is resolved, so it records the
noise the run actually used rather than the `None` that meant "draw one".

### The fixed seed panel

`generate.py` seeds image *i* with `batch_seed + i`. That makes the panel exactly
one flag pair — **`--seed 1000 --n 16`** → image seeds 1000–1015 — which is why
`labels.SEED_PANEL` is defined as a base+count rather than an arbitrary list, and
why `used_seed_panel` accepts `n > 16` (a superset still pairs) but not `n < 16`.

*Comparative* batches (A/B on one variable) must use it or the noise difference
swamps the idea being tested. Free exploration uses random seeds. The sidebar's
Seed hint says which of the two you are currently set up for; a batch that used
the panel gets a ⧉ badge in the queue selector and the report card.

## The dataset

`labels/labels.jsonl` at the repo root, **committed to git** — see
[labels/README.md](../../labels/README.md) for why it isn't under `outputs/`.

Schema, score range, keeper threshold and the summary maths all live in
`semantic_anarchy/labels.py` (torch-free, stdlib only), so `webui/app.py`,
`scripts/experiment_report.py` and the tests cannot disagree about what a label
is. `SA_LABELS_FILE` relocates the file.

Two invariants worth not breaking:

- **Append-only, latest wins.** `append_label` opens in `a` mode and writes one
  line per call, so concurrent writers can't interleave; `latest_by_rel` resolves
  relabels. Nothing in the codebase rewrites or reorders the file.
- **A record is self-contained.** `make_record` snapshots the experiment,
  backend, `ckpt_slug`, `dist`, `distance`, seeds and the sampler `knobs` dict
  out of the sidecar. Wiping the gallery costs images, never data points.

Summary statistics are deliberately tail-weighted — `keeper_rate` (share ≥ 7) and
`p90` first, mean last. Forty 3s and ten 9s beats fifty 6s, and only the tail
metrics can tell those apart. `tests/test_labels.py` asserts exactly that case.

## The label page

Its own browser tab at **`/label`**, not a gallery tab — labeling is a different
mode of attention from generating, it wants the whole window, and you keep it
open beside the dashboard while the next batch renders. The way in is the
`🏷 Label (N) ↗` button in the dashboard's action row (`LabelLink.tsx`), badged
with how much of the gallery is still unlabeled.

Routing is two pages and no router: `app.py` serves the same bundle at `/` and
`/label`, and `main.tsx` picks the shell off `location.pathname`. A route of its
own is required because `StaticFiles(html=True)` only answers `/` with
index.html.

| file | what it is |
|---|---|
| `LabelApp.tsx` | the shell + the page's ONE status line (cursor, controls, dataset numbers) |
| `QueueBar.tsx` | *which* images — facets, time window, scope, order (collapsed by default; the summary names the active filters so the closed state still reads) |
| `LabelPage.tsx` | *the* image — keyboard, scoring, prefetch, knob readout |
| `selection.ts` | `useLabelSelection()` — the store → the exact server query |

### Choosing the queue

`QueueBar` is a row of equality filters over dimensions each image already
carries, plus a window on when it was rendered:

**experiment · backend · checkpoint · resolution · folder · made-by · sampler**,
then **scope** (unlabeled / all / labeled), **starred only**, **order**
(shuffled / newest / oldest), and **rendered** (all time · 1h · 6h · 24h · 7d ·
30d, or an explicit from/to).

`GET /api/label/facets` supplies the options **with counts** (`unlabeled/total`),
computed over the whole labelable set — so a pick never makes another option
disappear; how many images a combination actually matches is reported live by
the queue itself, in the panel's own summary line. The `experiment` picker is
the one hand-built facet: it comes from `/api/experiments` instead, because that
knows each id's hypothesis (hover) and whether it ran on the seed panel (⧉).

Two things that look like details and aren't:

- **A relative window is frozen to an absolute instant when clicked.** A `since`
  that tracked the clock would change the query key on every render and refetch
  the queue forever.
- **The queue panel's open state is controlled from the store.** React re-applies
  a hard-coded `open` on every render, so a panel folded away by hand would
  spring back open the next time a label lands and the facet counts refresh.

Server-side, `_label_index()` walks `outputs/**/anarchy_*` across every
extension in `io_utils.IMAGE_EXTS` (that prefix is exactly the set with
per-image sidecars — a label on anything else would record no knobs), deriving
each row's facets from the sidecar and, for sizes older sidecars never recorded,
the file's own header (`_image_size`, PNG IHDR or JPEG SOF). Rows are cached on
mtime, so a facet refresh over 10k images is cheap.

A label's `rel` records the extension the image had when it was scored. That is
why the report resolves thumbnails by *stem* (`io_utils.find_image`) and the
dashboard's `_resolve_output_image` does the same: renders are JPEG since the
format switch, everything mined before it is a PNG, and neither a stale label
nor a bookmarked URL should 404 over that.

### Labeling

The page is a vertical budget in which the **image gets whatever is left**, so
everything else is one line: a single status bar (title, cursor, reload, knobs
toggle, key legend, dataset numbers), the collapsed-by-default queue builder, the
0-9 strip spread across the full width directly above the image, and a one-line
filename/knob readout under it. The image is `h-full w-full object-contain`, so a
512-square render fills the space rather than sitting marooned in the middle of it.

Keyboard-first: `0`–`9` score and advance, `←`/`→` navigate (relabeling allowed),
`Space` skip, `s` star, `k` knobs. One window listener — the page has no text
input, so digits can't be swallowed by a focused box; an open Radix popup does
win the keyboard, because its typeahead would otherwise read a score as a search.

Three decisions that are load-bearing:

- **The queue is fetched once and frozen** (`staleTime: Infinity`, no refetch on
  focus, no invalidation on submit). The page is a cursor walking a list; a
  background refetch would renumber the progress and slide a different image
  under a keypress already on its way. ↻ reload is the only thing that pulls in
  new arrivals.
- **The write is optimistic and the cursor advances immediately.** At one
  keypress per image, waiting on a round trip is the difference between fluent
  and unusable. `useSubmitLabel` patches the cached row.
- **Server-side order is a stable hash of `(seed, rel)`, not a shuffle.**
  Generation order correlates with everything (seed, sweep position), so labeling
  in it invites the eye to find a trend that isn't there — but re-shuffling on
  every fetch would teleport you mid-batch. Hashing means the order of what's
  *left* is unchanged as images drop out. ⤨ reshuffle bumps the salt on purpose.

Knobs are hidden by default (blind labeling); the readout is small type when
shown, so the label stays perceptual rather than analytical.

### Endpoints (`webui/app.py`)

| route | what it does |
|---|---|
| `GET /label` | the page itself (same bundle, no dashboard chrome) |
| `GET /api/label/facets` | every facet value with `count`/`unlabeled`, plus the gallery's mtime span |
| `GET /api/label/queue` | the selection: the seven facets + `scope`, `bucket`, `order`, `seed`, `since`, `until`, `limit` |
| `POST /api/label` | `{rel, score}` → one appended record |
| `GET /api/labels` | dataset summary: overall + one tail-weighted row per experiment |
| `GET /api/experiments` | every id the sidecars *or* the manifests know, with images/labeled counts |

`app.py` keeps its usual posture: allow-listed scope/bucket, equality-only
filters (nothing here interprets a user-supplied expression), path sandboxing on
every write, no computation, no torch. The sidecar cache it reads through
(`sidecar_for`, keyed on mtime) is shared with the gallery's distance column.

## The report card

`python scripts/experiment_report.py [ids…]` → a self-contained
`experiment_report.html` at the repo root (untracked, like the other reports).
Torch-free: numpy for the duplicate check, PIL only to shrink thumbnails, and the
page still builds without PIL.

Per experiment: keeper-rate and P90 first, then median/mean/best, the label
histogram (inline SVG — no plotting stack), median novelty and the share of
near-duplicates *within* the batch (both read from the caches `resonance.py`
already writes; the columns are simply absent when those don't exist), a per-knob
breakdown, and thumbnails ranked best-first.

The per-knob breakdown only prints knobs the batch actually *varied* — a constant
knob carries no information about the labels, so what remains is the swept
variable (or a confounder worth seeing). The near-duplicate share is the number
that catches a strategy producing fifty renders of one idea, however good those
fifty are.

## Where the pieces live

| piece | file |
|---|---|
| record schema, seed panel, summaries | `semantic_anarchy/labels.py` (+ `tests/test_labels.py`) |
| `--experiment` / `--hypothesis`, manifest writing | `semantic_anarchy/cli_args.py` |
| endpoints, queue ordering, sandboxing | `webui/app.py` |
| the page (`/label`) | `webui/frontend/src/features/labeling/` |
| sidebar knobs (id, hypothesis, seed-panel hint) | `webui/frontend/src/params/schema.ts` |
| report card | `scripts/experiment_report.py` |
| the dataset | `labels/labels.jsonl` |
| what we learned | `docs/TODO/02_experiment_ledger.md` |
