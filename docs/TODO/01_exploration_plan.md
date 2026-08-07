# 01 · The Exploration Sprint — a multi-week systematic hunt through SD1.5 conditioning space

**Status: Phase 0 (instrumentation) shipped. Phase 1's *code* has shipped — every
corrected sampler and knob exists, defaults unchanged
([reference/sampler-corrections.md](../reference/sampler-corrections.md)); the E00–E04
*experiments* are the next thing to run. Phases 2–4 not started.**
Source material: `algorithm_report.html` §13 (24-experiment backlog), `distribution_report.html`
(the measured corpus, 4,144 prompts through juggernaut_reborn + base-checkpoint control),
the griffonage "numerical prompting / PROC skew" writeup, and Goodfire's Block-Sparse
Featurizers work (concepts as multi-dimensional manifolds, not 1D directions).

---

## 0 · The compass — read this before every session

**This is an art project, not a statistics course.** The distribution report found real,
large modelling errors in the shipped samplers (the past-EOS mixture, the dead sequence-axis
coherence, N(0,1) mass in PC1's empty gap). We fix them — but *not* because closeness to the
corpus distribution is the goal. We fix them because:

1. A sampler that wastes 57% of its entropy budget on a subspace the corpus treats as rigid
   (fig 09) is spending its weirdness where the UNet can't read it as *intentional* weirdness.
2. Coherent knobs (ρ, length, radius, cluster) are **navigable**; incoherent noise is not.
   You can't steer a sampler whose failure mode is uniform static.
3. The corrected samplers become the *launch pads*. The interesting images live in
   extrapolation territory (d > 1.3, where zero real prompts exist) — you want to arrive
   there by a controlled trajectory, not by accident.

**The final judge is always Xander's subjective aesthetic resonance with a set of images
sampled a certain way.** Not the taste model, not the novelty score, not any statistic.
Metrics exist to *compress labeling effort into reusable direction*, never to overrule the
eye. It rode in the labeling UI as a banner for a while and came back off once
read — a standing rule doesn't need restating on screen four hundred times.

Corollaries that shape everything below:

- **Optimize for the tail, not the mean.** A strategy producing forty 3s and ten 9s beats
  one producing fifty 6s. Track keeper-rate (% ≥ 7) and P90, not just mean label.
- **Broken samplers stay in the toolbox.** The diagonal's 0.00 row coherence is a *texture*
  (77 unrelated prompt-summaries as one tensor). Every "fix" ships as a new option beside
  the old behavior, never a replacement.
- **One variable at a time, everything else frozen** (the griffonage method: same seeds,
  same checkpoint, same knobs, vary exactly one thing). Paired comparisons are the only
  batches that produce learnings instead of vibes.
- **Name and catalog what you find.** Griffonage's "283ish / anti-605ish" lexicon is the
  model: a discovered direction/region/recipe gets a name, a one-line description, and an
  entry in the ledger, so it becomes a reusable word in the project's vocabulary.

---

## 1 · The loop we are building

```
  ┌─▶ EXPERIMENT (one strategy / one sweep · ~50 images · fixed seed panel)
  │        scripts run through the dashboard queue, tagged with an experiment id
  │
  ├──▶ LABEL (the new 0–9 labeling page · ~5 min per batch)
  │        every label appended to the project-wide dataset, forever
  │
  ├──▶ LEARN (report card per experiment + 3-line ledger entry)
  │        keeper-rate, P90, label histogram, novelty, per-knob breakdown
  │
  └──── DECIDE (next experiment: follow the gradient, or jump to new terrain)

  … and once the dataset is big enough (~500+ labels):
  TRAIN a CLIP/DINO score regressor on all accumulated labels
  → pre-rank candidates, propose experiments, CMA-ES fitness → self-improving loop
  (the human stays the judge; the model only decides what gets rendered first)
```

Cost reality: ~50 images at 512×512/30 steps ≈ 25–35 min on the 3090; labeling ≈ 5 min.
2–3 experiments per day is comfortable; a week is 10–15 experiments ≈ 500–750 labels.

---

## 2 · Phase 0 — Instrumentation — **done**

The referee is built. Full mechanics: [reference/labeling.md](../reference/labeling.md).

- **The label page** at `/label` — its own browser tab, reached from the dashboard's
  `🏷 Label (N) ↗` button (`features/labeling/` + `/api/label*` in `app.py`).
  Keyboard-first 0–9 scoring, queue frozen once fetched, stable-hash (not re-shuffled)
  order, blind by default. Dataset at `labels/labels.jsonl` — append-only,
  latest-wins, **git-tracked**.
- **Queue builder**: filter by experiment / backend / checkpoint / resolution / folder /
  source / sampler + a rendered-since window, with per-option `unlabeled/total` counts
  from `/api/label/facets`.
- **Experiment identity**: `--experiment`/`--hypothesis` on `generate`/`explore`/
  `evolve_favorites`, id in every sidecar, manifest at `outputs/experiments/<id>.json`.
  Sidebar fields for both; the sweeps don't take the flag (no per-image sidecar).
- **Fixed seed panel** = `--seed 1000 --n 16` (image *i* uses `seed + i`), named in
  `labels.SEED_PANEL`, hinted in the sidebar, badged in the queue + report.
- **Report card**: `scripts/experiment_report.py` → `experiment_report.html`.
  Keeper-rate/P90 first, histogram, novelty + within-batch near-dupes (from the caches
  `resonance.py` writes), per-knob breakdown of what was actually swept, ranked thumbs.
- **Ledger**: [02_experiment_ledger.md](02_experiment_ledger.md), with the Lexicon table.
- Torch-free core + tests: `semantic_anarchy/labels.py`, `tests/test_labels.py`.

**Still open here** — the DINOv2 feature cache (`outputs/features/dino.npz`) beside the
existing CLIP one. Deferred deliberately: it only pays off at E16, and the CLIP cache
already covers the novelty/duplicate columns the report card needs today.

---

## 3 · Phase 1 — Fix the measured mismatches (the "core upgrades" wave)

**The code has shipped; the experiments have not.** Every knob below exists, is off by
default, is covered by tests and is exposed in the sidebar —
see [reference/sampler-corrections.md](../reference/sampler-corrections.md) for what each
one measured and how it works. What remains is the part that was always the point:
*rendering with them and labeling the result.*

**E00 · Baseline census.** Before using any of the new knobs: ~50 images across the
current samplers (diagonal T∈{1,1.4,2}, pca T∈{1.4,2}, blend 0.5, hybrid), seed panel,
labeled. This is the reference row every later experiment is judged against — and the
first real content of the labels dataset.

**Prerequisite for E01/E03: re-mine.** The length split, the empirical PCA head and the
radius band are recorded at fit time. The existing `outputs/dist*` and
`xander_prompts__*` fits predate them; the dashboard's distribution card says so, and the
CLI warns rather than silently ignoring the knob. Re-encode the working corpus first.

**E01 · Length-conditional fit.** `--length-mode corpus` (draw each sample's content
length from the histogram) vs `off`, seed panel, at a couple of temperatures — does
drawing from one lobe instead of the gap between them actually read better?
Then the semantic sweep, which is the bigger prize: `--length ∈ {5, 15, 30, 50, 76}` at
fixed T, one batch each. Prompt length was PC1/PC2 in disguise (|r| 0.69/0.64).

**E02 · Row-coherent diagonal.** **ρ is a brand-new aesthetic axis nobody has ever seen
images from.** Sweep `--rho ∈ {0, 0.33, 0.66, 0.9, 1.0}` at T ∈ {1.2, 1.8}: ρ=0 is today's
static, ρ≈0.66 is corpus-like, ρ→1 is one deviation smeared through the whole sentence
(the statistical cousin of the mantra, §5). Then the E01×E02 interaction, since ρ shares
its deviation within the content span only.

**E03 · Empirical PC1/PC2 head + radius band.** Two independent A/Bs on the pca sampler:
`--empirical-head 2` vs off, and `--radius-band` vs a fixed `--target-distance`. Also
`--radius-scale ∈ {1.0, 1.3, 1.6}` — the band's shape held while its centre moves out.

**E04 · Two-temperature on/off-manifold split — the diagnostic.** `--sampler split`, four
corners of (`--temp-on`, `--temp-off`) ∈ {0.5, 2} × {0.5, 2}, one batch each, seed panel.
Answers the deepest open question — **does the good weirdness live on the manifold, off
it, or on the diagonal between?** — and every later strategy inherits the answer. Run the
same day as E02: they probe the same structure by projection vs by construction.
Note the "manifold" is only the *retained* rank (512 axes by default), so re-run at
`--components 0` if the answer looks rank-sensitive.

**Hygiene — done.** The noise floor is now derived per fit (Marchenko–Pastur on the
shuffled null, no second PCA) instead of a hardcoded 416; `--equalize` stops its band
there; `--comp-lo` past it warns; truncation stays off by default; and the sd15
`--neg-mode` no-op is fixed (the sweeps and explore/evolve never passed `dists` to
`generate`, so `mean`/`zeros` silently fell back to the house negative).

**Gate at the end of Phase 1:** compare keeper-rates E00 vs E01–E03. If the fixed
samplers *don't* beat baseline aesthetically, that is itself a headline finding (the
statistical mismatches were doing artistic work) — write it in the ledger and lean into
the broken modes as deliberate textures. Either way, the knobs (length, ρ, radius-band)
remain — they're navigation instruments regardless of which setting wins.

---

## 4 · Phase 2 — The high-gain levers: driving the UNet differently

Everything in conditioning space gets multiplied by guidance ≈ 7.5 in ε-space before it
reaches pixels. These are the cheapest large effects in the system (backlog lever 3).

**E05 · (T, g) response grid (backlog #2).** Zero new code. T ∈ {0.8, 1.2, 1.8} ×
g ∈ {1, 3, 7.5, 15, 30} + the exotic corners: g ∈ (0,1) (the sampled conditioning as a
whisper) and **g < 0 (anti-guidance — denoise *away* from the sample; genuinely unexplored
and costs one afternoon)**. 8 panel seeds per cell, report card contour of keeper-rate.
Calibrates the baseline every other experiment sits on.

**E06 · Fabricated negatives (backlog #1).** The negative branch is a free second
embedding slot, currently always the empty prompt on sd15. Three modes, in order:
(a) `neg = μ` (corpus mean) — guidance amplifies "how this sample differs from the average
prompt", actively suppressing generic content (port from SDXL);
(b) aligned negative `neg = μ − γ·(c − μ)` — amplifies the sampled deviation itself
in ε-space *without* raising T in embedding space (where high T degrades structure);
(c) `neg = an independent hot sample` — every image a tug-of-war between two anarchic
points (the old eden fabricated-`uc` trick reborn).
Run at E05's best g. Later composition: neg = −(taste direction) once E15 exists.

**E07 · Conditioning schedules (backlog #3) — the combinatorial explosion.** All 30 steps
currently see one c, but diffusion decides composition early and texture late. Via
diffusers' `callback_on_step_end` (mutate `prompt_embeds`, no fork): **hot→mild** (alien
layout rendered competently — the likeliest grail), **mild→hot** (normal scene, alien
surfaces), continuous slerp c(t) (a latent walk inside one image). Every pair of samplers
becomes an (early, late) grid cell; start with the Phase-1/2 winners as endpoints.
Schedule the negative too (composes with E06).

**E08 · The dream loop (backlog #4, optional this phase).** Iterated img2img: each output
becomes its own init (strength ≈ 0.55) with freshly sampled c each round, 5–8 rounds, keep
whole trajectories. Attractor-seeking drift through image × conditioning space.

---

## 5 · Phase 3 — New terrain: operators that don't approximate anything

These aren't corrections — they're conditioning states no text (and no corpus statistic)
can produce. This is where the map ends. All ship as new sampler modes / small scripts;
each gets a name in the lexicon if it earns one.

**E09 · PROC-skew channel sweeps (from griffonage — the direct import).** Add a constant
δ to one channel (one of 768 columns) across all 77 rows of an otherwise-normal sample:
`c[:, ch] += δ`. Griffonage found single columns carry consistent, nameable qualities
("283ish", "anti-605ish") that generalize across prompts, seeds, and even *models*.
Protocol (theirs, adapted): fix one base sample + panel seeds; sweep δ ∈ {±3, ±6, ±9}·σ_ch
over a batch of random channels; magnitude-gradient strips for channels that hit;
**name the winners and add them to the lexicon**. Note: fig 11 says UNet *sensitivity* is
flat across channels — that measures variance readability, not what a large constant
*bias* does; these are different questions, which is exactly why this needs images, not
statistics. Labels build a per-channel leaderboard for free. Variants: skew content rows
only; skew in PCA basis instead (constant offset along axis k = a walk endpoint of E11).

**E10 · Sparse axis cocktails (backlog #8).** Real prompts are sparse in concept space; a
999-component Gaussian draw is diffuse soup. Pick K ∈ {2…8} axes (< 416), amplitude
±(3–6)·s_j each, zero elsewhere — few strong alien concepts instead of uniform
strangeness. The label-driven follow-up is the point: a per-axis leaderboard → bias
cocktails toward proven axes → a bandit over basis vectors, taste-directed with no model.

**E11 · Manifold surfaces & grid sheets (the BSF import, part 1).** Goodfire's core
finding (BSF paper, arXiv 2606.25234): visual concepts live on **curved 2–4-dimensional
manifold regions**, not 1D directions — and their strongest empirical claim is directly
usable: *arbitrary directions degrade generation, but movement confined to high-density
regions of a manifold stays semantically coherent even at large displacement*. Three
translations, cheapest first:
(a) **Grid sheets** — take 2 well-labeled images (or 2 lexicon directions), span a local
2D patch (SLERP × orthogonalized second axis), render an n×n contact sheet, label cells
coarsely. Coherently-varying regions are manifold patches worth naming; shattering
regions are cliff edges. The standard artifact for "what is this neighborhood like?".
(b) **Density-constrained bold steps** — a new post-op beside `retarget()`: take a *big*
step (high T, long walk), then pull back toward the nearest high-density region (kNN
density over the corpus + labeled keepers) instead of pinning radius only. Their recipe
for large-but-coherent displacement.
(c) **A 2D SOM map of the corpus** — fit a Kohonen map (torch-free, fits the
`neighborhood()`/`walk()` seam), walk its waypoints instead of straight SLERP so paths
follow the *curved* manifold rather than cutting through the low-density interior.
Natural future UI: a clickable 2D map of the corpus in the dashboard.

**E11½ · Block-sparse concept dictionary (BSF part 2 — ambitious, gate on E11 results).**
Fit a small Grassmannian block-sparse featurizer on the flattened corpus embeddings
(G ≈ dozens of blocks, b = 2–4 dims, k ≈ 4 active; signed codes, no ReLU; tiny training
job — 4,144 rows). This replaces "one global PCA subspace" with "a sum of a few small
concept manifolds", and yields a genuinely new two-level sampler the current zoo can't
express: **choose the support** (which k blocks fire — discrete, where novelty lives:
recombine blocks that never co-fired in the corpus) **then sample each block's
coordinates from its own empirical density** (continuous, stays inside a real mode —
sidesteps the PC1-gap problem by construction). Out-of-distribution *support* with
in-distribution *coordinates* is the paper's formula for coherent novelty — arguably the
statistical formalization of what this whole project is chasing. Also gives temperature a
two-resolution split: scale block *norms* (concept intensity) separately from *intrinsic
coordinates* (within-concept variation). Caveat from the authors: untested on
text-encoder space, and 4k rows is small for wide dictionaries — treat as an experiment,
not a bet.

**E12 · Cluster/mixture sampling (backlog #15 — upgraded by measurement).** GMM BIC picks
k=30; the corpus genuinely is a mixture. Fit k-means/GMM on (length-residualized — E01
owns the length split) PCA coefficients. New moves: sample *within* a far-from-center
cluster (themed anarchy — all draws share a flavor); sample the **chord between two
cluster centers** ± noise (hybridizing prompt *regions* instead of prompt pairs);
per-cluster temperature maps (which neighborhoods tolerate heat?). Clusters that produce
distinctive imagery get lexicon names. When choosing k / PCA rank here, the BSF paper's
description-length criterion (torch-free, fully specified in their Eq. 5/6) gives a sharp
interior optimum where scree plots and raw R² stay mute — worth the afternoon.

**E13 · Token-axis operators (backlog #9, #10, #11).** Three cheap ones in one session:
(a) **mantra** — one plausible content row tiled across all content positions, a chord of
2–3 rows, or a row dissolving into noise along the sequence; (b) **temperature profiles**
— T as a 77-vector: smooth random spline profiles ("where in the sentence the madness
lives"), or one-hot (calm sentence, 3 positions at T=4); (c) **mix_latents resurrected**
— per-token row crossover from n parents, interesting regime n ∈ {2…8} (large n is just
diagonal incoherence with nicer marginals). Profiles are nameable, starrable objects.

**E14 · Distribution arithmetic (backlog #16).** Mine contrast corpora (minutes each):
`dist_boring` (500 dull literal captions), `dist_weird` (surrealist poetry, glossolalia,
unicode soup). Then operate on *fits*: repulsion (project boring-vs-base LDA directions
out of every deviation; or neg = μ_boring — guidance flees boredom directly), fit
interpolation μ + β(μ_weird − μ) as an "anarchy prior" slider.

---

## 6 · Phase 4 — Taste geometry & the self-improving loop

By now the labels dataset has 500–1500 graded rows spanning many strategies — far richer
than binary stars. This phase converts it into instruments. (The human stays the judge;
models only decide what gets *rendered first*.)

**E15 · Contrastive steering vector, graded edition (backlog #12).** Regress labels (not
star/non-star) on top-50 PCA coordinates → direction w of steepest taste ascent.
Every sampler gets `c ← c + α·w`; α is signed — **−α·w is anti-taste**, exactly what
novelty hunting wants when the gallery starts repeating. Recompute in seconds as labels
grow. Also feeds E06: neg = μ − α·w makes guidance itself climb the taste gradient.

**E16 · The score regressor (the project-wide payoff of every label).**
`scripts/train_regressor.py`: features = CLIP ViT-L/14 (already cached by resonance) ⊕
DINOv2 (the cache still open in §2) → ridge regression first, tiny MLP only if it earns it.
Evaluate honestly: held-out Spearman + per-experiment generalization (train on E00–E10,
test on E11+ — does taste transfer to unseen strategies?). Ship when Spearman > ~0.5:
- report cards gain a predicted-score column for unlabeled images;
- the labeling queue gains a "model-uncertain first" ordering (label where the model
  learns most — active learning for free);
- `resonance.py`'s logistic head gets superseded by the regressor in the hunter.

**E17 · CMA-ES in coefficient space (backlog #14).** 40-dim PCA coefficient space,
fitness = w₁·regressor + w₂·novelty, population 24, ~15 generations ≈ 360 decodes — one
overnight run. Unlike the hunter (which samples strategies) this *climbs*. Seed from the
labeled winners. Human-audit the result: label the final population next morning; if the
optimizer's 9s aren't your 9s, that gap is the most valuable label data there is.

**E18 · Knob-response surface (backlog #21).** Every label row carries its full knob
vector. Fit label ~ knobs (logistic/GBM) over the whole ledger; print the argmax cell and
the steepest uphill direction from well-explored territory. Feeds the next week's
experiment queue — this is "gradient descent in idea space" made literal.

**E19 · Legibility-cliff mapping (backlog #20).** Per direction (lexicon entries, cocktail
axes, steering vector): render at d ∈ {1.0 … 3.0}, find where coherence collapses (CLIP
similarity to corpus, or regressor confidence). Keepers should cluster just before their
cliff; per-direction cliffs turn shell targeting from a global knob into an adaptive one.
Remember: everything past d ≈ 1.3 is pure extrapolation — which is the point.

**The standing loop (steady state after week 4):** each week — the response surface + the
regressor propose ~⅓ of experiments (exploit), ~⅓ come from the ledger's "suggests next"
lines (follow-up), ~⅓ are wildcards from Phase-5 terrain nobody proposed (explore).
Every batch labeled, every label compounds. Retrain the regressor weekly; audit it
monthly against a blind-labeled batch so it never silently drifts from the eye it serves.

---

## 7 · Week-by-week shape (revise against the ledger, don't obey)

| Week | Focus | Exit criterion |
|---|---|---|
| **1** | Phase 0 complete + E00 baseline + E01 length-conditional | Labeling loop airtight; first 150 labels banked; length knob live |
| **2** | Finish Phase 1 (E02–E04 + hygiene) + start Phase 2 (E05 grid) | The four-corner manifold question answered; (T,g) map drawn |
| **3** | Phase 2 (E06 negatives, E07 schedules) + first terrain (E09 proc-skew, E10 cocktails) | ≥1 lexicon entry that reliably produces keepers; schedules judged |
| **4** | Phase 3 terrain (E11 grids, E12 clusters, E13 token ops, E14 arithmetic as time allows) | Grid-sheet workflow standard; ~800+ labels banked |
| **5** | Phase 4: regressor + steering vector + CMA-ES overnight + response surface | Regressor Spearman known honestly; standing loop running |
| **6+** | Steady state: ⅓ exploit / ⅓ follow-up / ⅓ wildcard, weekly retrain | The loop feeds itself; this doc drains into reference docs |

Deliberately unscheduled (pull in when a week under-runs or the ledger points at them):
E08 dream loop, E11½ block dictionary, SDXL/Flux transfer (backlog #17), dual-stream
dissonance (#18) — the cross-model cards are a natural Sprint 02 once the SD1.5 loop is
humming. One film-side dessert from the BSF paper: probe **closed loops** in conditioning
space (sweep a periodic 1-parameter prompt family — clock times, color wheel, compass
directions — through the encoder, fit the harmonic structure of the resulting loop) →
periodic travel paths for the Timeline, films that return to their first frame with no
seam.

---

## 8 · Standing experiment protocol (copy into each ledger entry)

1. **Hypothesis** — one sentence, falsifiable by eye ("row coherence ρ≈0.66 reads as more
   *intentional* than ρ=0 at the same T").
2. **Design** — strategy + the ONE swept variable + fixed everything else; seed panel iff
   comparative; ~50 images (25–35 min GPU).
3. **Run** — through the dashboard queue, `--experiment E##-slug` on every image.
4. **Label** — same day, blind to knobs where feasible (collapse the knob readout).
5. **Report** — `experiment_report.py E##`; look at the *tail*, not the mean.
6. **Ledger** — 3–6 lines: learned / lexicon additions / suggests-next. If a result is
   surprising, schedule the follow-up before enthusiasm decays.

## 9 · Implementation seams (where code lands — keep the architecture)

- Fit/samplers: `semantic_anarchy/distribution.py` (+ tests — stays torch-free);
  new knobs in `cli_args.py` only; every knob = one `schema.ts` entry in the dashboard.
- Labeling: `webui/frontend/src/features/labeling/` + `/api/label*` endpoints in
  `webui/app.py` (orchestration only, allow-lists, no torch); dataset at
  `labels/labels.jsonl` (git-tracked — labels are the one non-regenerable output).
- New generation behaviors = new/extended `scripts/*.py` + a `build_argv` branch —
  never inline model code in app.py.
- Conditioning schedules touch `pipeline.py` (the one place torch belongs).
- Every image-producing path keeps writing both sidecars (`.npz` + `.json`) — the
  sidecars are what make labels, report cards, steering vectors and CMA-ES possible.

*When a phase ships: run the CLAUDE.md/global docs condensation pass — durable findings
migrate to `docs/reference/`, this plan shrinks, the ledger stays.*
