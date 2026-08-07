# The sampler corrections

The corpus autopsy (`semantic_anarchy/analysis.py` → `distribution_report.html`)
measured four places where the fitted model is provably wrong about the corpus it
was fitted to. Each has a correction in `semantic_anarchy/distribution.py`.

**They are knobs, not replacements.** Every default reproduces the original
sampler bit-for-bit (`test_defaults_reproduce_the_original_samplers_bit_for_bit`
is the guard). This is deliberate: the diagonal's 0.00 row coherence is a
*texture* — 77 unrelated prompt-summaries stapled into one tensor — and the whole
point of the project is that the statistically wrong settings sometimes win. The
corrections exist to make the model **navigable**, not correct.

None of them are on unless you ask. All of them require a distribution mined
since they existed; an older `.npz` loads fine and the CLI says which knob it is
ignoring rather than silently doing nothing (`cli_args.warn_sampler_args` /
`resolve_lengths`), and the dashboard's distribution card carries the same
warning.

## What was wrong, and what each knob does

### Length — the past-EOS mixture

CLIP pads to 77 with EOS, so token position *t* of the corpus is not one
population but two: the prompts still writing content there, and the prompts that
ran out and are padding. The report found that binary explaining up to **89%** of
a coordinate's variance. A single Gaussian fitted to a two-lobed mixture peaks in
the gap — the fitted model's densest region is where no prompt has ever been.

`fit(..., lengths=)` estimates every position's `(μ, σ)` twice, once per side,
and stores the corpus's own length histogram. `sample(..., lengths=)` then draws
each sample conditional on where its EOS falls. Positions whose thinner side has
fewer than `MIN_LENGTH_GROUP` members (position 0 is always BOS; the tail is
always padding) keep the pooled fit on both sides.

The bonus is the bigger prize: prompt length **was** PC1/PC2 in disguise
(|r| = 0.69 / 0.64), so `--length 60` — "sample me a 60-token image" — is the
single largest semantic dial the corpus owns.

- `--length-mode corpus` bootstraps the length histogram, so a batch has the mix
  of long and short prompts the corpus does. It is the **dashboard's default**
  (the CLI's is still `off`); a fit mined before the split falls back to `off`
  with a printed reason, so an old `.npz` still runs.
- `--length-mode fixed --length N` pins every sample. `--length N` alone implies it.
- CLI: `cli_args.resolve_lengths` draws once per batch and hands the same array to
  every named tensor, so a multi-tensor conditioning set stays coherent. The
  drawn length is recorded **per image** in the `.json` sidecar, because it varies
  within a batch in corpus mode.

### ρ — row coherence

The corpus's 77 token rows agree at **0.65**; diagonal samples agree at **0.00**.

`--rho` replaces the standard normal in the diagonal draw with
`√ρ·u + √(1−ρ)·v_t`, where `u` is drawn once per sample and shared across token
positions and `v_t` is fresh per position. Per-coordinate marginals are exactly
unchanged (`ρ + (1−ρ) = 1`); only the between-row correlation moves, and it lands
at ρ by construction.

ρ=0 is the historical static, ρ≈0.65 is corpus-like, ρ→1 is one deviation smeared
through the whole sentence. It shapes the **diagonal** draw only — so it reaches
`blend` and `split` through their diagonal halves, and `pca`/`hybrid` ignore it.

With length conditioning on, `u` is shared **within the content span only**:
padding positions are not part of the sentence, and smearing the sentence's
deviation into them just blurs the boundary the length split exists to sharpen.

### The empirical PCA head and the radius band

Two separate findings, both about the *shape* of a draw rather than its direction:

- **PC1 is bimodal** (it is prompt length in disguise), so an N(0,1) coefficient
  puts its densest mass in an empty gap. `fit` stores the corpus's own sorted,
  unit-variance coefficients along the leading `HEAD_AXES` axes; `--empirical-head K`
  draws the first K coefficients from those CDFs by interpolated inverse lookup.
  Only the axes actually selected by `--comp-lo`/`--components` are affected.
- **The corpus is a band of radii** (mean 0.99, sd 0.031, range 0.89–1.11 on the
  sd15 corpus) while every sampler produces a ~9× tighter spike. `fit` records
  each corpus embedding's own `distance()` gauge; `--radius-band` bootstraps a
  target radius per sample and lets the existing `retarget()` pin it.
  `--radius-scale` shifts the whole band outward. An explicit `--target-distance`
  is more specific and wins, with a warning.

### `split` — the on/off-manifold diagnostic

The retained PCA components are an orthonormal basis of the corpus subspace, so a
diagonal deviation decomposes exactly into `dev_on = (dev·Vᵀ)V` (what the corpus
could have produced) and `dev − dev_on` (what it never does). `--sampler split`
gives the two halves separate temperatures.

This is not a better sampler, it is an instrument: the four corners of
(`--temp-on`, `--temp-off`) answer *does the good weirdness live on the manifold,
off it, or on the diagonal between?* — and every later strategy inherits that
answer. `temp_on = temp_off = 1` is bit-identical to `diagonal`.

The "manifold" here is only what the fit **retained** (512 axes by default, not
the full N−1 rank), so a low-rank mine widens what counts as off-manifold — the
same caveat `--components` has always had.

## The noise floor

`EmbeddingDistribution.noise_floor_axes()` estimates how many PCA axes rise above
a shuffled-coordinate null, in closed form and without a second PCA. Shuffling
each coordinate independently destroys every correlation while preserving the
per-coordinate variances; the resulting Gram spectrum is Marchenko–Pastur with
ratio n/D, whose top edge sits at `total_var·(1 + √(n/D))² / (n−1)`.

It is mildly conservative, because MP assumes equal coordinate variances: on the
4,144-prompt sd15 corpus it says **379** where the measured shuffle null says
**416**. That is close enough for what it is used for:

- `--equalize` with no explicit `--components` now stops the axis band at the
  floor. Equalising means "express every selected axis at full strength", and past
  the floor there is no corpus structure to express — an unbounded equalised band
  spent most of its budget amplifying stored noise. An explicit `--components`
  still reaches as far as it is told to.
- `warn_sampler_args` flags a `--comp-lo` that starts at or past the floor, and
  notes when a band's tail extends beyond it. Warnings, never errors: riding the
  noise on purpose is a legitimate experiment.
- `mine_distribution.py` prints it, and it lands in each fit's `.meta.json`.

## What a mine records now

`mine_distribution.py` calls `Backend.token_lengths(prompts)` beside `encode` —
tokenizer only, no GPU, so it is free and unconditional rather than a flag. It is
implemented for the CLIP-family backends (`length_conditional = True` on sd15 /
sd2 / sdxl; the flow models mine Qwen hidden states at a fixed length and have no
EOS boundary to key on).

Beyond the original `mean`/`std`/`pca_components`/`pca_std`/`corpus_embeddings`,
a `.npz` now carries `mean_content`/`std_content`/`mean_pad`/`std_pad`,
`lengths`, `pca_head` and `corpus_distance` — four extra feature rows plus a few
N-vectors, negligible beside the PCA basis. Every one is optional on load, so
older files work untouched; `has_length_stats` / `corpus_distance is None` are
what the warnings key on. The `.meta.json` gains `has_length_stats`,
`has_radius_band` and `noise_floor_axes`.

Tensors without a token axis (sdxl's `pooled`) skip the length split
automatically, as does the very wide flow-model conditioning where four extra
full-size arrays would not pay.

## Where the knobs live

`cli_args.py` is still the single source of CLI truth, and now owns two helpers
that keep the scripts from drifting:

- `sampler_kwargs(args, lengths)` packs everything `EmbeddingDistribution.sample`
  takes. A test asserts it covers every parameter of that signature, so a new
  sampler knob reaches generate / the sweeps by being added to
  `add_backend_args` rather than copy-pasted four times.
- `neg_dists_kwarg(args, dists)` passes the fitted distributions to `generate()`
  on the backends whose negative branch can use them. Without it `--neg-mode mean`
  and `zeros` silently degraded to the backend default — which is what they did
  everywhere except `generate.py` until this landed.

Dashboard exposure is one `schema.ts` entry per knob as usual (ρ, prompt length,
the two split temperatures under Sampler; the empirical head under Advanced; the
radius band as a mode of the Reach group), plus the `split` sampler option and the
pass-through fields on `RunRequest` / `_common_sampler_flags` in `webui/app.py`.

Which of these knobs is actually live for a given sampler and action — and why
temperature, a target shell and the radius band are one control rather than
three — is [sampler-knobs.md](sampler-knobs.md).
