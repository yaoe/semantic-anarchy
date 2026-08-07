# The CFG negative branch

Promptless generation still runs classifier-free guidance, and CFG needs *two*
conditioning tensors: the sampled one, and something to push away from. That
second tensor is the only place a human-written string still enters an otherwise
promptless pipeline — which makes it worth pinning deliberately rather than
letting it default to whatever diffusers encodes for `""`.

## The house SD1.5 negative

```
nude, naked, poorly drawn face, ugly, tiling, out of frame, extra limbs,
disfigured, deformed body, blurry, blurred, watermark, text, grainy,
signature, cut off, draft
```

Lives in `semantic_anarchy/pipeline.py` as `SD15_NEGATIVE_PROMPT`. It is
**inherited, not invented** — it is the string the surrounding Eden SD1.5 stack
has always sampled with, recovered by tallying every recorded run config on this
machine:

| model family | runs with a recorded negative | using this exact string |
|---|---:|---:|
| SD1.5 lineage (`eden:eden-v1`, `dreamlike-photoreal-2.0`) | 1,153 | **1,140** (98.9%) |
| SDXL (`sdxl-v1.0`, `juggernaut_XL2`) | 82 | 0 — SDXL moved to a newer string |

It is additionally hardcoded at seven sites across `Eden/sd-lora-trainer`,
`diffusion_trainer` and `cog/eden-sd-pipelines` (where it survives, commented
out, as the pre-SDXL default in `eden/settings.py`).

Adopting it here means promptless samples land in the same aesthetic basin as
every other image ever rendered off these checkpoints. The visible difference on
a fixed seed is exactly what the string names: the empty-prompt negative happily
returns matted gallery prints with visible borders, captions and signatures,
because nothing is pushing against `out of frame` / `text` / `signature` /
`cut off` / `draft`.

## Which negative each backend uses

`neg_mode` is resolved in `cli_args.resolve_gen_defaults` and dispatched in
`SD15Backend._negative` / `SDXLBackend._negatives`.

| mode | tensor used | default for |
|---|---|---|
| `text` | the model's `negative_prompt`, encoded once and cached | **sd15** |
| `mean` | the fitted corpus mean — push away from the average prompt | **sdxl** |
| `empty` | the empty-prompt encoding | **sd2, flux2, krea2** |
| `zeros` | a zero tensor | — |

Only sd15 sets `SDModel.negative_prompt`; sd2 leaves it `None` (its OpenCLIP-H
encoder was never sampled with this string, and the tally above says nothing
about it), so `text` and `empty` collapse to the same thing there. `mean` and
`zeros` need a fitted distribution passed through `generate(dists=...)`; without
one they degrade to the backend default rather than failing.

Every image's `.json` sidecar records both the `neg_mode` and the effective
`negative` text it was made with, and `morph_film.py` reads the mode back so a
film renders its frames against the *same* negative as its own keyframes.

## Overriding it

Three layers, narrowest first:

* **Per run, the text itself** — `--negative "monochrome, greyscale"`, or the
  dashboard's Advanced → **negative prompt** box. Blank/omitted = the house
  default. Applies to sd15 and sd2 (`load_backend` sets it on any backend whose
  model has a `negative_prompt`; sdxl/flux2/krea2 have none and are skipped).
* **Per run, the mode** — `--neg-mode empty` (or `mean`/`zeros`), or the
  Advanced → neg-mode select. This is how you sample against *no* negative text.
* **Per machine** — `SA_SD15_NEGATIVE="..."` replaces the string;
  `SA_SD15_NEGATIVE=""` disables negative text entirely. Read by
  `pipeline.default_sd15_negative()`, and reflected in `/api/config` so the
  dashboard box shows whatever is actually in force.

### In the dashboard

The sidebar's neg-mode select defaults to **`mean`** (the corpus mean) on every
backend — that is a dashboard default only; the CLI still resolves a blank mode
per backend (`text` on sd15).

The negative-prompt box lives in the sidebar's **Advanced** section, below
neg-mode, and only appears for sd15/sd2 with neg-mode blank or set to `text` —
the other backends and modes use a tensor negative, so there is no text to show.
With the `mean` default in force it is therefore hidden until you pick a text
mode. It is prefilled with the real string (from `/api/config`, never a
copy hardcoded in the frontend) as a placeholder; clicking into it turns that
into editable text (`seedFromPlaceholder` in `params/schema.ts`, shared with
steps/guidance/width/height). Clearing the box restores the default.

The neg-mode select never says "auto": its blank entry is labelled with the mode
that backend actually resolves to (`NEG_AUTO`/`NEG_LABELS` in `schema.ts`,
mirroring `resolve_gen_defaults`), so the picker states the behaviour rather
than naming a fallback.

Free text is the one knob `webui/app.py` cannot allow-list, so it gets
`_clean_prompt` instead: it travels as a single element of a list-form argv (no
shell), newlines and control characters collapse to spaces so it can't forge
lines in the job log, and it is capped at `MAX_NEGATIVE_CHARS` (1000 — CLIP
truncates at 77 tokens long before that).

Every generated image's `.json` sidecar records the effective text under
`negative` (via `cli_args.effective_negative`), so the lightbox param dump shows
which words that image was actually pushed away from — `neg_mode` alone stopped
being enough once the text became editable.

## `uncond` is not the negative

`SDModel.uncond_embedding()` (the empty prompt) and
`SDModel.negative_embedding()` (the house negative) are deliberately separate,
and both are cached independently. `uncond` is the corpus's **geometric origin** —
`analysis.encode_corpus` saves it into `corpus_<backend>.npz` as the point every
distribution figure measures against. Repointing it at the negative prompt would
silently move the origin of every plot in the distribution report. Only
generation reads `negative_embedding()`.
