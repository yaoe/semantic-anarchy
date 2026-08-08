# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Promptless image generation ("Semantic Anarchy"): encode a corpus of prompts through a diffusion model's text encoder **once**, fit a distribution over the resulting conditioning tensors, then sample brand-new conditioning straight from that distribution and feed it to the UNet via `prompt_embeds`. The text encoder is never touched at generation time. Stock `diffusers` pipelines only — no SAE, no hooks, no forks.

## Commands

The repo uses a uv-managed Python 3.12 `.venv` (torch + diffusers + fastapi). Scripts are run as files, not as an installed package (each does `sys.path.insert(0, repo_root)`); there is no `pyproject.toml`.

The `Makefile` wraps the everyday ones — `make` alone lists them (`run`/`start`, `dev`, `build`, `restart`, `stop`, `logs`, `test`). Everything below still works directly.

```bash
pytest -q                                   # full suite (94 tests, torch-free, ~6s)
pytest tests/test_distribution.py -q        # one file
pytest -q -k pca_samples_lie_in_subspace    # one test
python scripts/demo_no_sd.py                # end-to-end demo on synthetic embeddings, no torch/GPU

# mine → generate (needs the full tier: torch + diffusers + weights)
python scripts/mine_distribution.py --backend sd15 --ckpt MODEL.safetensors \
    --prompts prompts_1000.txt --out outputs/dist
python scripts/generate.py --backend sd15 --dist outputs/dist --n 8
python scripts/temperature_sweep.py --backend sdxl --dist outputs/dist --sampler pca --temps 1,2,3,4

# fit a distribution on the latents of images you PICKED (no GPU, ~1s) and sample it
python scripts/fit_selection.py --backend sd15 --name keepers --images outputs/generated/*.jpg
python scripts/generate.py --backend sd15 --dist outputs/dist_fits/keepers --sampler pca --n 8

# same-latent hires fix (the default upscaler): enlarge, then re-run the tail of
# the ORIGINAL schedule with that image's own conditioning
python scripts/upscale.py --src outputs/generated/anarchy_sd15_7_000.jpg --factor 2.0 --denoise 0.3

# deep-dive the corpus statistics -> outputs/analysis/plots/*.png + stats.json
# first run encodes + caches the corpus (needs the ckpt); later runs are seconds
# --dist must be fitted on the SAME prompts, or figures 10/12 compare nothing
python scripts/analyze_distribution.py --prompts PROMPTS.txt --ckpt MODEL.safetensors \
    --dist outputs/dist_xander
python scripts/analyze_distribution.py --only 05,09      # iterate on two figures
python scripts/build_distribution_report.py             # -> distribution_report.html
# same prompts through different weights -> adds the checkpoint-robustness section
python scripts/analyze_distribution.py --outdir outputs/analysis_base --ckpt BASE.safetensors ...
python scripts/build_distribution_report.py --compare outputs/analysis_base

# tag a batch, label it in the dashboard's 🏷 tab, then measure it
python scripts/generate.py --backend sd15 --dist outputs/dist --seed 1000 --n 16 \
    --experiment E00-census --hypothesis "the reference row everything is judged against"
python scripts/experiment_report.py            # -> experiment_report.html (all labeled ids)
python scripts/experiment_report.py E00-census # one experiment

# latent travel film through keyframes -> outputs/films/<name>/<name>.mp4
python scripts/morph_film.py --name drift --refine none --frames-per 24 --fps 16 \
    --images generated/a.jpg generated/b.jpg generated/c.jpg

python -m semantic_anarchy.cli {demo,mine,generate,evolve} [args]   # thin dispatcher to scripts/

./webui/run.sh          # dashboard on the Tailscale IP:8800 (SA_HOST/SA_PORT/SA_PYTHON to override)
./webui/restart.sh      # safe detached restart; refuses while a GPU job runs (--force overrides)

# the dashboard frontend (Vite + React + TS). app.py serves webui/frontend/dist at /
cd webui/frontend && npm install && npm run build
cd webui/frontend && npm run dev      # :5173, proxies /api + /img to 127.0.0.1:8800
```

Model weights are never committed. `outputs/`, `*.npz`, `*.safetensors`, `init_images/` are gitignored — anything under `outputs/` is regenerable.

## Architecture

**Two tiers, enforced by lazy imports.** Everything that doesn't strictly need a GPU is pure NumPy. `torch`/`diffusers` are imported *inside* `Backend.load` / `pipeline.py` functions, never at module scope, so `fit`/`sample`/`save`/`load`, the plots, the evolution loop and the **entire test suite** run with neither installed. Preserve this when editing `backend.py`, `distribution.py`, `evolve.py`, `aesthetic.py` — a top-level `import torch` there breaks the tests.

**`distribution.py` — `EmbeddingDistribution`** is the statistical core and where most real logic lives. A per-coordinate Gaussian fit (`mean`, `std`) *plus* low-rank PCA of the corpus (`pca_components`, `pca_std`) *plus* a bounded subset of raw corpus embeddings. That triple is what makes the five samplers possible:

| sampler | draws from | notes |
|---|---|---|
| `diagonal` | independent per-coordinate Gaussians | the original model; off-manifold |
| `pca` | the low-rank corpus subspace | `temperature > 1` extrapolates coherently *outside* the corpus hull |
| `blend` | sqrt-weighted mix of the two **covariances** | `coherence=0` ≡ diagonal, `1` ≡ pca, bit-exact at the endpoints |
| `hybrid` | SLERP between two real corpus embeddings | needs `corpus_embeddings`; stays inside the hull |
| `split` | a diagonal draw cut into its PCA-subspace projection + the orthogonal remainder | separate `--temp-on`/`--temp-off`; `1,1` ≡ diagonal |

The fit also carries four **corrections** measured by the corpus autopsy — the past-EOS content/padding split (`--length-mode`/`--length`), row coherence (`--rho`), the empirical PCA head (`--empirical-head`) and the corpus radius band (`--radius-band`). All are off by default in the CLI (the dashboard sidebar starts length-mode at `corpus` and neg-mode at `mean`) and every CLI default is bit-identical to the original samplers, because the statistically wrong settings are textures the project keeps; a fit mined before they existed loads fine and the CLI says which knob it is ignoring. → [docs/reference/sampler-corrections.md](docs/reference/sampler-corrections.md)

**Reach is one control, not four.** `temperature` multiplies the whole deviation, but `retarget()` rescales a sample to an exact radius — which divides that factor straight back out, so a temperature set alongside `--target-distance` or `--radius-band` is *exactly* cancelled, not merely overridden (`hybrid` is the exception: its T weights jitter, not scale). Likewise `--temp-on`/`--temp-off` are multiplied by `temperature`, making it a redundant common factor under `split`. The sidebar therefore picks a **reach mode** (temperature | shell | band) and hides the knobs the current sampler/action cannot read — and `buildRunRequest` sends a hidden field at its *default*, so what the UI hides can never reach argv or an image's sidecar. → [docs/reference/sampler-knobs.md](docs/reference/sampler-knobs.md)

Also here: `distance()` (RMS z-score from the corpus center — the "how far off-grid" gauge, ~1.0 for a typical corpus sample, ~T for a temperature-T diagonal sample), `retarget()` (shell sampling: keep direction, pin radius), `neighborhood()`/`walk()` (local navigation), `refit_from_elites()` (evolutionary branches). `fit()` makes **two independent** choices, and conflating them is a trap: *dtype* drops to float32 when the feature dim exceeds 200k (flow-model Qwen conditioning — float64 copies won't fit), while the *PCA algorithm* keys on `n <= D`, i.e. fewer samples than coordinates, which every real corpus is. In that regime it eigendecomposes the (N,N) Gram matrix instead of SVD-ing the (N,D) data — identical top-k right singular vectors (`X = U S Vᵀ ⟹ X Xᵀ = U S² Uᵀ`, `Vᵀ = S⁻¹ Uᵀ X`), ~16× faster for sd15's 4k×59k shape, and the only tractable option at all for the flow models.

**`backend.py` — the model-agnostic seam.** A model is described purely by its *named conditioning tensors*, and every script drives four generic verbs: `encode → fit → sample → generate`. One `EmbeddingDistribution` is fitted per named tensor, and all tensors are sampled with the *same* knobs so the conditioning set stays coherent.

| backend | tensors | pipeline |
|---|---|---|
| `sd15` | `embeds` (77,768) | `StableDiffusionPipeline` |
| `sd2` | `embeds` (77,1024) | same, OpenCLIP-H; subclasses `SD15Backend` |
| `sdxl` | `prompt_embeds` (77,2048) + `pooled` (1280) | `StableDiffusionXLPipeline` |
| `flux2`, `krea2` | `embeds` — multi-layer Qwen3 hidden states | flow models; run in a **separate `.venv-flux`** (diffusers ≥0.39) |

Adding a backend means: a `Backend` subclass (`tensor_names` + `encode`/`generate`), a `*Model` wrapper in `pipeline.py`, a `BACKEND_DEFAULTS` entry, and the name in `make_backend`, `dist_backend`, `cli_args.add_backend_args` choices, and `webui/app.py`'s `BACKENDS` allow-list. `dist_backend(name)` gives a model-less instance exposing only the NumPy verbs — that's what the tests use.

**Mining loads no denoiser.** `make_backend(..., encode_only=True)` → `encode_only_kwargs()` passes every skippable component (`unet`/`transformer`/`vae`/safety stack) to `from_pretrained`/`from_single_file` as `None`, which makes diffusers skip loading it — an SD1.5 single-file checkpoint becomes a ~0.4s, ~250MB-VRAM text-encoder load. It's the default for `mine_distribution.py` (`--with-unet` opts out; a partial load that fails falls back to the full pipeline with a logged reason). The fit that follows keeps `MAX_COMPONENTS`=512 PCA axes by default (`--components 0` restores the full `N-1` rank): each axis costs a whole feature row on disk, and past ~400 the spectrum is noise, so full rank turned a 4k-prompt sd15 corpus into a ~1GB file for no gain. Encoding runs in batches of `--batch-size` (default 8) through `_encode_batches`, which drives `Progress` from `progress.py` — a throttled one-line bar with rate + ETA, because the dashboard turns every `\r` into a log line and an unthrottled bar would bury the job log. For the same reason `quiet_truncation_warnings()` mutes diffusers' per-batch "input was truncated" warning (it inlines the dropped text) and `report_truncation()` states the count once instead.

**Distribution files are backend-namespaced.** Three naming layers stack, and `dist_paths.py` (torch-free) owns all three so the CLIs and the dashboard can't drift: a *base* prefix is what `--dist`/`--out` receive → `dist_prefix()` in `cli_args.py` namespaces it (`sd15` keeps the bare `outputs/dist`, everything else gets `outputs/dist_<backend>`) → `save_dists` appends `__<tensor>` per file for multi-tensor backends (`outputs/dist_sdxl__pooled.npz`). Each `.npz` has a `.meta.json` sibling. Evolved taste branches use the `outputs/dist_evolved*` prefix with the same rules, and fits made from picked images live at `outputs/dist_fits/<name>` with the same rules plus a `.fit.json` manifest naming the images. `dist_files(base, backend)` walks that stack forwards (what *would* exist), `base_from_npz` walks it back (which base a file belongs to).

**A prompt corpus keeps its fit beside itself.** A `.txt` mined via the dashboard writes to `<corpus>__<ckpt-slug>` *next to the .txt*, not into `outputs/` — so one corpus carries a separate fit per checkpoint (`xander_prompts__juggernaut_reborn.npz` vs `xander_prompts__v1-5-pruned-emaonly.npz`) and switching checkpoints switches which fit is in play rather than silently reusing the wrong one. `prompt_dist_base()` is that rule.

**`travel.py` — keyframe films, torch-free.** `interpolate` (`slerp`/`lerp`), `ease` (`smooth`/`smoother`/`linear`), `frame_plan(n_keys, frames_per, easing)` → the `(segment, t)` list for every frame (each transition starts ON its left keyframe and stops short of its right, plus one final frame at t=1, so no keyframe renders twice), and `noise_t(t, window)` for the narrower window the *init noise* travels in. `scripts/morph_film.py` is the only renderer: it reconstructs each keyframe's init latent from the recorded `image_seed` exactly as diffusers' `randn_tensor` would (cuda generator for sd15/sd2, cpu for sdxl), interpolates conditioning + noise, renders every frame through the raw pipe with `latents=`, and muxes with the first ffmpeg that actually has an H.264 encoder (`pick_h264_encoder`; a conda `--disable-gpl` ffmpeg on PATH has no libx264 — `SA_FFMPEG` pins one). Endpoints therefore come back pixel-identical; the middle never existed. sd15/sd2/sdxl only (`FILM_TENSORS`). **A film has ONE resolution** (keyframe 1's own, or `--width/--height`): a keyframe of any other size is re-rendered at the film's size from a differently-shaped noise draw, so it is *not* reproduced — `keyframe_size` reads the true size off the image (sidecars from evolve branches record none) and the script names every off-size keyframe in the log and in `film.json.offsize_keyframes`.

**Three upscalers, one endpoint.** `POST /api/refine`'s `engine` picks between them; the UI is `features/refine/RefineBar.tsx`, and each engine hides the knobs the others own.

| engine | script | what it does |
|---|---|---|
| `hires` **(default)** | `scripts/upscale.py` | same model, same latents: enlarge (16px-aligned) → img2img the LAST `denoise` fraction of the ORIGINAL schedule |
| `flux` | `scripts/refine_flux.py` | klein reference-regeneration in `.venv-flux` — a different model, a stronger prior, free to reinterpret |
| `sd` | `scripts/refine.py` | general img2img refine, optionally tiled (Ultimate-SD-Upscale style) |

`upscale.py` is the faithful one and takes only two knobs (`--factor`, `--denoise`); *everything* else — step count, guidance, scheduler, seed, conditioning — is replayed from the source image's own sidecars, so the pass adds detail without drift. `semantic_anarchy/upscale.py` holds the torch-free half: `target_size` (snap to `LATENT_MULTIPLE`=16, cap the long side), `denoise_steps`/`clamp_denoise` (diffusers' own `int(steps*strength)` truncation, never zero steps), and `conditioning_source` (walk `refined_from` back to the ancestor that still owns a `.npz`, so upscaling an upscale still uses the true latents). sd15/sd2/sdxl only — a flux-origin image has no replayable SD conditioning and 400s with a pointer to the FLUX engine.

**`analysis.py` + `scripts/analyze_distribution.py` — the corpus autopsy.** Torch-free
statistics (moments, quantile envelope, half-sigmas, Gram-trick PCA with a
shuffled-coordinate null, the past-EOS gating `eta^2`, cross-attention `to_k`/`to_v`
column norms read straight out of the `.safetensors`) plus 12 figures that answer
"what should a sampler actually be doing?". `encode_corpus` is the one GPU-touching
function; it caches `(N, 77, 768)` + token ids + EOS positions to
`outputs/analysis/corpus_<backend>.npz`, and the heavy stats cache beside it, so
`--only NN` re-renders a figure in seconds. Headline results are written to
`outputs/analysis/stats.json`, and `scripts/build_distribution_report.py` folds the
figures + those numbers + the written conclusions into a single self-contained
`distribution_report.html` at the repo root (every PNG inlined as a data URI; ~4.7MB,
untracked like `algorithm_report.html`). `--compare <second analysis dir>` adds a
robustness section contrasting the headline numbers measured through two
checkpoints.

**Nothing in the figures or the report hardcodes the corpus.** Every caption,
title and prose number is derived from `stats.json` or measured on the spot, and
several captions branch on what they measured — because the findings genuinely
flip between corpora. Preserve that when editing: a literal `1000` or `59,136`
typed into a caption is a bug that only shows up on the next prompt file. The
findings on `xander_prompts_to_encode.txt` (4,144 prompts, sd15): the sigma
landscape is flat (nothing to freeze but BOS, whose sigma is exactly 0), the UNet
reads all 768 channels about equally, correlation lives across token positions
(median |r| 0.66) and not within them (0.06), ~416 of the 4,143 PCA axes clear
the noise floor, a past-EOS binary makes the widest coordinates bimodal, and the
corpus is *globally* multimodal too (GMM BIC picks k=30) because prompts long
enough to fill the 77-token window form a detached lobe — which also makes PC1
bimodal, so the pca sampler's N(0,1) draw on its first axis lands in a gap.

**`cli_args.py` is the single source of CLI truth.** `add_backend_args` defines the shared `--backend/--sampler/--temperature/--components/--comp-lo/--equalize/--truncation/--coherence/--neg-mode` set; `resolve_gen_defaults` fills steps/guidance/height/width from `BACKEND_DEFAULTS` when unset. New knobs belong here, not duplicated per script — and two helpers enforce that: `sampler_kwargs(args, lengths)` packs everything `EmbeddingDistribution.sample` takes (a test asserts it covers the whole signature) and `neg_dists_kwarg` hands `generate()` the fitted dists on the backends whose negative branch needs them, which is what `--neg-mode mean` was silently missing outside `generate.py`.

**Even promptless sampling has a negative prompt.** CFG needs something to push away from, and sd15 pushes away from the inherited Eden SD1.5 negative (`pipeline.SD15_NEGATIVE_PROMPT`), not the empty prompt — sdxl still pushes from the corpus mean. It's the one human-written string a "promptless" run still uses, so it's editable everywhere: `--negative`, the sidebar's Advanced box, `SA_SD15_NEGATIVE`. Note that `uncond_embedding()` (the analysis origin) and `negative_embedding()` (the CFG branch) are *different* tensors on sd15; don't collapse them. → [docs/reference/negative-prompt.md](docs/reference/negative-prompt.md)

**Every generated image carries its coordinates.** `scripts/generate.py` writes `anarchy_<backend>_<seed>_<NNN>.jpg` plus an `.npz` sidecar (the exact conditioning tensors) and a `.json` sidecar (all params + the `distance` gauge + `image_seed`). The `.npz` is what makes an image *explorable* later — `explore.py` (neighborhood/breed), `morph_film.py` (SLERP both conditioning *and* the reconstructed init noise, so keyframes reproduce exactly), and `evolve_favorites.py` all read it. Anything that produces images should write both sidecars, and must write them **before** the image — the gallery scan keys on the image files and then reads them, so an image that lands first shows up stripped of its params for a poll cycle. All writes go through `io_utils.unique_image_path` so re-running never clobbers a previous batch.

**The image *stem* is the identity; the extension is not.** Renders are written as high-quality JPEG (q95, `subsampling=0`, ~3.5× smaller than PNG on real output) — `io_utils` owns that: `image_ext()` (`SA_IMAGE_FORMAT=png` reverts, `SA_JPEG_QUALITY` tunes), `save_image()` (picks the encoder off the suffix, so a `--out foo.png` still writes PNG), `find_image()` (resolve by stem, whatever the extension) and `unique_image_path()` (a name is free only when no image of *any* extension and no `.json`/`.npz` sidecar claims that stem — plain `unique_path` would let a new `.jpg` overwrite an existing `.png`'s conditioning). Everything that *reads* the gallery — `app.py`'s scans and suffix checks, the labeling index, `resonance.py`, `experiment_report.py` — must accept all of `IMAGE_EXTS`, because the gallery mined before the switch is PNG forever and its labels/favorites still point at those names. Analysis plots and the report stay PNG (lossless line art); film frames follow the render format, with `frame_ext()` keeping a `--resume` run on whatever the directory already holds so ffmpeg still sees one numbered sequence.

**A batch is rendered one image at a time, and written that way too.** Every `generate_from_embeddings` loops with batch dim 1 (one pipe call, one `Generator` seeded `seed + i`), so `generate.py` passes `on_image=` down through `Backend.generate` and saves each image the moment it exists (`pipeline._stream`) instead of after the run. That's what fills the gallery live — you can judge an experiment at image 1 of 8 — and what lets a cancelled batch keep what it rendered; the dashboard's 2s image poll while busy (`useImages`) exists to match it. `explore.py`/`evolve_favorites.py` don't pass the callback yet and still write at the end.

**`webui/app.py` (FastAPI) orchestrates, it doesn't compute.** It builds argv for the *existing* `scripts/` CLIs and runs them as subprocesses through a **single worker thread** — one GPU, one job at a time. It imports no torch. User input reaches the command line only through allow-lists (`BACKENDS`, `SAMPLERS`, `SCHEDULERS`, `NEG_MODES`, `_clean_csv`) and path sandboxing (`resolve_init_dir`); keep it that way when adding actions. `python_for(backend)` routes `flux2`/`krea2` to `SA_FLUX_PYTHON` (`.venv-flux`) and everything else to `SA_PYTHON`. New capability = new script in `scripts/` + a branch in `build_argv` (or a dedicated `/api/*` endpoint), never inline model code.

**One gitignored `config.json` at the repo root is everything the dashboard remembers** — `models` (the hand-picked checkpoint per backend), `dists` (the selected base distribution per backend) and `ui` (the frontend's whole persisted store: every sidebar value, the timeline, the fit selection, the label-page settings). `SA_CONFIG` relocates it; `load_config`/`save_config_section` in `app.py` own it and fold in the superseded `webui/model_config.json` + `webui/dist_config.json` once, if config.json doesn't exist yet. The `ui` blob is opaque to the server (its shape, version and migrations are `store.ts`'s business) and is written by `src/lib/prefs.ts`, a zustand storage adapter that debounces `POST /api/prefs`, flushes on `pagehide` via `sendBeacon`, and keeps localStorage as a mirror it falls back to — read-only for the session — the moment a `GET` fails, so a hiccup can't upload defaults over a good server-side blob. Settings therefore follow the *server*: a phone that has never opened the dashboard gets the desktop's setup.

**The model picker** (`features/model/ModelPicker.tsx`, slotted under the sidebar's Model group via `ParamPanel`'s `after` prop) hand-picks the checkpoint a backend loads. `POST /api/model/native` pops a *real* OS file dialog **on the server host** (`run_native_picker` → zenity/kdialog/osascript; needs `DISPLAY`/`WAYLAND_DISPLAY`, or `SA_DISPLAY`) — useless from a remote tailnet device, so the UI falls back to `FileBrowser.tsx` over `GET /api/fs`, a server-side listing sandboxed to `browse_roots()` (`$HOME`, the repo, `/mnt`, `/media`, `/opt`, plus `SA_MODEL_ROOTS`). The pick is validated by `validate_model_path` (a `.safetensors`/`.ckpt` file, or a folder with `model_index.json`) and persisted per backend in the `models` section of the repo-root **`config.json`**. `_model_flags` gives that path priority over `SA_SD15_CKPT`/`SA_SD2_CKPT`/`SDXL_MODELS`/`SA_FLUX2_MODEL`, so every action (generate, mine, refine, explore, evolve) follows it.

**The base-distribution picker** (`features/distribution/DistPicker.tsx` + `DistModal.tsx`, slotted directly below the checkpoint picker in the sidebar's Model group — a corpus is fitted *per checkpoint*, so the two are picked as a pair) chooses what every sample is drawn from — it replaced the old base/evolved dropdown, so **`dist` is no longer a form value**: `/api/run` leaves it unset and the server reads the persisted choice. Four kinds: `base` (`outputs/dist`), `evolved` (`outputs/dist_evolved`), `prompts` (a `.txt` corpus anywhere under `browse_roots()`) and `file` (a `.npz` picked directly). Browsing is `GET /api/fs?pick=dist` (same sandbox as the model browser, listing `.txt`/`.npz` and flagging which corpora are already encoded); `GET /api/dist/probe` says where a candidate's latents would live and whether they exist for the *active checkpoint*; `POST /api/dist/encode` queues the mine job that creates them; `POST /api/dist` persists the pick per backend in `config.json`'s `dists` section. Only a *ready* distribution can be selected. `describe_dist`/`current_dist`/`resolve_dist_base` in `app.py` are the seam: `build_argv` resolves `--dist` through them, the sidebar's Mine action re-encodes the selected corpus (falling back to `prompts_1000.txt`), and `/api/evolve` branches off the selected base instead of always `outputs/dist`. Fits made from picked images get named one-click rows at the top of the modal, and `describe_dist` carries their manifest so the sidebar says "fitted from 42 images" rather than counting prompts that don't exist.

**A distribution can also be fitted to images instead of prompts.** The 🧬 Fit tab (`features/fit/`) filters the gallery down to a set — starred, scored 7+, one experiment, a time window — lets you add or drop individual images by eye (the ⊕ on any gallery card feeds the same persisted `fitSel`), and fits a real distribution to those `.npz` latents: its own mean, spread and N−1 PCA subspace, no corpus graft. Seconds of numpy, no model. That is what makes `--sampler pca` on it mean "more like these", and it is what `evolve_favorites.py` gets wrong by keeping the corpus basis. → [docs/reference/selection-fit.md](docs/reference/selection-fit.md)

**The 🎬 Timeline** is the film-making surface: `Card`/`Lightbox` push rel-paths into `timeline` in `src/store.ts` (persisted, duplicates allowed — A→B→A is a valid round trip), `features/timeline/Timeline.tsx` reorders them by HTML5 drag (the card's `onDrop` **must** `stopPropagation` or the strip's append-handler moves the keyframe a second time) and posts `/api/film`, which allow-lists every knob, resolves each keyframe through `_explorable_source`, rejects mixed backends, caps total frames and hands `scripts/morph_film.py` to the same single-worker queue. `POST /api/keyframes` is the pre-flight probe the strip calls on every edit (backend + true pixel size per keyframe, via a header read — `_image_size` parses PNG IHDR and JPEG SOF itself, no PIL in `app.py`), so mixed backends/resolutions surface as a warning banner and a ⚠ badge *before* GPU time is spent, with Advanced → Render size to pick which resolution wins. `features/timeline/FilmList.tsx` plays the results (also the 🎞 Films tab).

**`webui/frontend/` is the dashboard UI** (Vite + React + TS + Tailwind v4 + Radix). `app.py` mounts its `dist/` at `/` — that mount is registered **last**, after every `/api` route, so the SPA only catches what's left; the old inline `INDEX_HTML` still answers at `/legacy`, and becomes the `/` fallback when `dist/` hasn't been built. Server state is TanStack Query (`src/api/queries.ts`) — no hand-rolled polling; client state is Zustand (`src/store.ts`, persisted server-side into `config.json` — see above); the gallery is windowed with `@tanstack/react-virtual`. **The whole sidebar is generated from `src/params/schema.ts`**: one object per knob declares its type, group, `when` visibility predicate, placeholder/hint (static or a function of config + taste band) and how it casts into the `/api/run` body. Adding a control = adding one entry there. Job logs stream over SSE (`/api/log/{id}/stream`, resumable via `Last-Event-ID`), with the plain-text `/api/log/{id}` as the fallback.

**The labeling loop turns a batch into a data point.** `--experiment <id>` on the three scripts that write per-image sidecars (`generate`/`explore`/`evolve_favorites`) stamps the id into every `.json` and writes `outputs/experiments/<id>.json`; the dashboard's `🏷 Label ↗` button opens **`/label`** — its own browser tab, served by a route of its own because `StaticFiles(html=True)` only answers `/`; `main.tsx` picks the shell off `location.pathname` — where a facet bar (experiment / backend / checkpoint / resolution / folder / source / sampler + a time window, options and counts from `/api/label/facets`) chooses the queue and 0–9 keypresses score it into **`labels/labels.jsonl`** — append-only, latest-wins, and the one output that is **committed to git**, because a re-render is free and a human judgement isn't; `scripts/experiment_report.py` turns labels + sidecars into `experiment_report.html`. The record schema, the fixed seed panel (`--seed 1000 --n 16`, since image *i* uses `seed + i`) and the deliberately tail-weighted summaries (keeper-rate, P90 — never the mean alone) live in the torch-free `semantic_anarchy/labels.py`. → [docs/reference/labeling.md](docs/reference/labeling.md)

**The taste loop** is the project's real workflow, spread across three layers: star images in the dashboard (`outputs/favorites.json`) → `scripts/resonance.py` CLIP-embeds the gallery and computes novelty (distance to nearest gallery neighbor) + resonance (a logistic head trained on your stars) → `scripts/fit_selection.py` refits the distribution on the latents of whatever set you selected (the older `evolve_favorites.py` branch is kept for the fits already on disk, but its corpus graft is why its samples drift back toward the corpus) → the `webui/*_drive.py` unattended drivers (`explore_drive` = blind grid sweep, `guided_drive` = sample knobs from what you starred, `hunt_drive` = Pareto frontier of novelty × resonance) push jobs back through the dashboard queue one at a time. Drivers stop on a sentinel file (`outputs/STOP_HUNT`, `outputs/STOP_EXPLORE`).

## Notes

- `README.md` documents only `sd15`/`sdxl`; `sd2`, `flux2` and `krea2` exist in code and in the dashboard. Update the README when touching backend coverage.
- `algorithm_report.html` (untracked) is a long-form writeup of the algorithm; §12 compares this implementation against the user's original in `eden-sd-pipelines/eden/latent_magic.py`.
- Prompt corpora (`prompts_1000.txt` etc.) are deterministic and regenerable via `gen_prompts*.py`.
- In-flight plans live in `docs/TODO/`; `docs/TODO/02_experiment_ledger.md` is the append-only record of what each experiment taught, and is the first thing to read before proposing a new one.
