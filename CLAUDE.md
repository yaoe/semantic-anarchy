# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Promptless image generation ("Semantic Anarchy"): encode a corpus of prompts through a diffusion model's text encoder **once**, fit a distribution over the resulting conditioning tensors, then sample brand-new conditioning straight from that distribution and feed it to the UNet via `prompt_embeds`. The text encoder is never touched at generation time. Stock `diffusers` pipelines only — no SAE, no hooks, no forks.

## Commands

The repo uses a uv-managed Python 3.12 `.venv` (torch + diffusers + fastapi). Scripts are run as files, not as an installed package (each does `sys.path.insert(0, repo_root)`); there is no `pyproject.toml`.

```bash
pytest -q                                   # full suite (36 tests, torch-free, ~9s)
pytest tests/test_distribution.py -q        # one file
pytest -q -k pca_samples_lie_in_subspace    # one test
python scripts/demo_no_sd.py                # end-to-end demo on synthetic embeddings, no torch/GPU

# mine → generate (needs the full tier: torch + diffusers + weights)
python scripts/mine_distribution.py --backend sd15 --ckpt MODEL.safetensors \
    --prompts prompts_1000.txt --out outputs/dist
python scripts/generate.py --backend sd15 --dist outputs/dist --n 8
python scripts/temperature_sweep.py --backend sdxl --dist outputs/dist --sampler pca --temps 1,2,3,4

# latent travel film through keyframes -> outputs/films/<name>/<name>.mp4
python scripts/morph_film.py --name drift --refine none --frames-per 24 --fps 16 \
    --images generated/a.png generated/b.png generated/c.png

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

**`distribution.py` — `EmbeddingDistribution`** is the statistical core and where most real logic lives. A per-coordinate Gaussian fit (`mean`, `std`) *plus* low-rank PCA of the corpus (`pca_components`, `pca_std`) *plus* a bounded subset of raw corpus embeddings. That triple is what makes the four samplers possible:

| sampler | draws from | notes |
|---|---|---|
| `diagonal` | independent per-coordinate Gaussians | the original model; off-manifold |
| `pca` | the low-rank corpus subspace | `temperature > 1` extrapolates coherently *outside* the corpus hull |
| `blend` | sqrt-weighted mix of the two **covariances** | `coherence=0` ≡ diagonal, `1` ≡ pca, bit-exact at the endpoints |
| `hybrid` | SLERP between two real corpus embeddings | needs `corpus_embeddings`; stays inside the hull |

Also here: `distance()` (RMS z-score from the corpus center — the "how far off-grid" gauge, ~1.0 for a typical corpus sample, ~T for a temperature-T diagonal sample), `retarget()` (shell sampling: keep direction, pin radius), `neighborhood()`/`walk()` (local navigation), `refit_from_elites()` (evolutionary branches). `fit()` switches to float32 + a Gram-matrix PCA trick when the feature dim exceeds 200k (flow-model Qwen conditioning) — a direct SVD there needs tens of GB.

**`backend.py` — the model-agnostic seam.** A model is described purely by its *named conditioning tensors*, and every script drives four generic verbs: `encode → fit → sample → generate`. One `EmbeddingDistribution` is fitted per named tensor, and all tensors are sampled with the *same* knobs so the conditioning set stays coherent.

| backend | tensors | pipeline |
|---|---|---|
| `sd15` | `embeds` (77,768) | `StableDiffusionPipeline` |
| `sd2` | `embeds` (77,1024) | same, OpenCLIP-H; subclasses `SD15Backend` |
| `sdxl` | `prompt_embeds` (77,2048) + `pooled` (1280) | `StableDiffusionXLPipeline` |
| `flux2`, `krea2` | `embeds` — multi-layer Qwen3 hidden states | flow models; run in a **separate `.venv-flux`** (diffusers ≥0.39) |

Adding a backend means: a `Backend` subclass (`tensor_names` + `encode`/`generate`), a `*Model` wrapper in `pipeline.py`, a `BACKEND_DEFAULTS` entry, and the name in `make_backend`, `dist_backend`, `cli_args.add_backend_args` choices, and `webui/app.py`'s `BACKENDS` allow-list. `dist_backend(name)` gives a model-less instance exposing only the NumPy verbs — that's what the tests use.

**Distribution files are backend-namespaced.** `dist_prefix()` in `cli_args.py`: `sd15` keeps the bare `outputs/dist` (original layout), everything else gets `outputs/dist_<backend>`. Multi-tensor backends then append `__<tensor>` per file (`outputs/dist_sdxl__pooled.npz`). Each `.npz` has a `.meta.json` sibling. Evolved taste branches use the `outputs/dist_evolved*` prefix with the same rules.

**`travel.py` — keyframe films, torch-free.** `interpolate` (`slerp`/`lerp`), `ease` (`smooth`/`smoother`/`linear`), `frame_plan(n_keys, frames_per, easing)` → the `(segment, t)` list for every frame (each transition starts ON its left keyframe and stops short of its right, plus one final frame at t=1, so no keyframe renders twice), and `noise_t(t, window)` for the narrower window the *init noise* travels in. `scripts/morph_film.py` is the only renderer: it reconstructs each keyframe's init latent from the recorded `image_seed` exactly as diffusers' `randn_tensor` would (cuda generator for sd15/sd2, cpu for sdxl), interpolates conditioning + noise, renders every frame through the raw pipe with `latents=`, and muxes with the first ffmpeg that actually has an H.264 encoder (`pick_h264_encoder`; a conda `--disable-gpl` ffmpeg on PATH has no libx264 — `SA_FFMPEG` pins one). Endpoints therefore come back pixel-identical; the middle never existed. sd15/sd2/sdxl only (`FILM_TENSORS`). **A film has ONE resolution** (keyframe 1's own, or `--width/--height`): a keyframe of any other size is re-rendered at the film's size from a differently-shaped noise draw, so it is *not* reproduced — `keyframe_size` reads the true size off the PNG (sidecars from evolve branches record none) and the script names every off-size keyframe in the log and in `film.json.offsize_keyframes`.

**`cli_args.py` is the single source of CLI truth.** `add_backend_args` defines the shared `--backend/--sampler/--temperature/--components/--comp-lo/--equalize/--truncation/--coherence/--neg-mode` set; `resolve_gen_defaults` fills steps/guidance/height/width from `BACKEND_DEFAULTS` when unset. New knobs belong here, not duplicated per script.

**Every generated image carries its coordinates.** `scripts/generate.py` writes `anarchy_<backend>_<seed>_<NNN>.png` plus an `.npz` sidecar (the exact conditioning tensors) and a `.json` sidecar (all params + the `distance` gauge + `image_seed`). The `.npz` is what makes an image *explorable* later — `explore.py` (neighborhood/breed), `morph_film.py` (SLERP both conditioning *and* the reconstructed init noise, so keyframes reproduce exactly), and `evolve_favorites.py` all read it. Anything that produces images should write both sidecars. All writes go through `io_utils.unique_path` so re-running never clobbers a previous batch.

**`webui/app.py` (FastAPI) orchestrates, it doesn't compute.** It builds argv for the *existing* `scripts/` CLIs and runs them as subprocesses through a **single worker thread** — one GPU, one job at a time. It imports no torch. User input reaches the command line only through allow-lists (`BACKENDS`, `SAMPLERS`, `SCHEDULERS`, `NEG_MODES`, `_clean_csv`) and path sandboxing (`resolve_init_dir`); keep it that way when adding actions. `python_for(backend)` routes `flux2`/`krea2` to `SA_FLUX_PYTHON` (`.venv-flux`) and everything else to `SA_PYTHON`. New capability = new script in `scripts/` + a branch in `build_argv` (or a dedicated `/api/*` endpoint), never inline model code.

**The model picker** (`features/model/ModelPicker.tsx`, slotted under the sidebar's Model group via `ParamPanel`'s `after` prop) hand-picks the checkpoint a backend loads. `POST /api/model/native` pops a *real* OS file dialog **on the server host** (`run_native_picker` → zenity/kdialog/osascript; needs `DISPLAY`/`WAYLAND_DISPLAY`, or `SA_DISPLAY`) — useless from a remote tailnet device, so the UI falls back to `FileBrowser.tsx` over `GET /api/fs`, a server-side listing sandboxed to `browse_roots()` (`$HOME`, the repo, `/mnt`, `/media`, `/opt`, plus `SA_MODEL_ROOTS`). The pick is validated by `validate_model_path` (a `.safetensors`/`.ckpt` file, or a folder with `model_index.json`) and persisted per backend in **`webui/model_config.json`** (gitignored — everyone's weights live elsewhere; `SA_MODEL_CONFIG` relocates it). `_model_flags` gives that path priority over `SA_SD15_CKPT`/`SA_SD2_CKPT`/`SDXL_MODELS`/`SA_FLUX2_MODEL`, so every action (generate, mine, refine, explore, evolve) follows it.

**The 🎬 Timeline** is the film-making surface: `Card`/`Lightbox` push rel-paths into `timeline` in `src/store.ts` (persisted, duplicates allowed — A→B→A is a valid round trip), `features/timeline/Timeline.tsx` reorders them by HTML5 drag (the card's `onDrop` **must** `stopPropagation` or the strip's append-handler moves the keyframe a second time) and posts `/api/film`, which allow-lists every knob, resolves each keyframe through `_explorable_source`, rejects mixed backends, caps total frames and hands `scripts/morph_film.py` to the same single-worker queue. `POST /api/keyframes` is the pre-flight probe the strip calls on every edit (backend + true pixel size per keyframe, via a 24-byte PNG-header read — no PIL in `app.py`), so mixed backends/resolutions surface as a warning banner and a ⚠ badge *before* GPU time is spent, with Advanced → Render size to pick which resolution wins. `features/timeline/FilmList.tsx` plays the results (also the 🎞 Films tab).

**`webui/frontend/` is the dashboard UI** (Vite + React + TS + Tailwind v4 + Radix). `app.py` mounts its `dist/` at `/` — that mount is registered **last**, after every `/api` route, so the SPA only catches what's left; the old inline `INDEX_HTML` still answers at `/legacy`, and becomes the `/` fallback when `dist/` hasn't been built. Server state is TanStack Query (`src/api/queries.ts`) — no hand-rolled polling; client state is Zustand (`src/store.ts`); the gallery is windowed with `@tanstack/react-virtual`. **The whole sidebar is generated from `src/params/schema.ts`**: one object per knob declares its type, group, `when` visibility predicate, placeholder/hint (static or a function of config + taste band) and how it casts into the `/api/run` body. Adding a control = adding one entry there. Job logs stream over SSE (`/api/log/{id}/stream`, resumable via `Last-Event-ID`), with the plain-text `/api/log/{id}` as the fallback.

**The taste loop** is the project's real workflow, spread across three layers: star images in the dashboard (`outputs/favorites.json`) → `scripts/resonance.py` CLIP-embeds the gallery and computes novelty (distance to nearest gallery neighbor) + resonance (a logistic head trained on your stars) → `scripts/evolve_favorites.py` refits the distribution around starred latents (grafting the corpus PCA axes back on so all samplers keep working) → the `webui/*_drive.py` unattended drivers (`explore_drive` = blind grid sweep, `guided_drive` = sample knobs from what you starred, `hunt_drive` = Pareto frontier of novelty × resonance) push jobs back through the dashboard queue one at a time. Drivers stop on a sentinel file (`outputs/STOP_HUNT`, `outputs/STOP_EXPLORE`).

## Notes

- `README.md` documents only `sd15`/`sdxl`; `sd2`, `flux2` and `krea2` exist in code and in the dashboard. Update the README when touching backend coverage.
- `algorithm_report.html` (untracked) is a long-form writeup of the algorithm; §12 compares this implementation against the user's original in `eden-sd-pipelines/eden/latent_magic.py`.
- Prompt corpora (`prompts_1000.txt` etc.) are deterministic and regenerable via `gen_prompts*.py`.
