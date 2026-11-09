# Render replay: an image is its latents AND how it was rendered

The `.npz` sidecar alone does not reproduce an image's look. Measured
(2026-11-08, 10 random recent sd15 images, exact `.npz`, fresh noise seeds):

| changed vs the source's sidecar | effect on the look |
|---|---|
| nothing (same `image_seed` too) | pixel-identical (MAE 0–1.8/255) |
| noise seed only | new composition, clearly the same family — **this is what "more like this" should give** |
| checkpoint (juggernaut ↔ dreamlike) | look destroyed — the single biggest drift |
| render size (512 ↔ 1024 on SD1.5) | composition and texture change a lot |
| CFG negative (`neg_mode` empty ↔ text) | palette/finish shift (house negative = saturated, "clean") |
| scheduler (euler ↔ ddim) | barely visible |

The original bug ("explore gives a batch that is consistent with itself but
looks nothing like the seed"): `/api/explore` rendered children with the
*sidebar's* checkpoint, forced `--scheduler ddim`, 512², and the default
negative — one shared look for every seed.

## Who replays what (keep it this way)

| action | replays from the source sidecar |
|---|---|
| explore / walk / breed (`scripts/explore.py`) | `replay_source_params`: checkpoint, steps, guidance, scheduler, neg_mode, negative, and the source's TRUE pixel size (read off the image — old explore sidecars recorded none). Explicit CLI flags win; `--stay-local` also reuses the noise seed and lets the source win over flags; `--fresh-render` opts out. `/api/explore` passes the source's checkpoint (`_source_checkpoint`) and no render knobs of its own. |
| hires upscale (`scripts/upscale.py`) | checkpoint (sidecar `model`, unless `--ckpt` given; `/api/refine` passes the origin's), steps, guidance, scheduler, seed, and the CFG negative (`_original_negative` → `refine_image(negative=)`). |
| film (`morph_film.py`), animate | checkpoint from keyframe 1 / the source (already did). |
| universe sampling (`scripts/sample_collection.py`) | each child rendered with its anchor member's settings → [collection-sampling.md](collection-sampling.md). |
| plain Generate from a 🧬 fit | **does not** — it uses the sidebar. `DistPicker` warns in red when the fit's images came from a checkpoint other than the selected one. |

A remote ComfyUI lane keeps its own checkpoint; a comfy-rendered source can't be
replayed faithfully in-process (upscale.py says so).

Tests: `tests/test_explore_replay.py`, `tests/test_webui.py` (explore argv).
