# The 🌌 latent map

A 3D map of every image that has a conditioning `.npz`, for reading what the gallery contains and sampling from a chosen spot.

**Where things live.** `semantic_anarchy/latent_map.py` holds the projection math (NumPy at module scope; `umap`/`sklearn` imported inside functions). `scripts/latent_map.py` builds the JSON and thumbnail atlas. `GET /api/latent-map?backend&focus` reruns it when its signature changes, outside the GPU queue; `GET /api/latent-map/atlas` serves the atlas. Frontend is `webui/frontend/src/features/latentmap/`: `scene.ts` owns the three.js canvas, `LatentMap.tsx` maps data to typed arrays and holds the region list + sampling panel, `MapControls.tsx` is the control strip.

## The three spaces

| space | input | what near means | neighbours that look alike (trustworthiness vs CLIP; 0.5 = random) |
|---|---|---|---|
| `look` (default) | UMAP of CLIP image embeddings, cosine | looks alike | 0.83 |
| `cond` | UMAP of the conditioning, through the JL `sketch` | drawn alike | 0.54 |
| `corpus` | top 3 corpus PCA axes, asinh-squashed for display | same corpus direction | 0.55 |

The UI shows that number for the current space. Keep it there: it is what stops the conditioning spaces reading as structure they don't have.

**Why not UMAP of the conditioning alone**, which was the first idea. Measured on 1,003 real sd15 images (2026-11-02): an image's 15 nearest conditioning neighbours overlap its 15 nearest CLIP neighbours at chance (1.9% vs 1.5%), and conditioning distance has no useful correlation with visual distance (Spearman -0.15). 80% of the gallery is `diagonal` samples, independent draws in about 59k dimensions, so their distances concentrate and neighbours are close to arbitrary. No projection can recover what the input doesn't encode. Linear PCA of CLIP scores 0.77 against UMAP's 0.83, so the choice of input mattered far more than the choice of projection.

**The axis picker is gone.** Choosing 3 of 8 PCA axes showed about 4% of the variance whichever you picked. So was the marching-cubes hull (one lump in front of the images) and `off_manifold` paint (~0.97 for nearly every image).

## Layout details

- **Focus** presets (`MAP_FOCUS` in `app.py`) are `n_neighbors`/`min_dist` pairs: `islands` 8/0.02, `balanced` 20/0.1, `continents` 60/0.4. Each has its own `outputs/latent_map_<backend>_<focus>.json`; the atlas is shared.
- **Rebuilds continue from the previous layout** (`warm_start` + `align`), so images stay put: adding 20 images moved the look layout by a median of one nearest-neighbour gap (0.14 of a ~1.6 radius). The `cond` layout moves far more (~1.0) because it has no structure to hold it.
- **A full rebuild is ~15s**, so `useLatentMap` holds its query key while a job runs and catches up when it ends. The first version refetched on every 2s gallery poll.
- **Regions** are HDBSCAN clusters of the look layout, and colour every space. Default selection, not `leaf`: leaf splits plain Gaussian blobs into fragments. The default can collapse to one cluster holding nearly everything after a small layout shift (20 regions became 2), so a cluster over a third of the gallery triggers a re-run with `max_cluster_size` n/6, which held 11-23 regions across perturbed layouts. About 25-50% of images stay unclaimed (grey). Treat regions as a reading aid: an isotropic control also produced 6 clusters, though with 78% unclaimed and far lower trustworthiness (0.61).
- **New images have no look position until embedded.** The map counts them in `unembedded` and auto-queues `resonance.py` when idle. Unreadable files are excluded via the atlas' `broken` list, otherwise that retries on every load. `_map_signature` skips `taste_model.npz` because resonance rewrites it every run; counting it caused an endless rebuild in which focus changes never appeared.
- **Thumbnails switch on only after the atlas texture has loaded** (`applyThumbMode` in `scene.ts`). Sampling an empty texture paints every node black, which is what a 404 did. A new frontend against a not-yet-restarted server produces exactly that 404, so restart the server when deploying map changes.
- **The atlas** (`outputs/latent_map_<backend>.atlas.jpg` + `.json`) reuses pixels between builds, so only new images are decoded. The name must not start with a `GALLERY_BUCKETS` prefix or it appears in the gallery.

## Sampling from the map

A UMAP has no inverse, so a spot cannot become conditioning directly. Clicking empty space picks the nearest image (in the current space) and "Sample here" perturbs it via `POST /api/explore` with `stay_local: true`, so children land around that image rather than exactly on the clicked point. A region samples from its 4 most typical images.

`stay_local` → `explore.py --stay-local` replays the source's seed, size, steps, guidance, scheduler and negative, and `app.py` passes the source's checkpoint when it is still on disk. `--neg-mode mean` also needs the source's distribution. With radius 0 this reproduces the parent (mean pixel difference 0-1.6/255 across `ddim`, `euler`, `euler_a`, default), which is the regression check for this path.

**Measured, sd15, 60 spots, paired, same model/size as the anchor.** Rank = share of the gallery closer to the child than the anchor is (lower = closer).

| strategy | anchor rank (median, 95% CI) | CLIP cosine to anchor |
|---|---|---|
| radius 0.2, same seed | 0.1% [0.1, 0.8] | 0.77 |
| radius 0.2, new seed | 1.3% [0.5, 5.0] | 0.75 |
| radius 0, new seed | 2.0% [0.6, 3.9] | 0.74 |
| radius 0.6, same seed | 10.6% [3.6, 19.7] | 0.71 |
| radius 0.6, new seed | 16.7% [5.2, 25.0] | 0.67 |
| random diagonal sample | ~40% [30, 48] | 0.63 |

Conditioning does most of the work. The seed adds a modest amount, significant at radius 0.2 (p=0.008) and not at 0.6 (p=0.24). A first run with 26 spots suggested a far bigger seed effect, and a 22-spot rerun contradicted it: at these sample sizes the median rank is noisy, so don't draw conclusions from small runs.

**Rejected: blending the k nearest images' conditioning** (even radius-preserving), which is the natural way to sample a spot. Blends of 3 and 8 ranked 14% and 25% against the probe's neighbourhood, vs ~37% for random, in a 26-spot run; weighted anchor blends in a 22-spot run were no better than perturbing. Not retested at 60 spots.

**Placement on the map itself** (real end-to-end runs through the dashboard, tagged `map-sampling-check`; share of the map closer to the parent than the child is). Radius 0.2, 12 children of 6 sources: median 6.5%, four under 1%, three over 20% (worst 59%), and it varies by source. Radius 0.4, 4 children of 2 sources: 0.3%, 16%, 47%, 74%. So 0.2 is the default: 0.4 often changes the subject outright. New images are not misplaced by the warm start: 40 held-out images sat as near their closest visual neighbours (median rank 3.0%) as in a cold layout (3.6%). So a child landing far away mostly reflects a real change in look or the 3D layout's limits (map-vs-CLIP distance Spearman is only 0.36). Treat a sample as "starts here", not "lands here".
