# Sampling more images inside a collection ("aesthetic universe")

The job: the user curates a visually coherent set of generated images and wants
MORE in that look — tight inside it, or wider toward its edges. The 🧬 selection
fit ([selection-fit.md](selection-fit.md)) + Generate does not do this. What does:
**member-anchored sampling rendered with the anchor's own settings.**

| file | role |
|---|---|
| `semantic_anarchy/collection_sampler.py` | torch-free core: `load_members` (follows `refined_from`, dedupes, reads true pixel size), `Collection` (members in the collection's own frame), `sample(coll, method, spread, n)` → `Child(latents, anchor, partner, t)`, `resolve_seed_mode`, `group_by_checkpoint` |
| `scripts/sample_collection.py` | the CLI/job; renders each child with its anchor's checkpoint/size/steps/guidance/scheduler/negative, one checkpoint load per group, sidecars before the image |
| `tests/test_collection_sampler.py` | spread=0 ≡ anchor, monotone spread, BOS untouched, slerp keeps radius, crossover rows are member rows, torch-free |

```bash
python scripts/sample_collection.py --backend sd15 --from-file sel.json \
    --method auto --spread 0.5 --n 8 [--seed-mode auto|fresh|anchor] \
    [--dist <root corpus base>] [--experiment ID --hypothesis TEXT] [--outdir outputs/generated]
```

Sidecar `kind: "collection"` with `method`, `spread`, `seed_mode`, `anchor`,
`anchors` (anchor + partner), `t`, `collection_size`, the render fields,
`image_seed`, `distance` (vs `--dist`), `render_from`.

## Why the Gaussian fit fails (measured 2026-11-08)

Five hand-built sd15 universes (7–11 images: *mandala*, *statue*, *glowstage*,
*darktex*, *earthy*) plus the 53 ★ favourites; 8 renders per config.

1. **Render settings are the biggest single failure.** Generate from a fit uses
   the sidebar (one checkpoint, 512², the house `text` negative). Members drawn
   with `neg_mode empty` rendered through the house negative come out as
   saturated glitch; at T=1.7 as pure pixel noise. Same fit latents, each child
   rendered with a random member's settings: 3/40 → 17/40 in-universe by eye.
2. **Members are statistically independent.** In root-corpus z-space their
   deviations from the collection mean have pairwise cosine −1/(N−1) to three
   decimals (exactly what N random vectors give); CLIP-image similarity and
   conditioning similarity have rank correlation 0.03 over the gallery
   (top-5-NN overlap 15 %, and that is explore siblings). An aesthetic universe is
   **not a region a mean + covariance describes**: the collection mean sits at
   root-gauge ~0.3 (the bland corpus centre; members ~1.4–1.7), and a PCA draw
   is a superposition of all members — max |cos| to any member 0.63 at N≈10,
   0.41 at N=53. Rendered, that is grey/brown mush or a generic look
   (statue/glowstage/earthy: 2/8 each even with correct settings). T<1 (0.6)
   pulls toward the mean → coherent but generic "average" images.
3. **Temperature compounds.** A fit inherits the members' spread: T=1 lands at
   the members' gauge (1.4→1.4), T=1.7 at ~2.4–2.9. This is what the
   weirdness control fixes; it does not fix 1–2.
4. **The noise seed carries as much of the look as the conditioning.** Exact
   member conditioning + its settings + a NEW seed: CLIP cos to the original only
   0.67 (random pair ≈0.58–0.63). Replay with the recorded `image_seed`: 0.993.
   A 30° rotation of the init noise already drops to 0.72. So "same family" is
   cond × seed × settings; a universe partly lives in its members' seeds.

Controls: random root-corpus draws rendered with member settings score like
random gallery images (CLIP centroid 0.71 vs members' leave-one-out 0.87), so
settings alone do not make the universe — the members' own conditioning does.

## Strategy table (5 universes, 40 renders per row)

`eye` = my per-image "same universe?" count; CLIP = cosine to the members'
centroid / to the nearest member / 5-NN vote among members + 4× distractors.
Members leave-one-out: 0.87 / 0.83 / 0.83; random gallery 0.71 / 0.69 / 0.35.
CLIP is fooled by glitch textures (darktex fit-sidebar scores 0.95 vote while
looking nothing like it) — trust `eye`.

| config | eye /40 | CLIP cen / nn / vote |
|---|---|---|
| fit pca T1.7, sidebar settings (the user's run) | **0** | .75 / .74 / .39 |
| fit pca T1, sidebar settings | 3 | .75 / .73 / .50 |
| fit pca T1, anchor settings | 17 | .78 / .76 / .64 |
| fit diagonal T1, sidebar | ~4 | .75 / .72 / .57 |
| root corpus T1.5, anchor settings | ~5/24 | .71 / .69 / .45 |
| member (spread 0), fresh seed | 24 | .79 / .78 / .71 |
| jitter 0.5, fresh | 25 | .80 / .78 / .70 |
| crossover 0.5, fresh | 29 | .79 / .77 / .74 |
| slerp 0.5, fresh | 29 | .80 / .78 / .77 |
| slerp 1, fresh | 25 | .77 / .75 / .68 |
| **slerp 0.5, anchor seed** | **32** | **.81 / .81 / .78** |
| slerp 1, anchor seed | 29 | .80 / .78 / .76 |
| jitter 1, anchor seed | 24 | .77 / .74 / .72 |

On the broad 53-favourite set the ranking holds (fit T1.7 sidebar 0/8, slerp
0.5 ≈6–7/8) but every number is closer together — "my taste" is not one universe.

## The winner and the knobs

`--method auto` = **slerp**: each child takes an anchor (balanced: every member
once before any repeats), a random partner member, and moves `t ~ U(spread/8,
spread/2)` along the great circle between their deviations from the collection
mean. Slerp, not lerp: deviations are near-orthogonal, so a lerp midpoint shrinks
by ~1/√2 toward the bland mean; slerp keeps the child at a member's own distance
(the root gauge of children equals the members' to 2 decimals). The floor on `t`
stops an anchor-seeded child being a re-render of its anchor (happened at t≈0).

`--spread` (default **0.5**): 0 = the anchor itself (fresh seed only);
1 = halfway to another member; >1 additionally rotates toward collection-shaped
noise by `atan(spread−1)` — past the edges. Other methods: `jitter` rotates the
anchor toward noise scaled by the collection's own (shrunk) per-coordinate std by
`atan(spread)`, keeping its radius; `crossover` swaps token rows (probability
spread/2) for other members' same-position rows; `member` ignores spread.
Row 0 (CLIP BOS, constant) is never moved.

`--seed-mode auto` = the anchor's own `image_seed` while spread ≤ 1 (best
measured), fresh `seed + i` beyond (new compositions for exploring). `member`
is always fresh.

## Things that will bite

- **SD1.5 cross-attention is permutation-invariant over token rows** (no
  positional encoding on the context): shuffling rows 1–76 renders the same
  image (MAE 1–2.5/255). Row "position" only matters through CLIP's content vs
  past-EOS statistics, which is why crossover swaps same-position rows.
- **Size is part of the look.** Read it off the image (old explore sidecars
  have no width/height; an upscale's size is not its source's). `load_members`
  does both.
- **Mixed checkpoints are fine here** — each child uses its anchor's; the CLI
  groups children per checkpoint and loads each once. A missing checkpoint
  falls back to the default with a warning. `--ckpt` overrides all (breaks the
  replay; experiments only).
- `neg_mode mean` anchors need their recorded `dist`; unreadable → `--dist`.
- In-process only (`remote_ok=False`): a comfy-rendered anchor can't be replayed
  faithfully.
- `outputs/dist.npz` (the generic `--dist` default elsewhere) is a truncated
  zip on this box — pass the root corpus base (e.g.
  `xander_prompts_to_encode__juggernaut_reborn`) as `--dist`.

Experiment code, universes and contact sheets from the measurement: session
scratchpad only (regenerable; not kept).

→ related: [render-replay.md](render-replay.md) (why anchor settings),
[selection-fit.md](selection-fit.md), [sampler-knobs.md](sampler-knobs.md) (weirdness)
