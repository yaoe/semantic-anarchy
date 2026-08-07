# Fitting a distribution to images you picked

The second way to get a distribution. Mining answers *"what does this corpus of
**prompts** look like"*; a **selection fit** answers *"what do the images I
**kept** look like"* — using the `.npz` conditioning sidecar every generated
image already carries. No text encoder, no GPU, no torch: a set of images is
already a corpus of latents, so fitting one is a stack-and-fit that takes about
a second for 42 images.

Where the code is:

| file | role |
|---|---|
| `semantic_anarchy/selection_fit.py` | torch-free core: gather → stack → fit → save → manifest; also `list_fits` |
| `scripts/fit_selection.py` | the CLI/job. Takes `--images` or `--from-file`, writes `outputs/dist_fits/<name>` |
| `webui/app.py` → "Selection fits" section | `/api/fit/candidates`, `/api/fit`, `/api/fit/list`, `/api/fit/delete` |
| `webui/frontend/src/features/fit/` | the 🧬 Fit tab: `FitFilters` (who), `FitPage` (selection + the button), `SavedFits` (what exists) |

## Why not `evolve_favorites.py`

That script is still there and still works, but it answers a different question
badly. It re-centres the per-coordinate Gaussian on the elite mean and then
**grafts the corpus PCA basis** (components, `pca_std`, `pca_head`) back on, so
the `pca` sampler goes on drawing corpus-sized deviations along corpus axes
around a centre that basis never saw. The mean is your taste; the geometry is
still the corpus's. That is the "the evolved branch looks like crap" report.

A selection fit has no graft: mean, std, PCA subspace and radius band all come
from the selected latents. With N images the subspace is the N−1 dimensional
affine span of exactly those latents, so a `pca` draw is a combination of images
you picked and nothing else. Measured on 42 starred sd15 images: the favourites
themselves sit at corpus-distance 1.38, `pca` draws from their selection fit at
1.44, and draws from the old evolved branch at 0.99 — i.e. the branch drifts back
toward the corpus centre, the fit doesn't. `tests/test_selection_fit.py` pins the
span property.

**Tell people to sample it with `--sampler pca`.** `diagonal` on a 40-image fit
is per-coordinate independent noise around the taste centre — off-manifold for
the same reason it is off-manifold on the corpus, only now with a much thinner
estimate of the spread behind it.

## Things that will bite

- **A selection fit has no length statistics.** There are no prompts, so there
  is no EOS position to split on: `--length-mode` and the radius band's corpus
  semantics don't apply. The picker says so instead of showing the "re-encode to
  fix it" warning that a stale *mined* fit gets — re-encoding is not the fix,
  and never will be.
- **Mixed checkpoints are allowed, not endorsed.** Conditioning shapes match
  across sd15 finetunes, so a selection spanning two checkpoints fits fine; the
  manifest records every checkpoint slug and the CLI says so once. Mixed
  *backends* are rejected up front (different shapes, and the minority would be
  silently dropped).
- **Upscales carry no latents of their own.** `latents_for` follows the
  `refined_from` chain back to the ancestor that does — but the candidate list
  filters them out by default, because that ancestor is usually already in the
  gallery and would otherwise be counted twice.
- **The fit runs on the single GPU worker queue** even though it needs no GPU,
  so it waits behind whatever is rendering. It is seconds of work; the wait can
  be minutes. Worth revisiting only if it becomes annoying.
- **Small N is legal and thin.** Three is the floor (PCA needs a direction);
  under eight the span is small enough that samples read as interpolations of
  the picked images. That is sometimes exactly what you want.

## Naming and where files land

`outputs/dist_fits/<name>` is an ordinary distribution *base*, so the three
naming layers in [`dist_paths.py`](../../semantic_anarchy/dist_paths.py) apply
unchanged and the result is selectable exactly like a mined corpus. Beside it:

- `<name>.fit.json` — the manifest. Which images went in, which checkpoints they
  came from, what was skipped and why, the note you typed. Not needed to sample
  (the `.npz` is self-contained) but it is the only record of what "my keepers,
  week 3" meant, and `describe_dist` reads it to label the picker.
- `<name>.sources.json` — the selection as the dashboard handed it to the job.
  Selections run to hundreds of paths, which is argv's problem, not JSON's.

## The selection itself

Candidates come from the **same index the labeling queue uses**
(`_label_index()` in `app.py`), so a facet means one thing in both places; the
fit endpoint adds the two dimensions you select *by* once you have labeled — the
star and the label-score band. The chosen images live in the client
(`fitSel` in `store.ts`, persisted), not on the server: a filtered set you then
prune by eye is the normal case, so the selection cannot be a re-derivable
function of the filters. The ⊕ button on any gallery card adds to that same
list, which is what makes "these six, plus everything I starred yesterday" a
thing you can express.

→ related: [labeling.md](labeling.md) (where the scores come from),
[sampler-knobs.md](sampler-knobs.md) (what to do with the fit once you have it)
