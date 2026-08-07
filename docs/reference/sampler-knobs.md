# How the sampling knobs interact

`EmbeddingDistribution.sample` takes a dozen knobs and they are not independent:
several are read by only some samplers, and four of them are four spellings of
one quantity. This is the map of which knob is live when — the rules the sidebar
encodes in `webui/frontend/src/params/schema.ts` and the CLI warns about in
`scripts/generate.py`.

What each correction *measured* is a separate subject →
[sampler-corrections.md](sampler-corrections.md).

## The one line that explains most of it

```python
samples = mu + temperature * dev          # distribution.py, sample()
```

`temperature` is a **scalar multiplying the whole deviation**, applied after the
sampler has produced `dev`. Everything below follows from that plus one fact
about `retarget`.

## Reach: temperature, shell, band — pick one

`retarget()` rescales a sample by `target / distance(sample)`. Distance is
linear in the deviation, so a sample drawn at temperature *T* has distance
`T·d₀` and comes back as

```
mean + (T·dev₀)·target/(T·d₀)  =  mean + dev₀·target/d₀
```

**The temperature cancels exactly.** Not "is overridden", not "matters less" —
the output tensors are bit-identical at T = 1.0 and T = 3.7 under the same shell.
So these three never compose:

| mode | flag | what fixes the radius |
|---|---|---|
| temperature | `--temperature` | nothing — the draw's own spread, scaled |
| shell | `--target-distance` | one radius for the whole batch |
| band | `--radius-band` (+`--radius-scale`) | a per-sample radius bootstrapped from the corpus's own |

Shell and band are the *same* pin — both call `retarget()`, keeping the sampled
direction and overwriting the radius — and differ only in where the radius comes
from: one typed scalar broadcast to the batch, versus one value resampled per
image from `corpus_distance` (`sample_radii`). So the batch-level difference is
spread: shell puts every image on one sphere, band reproduces the corpus's real
range of radii, which every sampler otherwise collapses into a ~9× tighter spike.

`--min-distance` is a **floor applied last**, after either pin. A floor at or
above the shell therefore replaces it silently — every sample lands on the floor
and the shell never shows. `generate.py` warns on both of these now.

The one exception is `hybrid`, which returns before the deviation machinery
exists: it SLERPs two real corpus embeddings, so its samples sit at corpus
distance (~1.0) whatever T says, and T weights only the 15% gaussian jitter added
on top. Measured: d(T=1)=0.98, d(T=2)=1.01. That jitter changes the *direction*,
so unlike a scale factor it survives a pin.

Under a pin with `split`, the same cancellation means only the **ratio**
`temp_on : temp_off` changes anything; their common scale divides out.

## Temperature vs the split temperatures

`_split_dev` returns `temp_on·on + temp_off·off`, and `sample` then multiplies
the result by `temperature`. Three knobs, two degrees of freedom:

```
T=2.0, (temp_on 1.5, temp_off 0.5)  ≡  T=1.0, (temp_on 3.0, temp_off 1.0)
```

The global temperature is a pure common factor, so the sidebar hides it in split
mode and lets `temp_on`/`temp_off` **be** the temperatures. Nothing about the
maths changed — a hidden knob is sent at its default, which for temperature is
1.0.

## Which sampler reads which knob

`hybrid` returns from `sample()` before length conditioning, ρ, truncation and
every PCA selector are consulted, so all of them are no-ops there. `equalize`
lives only in `_pca_dev`. ρ shapes only the diagonal draw.

| knob | diagonal | pca | blend | hybrid | split |
|---|---|---|---|---|---|
| `temperature` | ✅ | ✅ | ✅ | jitter only | via temp_on/off |
| `rho` | ✅ | — | diagonal half | — | ✅ (pre-split) |
| `coherence` | — | — | ✅ | — | — |
| `components`, `comp_lo` | — | ✅ | ✅ | — | ✅ (picks the basis) |
| `equalize`, `empirical_head` | — | ✅ | ✅ | — | — |
| `truncation` | ✅ | ✅ | ✅ | — | ✅ |
| `length_mode` | ✅ | ✅ | ✅ | — | ✅ |
| `temp_on`, `temp_off` | — | — | — | — | ✅ |

Two endpoint cases fold a row away: `blend` at λ=1 **is** the pca sampler
(bit-exactly), so ρ stops doing anything there; at λ=0 it is the diagonal one and
the PCA selectors stop. The sidebar warns on both.

`pca`/`blend` also land *below* the distance the temperature names, because they
only span the retained subspace — "distance ≈ T" is a diagonal-sampler statement.

## Which action forwards which flag

`build_argv` in `webui/app.py` does not hand the same flags to every script, and
this was invisible from the sidebar until the schema encoded it:

- **`generate`** — everything, and the only action that carries
  `--target-distance` / `--radius-band` / `--min-distance`. The sweep scripts do
  not define those arguments at all.
- **`temp_sweep`** — the full sampler set, but it owns temperature itself via
  `--temps`.
- **`sampler_sweep`** — only `--temperature`, `--coherence`, `--seeds`. It renders
  its own diagonal/blend/pca rows, so `--sampler` and every other sampler knob is
  dropped. `--coherence` is live here: it labels and drives the middle row.
- **`mine`** — no sampler knobs. `--components` is live but means something else:
  the PCA rank to *fit*, not the axis count to *sample*.

## How the sidebar encodes it

Two mechanisms, both in `schema.ts`:

- **`visible(values)`** — an optional predicate ANDed with the existing `when`
  map. `when` is a conjunction of value tests and cannot express "A or B", which
  several of these rules genuinely need (temperature is live under
  `sampler_sweep` *or* under `generate` with reach=temperature *or* under
  hybrid).
- **hidden ⇒ sent at its default** — `buildRunRequest` reads `f.default` instead
  of form state for any field the predicates hide. This is what makes the
  conditional logic *binding* rather than decorative: a temperature typed before
  switching to a shell, or a ρ left over from a diagonal run, can no longer reach
  argv and turn up in the `.json` sidecar of an image it had no effect on.
  Falling back to the default rather than to `null` also keeps the
  non-optional `RunRequest` fields (`sampler`, `init_mode`) valid while their
  control is hidden.

`reachMode(values)` is the single helper deciding which of the three reach knobs
is showing; `radius_band` is derived from it rather than being a second toggle
that could disagree.

Because the sidecar records what was *sent*, this also cleans up the label
corpus: a knob vector regressed against a score no longer contains columns that
were inert for that run.
