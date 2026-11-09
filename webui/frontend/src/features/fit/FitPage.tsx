/**
 * 🧬 Fit — turn a set of images into the distribution the next batch is sampled
 * from.
 *
 * The loop this closes: generate → star/label → *select* → fit → sample from the
 * fit. Every generated image carries its conditioning in a `.npz` sidecar, so a
 * selection of images already IS a corpus of latents; fitting it is a few
 * seconds of numpy with no model involved.
 *
 * Selection is deliberately two things at once. The filter bar picks a set by
 * *description* (starred, scored 7+, experiment E07, last 24h) and the grid lets
 * you add or drop individual images by eye — a filtered set you then prune is
 * the normal case, so the selection is kept as its own list rather than being
 * recomputed from the filters. It persists across reloads and across tabs (the
 * + box on any gallery card adds to this same list).
 *
 * What comes out is an ordinary distribution file, so every sampler, temperature
 * and correction works on it unchanged — see `semantic_anarchy/selection_fit.py`
 * for why that matters more than it sounds.
 */
import { useMemo, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'

import { api } from '../../api/client'
import {
  useCreateFit,
  useDistConfig,
  useJobWatcher,
  useFitCandidates,
  useSetDist,
} from '../../api/queries'
import type { BackendId, FitCandidate, FitOrder, FitQueryParams, FitScored } from '../../api/types'
import { UNSET } from '../../api/types'
import { backendOf, cn, imgSrc } from '../../lib/utils'
import { useUI } from '../../store'
import { stubItem, useViewerList } from '../gallery/viewerLists'
import { FitFilters } from './FitFilters'
import { SavedFits } from './SavedFits'

/** Server-side minimum (`selection_fit.MIN_SAMPLES`) — PCA needs a direction. */
const MIN_SAMPLES = 3
/** Below this a fit is technically valid but the span is tiny; warn, don't block. */
const THIN_SAMPLES = 8
/** How many candidates the grid asks for; the server caps select-all at 2000. */
const PAGE = 600
const SELECT_ALL_LIMIT = 2000

const numOrNull = (v: string) => (v.trim() === '' ? null : Number(v))

/** The store's strings → the query the server actually answers. */
function toQuery(fit: Record<string, string>): FitQueryParams {
  return {
    experiment: fit.experiment || undefined,
    backend: fit.backend || undefined,
    ckpt: fit.ckpt || undefined,
    folder: fit.folder || undefined,
    size: fit.size || undefined,
    kind: fit.kind || undefined,
    sampler: fit.sampler || undefined,
    tag: fit.tag || undefined,
    starred: fit.starred === '1' || undefined,
    scored: (fit.scored as FitScored) || 'any',
    min_score: numOrNull(fit.minScore),
    max_score: numOrNull(fit.maxScore),
    since: numOrNull(fit.since),
    until: numOrNull(fit.until),
    order: (fit.order as FitOrder) || 'new',
    limit: PAGE,
  }
}

/** "selection-0804-1530" (or "abstract-0804-1530" while a universe filter is
 *  on) — a name you can tell apart a week later. */
function defaultName(universe?: string): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  const head = universe && universe !== UNSET ? universe : 'selection'
  return `${head}-${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`
}

function Tile({
  rel,
  row,
  picked,
  onToggle,
  onOpen,
}: {
  rel: string
  row?: FitCandidate
  picked: boolean
  onToggle: () => void
  /** Open the fullscreen viewer here (a click on the tile toggles the pick). */
  onOpen: () => void
}) {
  return (
    <div className="group relative">
      <button
        className={cn(
          'relative block w-full overflow-hidden rounded-md border bg-black text-left',
          picked ? 'border-run ring-1 ring-run' : 'border-line hover:border-line2',
        )}
        onClick={onToggle}
        title={rel}
      >
        <img
          loading="lazy"
          src={imgSrc(`/img?path=${rel}`, row?.mtime ?? 0)}
          alt={rel}
          className={cn('aspect-square w-full object-contain',
            !picked && 'opacity-80 group-hover:opacity-100')}
        />
        <span
          className={cn(
            'absolute left-1 top-1 flex h-5 w-5 items-center justify-center rounded border',
            'text-[11px] leading-none',
            picked ? 'border-run bg-run text-black' : 'border-line bg-black/70 text-dim',
          )}
          aria-hidden
        >
          {picked ? '✓' : ''}
        </span>
        <span className="absolute right-1 top-1 flex gap-1 text-[10px]">
          {row?.fav ? <span className="text-star">★</span> : null}
          {row?.score != null ? (
            <span className="rounded bg-black/70 px-1 text-score tabular-nums">{row.score}</span>
          ) : null}
        </span>
      </button>
      <button
        className="absolute bottom-1 right-1 flex h-5 w-5 items-center justify-center rounded border
                   border-line bg-black/70 text-[12px] leading-none text-ink opacity-0
                   transition-opacity hover:border-line2 focus:opacity-100 group-hover:opacity-100"
        title="view full size, with every per-image action"
        aria-label="view full size"
        onClick={onOpen}
      >
        ⤢
      </button>
    </div>
  )
}

export function FitPage() {
  const fit = useUI((s) => s.fit)
  const setFit = useUI((s) => s.setFit)
  const sel = useUI((s) => s.fitSel)
  const toggleSel = useUI((s) => s.toggleFitSel)
  const openLightbox = useUI((s) => s.openLightbox)
  const addSel = useUI((s) => s.addFitSel)
  const removeSel = useUI((s) => s.removeFitSel)
  const clearSel = useUI((s) => s.clearFitSel)
  const backend = useUI((s) => s.params.backend) as BackendId
  const modelKey = useUI((s) => s.params.model)

  const [view, setView] = useState<'matching' | 'selected'>('matching')
  const [note, setNote] = useState<string | null>(null)
  const [useAfter, setUseAfter] = useState(true)
  const [fitting, setFitting] = useState<number | null>(null)
  const [selectingAll, setSelectingAll] = useState(false)

  const query = useMemo(() => toQuery(fit), [fit])
  const { data, isFetching } = useFitCandidates(query)
  const create = useCreateFit()
  const setDist = useSetDist()
  const watchJob = useJobWatcher()
  const qc = useQueryClient()
  const { data: current } = useDistConfig(backend, backend === 'sdxl' ? modelKey : null)

  const rows = useMemo(() => data?.rows ?? [], [data])
  const byRel = useMemo(() => new Map(rows.map((r) => [r.rel, r])), [rows])
  const picked = useMemo(() => new Set(sel), [sel])

  // The fit uses exactly the checkmarks on screen. The stored selection can
  // also hold picks made under an EARLIER filter (it is its own persisted
  // list, fed from cards and the viewer too); in the matching view those are
  // invisible, so they are left out of the fit — not silently added to it —
  // and named below. The "all picked" view shows (and fits) every pick.
  // Only decidable when every match is on screen; past that, all picks count.
  const complete = data != null && data.total <= rows.length
  const matchSet = useMemo(() => new Set(rows.map((r) => r.rel)), [rows])
  const outside = useMemo(
    () => (view === 'matching' && complete ? sel.filter((r) => !matchSet.has(r)) : []),
    [view, complete, sel, matchSet],
  )
  const fitRels = useMemo(
    () => (outside.length ? sel.filter((r) => matchSet.has(r)) : sel),
    [outside, sel, matchSet],
  )

  /**
   * The selection, split by the model that made each image.
   *
   * A fit is PER BACKEND -- sd15 conditioning is (77, 768) and sdxl's is
   * (77, 2048) plus a pooled vector, so one distribution cannot span both. The
   * server refuses a mixed selection, but by then you have already built it;
   * this is the same fact shown while you are choosing.
   */
  const byBackend = useMemo(() => {
    const m = new Map<string, string[]>()
    for (const rel of fitRels) {
      const b = backendOf(rel) ?? 'unknown'
      m.set(b, [...(m.get(b) ?? []), rel])
    }
    return m
  }, [fitRels])
  const mixed = byBackend.size > 1
  const keepOnly = (b: string) => removeSel(sel.filter((r) => (backendOf(r) ?? 'unknown') !== b))
  const shownMatching = rows.map((r) => r.rel)
  const allMatchingPicked = shownMatching.length > 0 && shownMatching.every((r) => picked.has(r))

  /** Check every match, not just the page on screen — hence the second fetch. */
  async function selectAll() {
    setNote(null)
    if ((data?.total ?? 0) <= rows.length) {
      return addSel(shownMatching)
    }
    setSelectingAll(true)
    try {
      const all = await api.fitCandidates({ ...query, limit: SELECT_ALL_LIMIT })
      addSel(all.rows.map((r) => r.rel))
      if (all.total > all.rows.length) {
        setNote(`selected the first ${all.rows.length} of ${all.total} matches — narrow the filter to reach the rest`)
      }
    } catch (e) {
      setNote((e as Error).message)
    } finally {
      setSelectingAll(false)
    }
  }

  /** `force` rather than reading state: the overwrite confirm submits in the
   *  same tick it is granted, and a setState wouldn't have landed yet. */
  function submit(force = false) {
    setNote(null)
    const name = fit.name.trim() || defaultName(fit.tag)
    if (!fit.name.trim()) setFit('name', name)
    create.mutate(
      {
        name,
        rels: fitRels,
        note: fit.note.trim() || null,
        components: numOrNull(fit.components),
        overwrite: force,
      },
      {
        onSuccess: (res) => {
          setFitting(res.job_id)
          watchJob(res.job_id, (status) => {
            setFitting(null)
            qc.invalidateQueries({ queryKey: ['fitList'] })
            if (status !== 'done') {
              setNote(`fit ${status} — see the job log`)
              return
            }
            if (useAfter) {
              setDist.mutate({
                backend: res.backend,
                kind: 'file',
                path: res.file,
                model: res.backend === 'sdxl' ? modelKey : null,
              })
            }
          })
        },
        onError: (e) => setNote((e as Error).message),
      },
    )
  }

  const conflict = create.isError && (create.error as Error).message.includes('already exists')
  const tooFew = fitRels.length < MIN_SAMPLES
  const nFit = fitRels.length
  const busy = create.isPending || fitting != null

  const gallery = view === 'selected' ? sel : shownMatching
  // The fullscreen viewer walks the pool in the order it is drawn (the
  // selected view is grouped by backend).
  const viewerItems = useMemo(
    () =>
      (view === 'selected' ? [...byBackend.values()].flat() : rows.map((r) => r.rel)).map((rel) =>
        stubItem(rel, { mtime: byRel.get(rel)?.mtime ?? 0, fav: byRel.get(rel)?.fav ?? false }),
      ),
    [view, byBackend, rows, byRel],
  )
  useViewerList('fit', viewerItems)

  return (
    <div className="flex flex-col gap-3 pb-4">
      <FitFilters matched={data?.total} />

      {/* what will be fitted, and the button that does it — one row, plus one
          dim line of context */}
      <section className="sa-panel shrink-0 px-3 py-2">
        <div className="flex flex-wrap items-center gap-1.5">
          {/* Checking never forgets picks made under other filters: they are
              simply not in this view's fit (see `fitRels`). */}
          <button
            className={cn('sa-btn sa-btn-sm', !allMatchingPicked && rows.length > 0 && 'sa-btn-sel')}
            disabled={selectingAll || !rows.length || allMatchingPicked}
            onClick={selectAll}
            title="check every image matching the filter"
          >
            ✓ check all {data?.total ?? 0}
          </button>
          <button
            className="sa-btn sa-btn-sm"
            disabled={!allMatchingPicked && !shownMatching.some((r) => picked.has(r))}
            onClick={() => removeSel(shownMatching)}
            title="uncheck the images on screen"
          >
            uncheck
          </button>
          <div className="flex overflow-hidden rounded-md border border-line">
            {(['matching', 'selected'] as const).map((v) => (
              <button
                key={v}
                className={cn('px-2 py-[3px] text-[12px]',
                  view === v ? 'bg-panel2 text-ink' : 'text-dim hover:text-ink')}
                onClick={() => setView(v)}
                title={v === 'matching' ? 'the images this filter matches' : 'every image you have picked, under any filter'}
              >
                {v === 'matching' ? `matching ${data?.total ?? 0}` : `all picked ${sel.length}`}
              </button>
            ))}
          </div>

          <span className="ml-auto" />
          <input
            className="sa-input !w-[200px] !py-[4px] !text-[12px]"
            placeholder={defaultName(fit.tag)}
            value={fit.name}
            onChange={(e) => setFit('name', e.target.value)}
            title="what the fit is called on disk (outputs/dist_fits/<name>)"
          />
          <button
            className={cn('sa-btn', conflict ? 'border-err text-danger' : 'sa-btn-sel')}
            disabled={busy || tooFew || mixed}
            onClick={() => submit(conflict)}
            title={
              mixed
                ? 'the checked images mix models — a distribution is per backend'
                : tooFew
                  ? `check at least ${MIN_SAMPLES} images`
                  : `fit a distribution to these ${nFit} latents (no GPU — a few seconds)`
            }
          >
            {fitting != null
              ? `fitting… #${fitting}`
              : mixed
                ? '🧬 one model at a time'
                : conflict
                  ? '↻ Overwrite that fit?'
                  : `🧬 Fit ${nFit} checked`}
          </button>
        </div>

        {/* One dim line: what the fit will be, what happens after, and any
            picks the fit is leaving out. */}
        <div className="mt-1.5 flex flex-wrap items-center gap-x-3 gap-y-1 text-[12px] text-dim">
          <span
            className={cn(tooFew && 'text-err')}
            title={
              'a fit gets its own mean, spread and (N−1)-axis PCA subspace. Sample it with ' +
              'sampler = pca to stay inside the span of what you picked ("more like these"); ' +
              'diagonal on a small set goes off-manifold. Only images carrying conditioning ' +
              'are offered — an upscale carries none.'
            }
          >
            {tooFew
              ? `a fit needs at least ${MIN_SAMPLES} checked images`
              : nFit < THIN_SAMPLES
                ? `${nFit} checked — a thin fit (${nFit - 1} directions); samples stay close to these`
                : `${nFit} checked → ${nFit - 1}-axis fit · sample it with sampler = pca`}
          </span>
          <label
            className="flex items-center gap-1"
            title={current ? `sets the base distribution, replacing “${current.label}”` : 'sets the base distribution'}
          >
            <input type="checkbox" checked={useAfter} onChange={(e) => setUseAfter(e.target.checked)} />
            use it once fitted
          </label>
          <input
            className="sa-input !w-[240px] !py-[2px] !text-[12px]"
            placeholder="note (optional)"
            value={fit.note}
            onChange={(e) => setFit('note', e.target.value)}
            title="what this set is — shown with the saved fit"
          />
          {outside.length ? (
            <span className="flex items-center gap-1.5">
              + {outside.length} picked under other filters, not in this fit
              <button className="sa-btn sa-btn-sm" onClick={() => setView('selected')}>
                show
              </button>
              <button className="sa-btn sa-btn-sm" onClick={() => removeSel(outside)}>
                forget
              </button>
            </span>
          ) : null}
          {view === 'selected' && sel.length ? (
            <button className="sa-btn sa-btn-sm" onClick={clearSel} title="uncheck every pick, under any filter">
              clear all picks
            </button>
          ) : null}
        </div>

        {mixed ? (
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5 text-[12px] text-err">
            a fit is per model (sd15 conditioning is 77×768, sdxl's 77×2048 + pooled):
            {[...byBackend.entries()].map(([b, rels]) => (
              <button key={b} className="sa-btn sa-btn-sm" onClick={() => keepOnly(b)}>
                keep only {b} ({rels.length})
              </button>
            ))}
          </div>
        ) : null}
        {note ? <p className="sa-hint text-err">{note}</p> : null}
        {create.isError && !conflict ? (
          <p className="sa-hint text-err">{(create.error as Error).message}</p>
        ) : null}
      </section>

      <SavedFits />

      {/* the pool */}
      {!gallery.length ? (
        <p className="p-6 text-center text-[13px] text-dim">
          {view === 'selected'
            ? 'nothing selected yet — pick images from the grid, or hit the + box on any gallery card.'
            : isFetching
              ? 'loading…'
              : 'nothing matches these filters.'}
        </p>
      ) : (
        <>
          {view === 'selected' ? (
            // Grouped by model, because that is the axis along which the
            // selection is either usable or not.
            [...byBackend.entries()].map(([b, rels]) => (
              <div key={b} className="mb-3">
                <div className="mb-1.5 flex items-center gap-2 text-[12px] text-dim">
                  <span className="text-ink">{b}</span>
                  <span>
                    {rels.length} image{rels.length === 1 ? '' : 's'}
                  </span>
                  {mixed ? (
                    <button className="sa-btn sa-btn-sm" onClick={() => keepOnly(b)}>
                      keep only this
                    </button>
                  ) : null}
                  <span className="h-px flex-1 bg-line" />
                </div>
                <div
                  className="grid gap-2"
                  style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(112px, 1fr))' }}
                >
                  {rels.map((rel) => (
                    <Tile
                      key={rel}
                      rel={rel}
                      row={byRel.get(rel)}
                      picked={picked.has(rel)}
                      onToggle={() => toggleSel(rel)}
                      onOpen={() => openLightbox(rel, 'fit')}
                    />
                  ))}
                </div>
              </div>
            ))
          ) : (
            <div
              className="grid gap-2"
              style={{ gridTemplateColumns: 'repeat(auto-fill, minmax(112px, 1fr))' }}
            >
              {gallery.map((rel) => (
                <Tile
                  key={rel}
                  rel={rel}
                  row={byRel.get(rel)}
                  picked={picked.has(rel)}
                  onToggle={() => toggleSel(rel)}
                  onOpen={() => openLightbox(rel, 'fit')}
                />
              ))}
            </div>
          )}
          {view === 'matching' && (data?.total ?? 0) > rows.length ? (
            <p className="text-center text-[12px] text-dim">
              showing the first {rows.length} of {data?.total} matches — “select all”
              reaches up to {SELECT_ALL_LIMIT}
            </p>
          ) : null}
        </>
      )}
    </div>
  )
}
