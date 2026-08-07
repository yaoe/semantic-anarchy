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
 * ⊕ on any gallery card adds to this same list).
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
import { cn, imgSrc } from '../../lib/utils'
import { useUI } from '../../store'
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

/** "selection-0804-1530" — a name you can tell apart a week later. */
function defaultName(): string {
  const d = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  return `selection-${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`
}

function Tile({
  rel,
  row,
  picked,
  onToggle,
}: {
  rel: string
  row?: FitCandidate
  picked: boolean
  onToggle: () => void
}) {
  return (
    <button
      className={cn(
        'group relative block overflow-hidden rounded-md border bg-black text-left',
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
  )
}

export function FitPage() {
  const fit = useUI((s) => s.fit)
  const setFit = useUI((s) => s.setFit)
  const sel = useUI((s) => s.fitSel)
  const toggleSel = useUI((s) => s.toggleFitSel)
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
  const shownMatching = rows.map((r) => r.rel)
  const allMatchingPicked = shownMatching.length > 0 && shownMatching.every((r) => picked.has(r))

  /** Select every match, not just the page on screen — hence the second fetch. */
  async function selectAll() {
    setNote(null)
    if ((data?.total ?? 0) <= rows.length) return addSel(shownMatching)
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
    const name = fit.name.trim() || defaultName()
    if (!fit.name.trim()) setFit('name', name)
    create.mutate(
      {
        name,
        rels: sel,
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
  const tooFew = sel.length < MIN_SAMPLES
  const busy = create.isPending || fitting != null

  const gallery = view === 'selected' ? sel : shownMatching

  return (
    <div className="flex flex-col gap-3 pb-4">
      <FitFilters matched={data?.total} />

      {/* what will be fitted, and the button that does it */}
      <section className="sa-panel shrink-0 px-3 py-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <button
            className="sa-btn sa-btn-sm"
            disabled={selectingAll || !rows.length}
            onClick={selectAll}
            title="add every image matching the filter to the selection"
          >
            ＋ select all {data?.total ?? 0} matching
          </button>
          <button
            className="sa-btn sa-btn-sm"
            disabled={!allMatchingPicked && !shownMatching.some((r) => picked.has(r))}
            onClick={() => removeSel(shownMatching)}
            title="drop the images on screen from the selection"
          >
            − deselect these
          </button>
          <button
            className="sa-btn sa-btn-sm"
            disabled={!sel.length}
            onClick={clearSel}
            title="empty the selection"
          >
            clear selection
          </button>
          <div className="ml-1 flex overflow-hidden rounded-md border border-line">
            {(['matching', 'selected'] as const).map((v) => (
              <button
                key={v}
                className={cn('px-2 py-[5px] text-[12px]',
                  view === v ? 'bg-panel2 text-ink' : 'text-dim hover:text-ink')}
                onClick={() => setView(v)}
              >
                {v === 'matching'
                  ? `matching ${data?.total ?? 0}`
                  : `selected ${sel.length}`}
              </button>
            ))}
          </div>

          <input
            className="sa-input ml-auto !w-[190px] !py-[5px] !text-[12px]"
            placeholder={defaultName()}
            value={fit.name}
            onChange={(e) => setFit('name', e.target.value)}
            title="what the fit is called on disk (outputs/dist_fits/<name>)"
          />
          <input
            className="sa-input !w-[220px] !py-[5px] !text-[12px]"
            placeholder="note — what this set is (optional)"
            value={fit.note}
            onChange={(e) => setFit('note', e.target.value)}
          />
          <button
            className={cn('sa-btn', conflict ? 'border-err text-danger' : 'sa-btn-sel')}
            disabled={busy || tooFew}
            onClick={() => submit(conflict)}
            title={
              tooFew
                ? `select at least ${MIN_SAMPLES} images`
                : 'fit a distribution to these latents (no GPU — a few seconds)'
            }
          >
            {fitting != null
              ? `fitting… #${fitting}`
              : conflict
                ? '↻ Overwrite that fit?'
                : `🧬 Fit latent distribution on ${sel.length} selected sample${sel.length === 1 ? '' : 's'}`}
          </button>
        </div>

        <p className="sa-hint">
          {tooFew ? (
            <>
              a fit needs at least {MIN_SAMPLES} images — {sel.length} selected.
            </>
          ) : sel.length < THIN_SAMPLES ? (
            <>
              {sel.length} selected — that works, but the fit spans only{' '}
              {sel.length - 1} directions, so samples will stay close to these few.
            </>
          ) : (
            <>
              {sel.length} selected — the fit gets its own mean, spread and{' '}
              {sel.length - 1}-axis PCA subspace. Sample it with{' '}
              <b>sampler = pca</b>: those draws stay inside the span of the latents
              you picked, which is what “more like these” means. (diagonal on a
              small selection goes off-manifold, same as on the corpus.)
            </>
          )}
          {' '}Only images carrying conditioning are offered — an upscale carries none.
        </p>
        {note ? <p className="sa-hint text-err">{note}</p> : null}
        {create.isError && !conflict ? (
          <p className="sa-hint text-err">{(create.error as Error).message}</p>
        ) : null}
        <label className="mt-1 flex items-center gap-1.5 text-[12px] text-dim">
          <input
            type="checkbox"
            checked={useAfter}
            onChange={(e) => setUseAfter(e.target.checked)}
          />
          sample from it once it’s fitted (sets the base distribution
          {current ? `, replacing “${current.label}”` : ''})
        </label>
      </section>

      <SavedFits />

      {/* the pool */}
      {!gallery.length ? (
        <p className="p-6 text-center text-[13px] text-dim">
          {view === 'selected'
            ? 'nothing selected yet — pick images from the grid, or hit ⊕ on any gallery card.'
            : isFetching
              ? 'loading…'
              : 'nothing matches these filters.'}
        </p>
      ) : (
        <>
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
              />
            ))}
          </div>
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
