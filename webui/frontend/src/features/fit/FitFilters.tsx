/**
 * Which images are candidates for the next distribution fit.
 *
 * Same facet vocabulary as the labeling queue — deliberately: "everything from
 * E07, rendered in the last 24h" has to mean one thing across the app, and the
 * options come from the same `/api/label/facets` computed over the whole
 * gallery. What this adds is the two dimensions you select *by* once you have
 * labeled: the star and the label-score band.
 */
import { useLabelFacets } from '../../api/queries'
import type { FacetCell, FacetDim } from '../../api/types'
import { UNSET } from '../../api/types'
import { Select } from '../../components/ui/Select'
import type { Option } from '../../params/schema'
import { cn } from '../../lib/utils'
import { useUI } from '../../store'
import { universeColor } from '../universes/universes'

/** `labels.KEEPER_MIN` — the score at which a batch counts as a keeper. */
export const KEEPER_MIN = 7

const DIMS: { dim: FacetDim; label: string; any: string; width: number }[] = [
  { dim: 'backend', label: 'Backend', any: 'any backend', width: 120 },
  { dim: 'ckpt', label: 'Checkpoint', any: 'any checkpoint', width: 180 },
  { dim: 'size', label: 'Resolution', any: 'any size', width: 130 },
  { dim: 'folder', label: 'Folder', any: 'any folder', width: 140 },
  { dim: 'kind', label: 'Made by', any: 'any source', width: 130 },
  { dim: 'sampler', label: 'Sampler', any: 'any sampler', width: 130 },
]
// The universe ('tag') is NOT a dropdown here: fitting one aesthetic universe
// is the commonest reason to open this tab, so it gets its own chip row on top.

const SCORED: Option[] = [
  { value: 'any', label: 'labeled or not' },
  { value: 'labeled', label: 'labeled only' },
  { value: 'unlabeled', label: 'unlabeled only' },
]

const ORDERS: Option[] = [
  { value: 'new', label: 'newest first' },
  { value: 'old', label: 'oldest first' },
  { value: 'score', label: 'label score ↓' },
  { value: 'distance', label: 'distance ↓' },
]

/** 0–9, plus "any". Scores are integers, so a band is two dropdowns. */
function scoreOptions(anyLabel: string): Option[] {
  return [
    { value: '', label: anyLabel },
    ...Array.from({ length: 10 }, (_, i) => ({ value: String(i), label: String(i) })),
  ]
}

export const WINDOWS: { label: string; hours: number | null }[] = [
  { label: 'all time', hours: null },
  { label: '1h', hours: 1 },
  { label: '6h', hours: 6 },
  { label: '24h', hours: 24 },
  { label: '7d', hours: 24 * 7 },
  { label: '30d', hours: 24 * 30 },
]

export function toLocalInput(sec: string): string {
  if (!sec) return ''
  const d = new Date(Number(sec) * 1000)
  if (Number.isNaN(d.getTime())) return ''
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(
    d.getHours(),
  )}:${pad(d.getMinutes())}`
}
export function fromLocalInput(v: string): string {
  if (!v) return ''
  const t = new Date(v).getTime()
  return Number.isNaN(t) ? '' : String(Math.floor(t / 1000))
}

export function facetOptions(cells: FacetCell[] | undefined, anyLabel: string): Option[] {
  const opts: Option[] = [{ value: '', label: anyLabel }]
  for (const c of cells ?? []) {
    opts.push({
      value: c.value,
      label: `${c.value === UNSET ? '(none recorded)' : c.value} · ${c.count}`,
      title: `${c.count} image(s)`,
    })
  }
  return opts
}

export function FitFilters({ matched }: { matched?: number }) {
  const fit = useUI((s) => s.fit)
  const setFit = useUI((s) => s.setFit)
  const reset = useUI((s) => s.resetFitFilters)
  const { data: facets } = useLabelFacets()

  const activeWindow = (() => {
    if (!fit.since && !fit.until) return 'all time'
    if (fit.until) return 'custom'
    const hours = (Date.now() / 1000 - Number(fit.since)) / 3600
    const hit = WINDOWS.find((w) => w.hours && Math.abs(w.hours - hours) < w.hours * 0.02)
    return hit?.label ?? 'custom'
  })()

  // Frozen to an absolute instant on click, exactly as the labeling queue does:
  // a `since` that tracked the clock would refetch on every render.
  const pickWindow = (hours: number | null) => {
    setFit('since', hours == null ? '' : String(Math.floor(Date.now() / 1000 - hours * 3600)))
    setFit('until', '')
  }


  const tagCount = facets?.facets?.tag?.find((c) => c.value === fit.tag)?.count ?? 0
  const otherFilters = ['experiment', ...DIMS.map((d) => d.dim), 'since', 'until', 'starred',
    'minScore', 'maxScore'].some((k) => fit[k]) || fit.scored !== 'any'
  const dirty = !!fit.tag || otherFilters

  return (
    <div className="sa-panel shrink-0 px-3 py-2.5">
      {/* One click = "the images of this universe". Counts are over the whole
          gallery; the grid below then keeps only those that carry latents. */}
      <div className="mb-2 flex flex-wrap items-center gap-1.5 border-b border-line pb-2">
        <span className="mr-1 text-[12px] text-ink">🪐 Universe</span>
        <button
          className={cn('sa-btn sa-btn-sm', !fit.tag && 'sa-btn-sel')}
          onClick={() => setFit('tag', '')}
        >
          any
        </button>
        {(facets?.facets?.tag ?? []).map((c) => {
          const on = fit.tag === c.value
          const none = c.value === UNSET
          return (
            <button
              key={c.value}
              className={cn('sa-btn sa-btn-sm inline-flex items-center gap-1.5', on && 'sa-btn-sel')}
              onClick={() => setFit('tag', on ? '' : c.value)}
              title={none ? 'images in no universe yet' : `only images in “${c.value}”`}
            >
              {none ? null : (
                <span className="h-2 w-2 rounded-full" style={{ background: universeColor(c.value) }} />
              )}
              {none ? '(no universe)' : c.value}
              <span className="tabular-nums text-dim">{c.count}</span>
            </button>
          )
        })}
        {!facets?.facets?.tag?.length ? (
          <span className="text-[12px] text-dim">
            none yet — tag images into a universe from any card or the viewer
          </span>
        ) : null}
        {/* "abstract 12" but 5 match: say the other filters are why, and offer
            the one click that shows the whole universe. (Images without latents,
            e.g. upscales, never match — that part of a gap can't be cleared.) */}
        {fit.tag && matched != null && otherFilters && matched < tagCount ? (
          <span className="ml-auto flex items-center gap-1.5 text-[12px] text-dim">
            {matched} of {tagCount} match with the other filters below
            <button
              className="sa-btn sa-btn-sm"
              onClick={() => {
                const tag = fit.tag
                reset()
                setFit('tag', tag)
              }}
            >
              clear the others
            </button>
          </span>
        ) : null}
      </div>

      <div className="flex flex-wrap items-end gap-x-3 gap-y-2">
        <div style={{ width: 170 }}>
          <span className="sa-label !mt-0">Experiment</span>
          <Select
            value={fit.experiment}
            onChange={(v) => setFit('experiment', v)}
            options={facetOptions(facets?.facets?.experiment, 'any experiment')}
            ariaLabel="Experiment"
            className="!py-[5px] !text-[12px]"
          />
        </div>
        {DIMS.map(({ dim, label, any, width }) => (
          <div key={dim} style={{ width }}>
            <span className="sa-label !mt-0">{label}</span>
            <Select
              value={fit[dim] ?? ''}
              onChange={(v) => setFit(dim, v)}
              options={facetOptions(facets?.facets?.[dim], any)}
              ariaLabel={label}
              className="!py-[5px] !text-[12px]"
            />
          </div>
        ))}
        <div style={{ width: 140 }}>
          <span className="sa-label !mt-0">Order</span>
          <Select
            value={fit.order}
            onChange={(v) => setFit('order', v)}
            options={ORDERS}
            ariaLabel="order"
            className="!py-[5px] !text-[12px]"
          />
        </div>
      </div>

      {/* the two taste dimensions: what you starred, and what you scored */}
      <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-line pt-2">
        <button
          className={cn('sa-btn sa-btn-sm', fit.starred === '1' && 'sa-btn-sel')}
          onClick={() => setFit('starred', fit.starred === '1' ? '' : '1')}
          title="only images you starred"
        >
          ★ starred only
        </button>
        <span className="ml-1 text-[12px] text-dim">Label score</span>
        <Select
          value={fit.minScore}
          onChange={(v) => setFit('minScore', v)}
          options={scoreOptions('any')}
          ariaLabel="minimum label score"
          title="lowest label score to include"
          className="!w-[86px] !py-[4px] !text-[12px]"
        />
        <span className="text-[12px] text-dim">to</span>
        <Select
          value={fit.maxScore}
          onChange={(v) => setFit('maxScore', v)}
          options={scoreOptions('any')}
          ariaLabel="maximum label score"
          title="highest label score to include"
          className="!w-[86px] !py-[4px] !text-[12px]"
        />
        <button
          className={cn('sa-btn sa-btn-sm',
            fit.minScore === String(KEEPER_MIN) && !fit.maxScore && 'sa-btn-sel')}
          onClick={() => {
            setFit('minScore', String(KEEPER_MIN))
            setFit('maxScore', '')
          }}
          title={`the keeper band — score ${KEEPER_MIN} and up`}
        >
          keepers {KEEPER_MIN}+
        </button>
        <Select
          value={fit.scored}
          onChange={(v) => setFit('scored', v)}
          options={SCORED}
          ariaLabel="labeled or not"
          className="!w-[150px] !py-[4px] !text-[12px]"
        />
        {/* A score band only ever admits scored images, so say so rather than
            letting an empty grid look like a bug. */}
        {(fit.minScore || fit.maxScore) && fit.scored === 'unlabeled' ? (
          <span className="text-[11px] text-err">
            a score band and “unlabeled only” can never overlap
          </span>
        ) : null}
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-line pt-2">
        <span className="text-[12px] text-dim">Rendered</span>
        {WINDOWS.map((w) => (
          <button
            key={w.label}
            className={cn('sa-btn sa-btn-sm', activeWindow === w.label && 'sa-btn-sel')}
            onClick={() => pickWindow(w.hours)}
          >
            {w.label}
          </button>
        ))}
        <input
          type="datetime-local"
          className="sa-input !w-[180px] !py-[4px] !text-[12px]"
          value={toLocalInput(fit.since)}
          onChange={(e) => setFit('since', fromLocalInput(e.target.value))}
        />
        <span className="text-[12px] text-dim">to</span>
        <input
          type="datetime-local"
          className="sa-input !w-[180px] !py-[4px] !text-[12px]"
          value={toLocalInput(fit.until)}
          onChange={(e) => setFit('until', fromLocalInput(e.target.value))}
        />
        <span className="ml-auto text-[12px] text-dim">
          {matched == null ? '…' : (
            <>
              <span className="text-ink tabular-nums">{matched}</span> image
              {matched === 1 ? '' : 's'} match
            </>
          )}
        </span>
        <button
          className="sa-btn sa-btn-sm"
          onClick={reset}
          disabled={!dirty}
          title="clear every filter (keeps the order and the fit’s name)"
        >
          clear filters
        </button>
      </div>
    </div>
  )
}
