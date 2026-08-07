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

const WINDOWS: { label: string; hours: number | null }[] = [
  { label: 'all time', hours: null },
  { label: '1h', hours: 1 },
  { label: '6h', hours: 6 },
  { label: '24h', hours: 24 },
  { label: '7d', hours: 24 * 7 },
  { label: '30d', hours: 24 * 30 },
]

function toLocalInput(sec: string): string {
  if (!sec) return ''
  const d = new Date(Number(sec) * 1000)
  if (Number.isNaN(d.getTime())) return ''
  const pad = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(
    d.getHours(),
  )}:${pad(d.getMinutes())}`
}
function fromLocalInput(v: string): string {
  if (!v) return ''
  const t = new Date(v).getTime()
  return Number.isNaN(t) ? '' : String(Math.floor(t / 1000))
}

function facetOptions(cells: FacetCell[] | undefined, anyLabel: string): Option[] {
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

  const dirty = ['experiment', ...DIMS.map((d) => d.dim), 'since', 'until', 'starred',
    'minScore', 'maxScore'].some((k) => fit[k]) || fit.scored !== 'any'

  return (
    <div className="sa-panel shrink-0 px-3 py-2.5">
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
