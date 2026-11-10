/**
 * Which images are candidates for the next distribution fit.
 *
 * Same facet vocabulary as the labeling queue — deliberately: "everything from
 * E07, rendered in the last 24h" has to mean one thing across the app, and the
 * options come from the same `/api/label/facets` computed over the whole
 * gallery. What this adds is the two dimensions you select *by* once you have
 * labeled: the star and the label-score band.
 */
import { useState } from 'react'

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
  // Everything but the universe, the star, the keeper band and the time window
  // folds away: those four are what a fit is usually chosen by.
  const [more, setMore] = useState(false)

  const activeWindow = (() => {
    if (!fit.since && !fit.until) return 'all time'
    if (fit.until) return 'custom'
    const hours = (Date.now() / 1000 - Number(fit.since)) / 3600
    const hit = WINDOWS.find((w) => w.hours && Math.abs(w.hours - hours) < w.hours * 0.02)
    return hit?.label ?? 'custom'
  })()

  // Frozen to an absolute instant on click, exactly as the labeling queue does:
  // a `since` that tracked the clock would refetch on every render.
  const pickWindow = (label: string) => {
    const w = WINDOWS.find((x) => x.label === label)
    if (!w) return setMore(true) // "custom" lives with the date inputs
    setFit('since', w.hours == null ? '' : String(Math.floor(Date.now() / 1000 - w.hours * 3600)))
    setFit('until', '')
  }

  const keepers = fit.minScore === String(KEEPER_MIN) && !fit.maxScore
  const tagCount = facets?.facets?.tag?.find((c) => c.value === fit.tag)?.count ?? 0
  const otherFilters = ['experiment', ...DIMS.map((d) => d.dim), 'since', 'until', 'starred',
    'minScore', 'maxScore'].some((k) => fit[k]) || fit.scored !== 'any'
  const dirty = !!fit.tag || otherFilters

  // The folded filters that are ON, as removable chips — so folding never hides
  // why the grid is smaller than the universe.
  const hidden: { key: string; text: string; clear: () => void }[] = [
    ...(['experiment', ...DIMS.map((d) => d.dim)] as string[])
      .filter((k) => fit[k])
      .map((k) => ({
        key: k,
        text: fit[k] === UNSET ? `${k}: none` : fit[k],
        clear: () => setFit(k, ''),
      })),
    ...((fit.minScore || fit.maxScore) && !keepers
      ? [{
          key: 'score', text: `score ${fit.minScore || '0'}–${fit.maxScore || '9'}`,
          clear: () => { setFit('minScore', ''); setFit('maxScore', '') },
        }]
      : []),
    ...(fit.scored !== 'any'
      ? [{ key: 'scored', text: fit.scored, clear: () => setFit('scored', 'any') }]
      : []),
  ]

  const win = activeWindow
  return (
    <div className="sa-panel shrink-0 px-3 py-2">
      {/* One click = "the images of this universe". Counts are over the whole
          gallery; the grid below then keeps only those that carry latents. */}
      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-0.5 text-[12px] text-dim">🪐 universe</span>
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
              onClick={() => {
                setFit('tag', on ? '' : c.value)
                // A name typed for another set would overwrite that fit; let
                // the placeholder ("<universe>-MMDD-HHMM") take over instead.
                setFit('name', '')
              }}
              title={none ? 'images in no universe yet' : `only images in “${c.value}”`}
            >
              {none ? null : (
                <span className="h-2 w-2 rounded-full" style={{ background: universeColor(c.value) }} />
              )}
              {none ? 'no universe' : c.value}
              <span className="tabular-nums text-dim">{c.count}</span>
            </button>
          )
        })}
        {!facets?.facets?.tag?.length ? (
          <span className="text-[12px] text-dim">
            no universes yet — tag images from any card or the viewer
          </span>
        ) : null}
      </div>

      <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
        <button
          className={cn('sa-btn sa-btn-sm', fit.starred === '1' && 'sa-btn-sel')}
          onClick={() => setFit('starred', fit.starred === '1' ? '' : '1')}
          title="only images you starred"
        >
          ★ starred
        </button>
        <button
          className={cn('sa-btn sa-btn-sm', keepers && 'sa-btn-sel')}
          onClick={() => {
            setFit('minScore', keepers ? '' : String(KEEPER_MIN))
            setFit('maxScore', '')
          }}
          title={`only images you scored ${KEEPER_MIN} or higher`}
        >
          keepers {KEEPER_MIN}+
        </button>
        <Select
          value={win}
          onChange={pickWindow}
          options={[
            ...WINDOWS.map((w) => ({ value: w.label, label: w.label === 'all time' ? 'any time' : `last ${w.label}` })),
            ...(win === 'custom' ? [{ value: 'custom', label: 'custom range' }] : []),
          ]}
          ariaLabel="rendered within"
          className="!w-[118px] !py-[3px] !text-[12px]"
        />
        <button
          className={cn('sa-btn sa-btn-sm', more && 'sa-btn-sel')}
          onClick={() => setMore((v) => !v)}
          aria-expanded={more}
        >
          {more ? '▾' : '▸'} more filters
        </button>
        {!more
          ? hidden.map((h) => (
              <button
                key={h.key}
                className="inline-flex items-center gap-1 rounded-full border border-line2 px-2 py-px
                           text-[11px] text-dim hover:border-err hover:text-ink"
                onClick={h.clear}
                title="active filter — click to remove"
              >
                {h.text} <span aria-hidden>×</span>
              </button>
            ))
          : null}

        <span className="ml-auto flex items-center gap-2 text-[12px] text-dim">
          {/* "abstract 12" but 5 match: name the other filters as the reason.
              (Upscales count via their original's latents; an image whose
              chain leads to no .npz never matches.) */}
          {fit.tag && matched != null && otherFilters && matched < tagCount ? (
            <button
              className="sa-btn sa-btn-sm"
              onClick={() => {
                const tag = fit.tag
                reset()
                setFit('tag', tag)
              }}
              title="the other filters hide part of this universe"
            >
              show all {tagCount} in this universe
            </button>
          ) : null}
          <span>
            <span className="tabular-nums text-ink">{matched ?? '…'}</span> match
          </span>
          {dirty ? (
            <button className="text-dim underline-offset-2 hover:text-ink hover:underline" onClick={reset}>
              reset
            </button>
          ) : null}
        </span>
      </div>

      {more ? (
        <div className="mt-2 flex flex-wrap items-end gap-x-3 gap-y-2 border-t border-line pt-2">
          <div style={{ width: 170 }}>
            <span className="sa-label !mt-0">Experiment</span>
            <Select
              value={fit.experiment}
              onChange={(v) => setFit('experiment', v)}
              options={facetOptions(facets?.facets?.experiment, 'any experiment')}
              ariaLabel="Experiment"
              className="!py-[4px] !text-[12px]"
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
                className="!py-[4px] !text-[12px]"
              />
            </div>
          ))}
          <div>
            <span className="sa-label !mt-0">Label score</span>
            <div className="flex items-center gap-1">
              <Select
                value={fit.minScore}
                onChange={(v) => setFit('minScore', v)}
                options={scoreOptions('any')}
                ariaLabel="minimum label score"
                className="!w-[70px] !py-[4px] !text-[12px]"
              />
              <span className="text-[12px] text-dim">–</span>
              <Select
                value={fit.maxScore}
                onChange={(v) => setFit('maxScore', v)}
                options={scoreOptions('any')}
                ariaLabel="maximum label score"
                className="!w-[70px] !py-[4px] !text-[12px]"
              />
              <Select
                value={fit.scored}
                onChange={(v) => setFit('scored', v)}
                options={SCORED}
                ariaLabel="labeled or not"
                className="!w-[140px] !py-[4px] !text-[12px]"
              />
            </div>
          </div>
          <div>
            <span className="sa-label !mt-0">Rendered between</span>
            <div className="flex items-center gap-1">
              <input
                type="datetime-local"
                className="sa-input !w-[176px] !py-[3px] !text-[12px]"
                value={toLocalInput(fit.since)}
                onChange={(e) => setFit('since', fromLocalInput(e.target.value))}
              />
              <span className="text-[12px] text-dim">–</span>
              <input
                type="datetime-local"
                className="sa-input !w-[176px] !py-[3px] !text-[12px]"
                value={toLocalInput(fit.until)}
                onChange={(e) => setFit('until', fromLocalInput(e.target.value))}
              />
            </div>
          </div>
          <div style={{ width: 140 }}>
            <span className="sa-label !mt-0">Order</span>
            <Select
              value={fit.order}
              onChange={(v) => setFit('order', v)}
              options={ORDERS}
              ariaLabel="order"
              className="!py-[4px] !text-[12px]"
            />
          </div>
          {/* A score band only ever admits scored images. */}
          {(fit.minScore || fit.maxScore) && fit.scored === 'unlabeled' ? (
            <span className="basis-full text-[11px] text-err">
              a score band and “unlabeled only” can never overlap
            </span>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
