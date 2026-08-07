/**
 * Which images to label — the queue builder above the labeling page.
 *
 * Every control is an equality filter on something the image itself carries
 * (experiment, backend, checkpoint, folder, resolution, kind, sampler) plus a
 * time window on when it was rendered. The options come from
 * `/api/label/facets`, computed over the WHOLE labelable set with counts, so a
 * choice never makes another option disappear — how many images a given
 * combination actually matches is reported live by the queue itself.
 */
import { useExperiments, useLabelFacets } from '../../api/queries'
import type { FacetCell, FacetDim, LabelQueue } from '../../api/types'
import { UNSET } from '../../api/types'
import { Select } from '../../components/ui/Select'
import type { Option } from '../../params/schema'
import { cn } from '../../lib/utils'
import { useUI } from '../../store'

/** Facet dimensions in the order they are offered, with their "any" wording.
 *  `experiment` is not here: it gets its options from /api/experiments instead,
 *  which knows each id's hypothesis and whether it ran on the seed panel. */
const DIMS: { dim: FacetDim; label: string; any: string; width: number }[] = [
  { dim: 'backend', label: 'Backend', any: 'any backend', width: 130 },
  { dim: 'ckpt', label: 'Checkpoint', any: 'any checkpoint', width: 200 },
  { dim: 'size', label: 'Resolution', any: 'any size', width: 150 },
  { dim: 'folder', label: 'Folder', any: 'any folder', width: 150 },
  { dim: 'kind', label: 'Made by', any: 'any source', width: 140 },
  { dim: 'sampler', label: 'Sampler', any: 'any sampler', width: 140 },
]

const SCOPES = [
  { value: 'unlabeled', label: 'unlabeled only' },
  { value: 'all', label: 'all (relabel)' },
  { value: 'labeled', label: 'already labeled' },
]
const BUCKETS = [
  { value: 'generated', label: 'everything' },
  { value: 'favorites', label: '★ favorites only' },
]
const ORDERS = [
  { value: 'shuffle', label: 'shuffled', title: 'generation order correlates with the knobs — shuffle so it cannot bias the eye' },
  { value: 'new', label: 'newest first' },
  { value: 'old', label: 'oldest first' },
]

/** Relative windows, resolved to an ABSOLUTE `since` the moment they're picked. */
const WINDOWS: { label: string; hours: number | null }[] = [
  { label: 'all time', hours: null },
  { label: '1h', hours: 1 },
  { label: '6h', hours: 6 },
  { label: '24h', hours: 24 },
  { label: '7d', hours: 24 * 7 },
  { label: '30d', hours: 24 * 30 },
]

/** unix seconds <-> the value a `datetime-local` input holds (local time). */
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
    const name = c.value === UNSET ? '(none recorded)' : c.value
    opts.push({
      value: c.value,
      label: `${name} · ${c.unlabeled}/${c.count}`,
      title: `${c.count} image(s), ${c.unlabeled} still unlabeled`,
    })
  }
  return opts
}

export function QueueBar({ queue }: { queue?: LabelQueue }) {
  const label = useUI((s) => s.label)
  const setLabel = useUI((s) => s.setLabel)
  const reset = useUI((s) => s.resetLabelFilters)
  const { data: facets } = useLabelFacets()
  const { data: experiments } = useExperiments()

  // The one hand-built facet: an experiment is the unit the whole loop turns
  // on, so its picker shows the hypothesis on hover and flags the batches that
  // ran against the fixed seed panel (⧉ = comparable to every other one).
  const expOptions: Option[] = [{ value: '', label: 'any experiment' }]
  for (const e of experiments ?? []) {
    expOptions.push({
      value: e.id || UNSET,
      label: e.id
        ? `${e.id}${e.seed_panel ? ' ⧉' : ''} · ${e.images - e.labeled}/${e.images}`
        : `(untagged) · ${e.images - e.labeled}/${e.images}`,
      title: e.hypothesis ?? (e.id ? undefined : 'rendered without an --experiment id'),
    })
  }

  const activeWindow = (() => {
    if (!label.since && !label.until) return 'all time'
    if (label.until) return 'custom'
    const hours = (Date.now() / 1000 - Number(label.since)) / 3600
    const hit = WINDOWS.find((w) => w.hours && Math.abs(w.hours - hours) < w.hours * 0.02)
    return hit?.label ?? 'custom'
  })()

  const pickWindow = (hours: number | null) => {
    // Frozen to an absolute instant on click: a `since` that tracked the clock
    // would change the query key on every render and refetch the queue forever.
    setLabel('since', hours == null ? '' : String(Math.floor(Date.now() / 1000 - hours * 3600)))
    setLabel('until', '')
  }

  // Named, not counted: closed, "1024x1024 · pca · 24h" tells you what you are
  // looking at, where "3 filters" would send you back in to find out.
  const active: string[] = [
    label.experiment === UNSET ? 'untagged' : label.experiment,
    ...DIMS.map((d) => (label[d.dim] === UNSET ? `no ${d.dim}` : label[d.dim])),
    activeWindow === 'all time' ? '' : activeWindow,
    label.scope === 'unlabeled' ? '' : label.scope,
    label.bucket === 'favorites' ? '★ only' : '',
  ].filter(Boolean)
  const open = label.queueOpen === '1'

  return (
    // Controlled, not just `open`: React re-applies a hard-coded `open` on
    // every render, so a panel folded away by hand would spring back open the
    // next time a label lands and the facet counts refresh.
    <details
      className="sa-panel shrink-0 px-3 py-2"
      open={open}
      onToggle={(e) =>
        setLabel('queueOpen', (e.currentTarget as HTMLDetailsElement).open ? '1' : '')
      }
    >
      {/* Collapsed by default: the filters are set once at the start of a pass
          and then just take room the image wants. The summary has to carry
          enough for that closed state to be readable on its own — what the
          selection is, and how big it came out. */}
      <summary
        className="group flex cursor-pointer list-none items-center gap-2 text-[12px] text-dim
                   hover:text-ink"
      >
        {/* The whole summary is the hit target, but the chevron is what says
            so — at 10px it read as punctuation, so it gets its own boxed,
            accent-inked control. */}
        <span
          className="flex h-6 w-6 shrink-0 items-center justify-center rounded-md border
                     border-line bg-panel2 text-[13px] leading-none text-accent
                     transition-colors group-hover:border-line2"
          aria-hidden
        >
          {open ? '▾' : '▸'}
        </span>
        <span className="text-[13px] text-ink">Queue</span>
        <span>·</span>
        {queue ? (
          <span>
            <span className="text-ink tabular-nums">{queue.total}</span> image
            {queue.total === 1 ? '' : 's'} match
          </span>
        ) : (
          <span>…</span>
        )}
        <span className="truncate">
          {active.length ? `· ${active.join(' · ')}` : '· no filters'}
        </span>
        {facets ? (
          <span className="ml-auto shrink-0 pl-3">
            {facets.unlabeled} of {facets.total} unlabeled overall
          </span>
        ) : null}
      </summary>

      <div className="mt-2 flex flex-wrap items-end gap-x-3 gap-y-2">
        <div style={{ width: 230 }}>
          <span className="sa-label !mt-0">Experiment</span>
          <Select
            value={label.experiment}
            onChange={(v) => setLabel('experiment', v)}
            options={expOptions}
            ariaLabel="Experiment"
            className="!py-[5px] !text-[12px]"
          />
        </div>
        {DIMS.map(({ dim, label: name, any, width }) => (
          <div key={dim} style={{ width }}>
            <span className="sa-label !mt-0">{name}</span>
            <Select
              value={label[dim] ?? ''}
              onChange={(v) => setLabel(dim, v)}
              options={facetOptions(facets?.facets?.[dim], any)}
              ariaLabel={name}
              className="!py-[5px] !text-[12px]"
            />
          </div>
        ))}

        <div style={{ width: 150 }}>
          <span className="sa-label !mt-0">Scope</span>
          <Select
            value={label.scope}
            onChange={(v) => setLabel('scope', v)}
            options={SCOPES}
            ariaLabel="scope"
            className="!py-[5px] !text-[12px]"
          />
        </div>
        <div style={{ width: 150 }}>
          <span className="sa-label !mt-0">Starred</span>
          <Select
            value={label.bucket}
            onChange={(v) => setLabel('bucket', v)}
            options={BUCKETS}
            ariaLabel="bucket"
            className="!py-[5px] !text-[12px]"
          />
        </div>
        <div style={{ width: 150 }}>
          <span className="sa-label !mt-0">Order</span>
          <Select
            value={label.order}
            onChange={(v) => setLabel('order', v)}
            options={ORDERS}
            ariaLabel="order"
            className="!py-[5px] !text-[12px]"
          />
        </div>
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-2 border-t border-line pt-2">
        <span className="text-[12px] text-dim">Rendered</span>
        {WINDOWS.map((w) => (
          <button
            key={w.label}
            className={cn('sa-btn sa-btn-sm', activeWindow === w.label && 'sa-btn-sel')}
            onClick={() => pickWindow(w.hours)}
            title={w.hours ? `images rendered in the last ${w.label}` : 'no time filter'}
          >
            {w.label}
          </button>
        ))}
        <span className="ml-1 text-[12px] text-dim">from</span>
        <input
          type="datetime-local"
          className="sa-input !w-[190px] !py-[4px] !text-[12px]"
          value={toLocalInput(label.since)}
          onChange={(e) => setLabel('since', fromLocalInput(e.target.value))}
        />
        <span className="text-[12px] text-dim">to</span>
        <input
          type="datetime-local"
          className="sa-input !w-[190px] !py-[4px] !text-[12px]"
          value={toLocalInput(label.until)}
          onChange={(e) => setLabel('until', fromLocalInput(e.target.value))}
        />
        {facets?.oldest ? (
          <span className="text-[11px] text-dim">
            gallery spans {new Date(facets.oldest * 1000).toLocaleDateString()} →{' '}
            {new Date((facets.newest ?? facets.oldest) * 1000).toLocaleDateString()}
          </span>
        ) : null}
        <button
          className="sa-btn sa-btn-sm ml-auto"
          onClick={reset}
          disabled={!active.length}
          title="clear every filter (keeps scope, order and the knob readout)"
        >
          clear filters
        </button>
        <button
          className="sa-btn sa-btn-sm"
          onClick={() => setLabel('seed', String((Number(label.seed) || 0) + 1))}
          title="draw a different shuffle of the same selection"
        >
          ⤨ reshuffle
        </button>
      </div>
    </details>
  )
}
