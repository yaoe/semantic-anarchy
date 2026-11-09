/**
 * 🪐 "Sample more like these" — n + spread + go, for any set of real images
 * (a universe, the map selection). POST /api/collection/sample.
 *
 * Spread is measured in the members' OWN diversity in conditioning space, so
 * the same number means the same thing for a tight universe and a sprawling
 * one: 0 = re-draw on the members, 1 = anywhere across their range, >1 = past
 * their edges. The slider's ends say that in words.
 */
import { useSampleCollection } from '../../api/queries'
import { cn } from '../../lib/utils'
import { useUI } from '../../store'
import { oneBackend, spreadWords, useSampleBatches } from './universes'

const SPREAD_MAX = 2.5

export function SampleControls({
  rels: given,
  name,
  onQueued,
  disabledReason,
  tone = 'app',
}: {
  rels: string[]
  /** the universe's name, when the set is one (job label + notice) */
  name?: string
  onQueued?: (jobId: number) => void
  /** why sampling can't run for this set (e.g. a non-sd15 backend) */
  disabledReason?: string | null
  /** 'map' = the map's dark glass panel palette */
  tone?: 'app' | 'map'
}) {
  const spreadRaw = useUI((s) => s.map.spread)
  const nRaw = useUI((s) => s.map.sampleN)
  const setMap = useUI((s) => s.setMap)
  const addBatch = useSampleBatches((s) => s.add)
  const sample = useSampleCollection()

  const spread = Math.min(SPREAD_MAX, Math.max(0, Number.isFinite(Number(spreadRaw)) ? Number(spreadRaw) : 0.5))
  const n = Math.min(64, Math.max(1, Math.round(Number(nRaw)) || 8))
  const { backend, rels, skipped } = oneBackend(given)
  const blocked =
    disabledReason ??
    (!rels.length
      ? 'nothing selected'
      : backend && backend !== 'sd15'
        ? `universe sampling is sd15-only for now (these are ${backend})`
        : null)

  const go = () =>
    sample.mutate(
      { rels, spread, n, name: name ?? null },
      {
        onSuccess: (r) => {
          addBatch({
            jobId: r.job_id,
            name: name ?? `${r.n_members} selected image${r.n_members === 1 ? '' : 's'}`,
            members: r.n_members,
            queuedAt: Date.now() / 1000,
          })
          onQueued?.(r.job_id)
        },
      },
    )

  const map = tone === 'map'
  const dim = map ? 'text-[#e9dccb]/60' : 'text-dim'
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <span
          className={cn('text-[10px] uppercase tracking-[2px]', dim)}
          title={
            'How far new images may go from these, measured in their own diversity ' +
            '(in conditioning space, not on the map): 0 re-draws on the members, ' +
            '1 roams their whole range, above 1 explores past their edges.'
          }
        >
          spread
        </span>
        <span className={cn('font-mono text-[11px] tabular-nums', map ? 'text-[#efe6da]' : 'text-ink')}>
          {spread.toFixed(2)} · {spreadWords(spread)}
        </span>
      </div>
      <input
        type="range"
        min={0}
        max={SPREAD_MAX}
        step={0.05}
        value={spread}
        aria-label="spread"
        onChange={(e) => setMap({ spread: e.target.value })}
        className={map ? 'w-full accent-[#6fffd0]' : 'sa-range'}
      />
      <div className={cn('-mt-1 flex justify-between text-[10px]', dim)}>
        <span>tight — stay on these</span>
        <span>wide — explore the edges</span>
      </div>
      <div className="flex items-center gap-1.5">
        <label className={cn('flex items-center gap-1 text-[11px]', dim)}>
          ×
          <input
            type="number"
            min={1}
            max={64}
            value={n}
            onChange={(e) => setMap({ sampleN: e.target.value })}
            className={cn(
              'w-12 rounded-[6px] border px-1.5 py-0.5 text-right',
              map ? 'border-white/15 bg-black/40 text-[#efe6da]' : 'border-line bg-panel2 text-ink',
            )}
          />
        </label>
        <button
          type="button"
          disabled={!!blocked || sample.isPending}
          title={blocked ?? `queue ${n} new images drawn from these ${rels.length}`}
          onClick={go}
          className={cn(
            'flex-1 rounded-[8px] px-3 py-1.5 text-[12.5px] font-medium disabled:opacity-50',
            map ? 'bg-[#6fffd0] text-[#08221a] hover:bg-[#9effe0]' : 'sa-btn sa-btn-sel',
          )}
        >
          🪐 Sample {n} more like {name ? `“${name}”` : 'these'}
        </button>
      </div>
      {blocked && rels.length ? <div className="text-[11px] text-[#f3c27a]">{blocked}</div> : null}
      {skipped ? (
        <div className={cn('text-[11px]', dim)}>
          uses the {rels.length} {backend} images; {skipped} from other backends are left out
        </div>
      ) : null}
      {sample.isError ? <div className="text-[11px] text-err">{sample.error.message}</div> : null}
      {sample.isSuccess ? (
        <div className={cn('text-[11px]', map ? 'text-[#6fffd0]' : 'text-run')}>
          queued as job #{sample.data.job_id} from {sample.data.n_members} member
          {sample.data.n_members === 1 ? '' : 's'}
        </div>
      ) : null}
    </div>
  )
}
