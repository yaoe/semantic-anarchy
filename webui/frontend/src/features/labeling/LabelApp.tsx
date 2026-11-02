/**
 * The standalone labeling page, served at `/label` by `webui/app.py`.
 *
 * Its own tab on purpose: labeling is a different mode of attention from
 * generating — you want the whole window, no sidebar, no job log, and you want
 * to keep it open beside the dashboard while a batch renders.
 *
 * The page is a vertical budget where the IMAGE gets whatever is left, so
 * everything else is one line: a single status bar (cursor, controls, dataset
 * numbers), the collapsed-by-default queue builder, then the labeling surface.
 */
import { useLabelQueue, useLabelStats } from '../../api/queries'
import { cn } from '../../lib/utils'
import { useUI } from '../../store'
import { LabelPage } from './LabelPage'
import { QueueBar } from './QueueBar'
import { useLabelSelection } from './selection'

export function LabelApp() {
  const selection = useLabelSelection()
  // Same key as LabelPage's, so this is the one shared request — the cursor in
  // this bar is by construction the queue the keypresses walk.
  const { data: queue, refetch, isFetching } = useLabelQueue(selection)
  const { data: stats } = useLabelStats()

  const setIndex = useUI((s) => s.setLabelIndex)
  const index = useUI((s) => s.labelIndex)
  const showKnobs = useUI((s) => s.label.knobs) === '1'
  const setLabel = useUI((s) => s.setLabel)

  const rows = queue?.queue ?? []
  const at = Math.min(index, Math.max(0, rows.length - 1))

  return (
    <div className="flex h-full flex-col">
      {/* The page's ONE status line. */}
      <header className="flex shrink-0 flex-wrap items-center gap-x-3 gap-y-1.5 border-b border-line px-3 py-2 text-[12px]">
        <h1 className="m-0 text-[15px] tracking-[0.3px]">🏷 Label</h1>
        <a href="/" className="text-dim underline decoration-dotted hover:text-ink">
          ← dashboard
        </a>

        <span className="tabular-nums">
          <span className="text-ink">{rows.length ? at + 1 : 0}</span>
          <span className="text-dim">/{rows.length}</span>
          {queue && queue.total > rows.length ? (
            <span className="text-dim"> of {queue.total}</span>
          ) : null}
        </span>
        <button
          className="sa-btn sa-btn-sm"
          onClick={() => {
            setIndex(0)
            refetch()
          }}
          title="pull in newly generated images and restart at the top"
        >
          {isFetching ? '…' : '↻ reload'}
        </button>
        <button
          className={cn('sa-btn sa-btn-sm', showKnobs && 'sa-btn-sel')}
          onClick={() => setLabel('knobs', showKnobs ? '' : '1')}
          title="show the knobs this image was made with (k)"
        >
          knobs
        </button>
        <span className="text-[11px] text-dim">
          0–9 score · ←/→ move · Space skip · s star · k knobs
        </span>

        <span className="ml-auto flex flex-wrap items-baseline gap-3 text-dim">
          {stats ? (
            <>
              <span>
                <span className="text-ink tabular-nums">{stats.count}</span> labeled
                {stats.records > stats.count ? ` (${stats.records} records)` : ''}
              </span>
              {stats.overall.keeper_rate != null ? (
                <span>
                  keepers{' '}
                  <span className="text-score tabular-nums">
                    {(stats.overall.keeper_rate * 100).toFixed(0)}%
                  </span>
                </span>
              ) : null}
              {stats.overall.p90 != null ? (
                <span>
                  P90{' '}
                  <span className="text-ink tabular-nums">
                    {stats.overall.p90.toFixed(1)}
                  </span>
                </span>
              ) : null}
            </>
          ) : null}
        </span>
      </header>

      <div className="flex min-h-0 flex-1 flex-col gap-2 p-3">
        <QueueBar queue={queue} />
        <LabelPage />
      </div>
    </div>
  )
}
