/**
 * Where the last 🪐 sampling job's images went: a one-line notice that counts
 * them in as they land and, once there, opens them in the viewer (to star,
 * score and sort into universes with the usual keys) or selects them on the map.
 */
import { useMemo } from 'react'

import { cn } from '../../lib/utils'
import { useUI } from '../../store'
import { useViewerList } from '../gallery/viewerLists'
import { useLatestBatch, useSampleBatches } from './universes'

export function BatchNotice({ tone = 'app', className }: { tone?: 'app' | 'map'; className?: string }) {
  const { batch, job, made } = useLatestBatch()
  const dismiss = useSampleBatches((s) => s.dismiss)
  const openLightbox = useUI((s) => s.openLightbox)
  const setMapSel = useUI((s) => s.setMapSel)
  const setTab = useUI((s) => s.setTab)
  // The gallery is newest-first; the viewer walks them in the order they rendered.
  const ordered = useMemo(() => [...made].reverse(), [made])
  useViewerList('batch', ordered)
  if (!batch) return null

  const status = job?.status ?? 'queued'
  const failed = status === 'error' || status === 'cancelled'
  const map = tone === 'map'
  const btn = map
    ? 'rounded-[6px] border border-white/15 px-1.5 py-[1px] hover:border-white/40'
    : 'sa-btn sa-btn-sm !py-0'
  return (
    <div
      className={cn(
        'flex flex-wrap items-center gap-1.5 text-[12px]',
        map ? 'text-[#efe6da]' : 'text-ink',
        className,
      )}
      role="status"
    >
      <span className={failed ? 'text-err' : map ? 'text-[#6fffd0]' : 'text-run'}>🪐</span>
      <span>
        <b>{batch.name}</b>{' '}
        {failed
          ? `— job #${batch.jobId} ${status}${made.length ? ` after ${made.length} image${made.length === 1 ? '' : 's'}` : ''} (see its log)`
          : status === 'done'
            ? `— ${made.length} new image${made.length === 1 ? '' : 's'}`
            : status === 'running'
              ? `— rendering, ${made.length} landed so far`
              : `— job #${batch.jobId} waiting in the queue`}
      </span>
      {made.length ? (
        <>
          <button
            type="button"
            className={btn}
            title="open them in the viewer: s star, 0-9 score, shift+1…9 universe, ←/→ next"
            onClick={() => openLightbox(ordered[0].rel, 'batch')}
          >
            review ▸
          </button>
          <button
            type="button"
            className={btn}
            title="select them on the 🌌 map — see where they landed, sample again from the ones you like"
            onClick={() => {
              setMapSel(made.map((m) => m.rel))
              setTab('map')
            }}
          >
            select on map
          </button>
        </>
      ) : null}
      <button
        type="button"
        className={cn('ml-0.5', map ? 'text-[#e9dccb]/50 hover:text-[#e9dccb]' : 'text-dim hover:text-ink')}
        title="dismiss"
        onClick={() => dismiss(batch.jobId)}
      >
        ✕
      </button>
    </div>
  )
}
