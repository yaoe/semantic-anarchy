import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'

import { useImages } from '../../api/queries'
import type { GalleryKey, ImageItem, TabKey } from '../../api/types'
import { FitPage } from '../fit/FitPage'
import { FilmList } from '../timeline/FilmList'
import { Timeline } from '../timeline/Timeline'
import { useUI, type SortKey } from '../../store'
import { Card } from './Card'
import { Lightbox } from './Lightbox'

const SHEET_TABS: TabKey[] = ['temperature', 'sampler', 'marginals']
const MIN_COL = 220
const GAP = 12

const EMPTY_MSG: Partial<Record<TabKey, string>> = {
  favorites: 'no favorites yet — click the ☆ on any image to add it.',
  top: 'nothing scored yet — click “Score all ▶” to rank by aesthetic.',
  frontier: 'no analysis yet — click “🎯 Analyze” to compute novelty + resonance.',
}

/** Client-side reorder of a bucket. 'new' is the server's own mtime order. */
function sortItems(list: ImageItem[], sortBy: SortKey): ImageItem[] {
  if (sortBy === 'new') return list
  const key = (
    {
      score: (a: ImageItem) => a.score,
      dist: (a: ImageItem) => a.dist,
      dist_asc: (a: ImageItem) => a.dist,
      nov: (a: ImageItem) => a.nov,
      res: (a: ImageItem) => a.res,
    } as Record<string, (a: ImageItem) => number | null>
  )[sortBy]
  const asc = sortBy === 'dist_asc'
  return [...list].sort((a, b) =>
    asc
      ? (key(a) ?? 1e9) - (key(b) ?? 1e9)
      : (key(b) ?? -1) - (key(a) ?? -1),
  )
}

/** Tabs that render their own thing instead of the virtualized image grid. */
const CUSTOM_TABS: TabKey[] = ['films', 'timeline', 'fit']
const isGrid = (t: TabKey): t is GalleryKey => !CUSTOM_TABS.includes(t)

/**
 * Virtualized gallery. Rows are windowed with @tanstack/react-virtual, so the
 * full bucket (10k+ images) is in memory but only the visible rows are in the
 * DOM — there is no pagination and no "load more".
 */
export function Gallery({ busy }: { busy: boolean }) {
  const tab = useUI((s) => s.tab)
  const sortBy = useUI((s) => s.sortBy)
  const { data: images, isLoading } = useImages(busy)

  const list = isGrid(tab) ? (images?.[tab] ?? []) : []
  const view = useMemo(() => sortItems(list, sortBy), [list, sortBy])

  const parentRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)

  useLayoutEffect(() => {
    const el = parentRef.current
    if (!el) return
    setWidth(el.clientWidth)
    const ro = new ResizeObserver((entries) => setWidth(entries[0].contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const sheet = SHEET_TABS.includes(tab)
  const cols = sheet ? 1 : Math.max(1, Math.floor((width + GAP) / (MIN_COL + GAP)))
  const colW = cols > 0 && width > 0 ? (width - GAP * (cols - 1)) / cols : MIN_COL
  const rowCount = Math.ceil(view.length / cols)

  const virtualizer = useVirtualizer({
    count: rowCount,
    getScrollElement: () => parentRef.current,
    estimateSize: () => (sheet ? 720 : Math.round(colW + 60)),
    overscan: 3,
  })

  // Column count and bucket changes invalidate every cached row measurement.
  useEffect(() => {
    virtualizer.measure()
  }, [cols, tab, sortBy, virtualizer])

  useEffect(() => {
    if (parentRef.current) parentRef.current.scrollTop = 0
  }, [tab, sortBy])

  const rows = virtualizer.getVirtualItems()

  return (
    <div ref={parentRef} className="min-h-0 flex-1 overflow-auto pr-1">
      {tab === 'timeline' ? (
        <Timeline />
      ) : tab === 'films' ? (
        <FilmList />
      ) : tab === 'fit' ? (
        <FitPage />
      ) : isLoading ? (
        <div className="p-8 text-center text-dim">loading gallery…</div>
      ) : !view.length ? (
        <div className="p-8 text-center text-dim">
          {EMPTY_MSG[tab] ?? `no ${tab} artifacts yet — run something.`}
        </div>
      ) : (
        <>
          <div
            className="relative w-full"
            style={{ height: `${virtualizer.getTotalSize()}px` }}
          >
            {rows.map((row) => {
              const start = row.index * cols
              const rowItems = view.slice(start, start + cols)
              return (
                <div
                  key={row.key}
                  data-index={row.index}
                  ref={virtualizer.measureElement}
                  className="absolute left-0 top-0 w-full"
                  style={{ transform: `translateY(${row.start}px)` }}
                >
                  <div
                    className="grid pb-3"
                    style={{
                      gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
                      gap: `${GAP}px`,
                    }}
                  >
                    {rowItems.map((im, i) => (
                      <Card
                        key={im.rel}
                        im={im}
                        index={start + i}
                        sheet={sheet}
                        showNovRes={tab === 'frontier'}
                      />
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
          <div className="py-3.5 text-center text-[12px] text-dim">
            {view.length} image{view.length === 1 ? '' : 's'}
          </div>
        </>
      )}
      <Lightbox items={view} />
    </div>
  )
}
