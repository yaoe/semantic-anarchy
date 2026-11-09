import { Suspense, lazy, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { useVirtualizer } from '@tanstack/react-virtual'

import { useImages } from '../../api/queries'
import { UNSET, type GalleryKey, type ImageItem, type TabKey } from '../../api/types'
import { FitPage } from '../fit/FitPage'
import { FilmList } from '../timeline/FilmList'
import { Timeline } from '../timeline/Timeline'
import { useUI, type SortKey } from '../../store'
import { Card } from './Card'
import { useViewerList } from './viewerLists'

// three.js is ~600 kB -- only fetched when the 🌌 tab is first opened.
const LatentMap = lazy(() =>
  import('../latentmap/LatentMap').then((m) => ({ default: m.LatentMap })),
)

const SHEET_TABS: TabKey[] = ['temperature', 'sampler', 'marginals']
const MIN_COL = 220
const GAP = 6

const EMPTY_MSG: Partial<Record<TabKey, string>> = {
  favorites: 'no favorites yet — click the ☆ on any image to add it.',
  top: 'nothing scored yet — click “Score all ▶” to rank by aesthetic.',
  frontier: 'no analysis yet — click “🎯 Analyze” to compute novelty + resonance.',
}

/**
 * Narrow a bucket to one aesthetic universe. '' keeps everything; `UNSET` keeps
 * the images that have not been sorted into one yet — the useful inverse when
 * you are working through a backlog.
 *
 * Client-side, like the sort: /api/images already carries each row's tags, so
 * filtering here costs one pass over an array that is in memory anyway and
 * needs no round trip.
 */
function filterByTag(list: ImageItem[], tag: string): ImageItem[] {
  if (!tag) return list
  if (tag === UNSET) return list.filter((im) => !im.tags?.length)
  const want = tag.toLocaleLowerCase()
  return list.filter((im) => im.tags?.some((t) => t.toLocaleLowerCase() === want))
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
const CUSTOM_TABS: TabKey[] = ['films', 'timeline', 'fit', 'map']
const isGrid = (t: TabKey): t is GalleryKey => !CUSTOM_TABS.includes(t)

/**
 * Virtualized gallery. Rows are windowed with @tanstack/react-virtual, so the
 * full bucket (10k+ images) is in memory but only the visible rows are in the
 * DOM — there is no pagination and no "load more".
 */
export function Gallery({ busy }: { busy: boolean }) {
  const tab = useUI((s) => s.tab)
  const sortBy = useUI((s) => s.sortBy)
  const tagFilter = useUI((s) => s.tagFilter)
  const { data: images, isLoading } = useImages(busy)

  // Memoized so the `[]` fallback isn't a fresh array on every render, which
  // would re-run the filter+sort below for nothing.
  const list = useMemo(
    () => (isGrid(tab) ? (images?.[tab] ?? []) : []),
    [images, tab],
  )
  const view = useMemo(
    () => sortItems(filterByTag(list, tagFilter), sortBy),
    [list, tagFilter, sortBy],
  )
  // The fullscreen viewer (mounted once, in App) walks exactly this view.
  useViewerList('gallery', view)

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
    estimateSize: () => (sheet ? 720 : Math.round(colW + 30)),
    overscan: 3,
  })

  // Column count and bucket changes invalidate every cached row measurement.
  useEffect(() => {
    virtualizer.measure()
  }, [cols, tab, sortBy, tagFilter, virtualizer])

  useEffect(() => {
    if (parentRef.current) parentRef.current.scrollTop = 0
    setFresh(0)
  }, [tab, sortBy, tagFilter])

  // Keep the view still when the bucket changes under it. Starring refetches
  // the gallery, and with sort=new every image made since the last fetch lands
  // at the top at once, pushing rows down. Anchor the grid on the item it shows.
  const prevRef = useRef<{ view: ImageItem[]; tab: TabKey; sortBy: SortKey; cols: number } | null>(null)
  const [fresh, setFresh] = useState(0) // new items parked above the anchor
  useLayoutEffect(() => {
    const prev = prevRef.current
    prevRef.current = { view, tab, sortBy, cols }
    const el = parentRef.current
    if (!prev || !el || prev.view === view || !prev.view.length || !view.length) return
    if (prev.tab !== tab || prev.sortBy !== sortBy || prev.cols !== cols) return

    // Grid: the first row that is still on screen is the anchor. Row sizes are
    // cached per row *index*, and a prepend only adds rows at the end of the
    // index space, so measurements[r].start is still the pre-change start.
    const top = el.scrollTop
    const ms = virtualizer.measurementsCache
    const anchorRow = ms.find((m) => m.end > top)
    if (!anchorRow) return
    const oldIdx = anchorRow.index * prev.cols
    const rel = prev.view[oldIdx]?.rel
    const newIdx = rel ? view.findIndex((v) => v.rel === rel) : -1
    if (newIdx < 0) return
    const newRow = Math.floor(newIdx / cols)
    if (newRow === anchorRow.index) return
    const start = ms[newRow]?.start ?? anchorRow.start + (newRow - anchorRow.index) * anchorRow.size
    el.scrollTop = start + (top - anchorRow.start)
    if (newIdx > oldIdx) setFresh((f) => f + newIdx - oldIdx)
  }, [view, tab, sortBy, cols, virtualizer])

  const toTop = () => {
    parentRef.current?.scrollTo({ top: 0, behavior: 'smooth' })
    setFresh(0)
  }

  const rows = virtualizer.getVirtualItems()

  return (
    <div
      ref={parentRef}
      className="relative min-h-0 flex-1 overflow-auto pr-1"
      onScroll={(e) => {
        if (fresh && e.currentTarget.scrollTop < 8) setFresh(0)
      }}
    >
      {fresh > 0 && isGrid(tab) ? (
        // Zero-height sticky rail: floats over the grid without shifting it.
        <div className="sticky top-2 z-10 flex h-0 justify-center overflow-visible">
          <button type="button" className="sa-btn sa-btn-sm shadow-lg" onClick={toTop}>
            ↑ {fresh} new
          </button>
        </div>
      ) : null}
      {tab === 'map' ? (
        // The map fills the pane; its images open in the same app-level viewer.
        <Suspense fallback={<div className="p-8 text-center text-dim">loading map…</div>}>
          <LatentMap busy={busy} />
        </Suspense>
      ) : tab === 'timeline' ? (
        <Timeline />
      ) : tab === 'films' ? (
        <FilmList />
      ) : tab === 'fit' ? (
        <FitPage />
      ) : isLoading ? (
        <div className="p-8 text-center text-dim">loading gallery…</div>
      ) : !view.length ? (
        <div className="p-8 text-center text-dim">
          {tagFilter && list.length
            ? tagFilter === UNSET
              ? 'every image here is already sorted into a universe.'
              : `nothing in this tab wears “${tagFilter}”.`
            : (EMPTY_MSG[tab] ?? `no ${tab} artifacts yet — run something.`)}
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
                    className="grid pb-1.5"
                    style={{
                      gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))`,
                      gap: `${GAP}px`,
                    }}
                  >
                    {rowItems.map((im) => (
                      <Card
                        key={im.rel}
                        im={im}
                        sheet={sheet}
                        showNovRes={tab === 'frontier'}
                      />
                    ))}
                  </div>
                </div>
              )
            })}
          </div>
          <div className="py-1.5 text-center text-[12px] text-dim">
            {view.length} image{view.length === 1 ? '' : 's'}
            {tagFilter ? (
              <span className="text-tag">
                {' '}
                · {tagFilter === UNSET ? 'untagged' : tagFilter} of {list.length}
              </span>
            ) : null}
          </div>
        </>
      )}
    </div>
  )
}
