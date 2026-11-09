/**
 * The image lists the ONE fullscreen viewer can walk.
 *
 * The viewer (Lightbox) is mounted once, at the app root. Every surface that
 * can open it — the gallery grid, the 🌌 map, the 🧬 Fit pool, the 🎬 timeline
 * strip — registers the list it is showing under a source key, and opens the
 * viewer with `openLightbox(rel, source)`. ←/→ then walk exactly the list the
 * image was opened from, kept live as that list changes under it.
 *
 * Its own store rather than a field of `useUI`: zustand's persist middleware
 * serializes on EVERY set, and the gallery re-registers its view on each poll
 * while a batch renders — that would turn into a prefs POST every 2 s.
 */
import { useEffect } from 'react'
import { create } from 'zustand'

import type { ImageItem } from '../../api/types'
import { basename } from '../../lib/utils'

/** `selection` = the 🌌 map selection; `batch` = the images a 🪐 sampling job just made. */
export type ViewerSource = 'gallery' | 'map' | 'fit' | 'timeline' | 'selection' | 'batch'

interface ViewerLists {
  lists: Partial<Record<ViewerSource, ImageItem[]>>
  setList: (source: ViewerSource, items: ImageItem[] | null) => void
}

export const useViewerLists = create<ViewerLists>()((set) => ({
  lists: {},
  setList: (source, items) =>
    set((s) => {
      const lists = { ...s.lists }
      if (items) lists[source] = items
      else delete lists[source]
      return { lists }
    }),
}))

/** Keep `items` registered as `source` while the calling component is mounted. */
export function useViewerList(source: ViewerSource, items: ImageItem[]): void {
  const setList = useViewerLists((s) => s.setList)
  useEffect(() => {
    setList(source, items)
  }, [source, items, setList])
  useEffect(() => () => setList(source, null), [source, setList])
}

/**
 * An `ImageItem` for a rel the caller has no gallery row for (a timeline
 * keyframe, a Fit candidate). The viewer swaps in the live gallery row when
 * there is one, so the star and scores it shows are never stale.
 */
export function stubItem(rel: string, extra: Partial<ImageItem> = {}): ImageItem {
  return {
    name: basename(rel),
    rel,
    url: `/img?path=${encodeURIComponent(rel)}`,
    mtime: 0,
    size: 0,
    fav: false,
    score: null,
    dist: null,
    nov: null,
    res: null,
    ...extra,
  }
}
