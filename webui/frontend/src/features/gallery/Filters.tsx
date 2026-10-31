import { useMemo, type ReactNode } from 'react'

import { useTags } from '../../api/queries'
import { UNSET, type TabKey } from '../../api/types'
import { Segmented } from '../../components/ui/Segmented'
import { Select } from '../../components/ui/Select'
import type { Option } from '../../params/schema'
import { cn } from '../../lib/utils'
import { useUI, type SortKey } from '../../store'

const TABS = [
  { value: 'generated', label: 'Generated' },
  { value: 'frontier', label: '🎯 Frontier', title: 'Pareto front of novelty × your resonance' },
  { value: 'top', label: '🏆 Top', title: 'top rated by aesthetic score' },
  { value: 'favorites', label: '★ Favorites' },
  { value: 'temperature', label: 'Temp', title: 'temperature sweeps' },
  { value: 'sampler', label: 'Sampler', title: 'sampler sweeps' },
  { value: 'marginals', label: 'Marginals' },
  { value: 'timeline', label: '🎬 Timeline', title: 'keyframes to travel through' },
  { value: 'films', label: '🎞 Films' },
  {
    value: 'fit',
    label: '🧬 Fit',
    title: 'fit the next distribution to a set of images you pick',
  },
]

/** Tabs that aren't image grids, so the sort order doesn't apply. */
const UNSORTED: TabKey[] = ['films', 'timeline', 'fit']

const SORTS = [
  { value: 'new', label: 'newest first' },
  { value: 'score', label: 'score ↓' },
  { value: 'dist', label: 'distance ↓' },
  { value: 'dist_asc', label: 'distance ↑' },
  { value: 'nov', label: 'novelty ↓' },
  { value: 'res', label: 'resonance ↓' },
]

/** Gallery tab strip + universe filter + sort order + any action buttons. */
export function Filters({ actions }: { actions?: ReactNode }) {
  const tab = useUI((s) => s.tab)
  const setTab = useUI((s) => s.setTab)
  const sortBy = useUI((s) => s.sortBy)
  const setSortBy = useUI((s) => s.setSortBy)
  const tagFilter = useUI((s) => s.tagFilter)
  const setTagFilter = useUI((s) => s.setTagFilter)
  const { data: vocab } = useTags()

  // Counts are over the whole labelable set, not the current tab — the same
  // convention the label page's facets use, so a universe never appears to
  // vanish just because you are looking at ★ Favorites.
  const tagOptions: Option[] = useMemo(() => {
    const opts: Option[] = [{ value: '', label: 'any universe' }]
    for (const t of vocab?.tags ?? []) {
      opts.push({ value: t.tag, label: `${t.tag} (${t.count})`, title: `${t.count} images overall` })
    }
    if (vocab?.tags?.length) {
      opts.push({ value: UNSET, label: 'untagged', title: 'not sorted into a universe yet' })
    }
    return opts
  }, [vocab])

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Segmented
        value={tab}
        onChange={(v) => setTab(v as TabKey)}
        options={TABS}
        grow={false}
        itemClassName="!py-[5px] !text-[12px]"
      />
      {!UNSORTED.includes(tab) && tagOptions.length > 1 ? (
        <Select
          value={tagFilter}
          onChange={setTagFilter}
          options={tagOptions}
          ariaLabel="filter by aesthetic universe"
          title="filter by aesthetic universe (🏷 tags)"
          className="ml-auto !w-[140px] !py-[5px] !text-[12px]"
        />
      ) : null}
      {!UNSORTED.includes(tab) ? (
        <Select
          value={sortBy}
          onChange={(v) => setSortBy(v as SortKey)}
          options={SORTS}
          ariaLabel="gallery order"
          title="gallery order"
          className={cn(
            '!w-[120px] !py-[5px] !text-[12px]',
            tagOptions.length > 1 ? '' : 'ml-auto',
          )}
        />
      ) : (
        <span className="ml-auto" />
      )}
      {actions}
    </div>
  )
}
