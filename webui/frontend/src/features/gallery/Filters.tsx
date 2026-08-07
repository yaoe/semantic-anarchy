import type { ReactNode } from 'react'

import type { TabKey } from '../../api/types'
import { Segmented } from '../../components/ui/Segmented'
import { Select } from '../../components/ui/Select'
import { useUI, type SortKey } from '../../store'

const TABS = [
  { value: 'generated', label: 'Generated' },
  { value: 'frontier', label: '🎯 Frontier', title: 'Pareto front of novelty × your resonance' },
  { value: 'top', label: '🏆 Top rated' },
  { value: 'favorites', label: '★ Favorites' },
  { value: 'temperature', label: 'Temp sweeps' },
  { value: 'sampler', label: 'Sampler sweeps' },
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

/** Gallery tab strip + sort order + whatever action buttons are passed in. */
export function Filters({ actions }: { actions?: ReactNode }) {
  const tab = useUI((s) => s.tab)
  const setTab = useUI((s) => s.setTab)
  const sortBy = useUI((s) => s.sortBy)
  const setSortBy = useUI((s) => s.setSortBy)

  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Segmented
        value={tab}
        onChange={(v) => setTab(v as TabKey)}
        options={TABS}
        grow={false}
        itemClassName="!py-[5px] !text-[12px]"
      />
      {!UNSORTED.includes(tab) ? (
        <Select
          value={sortBy}
          onChange={(v) => setSortBy(v as SortKey)}
          options={SORTS}
          ariaLabel="gallery order"
          title="gallery order"
          className="ml-auto !w-[150px] !py-[5px] !text-[12px]"
        />
      ) : (
        <span className="ml-auto" />
      )}
      {actions}
    </div>
  )
}
