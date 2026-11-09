/**
 * 🪐 The universe bar — the ONE place above the gallery that lists every
 * aesthetic universe (= 🏷 tag) as a button.
 *
 *   click a universe     -> the gallery shows just its images (the tag filter)
 *   🪐 on a universe     -> sample more like it (n + spread)
 *   + new universe       -> name it, fill it from the 🧬 Fit selection, the 🌌
 *                           map selection, your ★ stars or the current view
 *
 * Plus the notice for the last sampling job, so its images are one click away.
 */
import { useState } from 'react'

import { useBulkTag, useImages } from '../../api/queries'
import { UNSET, type TabKey } from '../../api/types'
import { Popover } from '../../components/ui/Popover'
import { cn } from '../../lib/utils'
import { useUI } from '../../store'
import { useViewerLists } from '../gallery/viewerLists'
import { BatchNotice } from './BatchNotice'
import { SampleControls } from './SampleControls'
import { useUniverses } from './universes'

/** Tabs that are image grids the tag filter applies to. */
const GRID: TabKey[] = ['generated', 'frontier', 'top', 'favorites']

export function UniverseBar() {
  const tab = useUI((s) => s.tab)
  const tagFilter = useUI((s) => s.tagFilter)
  const setTagFilter = useUI((s) => s.setTagFilter)
  const { list } = useUniverses()
  if (!GRID.includes(tab)) return null

  const isOn = (t: string) => tagFilter.toLocaleLowerCase() === t.toLocaleLowerCase()
  return (
    <div className="flex shrink-0 flex-wrap items-center gap-1.5 px-0.5">
      <span className="mr-0.5 text-[10px] uppercase tracking-[2px] text-dim" title="aesthetic universes (🏷 tags)">
        universes
      </span>
      <button
        type="button"
        onClick={() => setTagFilter('')}
        className={cn(
          'h-[24px] rounded-full border px-2.5 text-[12px]',
          !tagFilter ? 'border-tag text-tag' : 'border-line text-dim hover:text-ink',
        )}
      >
        all
      </button>
      {list.map((u) => (
        <span
          key={u.tag}
          className={cn(
            'flex h-[24px] items-center overflow-hidden rounded-full border text-[12px]',
            isOn(u.tag) ? 'border-transparent' : 'border-line',
          )}
          style={isOn(u.tag) ? { boxShadow: `0 0 0 1px ${u.color}`, background: `${u.color}22` } : undefined}
        >
          <button
            type="button"
            onClick={() => setTagFilter(isOn(u.tag) ? '' : u.tag)}
            title={`show the ${u.count} image${u.count === 1 ? '' : 's'} in “${u.tag}”`}
            className={cn('flex h-full items-center gap-1.5 pl-2 pr-1.5', isOn(u.tag) ? 'text-ink' : 'text-dim hover:text-ink')}
          >
            <span className="h-2 w-2 rounded-full" style={{ background: u.color }} />
            {u.tag}
            <span className="tabular-nums opacity-60">{u.count}</span>
          </button>
          <Popover
            title={`🪐 sample more like “${u.tag}”`}
            width={290}
            triggerClassName="h-full border-l border-line px-1.5 text-[11px] text-dim hover:bg-panel2 hover:text-ink"
            trigger="🪐"
          >
            {(close) => (
              <div className="flex flex-col gap-2">
                <div className="text-[12px]">
                  <b>{u.tag}</b> · {u.rels.length} member{u.rels.length === 1 ? '' : 's'}
                </div>
                <SampleControls rels={u.rels} name={u.tag} onQueued={() => setTimeout(close, 900)} />
              </div>
            )}
          </Popover>
        </span>
      ))}
      {list.length ? (
        <button
          type="button"
          onClick={() => setTagFilter(tagFilter === UNSET ? '' : UNSET)}
          title="images not sorted into any universe yet"
          className={cn(
            'h-[24px] rounded-full border px-2.5 text-[12px]',
            tagFilter === UNSET ? 'border-tag text-tag' : 'border-line text-dim hover:text-ink',
          )}
        >
          untagged
        </button>
      ) : null}
      <NewUniverse />
      <BatchNotice className="ml-auto" />
    </div>
  )
}

/** Name a universe and fill it from a set you already have. */
function NewUniverse() {
  const fitSel = useUI((s) => s.fitSel)
  const mapSel = useUI((s) => s.mapSel)
  const setTagFilter = useUI((s) => s.setTagFilter)
  const view = useViewerLists((s) => s.lists.gallery)
  const { data: images } = useImages(false)
  const bulk = useBulkTag()
  const [name, setName] = useState('')

  const sources: { id: string; label: string; rels: string[]; title: string }[] = [
    { id: 'fit', label: '🧬 Fit selection', rels: fitSel, title: 'the images picked for the next fit' },
    { id: 'map', label: '🌌 map selection', rels: mapSel, title: 'the images selected on the map' },
    {
      id: 'star', label: '★ starred', rels: (images?.favorites ?? []).map((i) => i.rel),
      title: 'every starred image',
    },
    {
      id: 'view', label: 'this view', rels: (view ?? []).map((i) => i.rel),
      title: 'every image the gallery is showing right now (tab + filter)',
    },
  ]

  return (
    <Popover
      title="make a new universe from images you already picked"
      width={300}
      triggerClassName="h-[24px] rounded-full border border-dashed border-line2 px-2.5 text-[12px] text-dim hover:text-ink"
      trigger="+ new universe"
    >
      {(close) => (
        <div className="flex flex-col gap-2">
          <input
            autoFocus
            value={name}
            onChange={(e) => setName(e.target.value)}
            onKeyDown={(e) => e.stopPropagation()}
            placeholder="name it (e.g. ‘ink coral’)"
            className="rounded-[6px] border border-line bg-panel2 px-2 py-1 text-[12px] focus:border-accent focus:outline-none"
          />
          <div className="text-[10px] uppercase tracking-[2px] text-dim">fill it from</div>
          <div className="flex flex-col gap-1">
            {sources.map((s) => (
              <button
                key={s.id}
                type="button"
                title={s.title}
                disabled={!name.trim() || !s.rels.length || bulk.isPending}
                onClick={() =>
                  bulk.mutate(
                    { rels: s.rels, tag: name, on: true },
                    {
                      onSuccess: (r) => {
                        setTagFilter(r.tag)
                        setName('')
                        close()
                      },
                    },
                  )
                }
                className="sa-btn sa-btn-sm flex justify-between disabled:opacity-40"
              >
                <span>{s.label}</span>
                <span className="tabular-nums text-dim">{s.rels.length}</span>
              </button>
            ))}
          </div>
          <div className="text-[11px] text-dim">
            Or tag images one by one: the tag button on any card, <kbd>t</kbd> / shift+1…9 in the viewer.
          </div>
          {bulk.isPending ? <div className="text-[11px] text-dim">tagging…</div> : null}
          {bulk.isError ? <div className="text-[11px] text-err">{bulk.error.message}</div> : null}
        </div>
      )}
    </Popover>
  )
}
