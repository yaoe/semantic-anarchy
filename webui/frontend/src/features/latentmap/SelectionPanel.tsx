/**
 * The 🌌 map's selection panel: what you picked (⇧-click, lasso, circle), what
 * universes it spans, and what to do with it — sample more like it, sort it
 * into a universe, hand it to the 🧬 Fit tab, walk it in the viewer.
 */
import { useMemo } from 'react'

import type { ImageItem } from '../../api/types'
import { Popover } from '../../components/ui/Popover'
import { useUI } from '../../store'
import { stubItem, useViewerList } from '../gallery/viewerLists'
import { SampleControls } from '../universes/SampleControls'
import { UniversePicker } from '../universes/UniversePicker'
import { universeColor } from '../universes/universes'

const STRIP = 30

/** One image out of the atlas, as CSS: no extra request per thumbnail. */
export function Thumb({ cell, cols, url, size, className }: {
  cell: number; cols: number; url: string; size: number; className?: string
}) {
  return (
    <div
      className={`shrink-0 rounded-[4px] bg-black ${className ?? ''}`}
      style={{
        width: size,
        height: size,
        backgroundImage: cell >= 0 ? `url(${url})` : undefined,
        backgroundSize: `${cols * size}px ${cols * size}px`,
        backgroundPosition: `${-(cell % cols) * size}px ${-Math.floor(cell / cols) * size}px`,
      }}
    />
  )
}

const GLASS_BTN =
  'rounded-[7px] border border-white/12 bg-black/30 px-2 py-1 text-[11.5px] text-[#e9dccb]/85 hover:border-white/35'

export function SelectionPanel({
  rels,
  cellOf,
  byRel,
  atlasUrl,
  cols,
  onSampled,
  onFly,
}: {
  rels: string[]
  /** rel -> atlas cell (-1 none) */
  cellOf: (rel: string) => number
  byRel: Map<string, ImageItem>
  atlasUrl: string
  cols: number
  /** a sampling job was queued (the map grows the new images when they land) */
  onSampled: () => void
  onFly: () => void
}) {
  const setMapSel = useUI((s) => s.setMapSel)
  const removeMapSel = useUI((s) => s.removeMapSel)
  const addFitSel = useUI((s) => s.addFitSel)
  const addKeyframe = useUI((s) => s.addKeyframe)
  const openLightbox = useUI((s) => s.openLightbox)

  const items = useMemo(() => rels.map((r) => byRel.get(r) ?? stubItem(r)), [rels, byRel])
  useViewerList('selection', items)

  // Which universes the selection spans: the honest answer to "is this ONE world?"
  const spans = useMemo(() => {
    const m = new Map<string, { tag: string; rels: string[] }>()
    const none: string[] = []
    for (const it of items) {
      if (!it.tags?.length) none.push(it.rel)
      for (const t of it.tags ?? []) {
        const k = t.toLocaleLowerCase()
        const e = m.get(k) ?? { tag: t, rels: [] }
        e.rels.push(it.rel)
        m.set(k, e)
      }
    }
    return { tags: [...m.values()].sort((a, b) => b.rels.length - a.rels.length), none }
  }, [items])
  // A selection that is exactly one universe's members gets its name on the job.
  const whole = spans.tags.length === 1 && !spans.none.length ? spans.tags[0].tag : undefined

  return (
    <div className="rounded-[12px] border border-[#6fffd0]/25 bg-black/65 p-2.5 text-[12px] text-[#efe6da] backdrop-blur-md">
      <div className="mb-1.5 flex items-center gap-2">
        <span className="h-2 w-2 rounded-full bg-[#6fffd0]" />
        <b>{rels.length} selected</b>
        <button type="button" className="text-[11px] text-[#e9dccb]/55 hover:text-[#e9dccb]" onClick={onFly} title="fly to the selection">
          ◎ fly
        </button>
        <button
          type="button"
          className="ml-auto text-[#e9dccb]/50 hover:text-[#e9dccb]"
          onClick={() => setMapSel([])}
          title="clear the selection (Esc)"
        >
          ✕
        </button>
      </div>

      {spans.tags.length || spans.none.length ? (
        <div className="mb-1.5 flex flex-wrap gap-1" title="click one to keep only those">
          {spans.tags.map((s) => (
            <button
              key={s.tag}
              type="button"
              onClick={() => setMapSel(s.rels)}
              className="flex items-center gap-1 rounded-full border border-white/12 px-1.5 text-[10.5px] text-[#efe6da]/80 hover:border-white/35"
            >
              <span className="h-1.5 w-1.5 rounded-full" style={{ background: universeColor(s.tag) }} />
              {s.tag} {s.rels.length}
            </button>
          ))}
          {spans.none.length && spans.tags.length ? (
            <button
              type="button"
              onClick={() => setMapSel(spans.none)}
              className="rounded-full border border-white/12 px-1.5 text-[10.5px] text-[#efe6da]/60 hover:border-white/35"
            >
              untagged {spans.none.length}
            </button>
          ) : null}
        </div>
      ) : null}

      <div className="mb-2 flex flex-wrap gap-[3px]">
        {rels.slice(0, STRIP).map((r) => (
          <button
            key={r}
            type="button"
            title="open in the viewer · ⇧-click: drop from the selection"
            onClick={(e) => (e.shiftKey ? removeMapSel([r]) : openLightbox(r, 'selection'))}
          >
            <Thumb cell={cellOf(r)} cols={cols} url={atlasUrl} size={34} className="hover:ring-1 hover:ring-[#6fffd0]" />
          </button>
        ))}
        {rels.length > STRIP ? (
          <span className="self-center pl-1 text-[11px] text-[#e9dccb]/60">+{rels.length - STRIP}</span>
        ) : null}
      </div>

      <SampleControls
        tone="map"
        rels={rels}
        name={whole}
        onQueued={onSampled}
      />
      <div
        className="mt-1 text-[10.5px] leading-snug text-[#e9dccb]/50"
        title={
          'New images are drawn in conditioning space, around these images’ own latents, with ' +
          'spread measured in their own diversity. Where they land on this map depends on what ' +
          'they end up looking like. The mint halos only show how far the spread reaches compared ' +
          'with how far apart these images sit HERE — a map spot has no conditioning of its own ' +
          '(a UMAP has no inverse), which is why the region is always a set of real images.'
        }
      >
        ◌ halos are illustrative — the real region lives in conditioning space (hover for why)
      </div>

      <div className="mt-2 flex flex-wrap gap-1">
        <Popover
          title="put the selection into an aesthetic universe"
          width={280}
          triggerClassName={GLASS_BTN}
          trigger="🏷 add to universe…"
          className="!border-white/15 !bg-[#141012]/95"
        >
          {() => <UniversePicker rels={rels} tone="map" />}
        </Popover>
        <button type="button" className={GLASS_BTN} onClick={() => addFitSel(rels)} title="add these to the 🧬 Fit tab's selection">
          🧬 to Fit
        </button>
        <button
          type="button"
          className={GLASS_BTN}
          onClick={() => rels[0] && openLightbox(rels[0], 'selection')}
          title="walk the selection in the viewer (star, score, tag with the usual keys)"
        >
          ▸ viewer
        </button>
        <button
          type="button"
          className={GLASS_BTN}
          onClick={() => rels.forEach((r) => addKeyframe(r))}
          title="append the selection to the 🎬 timeline, in pick order"
        >
          🎬 timeline
        </button>
      </div>
    </div>
  )
}
