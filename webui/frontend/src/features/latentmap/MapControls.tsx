import type { LatentMap, MapSpace } from '../../api/types'
import { cn } from '../../lib/utils'
import { useUI } from '../../store'
import { LEGEND, ORIGIN_COLORS, RAMPS } from './palette'

export function Pill({
  on, onClick, children, title, className, disabled,
}: {
  on?: boolean
  onClick: () => void
  children: React.ReactNode
  title?: string
  className?: string
  disabled?: boolean
}) {
  return (
    <button
      type="button"
      title={title}
      disabled={disabled}
      onClick={onClick}
      className={cn(
        'flex items-center gap-1.5 rounded-[8px] border px-2.5 py-[5px] text-[11.5px] transition-colors disabled:opacity-40',
        on
          ? 'border-[#f3c27a]/70 bg-[#f3c27a]/15 text-[#fff1dc]'
          : 'border-white/12 bg-black/40 text-[#e9dccb]/80 hover:border-white/30',
        className,
      )}
    >
      {children}
    </button>
  )
}

const SPACES: { id: MapSpace; label: string; title: string }[] = [
  { id: 'look', label: 'Look', title: 'UMAP of CLIP image embeddings: near means the images look alike' },
  { id: 'cond', label: 'Conditioning', title: 'UMAP of the conditioning tensors: near means sampled alike, not that they look alike' },
  { id: 'corpus', label: 'Corpus PCA', title: 'The top 3 axes of the prompt corpus. Linear, with labelled axes' },
]
const FOCUS = [
  ['islands', 'Islands', 'tight local clusters with gaps between them'],
  ['balanced', 'Balanced', 'between the two'],
  ['continents', 'Continents', "the gallery's overall shape; local detail blurs"],
] as const
const PAINT = [
  ['region', 'Region', 'clusters of the look layout, in every space'],
  ['universe', 'Universe', 'each image in the colour of its aesthetic universe (🏷 tag); untagged grey'],
  ['origin', 'Sampler', 'what produced the image'],
  ['taste', 'Taste', 'predicted from your ★ (run by the resonance step)'],
  ['age', 'Age', 'oldest to newest'],
  ['fav', '★', 'your starred images'],
] as const

function Group({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <span className="mr-0.5 text-[10px] uppercase tracking-[2px] text-[#e9dccb]/50">{label}</span>
      {children}
    </div>
  )
}

/**
 * Every control here changes the picture in the current state. One that could
 * not -- paths when there are none, corpus dust outside its space -- isn't shown.
 */
export function MapControls({
  doc, pathLen, onResetView,
}: {
  doc?: LatentMap
  /** how many 🎬 timeline keyframes are on the map */
  pathLen: number
  onResetView: () => void
}) {
  const map = useUI((s) => s.map)
  const setMap = useUI((s) => s.setMap)
  const flip = (k: string) => setMap({ [k]: map[k] === '1' ? '' : '1' })
  const space = map.space as MapSpace
  const hasEdges = !!doc?.edges.length
  const hasCorpus = space === 'corpus' && !!doc?.corpus.length

  return (
    <div className="absolute bottom-3 right-3 flex max-w-[min(860px,calc(100%-24px))] flex-col gap-2 rounded-[12px] border border-white/10 bg-black/50 p-2.5 backdrop-blur-md">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <Group label="space">
          {SPACES.map((s) => (
            <Pill key={s.id} on={space === s.id} title={s.title} onClick={() => setMap({ space: s.id })}>
              {s.label}
            </Pill>
          ))}
        </Group>
        {space !== 'corpus' ? (
          <Group label="layout">
            {FOCUS.map(([id, label, title]) => (
              <Pill key={id} on={map.focus === id} title={title} onClick={() => setMap({ focus: id })}>
                {label}
              </Pill>
            ))}
          </Group>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
        <Group label="paint">
          {PAINT.map(([id, label, title]) => (
            <Pill key={id} on={map.paint === id} title={title} onClick={() => setMap({ paint: id })}>
              {label}
            </Pill>
          ))}
        </Group>
        <Group label="show">
          <Pill on={map.thumbs === '1'} title="draw the images themselves, or dots" onClick={() => flip('thumbs')}>
            Images
          </Pill>
          <label className="flex items-center gap-1.5 text-[11px] text-[#e9dccb]/70" title="size of every image on the map">
            size
            <input
              type="range" min={0.3} max={4} step={0.05} value={Number(map.size) || 1}
              onChange={(e) => setMap({ size: e.target.value })}
              className="w-20 accent-[#f3c27a]"
            />
          </label>
          {hasEdges ? (
            <Pill on={map.edges === '1'} title="lines from a parent image to its children" onClick={() => flip('edges')}>
              Lineage {doc?.edges.length}
            </Pill>
          ) : null}
          {hasCorpus ? (
            <Pill on={map.corpus === '1'} title="the prompts this distribution was fitted on" onClick={() => flip('corpus')}>
              Corpus
            </Pill>
          ) : null}
          {pathLen > 1 ? (
            <Pill on={map.path === '1'} title="your film's keyframes, in order" onClick={() => flip('path')}>
              🎬 path
            </Pill>
          ) : null}
          <Pill on={map.spin === '1'} title="slowly orbit, so depth is readable" onClick={() => flip('spin')}>
            Spin
          </Pill>
          <Pill onClick={onResetView} title="frame everything">Reset view</Pill>
        </Group>
      </div>
      {map.paint === 'origin' ? (
        <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[10.5px] text-[#e9dccb]/70">
          {Object.entries(ORIGIN_COLORS).map(([k, c]) => (
            <span key={k} className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ background: c }} />
              {k}
            </span>
          ))}
        </div>
      ) : LEGEND[map.paint] ? (
        <div className="flex items-center gap-2 text-[10.5px] text-[#e9dccb]/70">
          {LEGEND[map.paint][0]}
          <span
            className="h-1.5 w-28 rounded-full"
            style={{ background: `linear-gradient(90deg, ${RAMPS[map.paint as keyof typeof RAMPS].join(',')})` }}
          />
          {LEGEND[map.paint][1]}
        </div>
      ) : null}
    </div>
  )
}
