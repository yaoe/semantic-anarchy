import type { LatentMap } from '../../api/types'
import { cn } from '../../lib/utils'
import { useUI } from '../../store'
import { LEGEND, ORIGIN_COLORS, RAMPS, shortPrompt } from './palette'

const SLOTS = ['x', 'y', 'z'] as const

function Pill({
  on, onClick, children, title, className,
}: {
  on?: boolean
  onClick: () => void
  children: React.ReactNode
  title?: string
  className?: string
}) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={cn(
        'flex items-center gap-1.5 rounded-[8px] border px-2.5 py-[5px] text-[11.5px] transition-colors',
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

/**
 * The FEATURES panel: pick which axes span the view (click an axis slot, then
 * an axis), switch frame, colouring and layers. Everything persists in the
 * store's `map` slice.
 */
export function MapControls({ doc }: { doc?: LatentMap }) {
  const map = useUI((s) => s.map)
  const setMap = useUI((s) => s.setMap)
  const frame = map.frame === 'explored' ? 'explored' : 'corpus'
  const dims = doc?.dims ?? 0
  const share = doc ? (frame === 'corpus' ? doc.axes.corpus.share : doc.axes.explored.share) : []
  const ex = doc?.axes.corpus.extremes ?? []
  const slot = (SLOTS as readonly string[]).includes(map.slot) ? map.slot : 'x'

  const assign = (a: number) => {
    const next = SLOTS[(SLOTS.indexOf(slot as 'x') + 1) % 3]
    // An axis lives in one slot at a time: swap with whoever holds it.
    const holder = SLOTS.find((s) => s !== slot && Number(map[s]) === a)
    const patch: Record<string, string> = { [slot]: String(a), slot: next }
    if (holder) patch[holder] = map[slot]
    setMap(patch)
  }

  if (map.panel !== '1')
    return (
      <div className="absolute bottom-3 right-3">
        <Pill onClick={() => setMap({ panel: '1' })}>features ▴</Pill>
      </div>
    )

  return (
    <div className="absolute bottom-3 right-3 max-w-[min(700px,calc(100%-24px))] rounded-[12px] border border-white/10 bg-black/45 p-2.5 backdrop-blur-md">
      <button
        type="button"
        onClick={() => setMap({ panel: '' })}
        className="absolute right-2 top-1.5 text-[10px] uppercase tracking-[2px] text-[#e9dccb]/45 hover:text-[#e9dccb]"
        title="fold the panel away"
      >
        features ▾
      </button>
      <div className="mb-2 flex flex-wrap items-center gap-1.5 pr-20">
        <span className="mr-1 text-[10px] uppercase tracking-[2px] text-[#e9dccb]/50">frame</span>
        {(['corpus', 'explored'] as const).map((f) => (
          <Pill
            key={f}
            on={frame === f}
            onClick={() => setMap({ frame: f })}
            title={
              f === 'corpus'
                ? 'axes of the prompt corpus — stable, labelled, invertible'
                : 'PCA of your own explorations — spreads out where you travelled'
            }
          >
            {f}
          </Pill>
        ))}
        <span className="ml-3 mr-1 text-[10px] uppercase tracking-[2px] text-[#e9dccb]/50">colour</span>
        {(['origin', 'distance', 'off', 'age', 'fav'] as const).map((c) => (
          <Pill key={c} on={map.color === c} onClick={() => setMap({ color: c })}>
            {c === 'off' ? 'off-manifold' : c === 'fav' ? '★' : c}
          </Pill>
        ))}
        <span className="ml-3 mr-1 text-[10px] uppercase tracking-[2px] text-[#e9dccb]/50">show</span>
        {(
          [
            ['blob', 'hull'],
            ['edges', 'paths'],
            ['corpus', 'corpus'],
            ['path', '🎬'],
            ['squash', 'squash'],
          ] as const
        ).map(([k, label]) => (
          <Pill
            key={k}
            on={map[k] === '1'}
            onClick={() => setMap({ [k]: map[k] === '1' ? '' : '1' })}
            title={
              k === 'squash'
                ? 'radial asinh: linear near the corpus, logarithmic far out (tooltips keep true distances)'
                : undefined
            }
          >
            {label}
          </Pill>
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <span className="mr-1 text-[10px] uppercase tracking-[2px] text-[#e9dccb]/50">axes</span>
        {SLOTS.map((s) => (
          <Pill
            key={s}
            on={slot === s}
            onClick={() => setMap({ slot: s })}
            title={`next axis you click goes on ${s.toUpperCase()}`}
            className="!px-2 font-mono uppercase"
          >
            {s}
          </Pill>
        ))}
        <span className="mx-1 h-5 border-l border-white/10" />
        {Array.from({ length: dims }, (_, a) => {
          const held = SLOTS.find((s) => Number(map[s]) === a)
          const name = frame === 'corpus' ? `PC${a + 1}` : `E${a + 1}`
          const hint =
            frame === 'corpus' && ex[a]
              ? `${shortPrompt(ex[a].neg[0], 13)} ↔ ${shortPrompt(ex[a].pos[0], 13)}`
              : ''
          return (
            <Pill
              key={a}
              on={!!held}
              onClick={() => assign(a)}
              title={
                frame === 'corpus' && ex[a]
                  ? `+ ${ex[a].pos.join(' | ')}\n− ${ex[a].neg.join(' | ')}`
                  : undefined
              }
            >
              <span
                className={cn(
                  'flex h-4 w-4 items-center justify-center rounded-full border text-[9px] font-mono uppercase',
                  held ? 'border-[#f3c27a] text-[#f3c27a]' : 'border-white/25 text-transparent',
                )}
              >
                {held ?? '·'}
              </span>
              <span className="font-medium">{name}</span>
              <span className="opacity-60">{((share[a] ?? 0) * 100).toFixed(1)}%</span>
              {hint ? <span className="max-w-[190px] truncate opacity-70">{hint}</span> : null}
            </Pill>
          )
        })}
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[10.5px] text-[#e9dccb]/70">
        {map.color === 'origin' ? (
          Object.entries(ORIGIN_COLORS).map(([k, c]) => (
            <span key={k} className="flex items-center gap-1">
              <span className="h-2 w-2 rounded-full" style={{ background: c }} />
              {k}
            </span>
          ))
        ) : map.color === 'fav' ? (
          <span className="flex items-center gap-1">
            <span className="h-2 w-2 rounded-full bg-[#ffcf4d]" /> starred
          </span>
        ) : LEGEND[map.color] ? (
          <span className="flex items-center gap-2">
            {LEGEND[map.color]![0]}
            <span
              className="h-1.5 w-28 rounded-full"
              style={{
                background: `linear-gradient(90deg, ${RAMPS[map.color as keyof typeof RAMPS].join(',')})`,
              }}
            />
            {LEGEND[map.color]![1]}
          </span>
        ) : null}
        <span className="ml-auto flex items-center gap-2">
          <span className="h-px w-4 bg-[#f3c27a]" /> parent
          <span className="h-px w-4 bg-[#ffe2a8]" /> walk
          <span className="h-px w-4 bg-[#e07ab8]" /> breed
        </span>
      </div>
    </div>
  )
}
