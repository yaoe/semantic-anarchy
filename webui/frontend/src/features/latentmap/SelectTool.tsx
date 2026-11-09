/**
 * Lasso and circle selection over the 🌌 map, in SCREEN space: what is selected
 * is whatever images you see inside the shape from where you are looking.
 *
 *   lasso   drag a free shape. ⇧ adds, ctrl/⌘ removes, plain replaces.
 *           In the orbit tool, ⇧-drag lassos too (adding) — orbit's own
 *           ⇧-drag pan is still on right-drag.
 *   circle  a brush of fixed screen radius (wheel or [ ] resizes). Press and
 *           drag to MOVE the region: the selection becomes what is under it
 *           when you let go. ⇧-drag paints (adds everything swept), ctrl/⌘
 *           erases. This is the honest "move the sampling sphere": the region
 *           is always a set of real images, whose own conditioning a sampler
 *           can use — a spot on a UMAP has no conditioning of its own.
 *
 * Listeners sit on the host in the CAPTURE phase, so they run before
 * OrbitControls (on the canvas inside it) and can switch it off for the drag.
 */
import { useEffect, useRef, useState, type RefObject } from 'react'

import type { LatentScene } from './scene'

export type MapTool = 'orbit' | 'lasso' | 'circle'
export type SelectMode = 'replace' | 'add' | 'remove'

const MIN_DRAG = 5
export const CIRCLE_MIN = 24
export const CIRCLE_MAX = 400

function inPolygon(x: number, y: number, poly: number[][]): boolean {
  let inside = false
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i]
    const [xj, yj] = poly[j]
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside
  }
  return inside
}

function inCircle(xy: Float32Array, cx: number, cy: number, r: number): number[] {
  const out: number[] = []
  const r2 = r * r
  for (let i = 0; i < xy.length / 2; i++) {
    const dx = xy[i * 2] - cx
    const dy = xy[i * 2 + 1] - cy
    if (dx * dx + dy * dy <= r2) out.push(i)
  }
  return out
}

const modeOf = (e: PointerEvent | MouseEvent): SelectMode =>
  e.ctrlKey || e.metaKey ? 'remove' : e.shiftKey ? 'add' : 'replace'

export function SelectTool({
  host,
  scene,
  tool,
  radius,
  setRadius,
  onCommit,
  onPreview,
}: {
  host: RefObject<HTMLDivElement | null>
  scene: RefObject<LatentScene | null>
  tool: MapTool
  radius: number
  setRadius: (r: number) => void
  /** scene indexes the gesture chose, and how they combine with the selection */
  onCommit: (idx: number[], mode: SelectMode) => void
  /** live scene indexes under the brush while it drags (null = gesture over) */
  onPreview: (idx: number[] | null, mode: SelectMode) => void
}) {
  const [lasso, setLasso] = useState<number[][] | null>(null)
  const [cursor, setCursor] = useState<{ x: number; y: number; down: boolean } | null>(null)
  const [under, setUnder] = useState(0)

  // Everything the native listeners read, without re-binding them per render.
  const live = useRef({ tool, radius, onCommit, onPreview, setRadius })
  live.current = { tool, radius, onCommit, onPreview, setRadius }

  useEffect(() => {
    const el = host.current
    if (!el) return
    let drag: {
      kind: 'lasso' | 'circle'
      x0: number
      y0: number
      pts: number[][]
      mode: SelectMode
      moved: boolean
      xy: Float32Array
      swept: Set<number>
    } | null = null
    let raf = 0
    let last: PointerEvent | null = null

    const local = (e: MouseEvent) => {
      const r = el.getBoundingClientRect()
      return [e.clientX - r.left, e.clientY - r.top]
    }

    const circleAt = (x: number, y: number) => {
      const s = scene.current
      if (!s || !drag) return []
      return inCircle(drag.xy, x, y, live.current.radius)
    }

    const onDown = (e: PointerEvent) => {
      if (e.button !== 0) return
      const t = live.current.tool
      const kind = t === 'circle' ? 'circle' : t === 'lasso' || e.shiftKey ? 'lasso' : null
      if (!kind || !scene.current) return
      // Only gestures that start on the canvas, not on a panel over it.
      if (!(e.target instanceof HTMLCanvasElement)) return
      scene.current.setOrbitEnabled(false)
      const [x, y] = local(e)
      drag = {
        kind, x0: x, y0: y, pts: [[x, y]], mode: modeOf(e), moved: false,
        xy: scene.current.projectAll(), swept: new Set(),
      }
      if (kind === 'circle') {
        // The region is placed on press: a click alone selects what is under it.
        const idx = circleAt(x, y)
        idx.forEach((i) => drag!.swept.add(i))
        live.current.onPreview(idx, drag.mode)
      }
    }

    const step = () => {
      raf = 0
      const e = last
      if (!e) return
      const [x, y] = local(e)
      if (live.current.tool === 'circle') setCursor({ x, y, down: !!drag })
      if (!drag) {
        if (live.current.tool === 'circle' && scene.current) {
          const xy = scene.current.projectAll()
          setUnder(inCircle(xy, x, y, live.current.radius).length)
        }
        return
      }
      if (Math.hypot(x - drag.x0, y - drag.y0) > MIN_DRAG) drag.moved = true
      if (drag.kind === 'lasso') {
        const p = drag.pts[drag.pts.length - 1]
        if (Math.hypot(x - p[0], y - p[1]) > 3) drag.pts.push([x, y])
        if (drag.moved) setLasso([...drag.pts])
      } else {
        const idx = circleAt(x, y)
        if (drag.mode === 'replace') {
          drag.swept = new Set(idx)
          live.current.onPreview(idx, 'replace')
        } else {
          idx.forEach((i) => drag!.swept.add(i))
          live.current.onPreview([...drag.swept], drag.mode)
        }
        setUnder(idx.length)
      }
    }

    const onMove = (e: PointerEvent) => {
      last = e
      if (!raf) raf = requestAnimationFrame(step)
    }

    const onUp = () => {
      const d = drag
      drag = null
      scene.current?.setOrbitEnabled(true)
      if (!d) return
      setLasso(null)
      setCursor((c) => (c ? { ...c, down: false } : c))
      if (d.kind === 'lasso') {
        // A click that never moved is left to the scene's own click handling
        // (⇧-click toggles one image).
        if (d.moved && d.pts.length > 2) {
          const idx: number[] = []
          for (let i = 0; i < d.xy.length / 2; i++) {
            const x = d.xy[i * 2]
            const y = d.xy[i * 2 + 1]
            if (!Number.isNaN(x) && inPolygon(x, y, d.pts)) idx.push(i)
          }
          live.current.onCommit(idx, d.mode)
        }
      } else {
        live.current.onPreview(null, d.mode)
        live.current.onCommit([...d.swept], d.mode)
      }
    }

    const onWheel = (e: WheelEvent) => {
      if (live.current.tool !== 'circle') return
      e.preventDefault()
      e.stopPropagation()
      const r = live.current.radius * Math.exp(-e.deltaY * 0.0015)
      live.current.setRadius(Math.round(Math.min(CIRCLE_MAX, Math.max(CIRCLE_MIN, r))))
    }

    const onLeave = () => setCursor(null)

    el.addEventListener('pointerdown', onDown, true)
    el.addEventListener('wheel', onWheel, { capture: true, passive: false })
    el.addEventListener('pointerleave', onLeave)
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp)
    window.addEventListener('pointercancel', onUp)
    return () => {
      cancelAnimationFrame(raf)
      el.removeEventListener('pointerdown', onDown, true)
      el.removeEventListener('wheel', onWheel, true)
      el.removeEventListener('pointerleave', onLeave)
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
      window.removeEventListener('pointercancel', onUp)
    }
  }, [host, scene])

  // Leaving the circle tool hides its cursor.
  useEffect(() => {
    if (tool !== 'circle') setCursor(null)
  }, [tool])

  return (
    <svg className="pointer-events-none absolute inset-0 h-full w-full">
      {lasso && lasso.length > 1 ? (
        <polygon
          points={lasso.map((p) => p.join(',')).join(' ')}
          fill="rgba(111,255,208,0.08)"
          stroke="#6fffd0"
          strokeWidth={1.5}
          strokeDasharray="5 4"
        />
      ) : null}
      {tool === 'circle' && cursor ? (
        <g>
          <circle
            cx={cursor.x}
            cy={cursor.y}
            r={radius}
            fill={cursor.down ? 'rgba(111,255,208,0.10)' : 'rgba(111,255,208,0.04)'}
            stroke="#6fffd0"
            strokeOpacity={cursor.down ? 0.95 : 0.6}
            strokeWidth={1.5}
            strokeDasharray={cursor.down ? undefined : '6 5'}
          />
          <text
            x={cursor.x}
            y={cursor.y - radius - 6}
            textAnchor="middle"
            fill="#6fffd0"
            fontSize={11}
            style={{ paintOrder: 'stroke', stroke: 'rgba(0,0,0,0.7)', strokeWidth: 3 }}
          >
            {under} image{under === 1 ? '' : 's'}
          </text>
        </g>
      ) : null}
    </svg>
  )
}
