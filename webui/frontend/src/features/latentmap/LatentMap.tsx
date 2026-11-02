import { useEffect, useMemo, useRef, useState } from 'react'

import { useImages, useLatentMap } from '../../api/queries'
import type { BackendId, ImageItem, LatentMap as MapDoc, MapNode } from '../../api/types'
import { Lightbox } from '../gallery/Lightbox'
import { useUI } from '../../store'
import { MapControls } from './MapControls'
import { ORIGIN_COLORS, RAMPS, originOf, shortPrompt } from './palette'
import { LatentScene, type Hit, type SceneData } from './scene'

/** '#rrggbb' -> sRGB 0..1 (the node shader writes colour straight out). */
function rgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}

/** Piecewise-linear ramp through `stops` at t in [0, 1]. */
function ramp(stops: string[], t: number): [number, number, number] {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1)
  const i = Math.min(stops.length - 2, Math.floor(x))
  const a = rgb(stops[i])
  const b = rgb(stops[i + 1])
  const f = x - i
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f]
}

/**
 * Radial asinh squash for display: linear inside ~2.5σ, logarithmic beyond,
 * so a T=3 walk 70σ out still sits on screen next to a readable core.
 */
const SQUASH = 2.5
function squash(x: number, y: number, z: number, on: boolean): [number, number, number] {
  if (!on) return [x, y, z]
  const r = Math.hypot(x, y, z)
  if (r < 1e-9) return [x, y, z]
  const f = (SQUASH * Math.asinh(r / SQUASH)) / r
  return [x * f, y * f, z * f]
}

function colourFor(
  n: MapNode, by: string, fav: boolean, lo: Record<string, number>, hi: Record<string, number>,
): [number, number, number] {
  const norm = (k: string, v: number) => (hi[k] > lo[k] ? (v - lo[k]) / (hi[k] - lo[k]) : 0.5)
  switch (by) {
    case 'distance':
      return ramp(RAMPS.distance, norm('distance', n.distance ?? lo.distance))
    case 'off':
      return ramp(RAMPS.off, n.off)
    case 'age':
      return ramp(RAMPS.age, norm('t', n.t))
    case 'fav':
      return rgb(fav ? '#ffcf4d' : '#59617a')
    default:
      return rgb(ORIGIN_COLORS[originOf(n)] ?? '#c9ccd6')
  }
}

/**
 * 🌌 The explored latent space in 3D. Every image with a conditioning sidecar
 * is a node at its PCA coordinates; lineage draws the paths you travelled; the
 * glass hull is the territory covered so far. Click opens the image, ⇧-click
 * appends it to the 🎬 timeline (drawn as the dashed path).
 */
export function LatentMap({ busy }: { busy: boolean }) {
  const backend = useUI((s) => s.params.backend) as BackendId
  const map = useUI((s) => s.map)
  const timeline = useUI((s) => s.timeline)
  const addKeyframe = useUI((s) => s.addKeyframe)
  const openLightbox = useUI((s) => s.openLightbox)
  const { data: images } = useImages(busy)
  const { data, error, isFetching } = useLatentMap(backend, true)

  const hostRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<LatentScene | null>(null)
  const [hover, setHover] = useState<Hit | null>(null)
  const [flash, setFlash] = useState<string | null>(null)

  const doc: MapDoc | undefined = data?.backend === backend ? data : undefined
  const nodes = useMemo(() => doc?.nodes ?? [], [doc])

  const favs = useMemo(() => new Set((images?.favorites ?? []).map((i) => i.rel)), [images])
  const byRel = useMemo(() => {
    const m = new Map<string, ImageItem>()
    for (const list of Object.values(images ?? {})) for (const it of list) m.set(it.rel, it)
    return m
  }, [images])

  // The lightbox walks the map's nodes in time order.
  const items: ImageItem[] = useMemo(
    () =>
      nodes.map(
        (n) =>
          byRel.get(n.rel) ?? {
            name: n.name, rel: n.rel, url: `/img?path=${encodeURIComponent(n.rel)}`,
            mtime: n.t, size: 0, fav: favs.has(n.rel), score: null,
            dist: n.distance, nov: null, res: null,
          },
      ),
    [nodes, byRel, favs],
  )

  // Stable callbacks into the scene (it is created once).
  const pickRef = useRef<(i: number, e: { shiftKey: boolean }) => void>(() => {})
  pickRef.current = (i, e) => {
    const n = nodes[i]
    if (!n) return
    if (e.shiftKey) {
      addKeyframe(n.rel)
      setFlash(`added ${n.name} to 🎬 timeline`)
    } else openLightbox(i)
  }

  useEffect(() => {
    const s = new LatentScene(hostRef.current!, {
      hover: setHover,
      pick: (i, e) => pickRef.current(i, e),
    })
    sceneRef.current = s
    return () => s.dispose()
  }, [])

  useEffect(() => {
    if (!flash) return
    const t = setTimeout(() => setFlash(null), 1800)
    return () => clearTimeout(t)
  }, [flash])

  const frame = map.frame === 'explored' ? 'explored' : 'corpus'
  const dims = doc?.dims ?? 3
  const axes = [map.x, map.y, map.z].map((v) => Math.min(dims - 1, Math.max(0, Number(v) || 0)))

  // Re-frame the camera only when *what* is plotted changes, not on new images.
  const viewKey = `${backend}|${frame}|${axes.join(',')}|${map.squash}`
  const lastView = useRef('')

  useEffect(() => {
    const s = sceneRef.current
    if (!s || !doc) return
    const key = frame === 'corpus' ? 'c' : 'e'
    const sq = map.squash === '1'
    const n = nodes.length
    const positions = new Float32Array(n * 3)
    const colors = new Float32Array(n * 3)
    const sizes = new Float32Array(n)
    const lo = { distance: Infinity, t: Infinity }
    const hi = { distance: -Infinity, t: -Infinity }
    for (const nd of nodes) {
      if (nd.distance != null) {
        lo.distance = Math.min(lo.distance, nd.distance)
        hi.distance = Math.max(hi.distance, nd.distance)
      }
      lo.t = Math.min(lo.t, nd.t)
      hi.t = Math.max(hi.t, nd.t)
    }
    nodes.forEach((nd, i) => {
      const v = nd[key]
      positions.set(squash(v[axes[0]] ?? 0, v[axes[1]] ?? 0, v[axes[2]] ?? 0, sq), i * 3)
      const fav = favs.has(nd.rel)
      colors.set(colourFor(nd, map.color, fav, lo, hi), i * 3)
      sizes[i] = fav ? 0.42 : 0.28
    })
    let corpus: Float32Array | null = null
    if (frame === 'corpus' && map.corpus === '1' && doc.corpus.length) {
      corpus = new Float32Array(doc.corpus.length * 3)
      doc.corpus.forEach((c, i) => corpus!.set(squash(c.c[axes[0]], c.c[axes[1]], c.c[axes[2]], sq), i * 3))
    }
    const idx = new Map(nodes.map((nd, i) => [nd.rel, i]))
    const path = map.path === '1'
      ? timeline.map((r) => idx.get(r)).filter((i): i is number => i != null)
      : []
    const ex = doc.axes.corpus.extremes
    const axisLabels = axes.map((a): [string, string] =>
      frame === 'corpus'
        ? [`PC${a + 1}− ${shortPrompt(ex[a]?.neg[0] ?? '')}`, `PC${a + 1}+ ${shortPrompt(ex[a]?.pos[0] ?? '')}`]
        : [`E${a + 1}−`, `E${a + 1}+`],
    )
    const data: SceneData = {
      positions, colors, sizes, corpus, path, axisLabels,
      edges: map.edges === '1' ? doc.edges : [],
      showBlob: map.blob === '1',
      showEdges: map.edges === '1',
      refit: lastView.current !== viewKey,
    }
    lastView.current = viewKey
    s.update(data)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doc, nodes, favs, timeline, map, viewKey])

  useEffect(() => {
    sceneRef.current?.setHighlight(hover?.kind === 'node' ? hover.index : null)
  }, [hover])

  const hovered = hover?.kind === 'node' ? nodes[hover.index] : undefined
  const hoveredCorpus = hover?.kind === 'corpus' ? doc?.corpus[hover.index] : undefined
  const share = frame === 'corpus' ? doc?.axes.corpus.share : doc?.axes.explored.share
  const shown = share ? axes.reduce((s, a) => s + (share[a] ?? 0), 0) : 0
  const offMean = nodes.length ? nodes.reduce((s, n) => s + n.off, 0) / nodes.length : 0

  return (
    <div
      className="absolute inset-0 overflow-hidden rounded-[10px] border border-line"
      style={{
        background:
          'radial-gradient(120% 90% at 18% -10%, #7a4630 0%, #3a2219 28%, #14100f 58%, #08080b 100%)',
      }}
    >
      <div ref={hostRef} className="absolute inset-0" />

      {/* status, top-left */}
      <div className="pointer-events-none absolute left-4 top-3 text-[12px] leading-relaxed text-[#e9dccb]/80">
        <div className="text-[11px] uppercase tracking-[2px] text-[#e9dccb]/55">
          latent map · {backend} · {frame} frame
        </div>
        {doc ? (
          <>
            <div>
              {nodes.length} images · {doc.edges.length} paths
              {frame === 'corpus' ? ` · ${doc.corpus.length} corpus prompts` : ''}
            </div>
            <div>
              these 3 axes carry {(shown * 100).toFixed(1)}% of the{' '}
              {frame === 'corpus' ? 'corpus' : 'explored'} variance
            </div>
            {frame === 'corpus' && offMean > 0.5 ? (
              <div className="text-[#f3c27a]">
                {Math.round(offMean * 100)}% of the average image lies off the corpus
                manifold — this frame pulls those toward the centre; try Explored.
              </div>
            ) : null}
          </>
        ) : null}
        {isFetching ? <div className="text-[#e9dccb]/50">projecting…</div> : null}
      </div>

      {error ? (
        <div className="absolute inset-0 flex items-center justify-center p-8 text-center text-[13px] text-[#e9dccb]/80">
          {String((error as Error).message)}
        </div>
      ) : doc && !nodes.length ? (
        <div className="pointer-events-none absolute inset-x-0 top-1/3 text-center text-[13px] text-[#e9dccb]/70">
          no {backend} images with conditioning sidecars yet — generate some.
        </div>
      ) : null}

      {hovered && hover ? (
        <div
          className="pointer-events-none absolute z-10 w-[188px] rounded-[9px] border border-white/15 bg-black/75 p-1.5 text-[11px] text-[#efe6da] shadow-xl backdrop-blur"
          style={{
            left: Math.min(hover.x + 16, (hostRef.current?.clientWidth ?? 9999) - 200),
            top: Math.max(8, Math.min(hover.y - 90, (hostRef.current?.clientHeight ?? 9999) - 250)),
          }}
        >
          <img
            src={`/img?path=${encodeURIComponent(hovered.rel)}&t=${Math.floor(hovered.t)}`}
            alt=""
            className="mb-1 block aspect-square w-full rounded-[6px] object-cover"
          />
          <div className="truncate font-mono text-[10.5px] opacity-80">{hovered.name}</div>
          <div>
            {hovered.kind === 'explore' ? hovered.mode : hovered.sampler}
            {hovered.temperature != null && hovered.kind !== 'explore' ? ` · T ${hovered.temperature}` : ''}
            {hovered.distance != null ? ` · d ${hovered.distance.toFixed(2)}` : ''}
            {` · off ${Math.round(hovered.off * 100)}%`}
          </div>
        </div>
      ) : hoveredCorpus && hover ? (
        <div
          className="pointer-events-none absolute z-10 max-w-[280px] rounded-[8px] border border-white/10 bg-black/70 px-2 py-1 text-[11.5px] text-[#cfdcf0] backdrop-blur"
          style={{ left: hover.x + 14, top: hover.y + 10 }}
        >
          {hoveredCorpus.label}
        </div>
      ) : null}

      {flash ? (
        <div className="pointer-events-none absolute left-1/2 top-4 -translate-x-1/2 rounded-full border border-white/15 bg-black/70 px-3 py-1 text-[12px] text-[#efe6da]">
          {flash}
        </div>
      ) : null}

      <div className="pointer-events-none absolute bottom-3 left-4 text-[11px] text-[#e9dccb]/45">
        drag to orbit · right-drag to move · scroll to zoom · click opens · ⇧-click → 🎬
      </div>

      <MapControls doc={doc} />

      <Lightbox items={items} />
    </div>
  )
}
