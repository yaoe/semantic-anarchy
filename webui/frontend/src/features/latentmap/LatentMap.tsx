import { useEffect, useMemo, useRef, useState } from 'react'

import { useExplore, useImages, useLatentMap, useResonance } from '../../api/queries'
import type { BackendId, ImageItem, LatentMap as MapDoc, MapSpace } from '../../api/types'
import { cn } from '../../lib/utils'
import { useViewerList } from '../gallery/viewerLists'
import { useUI } from '../../store'
import { BatchNotice } from '../universes/BatchNotice'
import { universeColor, useSampleBatches, useUniverses } from '../universes/universes'
import { MapControls, Pill } from './MapControls'
import { ORIGIN_COLORS, RAMPS, UNCLAIMED, originOf, regionColor, shortPrompt } from './palette'
import { LatentScene, type Hit, type SceneData } from './scene'
import { SelectionPanel, Thumb } from './SelectionPanel'
import { CIRCLE_MAX, CIRCLE_MIN, SelectTool, type MapTool, type SelectMode } from './SelectTool'

const COLOR = new Map<string, [number, number, number]>()
/** any CSS colour -> sRGB 0..1 (the node shader writes colour straight out). */
function rgb(css: string): [number, number, number] {
  const hit = COLOR.get(css)
  if (hit) return hit
  const ctx = document.createElement('canvas').getContext('2d')!
  ctx.fillStyle = css
  ctx.fillRect(0, 0, 1, 1)
  const d = ctx.getImageData(0, 0, 1, 1).data
  const out: [number, number, number] = [d[0] / 255, d[1] / 255, d[2] / 255]
  COLOR.set(css, out)
  return out
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
 * Radial asinh squash for the corpus space only: linear inside ~2.5σ,
 * logarithmic beyond, so a T=3 walk 70σ out still sits next to a readable core.
 * A UMAP layout has no such outliers.
 */
const SQUASH = 2.5
function squash(v: number[]): [number, number, number] {
  const r = Math.hypot(v[0], v[1], v[2])
  if (r < 1e-9) return [v[0], v[1], v[2]]
  const f = (SQUASH * Math.asinh(r / SQUASH)) / r
  return [v[0] * f, v[1] * f, v[2] * f]
}

/**
 * Perturbation radius. Similarity to the source falls steadily with it, and even
 * the largest stays closer than a random sample (docs/reference/latent-map.md).
 */
const REACH = [
  ['0.2', 'Close'],
  ['0.4', 'Nearby'],
  ['0.8', 'Wander'],
  ['1.2', 'Far'],
] as const

const SPACE_KEY = { look: 'l', cond: 'u', corpus: 'c' } as const

function alikeLabel(v: number | null | undefined) {
  if (v == null) return { word: 'not measured yet', good: true }
  if (v >= 0.75) return { word: 'strongly', good: true }
  if (v >= 0.62) return { word: 'partly', good: true }
  return { word: 'barely', good: false }
}

const TOOLS: { id: MapTool; key: string; label: string; title: string }[] = [
  { id: 'orbit', key: 'o', label: '✥ Orbit', title: 'drag to orbit, right-drag to pan, wheel to zoom · ⇧-click selects one image · ⇧-drag lassos (o)' },
  { id: 'lasso', key: 'l', label: '◌ Lasso', title: 'drag a shape around images to select them · ⇧ adds · ctrl removes · right-drag still pans (l)' },
  {
    id: 'circle', key: 'c', label: '◯ Region', title:
      'a circle you move over the map: press and drag, and the selection becomes the images under it · ' +
      'wheel or [ ] resizes · ⇧-drag paints, ctrl erases (c)',
  },
]

/** How many nearest images "select around this spot" takes. */
const SPOT_K = [8, 16, 32] as const

/** What a click on the map chose: an image, or a spot whose nearest image anchors it. */
type Pick =
  | { kind: 'node'; index: number }
  | { kind: 'spot'; index: number; at: [number, number, number] }

/**
 * 🌌 The explored space in 3D. Every image sits where a UMAP of what it looks
 * like puts it, so the map is a thing to read: which worlds exist, how big they
 * are, what lies between them. Click anywhere to sample there.
 */
export function LatentMap({ busy }: { busy: boolean }) {
  const backend = useUI((s) => s.params.backend) as BackendId
  const map = useUI((s) => s.map)
  const setMap = useUI((s) => s.setMap)
  const timeline = useUI((s) => s.timeline)
  const addKeyframe = useUI((s) => s.addKeyframe)
  const openLightbox = useUI((s) => s.openLightbox)
  const lightboxOpen = useUI((s) => s.lightboxRel !== null)
  const mapSel = useUI((s) => s.mapSel)
  const setMapSel = useUI((s) => s.setMapSel)
  const toggleMapSel = useUI((s) => s.toggleMapSel)
  const addMapSel = useUI((s) => s.addMapSel)
  const removeMapSel = useUI((s) => s.removeMapSel)
  const mapUniverses = useUI((s) => s.mapUniverses)
  const setMapUniverses = useUI((s) => s.setMapUniverses)
  const { list: universes } = useUniverses()
  const lastBatchAt = useSampleBatches((s) => s.batches[0]?.queuedAt ?? 0)
  const { data: images } = useImages(busy)
  const { data, error, isFetching } = useLatentMap(backend, map.focus, busy)
  const explore = useExplore()
  const resonance = useResonance()

  const hostRef = useRef<HTMLDivElement>(null)
  const sceneRef = useRef<LatentScene | null>(null)
  const [hover, setHover] = useState<Hit | null>(null)
  const [flash, setFlash] = useState<string | null>(null)
  const [pick, setPick] = useState<Pick | null>(null)
  const [region, setRegion] = useState<number | null>(null)
  const [sampledAt, setSampledAt] = useState(0)
  const [tool, setTool] = useState<MapTool>('orbit')
  const [circleR, setCircleR] = useState(90)
  // A lasso/circle gesture still in progress: drawn as the selection, not yet stored.
  const [preview, setPreview] = useState<{ idx: number[]; mode: SelectMode } | null>(null)
  const freshAt = Math.max(sampledAt, lastBatchAt)

  const doc: MapDoc | undefined = data?.backend === backend ? data : undefined
  const nodes = useMemo(() => doc?.nodes ?? [], [doc])
  const space = (['look', 'cond', 'corpus'].includes(map.space) ? map.space : 'look') as MapSpace
  const key = SPACE_KEY[space]

  // An image can be absent from a space (no CLIP embedding yet), so the scene
  // draws a subset: scene index -> node index.
  const shown = useMemo(() => {
    const out: number[] = []
    nodes.forEach((n, i) => {
      if (n[key]) out.push(i)
    })
    return out
  }, [nodes, key])
  const shownRef = useRef(shown)
  shownRef.current = shown
  const nodeIndexOf = (i: number) => shownRef.current[i]

  const favs = useMemo(() => new Set((images?.favorites ?? []).map((i) => i.rel)), [images])
  const byRel = useMemo(() => {
    const m = new Map<string, ImageItem>()
    for (const list of Object.values(images ?? {})) for (const it of list) m.set(it.rel, it)
    return m
  }, [images])

  // rel -> scene index, for the selection and the universes.
  const sceneOf = useMemo(() => new Map(shown.map((ni, i) => [nodes[ni].rel, i])), [shown, nodes])
  const cellOf = useMemo(() => {
    const m = new Map(nodes.map((n) => [n.rel, n.a]))
    return (rel: string) => m.get(rel) ?? -1
  }, [nodes])
  const onMap = useMemo(() => new Set(nodes.map((n) => n.rel)), [nodes])
  // Universes with their members on THIS map (this backend, placed).
  const mapUnis = useMemo(
    () => universes.map((u, i) => ({ ...u, key: i, here: u.rels.filter((r) => sceneOf.has(r)) })),
    [universes, sceneOf],
  )
  const shownUnis = useMemo(() => new Set(mapUniverses.map((t) => t.toLocaleLowerCase())), [mapUniverses])
  // The stored selection, minus images this map doesn't place (another backend, not embedded yet).
  const selHere = useMemo(() => mapSel.filter((r) => onMap.has(r)), [mapSel, onMap])

  // The app-level viewer walks the map's nodes in time order.
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
  useViewerList('map', items)

  // Stable callbacks into the scene (it is created once).
  const handlers = useRef<{
    pick: (i: number, e: { shiftKey: boolean }) => void
    empty: (p: [number, number, number], e: { shiftKey: boolean }) => void
    open: (i: number) => void
  }>({ pick: () => {}, empty: () => {}, open: () => {} })
  handlers.current.pick = (i, e) => {
    const ni = nodeIndexOf(i)
    const n = nodes[ni]
    if (!n || tool === 'circle') return // the circle selected on press
    if (e.shiftKey) {
      toggleMapSel(n.rel)
    } else {
      // A click on an image opens THE viewer (every per-image action lives
      // there); the pick stays behind it, so the sampling panel is waiting
      // when the viewer closes.
      setPick({ kind: 'node', index: ni })
      openLightbox(n.rel, 'map')
    }
  }
  handlers.current.empty = (at, e) => {
    // ⇧ is the selection modifier; a ⇧-click that missed every image is a no-op.
    if (tool === 'circle' || e.shiftKey) return
    // Sampling needs an image to start from: the one nearest the clicked spot.
    let best = -1
    let bestD = Infinity
    shownRef.current.forEach((ni, i) => {
      const p = sceneRef.current?.positionOf(i)
      if (!p) return
      const d = Math.hypot(p[0] - at[0], p[1] - at[1], p[2] - at[2])
      if (d < bestD) {
        bestD = d
        best = ni
      }
    })
    if (best >= 0) setPick({ kind: 'spot', index: best, at })
  }
  handlers.current.open = (i) => {
    const n = nodes[nodeIndexOf(i)]
    if (n) openLightbox(n.rel, 'map')
  }

  useEffect(() => {
    const s = new LatentScene(hostRef.current!, {
      hover: setHover,
      pick: (i, e) => handlers.current.pick(i, e),
      pickEmpty: (p, e) => handlers.current.empty(p, e),
      open: (i) => handlers.current.open(i),
    })
    sceneRef.current = s
    return () => s.dispose()
  }, [])

  useEffect(() => {
    if (!flash) return
    const t = setTimeout(() => setFlash(null), 2400)
    return () => clearTimeout(t)
  }, [flash])

  // New images have no look position until CLIP has seen them. Embed them when
  // nothing else is running; once per count, so an unreadable file can't loop.
  const asked = useRef(-1)
  const missing = doc?.unembedded ?? 0
  useEffect(() => {
    // A doc being rebuilt may predate the embeddings, so its count can't be trusted.
    if (!missing || busy || doc?.rebuilding || asked.current === missing) return
    asked.current = missing
    resonance.mutate()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [missing, busy, doc?.rebuilding])

  // Re-frame the camera only when *what* is plotted changes, not on new images.
  const viewKey = `${backend}|${space}|${map.focus}`
  const lastView = useRef('')
  const atlasUrl = doc
    ? `/api/latent-map/atlas?backend=${encodeURIComponent(backend)}&v=${doc.atlas.v ?? Math.floor(doc.built)}`
    : ''

  useEffect(() => {
    const s = sceneRef.current
    if (!s || !doc) return
    const thumbs = map.thumbs === '1'
    const n = shown.length
    const positions = new Float32Array(n * 3)
    const colors = new Float32Array(n * 3)
    const sizes = new Float32Array(n)
    const tiles = new Float32Array(n)
    const fades = new Float32Array(n)
    let tLo = Infinity
    let tHi = -Infinity
    for (const nd of nodes) {
      tLo = Math.min(tLo, nd.t)
      tHi = Math.max(tHi, nd.t)
    }
    // Images a fixed share of the typical gap: every space and gallery size reads
    // alike, whatever scale its layout came out at.
    let spread = 0
    shown.forEach((ni) => {
      const v = nodes[ni][key] as number[]
      spread += v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    })
    const base = ((thumbs ? 1.5 : 1.2) * Math.sqrt(spread / Math.max(1, n))) / Math.cbrt(Math.max(1, n))
    // Taste scores bunch in a narrow band (0.26-0.65 on the real gallery): stretch
    // that band over the ramp, or every image paints the same colour.
    const scores = shown.map((ni) => byRel.get(nodes[ni].rel)?.res ?? 0).sort((p, q) => p - q)
    // The first universe an image wears that is highlighted (or, painting by
    // universe, the first it wears at all).
    const uniOf = (rel: string, only: boolean) => {
      for (const t of byRel.get(rel)?.tags ?? []) if (!only || shownUnis.has(t.toLocaleLowerCase())) return t
      return null
    }
    const resLo = scores[Math.floor(scores.length * 0.05)] ?? 0
    const resHi = scores[Math.floor(scores.length * 0.95)] ?? 1
    shown.forEach((ni, i) => {
      const nd = nodes[ni]
      const v = nd[key] as number[]
      positions.set(space === 'corpus' ? squash(v) : [v[0], v[1], v[2]], i * 3)
      const fav = favs.has(nd.rel)
      const res = byRel.get(nd.rel)?.res ?? 0
      const uni = shownUnis.size ? uniOf(nd.rel, true) : map.paint === 'universe' ? uniOf(nd.rel, false) : null
      colors.set(
        uni ? rgb(universeColor(uni))
          : map.paint === 'universe' ? rgb(UNCLAIMED)
          : map.paint === 'origin' ? rgb(ORIGIN_COLORS[originOf(nd)] ?? '#c9ccd6')
          : map.paint === 'taste' ? ramp(RAMPS.taste, resHi > resLo ? (res - resLo) / (resHi - resLo) : 0.5)
            : map.paint === 'age' ? ramp(RAMPS.age, tHi > tLo ? (nd.t - tLo) / (tHi - tLo) : 0.5)
              : map.paint === 'fav' ? rgb(fav ? '#ffcf4d' : UNCLAIMED)
                : rgb(regionColor(nd.r)),
        i * 3,
      )
      const fresh = freshAt > 0 && nd.t > freshAt
      const inFocus = (region == null || nd.r === region) && (!shownUnis.size || uni != null)
      // Faded images also shrink: at full size they would block the view of the
      // region from in front of it.
      sizes[i] = base * (fav ? 1.25 : 1) * (fresh ? 2 : 1) * (inFocus ? 1 : 0.3)
      tiles[i] = nd.a
      fades[i] = inFocus ? 1 : 0.35
    })
    const sceneIdx = new Map(shown.map((ni, i) => [ni, i]))
    const relIdx = new Map(shown.map((ni, i) => [nodes[ni].rel, i]))
    const pathIdx = timeline.map((r) => relIdx.get(r)).filter((i): i is number => i != null)
    let corpus: Float32Array | null = null
    if (space === 'corpus' && map.corpus === '1' && doc.corpus.length) {
      corpus = new Float32Array(doc.corpus.length * 3)
      doc.corpus.forEach((c, i) => corpus!.set(squash(c.c), i * 3))
    }
    const ex = doc.axes.corpus.extremes
    const axisLabels = space === 'corpus'
      ? [0, 1, 2].map((a): [string, string] => [
          `PC${a + 1}− ${shortPrompt(ex[a]?.neg[0] ?? '')}`,
          `PC${a + 1}+ ${shortPrompt(ex[a]?.pos[0] ?? '')}`,
        ])
      : []
    const edges = map.edges === '1'
      ? doc.edges.flatMap((e) => {
          const a = sceneIdx.get(e.a)
          const b = sceneIdx.get(e.b)
          return a != null && b != null ? [{ a, b, kind: e.kind }] : []
        })
      : []
    const scene: SceneData = {
      positions, colors, sizes, tiles, fades, corpus, axisLabels, edges,
      path: map.path === '1' ? pathIdx : [],
      showEdges: map.edges === '1',
      refit: lastView.current !== viewKey,
    }
    lastView.current = viewKey
    s.setAtlas(thumbs ? atlasUrl : null, doc.atlas.cols)
    s.update(scene)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [doc, nodes, shown, favs, byRel, timeline, map.paint, map.thumbs, map.edges, map.corpus, map.path, viewKey, region, freshAt, shownUnis])

  useEffect(() => {
    sceneRef.current?.setNodeScale(Number(map.size) || 1)
  }, [map.size])
  useEffect(() => {
    sceneRef.current?.setSpin(map.spin === '1')
  }, [map.spin])

  useEffect(() => {
    sceneRef.current?.setHover(hover?.kind === 'node' ? hover.index : null)
  }, [hover])

  // The selection as drawn: the stored one, with a gesture in progress applied.
  const selIdx = useMemo(() => {
    const stored = selHere.map((r) => sceneOf.get(r)!).filter((i) => i != null)
    if (!preview) return stored
    if (preview.mode === 'replace') return preview.idx
    if (preview.mode === 'add') return [...new Set([...stored, ...preview.idx])]
    const drop = new Set(preview.idx)
    return stored.filter((i) => !drop.has(i))
  }, [selHere, sceneOf, preview])

  // Typical gap between images in this layout (world units): the halo's floor.
  const gap = useMemo(() => {
    let sum = 0
    shown.forEach((ni) => {
      const v = nodes[ni][key] as number[]
      sum += v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    })
    return Math.sqrt(sum / Math.max(1, shown.length)) / Math.cbrt(Math.max(1, shown.length))
  }, [shown, nodes, key])
  const spread = Math.max(0, Number(map.spread) || 0)

  useEffect(() => {
    const s = sceneRef.current
    if (!s) return
    s.setSelection(selIdx)
    // Halo = an illustration of the spread, scaled by how far apart the
    // selected images sit HERE: at spread 1 each halo reaches the selection's
    // own RMS radius, so a group's halos merge into one region; at 0 they hug
    // the images. Not a region of conditioning space (see SelectionPanel).
    const pts = selIdx.map((i) => s.positionOf(i)).filter((p): p is [number, number, number] => !!p)
    let rms = 0
    if (pts.length > 1) {
      const c = [0, 1, 2].map((k) => pts.reduce((a, p) => a + p[k], 0) / pts.length)
      rms = Math.sqrt(pts.reduce((a, p) => a + (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2 + (p[2] - c[2]) ** 2, 0) / pts.length)
    }
    s.setHalo(gap * 0.7 + spread * Math.max(rms, gap * 2), pts.length > 0)
  }, [selIdx, gap, spread, doc])

  const commitSelection = (idx: number[], mode: SelectMode) => {
    const rels = idx.map((i) => nodes[shownRef.current[i]]?.rel).filter((r): r is string => !!r)
    if (mode === 'add') addMapSel(rels)
    else if (mode === 'remove') removeMapSel(rels)
    else setMapSel(rels)
    if (rels.length || mode === 'replace')
      setFlash(mode === 'remove' ? `dropped ${rels.length}` : `${mode === 'add' ? 'added' : 'selected'} ${rels.length} image${rels.length === 1 ? '' : 's'}`)
  }

  const flyToRels = (rels: string[]) => {
    const pts = rels.flatMap((r) => {
      const i = sceneOf.get(r)
      const p = i != null ? sceneRef.current?.positionOf(i) : null
      return p ? [p] : []
    })
    if (!pts.length) return
    const c = [0, 1, 2].map((k) => pts.reduce((a, p) => a + p[k], 0) / pts.length) as [number, number, number]
    const r = Math.max(...pts.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2])))
    sceneRef.current?.flyTo(c, Math.max(r * 1.3, 0.8))
  }

  const toggleUniverse = (tag: string) => {
    const low = tag.toLocaleLowerCase()
    const on = shownUnis.has(low)
    const next = on ? mapUniverses.filter((t) => t.toLocaleLowerCase() !== low) : [...mapUniverses, tag]
    setMapUniverses(next)
    if (!on) flyToRels(mapUnis.find((u) => u.tag === tag)?.here ?? [])
    else if (!next.length) sceneRef.current?.resetView()
  }

  // Map keys. Not while the viewer is up (it owns digits and letters) or typing.
  const keyState = useRef({ mapUnis, toggleUniverse, tool, selHere, mapUniverses })
  keyState.current = { mapUnis, toggleUniverse, tool, selHere, mapUniverses }
  useEffect(() => {
    if (lightboxOpen) return
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null
      if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const k = keyState.current
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key
      const toolHit = TOOLS.find((x) => x.key === key)
      if (toolHit && !e.shiftKey) {
        setTool(toolHit.id)
      } else if (/^[1-9]$/.test(key) && !e.shiftKey) {
        const u = k.mapUnis[Number(key) - 1]
        if (u) k.toggleUniverse(u.tag)
      } else if (key === '0') {
        setMapUniverses([])
      } else if (key === '[' || key === ']') {
        setCircleR((r) => Math.min(CIRCLE_MAX, Math.max(CIRCLE_MIN, Math.round(r * (key === ']' ? 1.2 : 1 / 1.2)))))
      } else if (key === 'Escape') {
        // Peel one layer at a time: tool, then selection, then highlights.
        if (k.tool !== 'orbit') setTool('orbit')
        else if (k.selHere.length) setMapSel([])
        else if (k.mapUniverses.length) setMapUniverses([])
        else return
      } else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [lightboxOpen, setMapSel, setMapUniverses])

  // A pick is a node index; a rebuild can reorder nothing but may drop images.
  const picked = pick ? nodes[pick.index] : undefined
  const pickedScene = picked ? shown.indexOf(pick!.index) : -1
  useEffect(() => {
    const s = sceneRef.current
    if (!s) return
    s.setSelected(pickedScene >= 0 ? pickedScene : null)
    s.setProbe(pick?.kind === 'spot' && pickedScene >= 0 ? pick.at : null, pickedScene >= 0 ? [pickedScene] : [])
  }, [pick, pickedScene])

  const regionRow = doc?.regions.find((r) => r.id === region)
  const reach = REACH.some(([v]) => v === map.reach) ? map.reach : '0.2'
  const count = Math.min(24, Math.max(1, Number(map.count) || 6))

  const sampleFrom = (rels: string[]) => {
    const each = Math.max(1, Math.ceil(count / rels.length))
    setSampledAt(Date.now() / 1000)
    rels.forEach((rel) =>
      explore.mutate({ src: rel, mode: 'neighborhood', radius: Number(reach), n: each, stay_local: true }),
    )
    setFlash(`queued ${each * rels.length} new image${each * rels.length > 1 ? 's' : ''}; they appear on the map when done`)
  }

  /** The k images nearest a spot, in the current space: a region of real images. */
  const selectAround = (at: [number, number, number], k: number) => {
    const d = shown.map((ni, i) => {
      const p = sceneRef.current?.positionOf(i)
      return { rel: nodes[ni].rel, d: p ? Math.hypot(p[0] - at[0], p[1] - at[1], p[2] - at[2]) : Infinity }
    })
    d.sort((a, b) => a.d - b.d)
    setMapSel(d.slice(0, k).map((x) => x.rel))
    setPick(null)
  }

  const chooseRegion = (id: number) => {
    const next = region === id ? null : id
    setRegion(next)
    if (next == null) return sceneRef.current?.resetView()
    setPick(null)
    const pts = shown.flatMap((ni, i) => {
      if (nodes[ni].r !== next) return []
      const p = sceneRef.current?.positionOf(i)
      return p ? [p] : []
    })
    if (!pts.length) return
    const c = [0, 1, 2].map((k) => pts.reduce((a, p) => a + p[k], 0) / pts.length) as [number, number, number]
    const r = Math.max(...pts.map((p) => Math.hypot(p[0] - c[0], p[1] - c[1], p[2] - c[2])))
    sceneRef.current?.flyTo(c, Math.max(r * 1.3, 0.8))
  }

  const hovered = hover?.kind === 'node' ? nodes[nodeIndexOf(hover.index)] : undefined
  const hoveredCorpus = hover?.kind === 'corpus' ? doc?.corpus[hover.index] : undefined
  const alike = doc?.frames[space]?.alike
  const quality = alikeLabel(alike)
  const cols = doc?.atlas.cols ?? 1

  return (
    <div
      className="absolute inset-0 overflow-hidden rounded-[10px] border border-line"
      style={{
        background:
          'radial-gradient(120% 90% at 18% -10%, #3a2a24 0%, #1d1614 28%, #100d0d 58%, #08080b 100%)',
      }}
    >
      <div ref={hostRef} className="absolute inset-0" />
      <SelectTool
        host={hostRef}
        scene={sceneRef}
        tool={tool}
        radius={circleR}
        setRadius={setCircleR}
        onCommit={commitSelection}
        onPreview={(idx, mode) => setPreview(idx ? { idx, mode } : null)}
      />

      {/* status + regions, left */}
      <div className="pointer-events-none absolute bottom-14 left-4 top-3 flex w-[248px] flex-col gap-2 text-[12px] leading-relaxed text-[#e9dccb]/80">
        <div>
          <div className="text-[11px] uppercase tracking-[2px] text-[#e9dccb]/55">
            latent map · {backend}
            {isFetching || doc?.rebuilding ? <span className="ml-2 normal-case tracking-normal text-[#f3c27a]">placing images…</span> : null}
          </div>
          {doc ? (
            <>
              <div>{shown.length} images{doc.regions.length ? ` · ${doc.regions.length} regions` : ''}</div>
              <div
                className={quality.good ? '' : 'text-[#f3c27a]'}
                title="Share of each image's map neighbours that really are similar images, by CLIP. 1 = all of them, 0.5 = random."
              >
                Nearby images {quality.word} look alike
                {alike != null ? ` (${alike.toFixed(2)}; random is 0.50)` : ''}
              </div>
              {doc.unembedded > 0 ? (
                <div className="text-[#e9dccb]/60">
                  {doc.unembedded} new image{doc.unembedded > 1 ? 's' : ''} waiting for an embedding
                </div>
              ) : null}
              {doc.warning ? <div className="text-[#f3c27a]">{doc.warning}</div> : null}
            </>
          ) : null}
        </div>

        {mapUnis.length ? (
          <div className="pointer-events-auto flex max-h-[42%] shrink-0 flex-col gap-1 overflow-y-auto pr-1">
            <div className="flex items-center text-[10px] uppercase tracking-[2px] text-[#e9dccb]/55">
              universes
              <span className="ml-auto normal-case tracking-normal text-[#e9dccb]/40">1–9 show · 0 none</span>
            </div>
            {mapUnis.map((u) => {
              const on = shownUnis.has(u.tag.toLocaleLowerCase())
              return (
                <div
                  key={u.tag}
                  className={cn(
                    'flex items-center gap-1.5 rounded-[7px] border bg-black/40 py-[3px] pl-1.5 pr-1 transition-colors',
                    on ? 'border-transparent' : 'border-white/10 hover:border-white/30',
                  )}
                  style={on ? { boxShadow: `inset 0 0 0 1px ${u.color}` } : undefined}
                >
                  <button
                    type="button"
                    className="flex min-w-0 flex-1 items-center gap-1.5 text-left"
                    onClick={() => toggleUniverse(u.tag)}
                    title={`show “${u.tag}” on the map: its images in its colour, the rest faded${u.key < 9 ? ` (${u.key + 1})` : ''}`}
                  >
                    <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: u.color }} />
                    {u.key < 9 ? <span className="text-[10px] tabular-nums text-[#e9dccb]/45">{u.key + 1}</span> : null}
                    <span className={cn('truncate', on ? 'text-[#fff1dc]' : '')}>{u.tag}</span>
                    <span
                      className="ml-auto shrink-0 text-[10.5px] tabular-nums text-[#e9dccb]/60"
                      title={`${u.here.length} on this map · ${u.count} overall`}
                    >
                      {u.here.length}
                      {u.here.length !== u.count ? <span className="opacity-60">/{u.count}</span> : null}
                    </span>
                  </button>
                  <button
                    type="button"
                    disabled={!u.here.length}
                    onClick={(e) => {
                      if (e.shiftKey) addMapSel(u.here)
                      else setMapSel(u.here)
                      flyToRels(u.here)
                    }}
                    title="select its images on the map (⇧ adds to the selection) — then sample more like them"
                    className="shrink-0 rounded-[5px] border border-white/10 px-1 text-[10.5px] text-[#6fffd0]/85 hover:border-[#6fffd0]/50 disabled:opacity-30"
                  >
                    select
                  </button>
                </div>
              )
            })}
          </div>
        ) : null}

        {doc?.regions.length ? (
          <div className="pointer-events-auto min-h-0 flex-1 overflow-y-auto pr-1">
            <div className="mb-1 text-[10px] uppercase tracking-[2px] text-[#e9dccb]/55" title="clusters of the look layout, found automatically">
              regions
            </div>
            <div className="flex flex-col gap-1">
              {doc.regions.map((r) => (
                <button
                  key={r.id}
                  type="button"
                  onClick={() => chooseRegion(r.id)}
                  title="fly here and fade the rest"
                  className={cn(
                    'flex items-center gap-1 rounded-[7px] border bg-black/40 p-1 text-left transition-colors',
                    region === r.id ? 'border-[#f3c27a]/80' : 'border-white/10 hover:border-white/30',
                  )}
                >
                  <span className="h-8 w-1 shrink-0 rounded-full" style={{ background: regionColor(r.id) }} />
                  {r.faces.slice(0, 4).map((ni) => (
                    <Thumb key={ni} cell={nodes[ni]?.a ?? -1} cols={cols} url={atlasUrl} size={38} />
                  ))}
                  <span className="ml-auto pr-1 text-[10.5px] text-[#e9dccb]/70">{r.count}</span>
                </button>
              ))}
            </div>
          </div>
        ) : null}
      </div>

      {/* selection + sampling, right */}
      <div className="absolute right-3 top-3 flex max-h-[calc(100%-150px)] w-[272px] flex-col gap-2 overflow-y-auto">
      <BatchNotice tone="map" className="rounded-[10px] border border-white/10 bg-black/60 px-2 py-1.5 backdrop-blur-md empty:hidden" />
      {selHere.length || (preview && selIdx.length) ? (
        <SelectionPanel
          rels={preview ? selIdx.map((i) => nodes[shown[i]]?.rel).filter((r): r is string => !!r) : selHere}
          cellOf={cellOf}
          byRel={byRel}
          atlasUrl={atlasUrl}
          cols={cols}
          onSampled={() => setFlash('queued — the new images grow on the map as they land')}
          onFly={() => flyToRels(selHere)}
        />
      ) : null}
      {picked || regionRow ? (
        <div className="relative shrink-0 rounded-[12px] border border-white/10 bg-black/60 p-2.5 text-[12px] text-[#efe6da] backdrop-blur-md">
          <button
            type="button"
            className="absolute right-2.5 top-1.5 text-[#e9dccb]/50 hover:text-[#e9dccb]"
            onClick={() => {
              if (region != null) sceneRef.current?.resetView()
              setPick(null)
              setRegion(null)
            }}
            title="clear"
          >
            ✕
          </button>
          {picked ? (
            <>
              <div className="mb-1.5 text-[11px] text-[#e9dccb]/70">
                {pick?.kind === 'spot' ? 'Empty spot. Samples start from the nearest image:' : 'Selected image'}
              </div>
              <button type="button" className="block w-full" onClick={() => openLightbox(picked.rel, 'map')} title="open full size">
                <img
                  src={`/img?path=${encodeURIComponent(picked.rel)}&t=${Math.floor(picked.t)}`}
                  alt=""
                  className="mb-1.5 block aspect-square w-full rounded-[7px] object-cover"
                />
              </button>
              <div className="mb-2 truncate font-mono text-[10.5px] opacity-70">{picked.name}</div>
            </>
          ) : regionRow ? (
            <div className="mb-2 text-[11px] text-[#e9dccb]/70">
              Region · {regionRow.count} images. Samples from its {Math.min(4, regionRow.faces.length)} most typical.
            </div>
          ) : null}

          {/* Turn the pick into a selection of REAL images: the region the
              universe sampler (selection panel) draws from. */}
          {picked && pick?.kind === 'node' ? (
            <button
              type="button"
              onClick={() => toggleMapSel(picked.rel)}
              className="mb-2 w-full rounded-[8px] border border-[#6fffd0]/35 px-3 py-1 text-[11.5px] text-[#6fffd0] hover:border-[#6fffd0]/70"
              title="⇧-click on the map does the same"
            >
              {mapSel.includes(picked.rel) ? '− drop from selection' : '＋ add to selection'}
            </button>
          ) : picked && pick?.kind === 'spot' ? (
            <div className="mb-2 flex items-center gap-1 text-[11px] text-[#e9dccb]/70" title="select the images nearest this spot ON THE MAP, then sample from them as a group">
              select nearest
              {SPOT_K.map((k) => (
                <Pill key={k} onClick={() => selectAround(pick.at, k)} className="!px-2 !py-[2px]">
                  {k}
                </Pill>
              ))}
            </div>
          ) : regionRow ? (
            <button
              type="button"
              onClick={() => {
                const rels = nodes.filter((n) => n.r === regionRow.id && sceneOf.has(n.rel)).map((n) => n.rel)
                setMapSel(rels)
              }}
              className="mb-2 w-full rounded-[8px] border border-[#6fffd0]/35 px-3 py-1 text-[11.5px] text-[#6fffd0] hover:border-[#6fffd0]/70"
            >
              ＋ select all {regionRow.count} → sample as a group
            </button>
          ) : null}

          <div className="mb-1 text-[10px] uppercase tracking-[2px] text-[#e9dccb]/50">how far</div>
          <div className="mb-2 grid grid-cols-2 gap-1">
            {REACH.map(([v, label]) => (
              <Pill key={v} on={reach === v} onClick={() => setMap({ reach: v })} className="justify-center">
                {label}
              </Pill>
            ))}
          </div>
          <label className="mb-2 flex items-center justify-between text-[11px] text-[#e9dccb]/70">
            images
            <input
              type="number" min={1} max={24} value={count}
              onChange={(e) => setMap({ count: e.target.value })}
              className="w-14 rounded-[6px] border border-white/15 bg-black/40 px-1.5 py-0.5 text-right text-[#efe6da]"
            />
          </label>
          <button
            type="button"
            disabled={explore.isPending}
            onClick={() =>
              sampleFrom(picked ? [picked.rel] : (regionRow?.faces ?? []).slice(0, 4).map((ni) => nodes[ni].rel))
            }
            className="w-full rounded-[8px] bg-[#f3c27a] px-3 py-1.5 text-[12.5px] font-medium text-[#1b130c] hover:bg-[#ffd594] disabled:opacity-50"
          >
            {pick?.kind === 'spot' || regionRow ? `Sample ${count} here` : `Sample ${count} around this image`}
          </button>
          {picked ? (
            <button
              type="button"
              onClick={() => { addKeyframe(picked.rel); setFlash(`added ${picked.name} to the 🎬 timeline`) }}
              className="mt-1 w-full rounded-[8px] border border-white/12 px-3 py-1 text-[11.5px] text-[#e9dccb]/80 hover:border-white/30"
            >
              Add to 🎬 timeline
            </button>
          ) : null}
        </div>
      ) : null}
      </div>

      {/* selection tools, top centre */}
      <div className="absolute left-1/2 top-3 flex -translate-x-1/2 items-center gap-1 rounded-[10px] border border-white/10 bg-black/50 p-1 backdrop-blur-md">
        {TOOLS.map((t) => (
          <Pill key={t.id} on={tool === t.id} onClick={() => setTool(t.id)} title={t.title} className="!py-[3px]">
            {t.label}
          </Pill>
        ))}
        {tool === 'circle' ? (
          <span className="px-1 text-[10.5px] text-[#e9dccb]/55">wheel / [ ] resize</span>
        ) : tool === 'lasso' ? (
          <span className="px-1 text-[10.5px] text-[#e9dccb]/55">⇧ add · ctrl remove</span>
        ) : null}
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
        <div className="pointer-events-none absolute left-1/2 top-14 -translate-x-1/2 rounded-full border border-white/15 bg-black/70 px-3 py-1 text-[12px] text-[#efe6da]">
          {flash}
        </div>
      ) : null}

      <div className="pointer-events-none absolute bottom-3 left-4 text-[11px] text-[#e9dccb]/45">
        click: open · ⇧-click: select · ⇧-drag: lasso · empty space: sample there · 1–9: universes · Esc: clear
      </div>

      <MapControls
        doc={doc}
        pathLen={timeline.filter((r) => nodes.some((n) => n.rel === r)).length}
        onResetView={() => sceneRef.current?.resetView()}
      />
    </div>
  )
}
