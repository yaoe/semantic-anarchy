/**
 * The 🌌 map's three.js scene, kept imperative and outside React: React hands
 * it typed arrays via `update()`, it hands back hover/pick events. Nothing in
 * here knows about the API or the store.
 *
 * Layers, back to front: a starfield, the faint corpus "dust", the explored
 * territory as a metaball hull (marching cubes, drawn three times over the same
 * geometry: faceted glass, wireframe, vertex dots), lineage edges, the timeline
 * path, and the image nodes as additive glow sprites. Node positions tween when
 * the axes change; the hull is rebuilt at the destination and fades in.
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'
import { MarchingCubes } from 'three/addons/objects/MarchingCubes.js'

export type EdgeKind = 'parent' | 'breed' | 'walk'

export interface SceneData {
  /** n*3 node positions (world units = z-scores) */
  positions: Float32Array
  /** n*3 linear RGB */
  colors: Float32Array
  /** n, world-unit diameter */
  sizes: Float32Array
  edges: { a: number; b: number; kind: EdgeKind }[]
  /** m*3 corpus dust positions, or null when hidden */
  corpus: Float32Array | null
  /** node indexes of the 🎬 timeline, in order */
  path: number[]
  /** [neg, pos] end labels for the x, y, z axes */
  axisLabels: [string, string][]
  showBlob: boolean
  showEdges: boolean
  /** re-frame the camera on the data (frame / axis change) */
  refit: boolean
}

export type Hit =
  | { kind: 'node'; index: number; x: number; y: number }
  | { kind: 'corpus'; index: number; x: number; y: number }

interface Callbacks {
  hover: (hit: Hit | null) => void
  pick: (index: number, e: { shiftKey: boolean }) => void
}

const EDGE_COLOR: Record<EdgeKind, THREE.Color> = {
  parent: new THREE.Color('#f3c27a'),
  walk: new THREE.Color('#ffe2a8'),
  breed: new THREE.Color('#e07ab8'),
}
const TWEEN_MS = 750
const BLOB_RES = 52
const ISO = 80
const SUBTRACT = 12

const NODE_VERT = /* glsl */ `
  attribute float size;
  attribute float highlight;
  attribute vec3 color;
  uniform float uProj;
  varying vec3 vColor;
  varying float vHi;
  void main() {
    vColor = color;
    vHi = highlight;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = max(4.0, size * (1.0 + 0.6 * highlight) * uProj / -mv.z);
  }
`
// A bright core, a thin ring, and a soft halo -- the "lit bead" look.
const NODE_FRAG = /* glsl */ `
  varying vec3 vColor;
  varying float vHi;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    if (d > 0.5) discard;
    float core = smoothstep(0.17, 0.10, d);
    float ring = smoothstep(0.035, 0.0, abs(d - 0.30)) * (0.55 + 0.45 * vHi);
    float halo = exp(-d * d * 26.0) * 0.55;
    vec3 c = mix(vColor, vec3(1.0), core * 0.55 + vHi * 0.25);
    float a = clamp(core + ring + halo, 0.0, 1.0);
    gl_FragColor = vec4(c * a, a);
  }
`

function textSprite(text: string, color = '#f1e6d6'): THREE.Sprite {
  const pad = 10
  const font = '500 26px ui-sans-serif, system-ui, sans-serif'
  const c = document.createElement('canvas')
  const ctx = c.getContext('2d')!
  ctx.font = font
  const w = Math.ceil(ctx.measureText(text).width) + pad * 2
  c.width = w
  c.height = 40
  ctx.font = font
  ctx.fillStyle = color
  ctx.textBaseline = 'middle'
  ctx.fillText(text, pad, 20)
  const tex = new THREE.CanvasTexture(c)
  tex.colorSpace = THREE.SRGBColorSpace
  const s = new THREE.Sprite(
    new THREE.SpriteMaterial({
      map: tex, transparent: true, depthWrite: false, opacity: 0.8, sizeAttenuation: false,
    }),
  )
  // constant on-screen size (fraction of viewport height), however close the camera
  s.scale.set((w / 40) * 0.028, 0.028, 1)
  return s
}

function concat(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(a.length + b.length)
  out.set(a)
  out.set(b, a.length)
  return out
}

/** Per-axis [q_lo, q_hi] quantile box of xyz triples, widened by `grow`. */
function coreBox(p: Float32Array, qlo: number, qhi: number, grow: number) {
  const n = p.length / 3
  const lo: [number, number, number] = [0, 0, 0]
  const hi: [number, number, number] = [0, 0, 0]
  for (let k = 0; k < 3; k++) {
    const v = Array.from({ length: n }, (_, i) => p[i * 3 + k]).sort((a, b) => a - b)
    const a = v[Math.floor(qlo * (n - 1))]
    const b = v[Math.ceil(qhi * (n - 1))]
    const mid = (a + b) / 2
    const h = Math.max((b - a) / 2, 0.5) * grow
    lo[k] = mid - h
    hi[k] = mid + h
  }
  return { lo, hi }
}

/** Median nearest-neighbour distance over (a sample of) the given nodes. */
function medianNN(p: Float32Array, idx: number[]): number {
  const s = idx.length > 1200 ? idx.filter((_, i) => i % Math.ceil(idx.length / 1200) === 0) : idx
  if (s.length < 2) return 0.3
  const nn = s.map((i) => {
    let best = Infinity
    for (const j of s) {
      if (j === i) continue
      const dx = p[i * 3] - p[j * 3]
      const dy = p[i * 3 + 1] - p[j * 3 + 1]
      const dz = p[i * 3 + 2] - p[j * 3 + 2]
      best = Math.min(best, dx * dx + dy * dy + dz * dz)
    }
    return Math.sqrt(best)
  })
  nn.sort((a, b) => a - b)
  return nn[Math.floor(nn.length / 2)]
}

const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2)

export class LatentScene {
  private renderer: THREE.WebGLRenderer
  private scene = new THREE.Scene()
  private camera = new THREE.PerspectiveCamera(45, 1, 0.05, 2000)
  private controls: OrbitControls
  private raycaster = new THREE.Raycaster()
  private frame = 0
  private ro: ResizeObserver

  private nodes: THREE.Points
  private nodeMat: THREE.ShaderMaterial
  private edges: THREE.LineSegments
  private path: THREE.Line
  private dust: THREE.Points
  private blob: MarchingCubes
  private blobMats: (THREE.Material & { opacity: number })[] = []
  private blobBase: number[] = []
  private axes = new THREE.Group()

  private from = new Float32Array(0)
  private to = new Float32Array(0)
  private cur = new Float32Array(0)
  private t0 = 0
  private data: SceneData | null = null
  private down: { x: number; y: number } | null = null
  private hoverQueued: PointerEvent | null = null

  private host: HTMLElement
  private cb: Callbacks

  constructor(host: HTMLElement, cb: Callbacks) {
    this.host = host
    this.cb = cb
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.renderer.setClearColor(0x000000, 0)
    host.appendChild(this.renderer.domElement)
    this.renderer.domElement.style.display = 'block'

    this.camera.position.set(6, 4, 9)
    this.controls = new OrbitControls(this.camera, this.renderer.domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.08
    this.controls.zoomToCursor = true

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.55))
    const key = new THREE.DirectionalLight(0xffe6c8, 1.4)
    key.position.set(4, 8, 6)
    this.scene.add(key)
    const rim = new THREE.DirectionalLight(0x9fd6ff, 0.6)
    rim.position.set(-6, -2, -5)
    this.scene.add(rim)

    this.scene.add(this.stars())

    // corpus dust
    this.dust = new THREE.Points(
      new THREE.BufferGeometry(),
      new THREE.PointsMaterial({
        color: 0x9db4d8, size: 2.2, sizeAttenuation: false,
        transparent: true, opacity: 0.45, depthWrite: false,
      }),
    )
    this.scene.add(this.dust)

    // explored hull: one marching-cubes field, three materials over its geometry
    const glass = new THREE.MeshPhongMaterial({
      color: 0xcfe9b4, emissive: 0x16240f, specular: 0xffffff, shininess: 70,
      flatShading: true, transparent: true, opacity: 0.2,
      side: THREE.DoubleSide, depthWrite: false,
    })
    this.blob = new MarchingCubes(BLOB_RES, glass, false, false, 90000)
    this.blob.isolation = ISO
    this.blob.renderOrder = 1
    const wire = new THREE.MeshBasicMaterial({
      color: 0xe9f8d2, wireframe: true, transparent: true, opacity: 0.3, depthWrite: false,
    })
    const dots = new THREE.PointsMaterial({
      color: 0xf4ffe6, size: 1.6, sizeAttenuation: false,
      transparent: true, opacity: 0.55, depthWrite: false,
    })
    this.blob.add(new THREE.Mesh(this.blob.geometry, wire))
    this.blob.add(new THREE.Points(this.blob.geometry, dots))
    this.blobMats = [glass, wire, dots]
    this.blobBase = this.blobMats.map((m) => m.opacity)
    this.scene.add(this.blob)

    this.edges = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({
        vertexColors: true, transparent: true, opacity: 0.6,
        blending: THREE.AdditiveBlending, depthWrite: false,
      }),
    )
    this.edges.renderOrder = 2
    this.scene.add(this.edges)

    this.path = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineDashedMaterial({
        color: 0x6fd3ff, dashSize: 0.18, gapSize: 0.1, transparent: true, opacity: 0.9,
      }),
    )
    this.path.renderOrder = 3
    this.scene.add(this.path)

    this.nodeMat = new THREE.ShaderMaterial({
      vertexShader: NODE_VERT,
      fragmentShader: NODE_FRAG,
      uniforms: { uProj: { value: 500 } },
      transparent: true,
      depthWrite: false,
      blending: THREE.CustomBlending,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneMinusSrcAlphaFactor,
    })
    this.nodes = new THREE.Points(new THREE.BufferGeometry(), this.nodeMat)
    this.nodes.renderOrder = 4
    this.scene.add(this.nodes)
    this.scene.add(this.axes)

    const el = this.renderer.domElement
    el.addEventListener('pointermove', this.onMove)
    el.addEventListener('pointerdown', this.onDown)
    el.addEventListener('pointerup', this.onUp)
    el.addEventListener('pointerleave', this.onLeave)

    this.ro = new ResizeObserver(() => this.resize())
    this.ro.observe(host)
    this.resize()
    this.loop()
  }

  // ------------------------------------------------------------- public ---
  update(d: SceneData) {
    const n = d.positions.length / 3
    const sameCount = this.cur.length === d.positions.length
    this.from = sameCount ? this.cur.slice() : d.positions.slice()
    this.to = d.positions.slice()
    this.cur = this.from.slice()
    this.t0 = performance.now()
    this.data = d

    const g = this.nodes.geometry
    // `cur` is a fresh array each update, so the attribute must be rebound.
    g.setAttribute('position', new THREE.BufferAttribute(this.cur, 3))
    if (!sameCount || !g.getAttribute('highlight'))
      g.setAttribute('highlight', new THREE.BufferAttribute(new Float32Array(n), 1))
    g.setAttribute('color', new THREE.BufferAttribute(d.colors, 3))
    g.setAttribute('size', new THREE.BufferAttribute(d.sizes, 1))

    // edges: two vertices per edge, coloured by kind
    const eg = this.edges.geometry
    const ep = new Float32Array(d.edges.length * 6)
    const ec = new Float32Array(d.edges.length * 6)
    d.edges.forEach((e, i) => {
      const c = EDGE_COLOR[e.kind]
      ec.set([c.r * 0.5, c.g * 0.5, c.b * 0.5, c.r, c.g, c.b], i * 6)
    })
    eg.setAttribute('position', new THREE.BufferAttribute(ep, 3))
    eg.setAttribute('color', new THREE.BufferAttribute(ec, 3))
    this.edges.visible = d.showEdges

    this.path.geometry.setAttribute(
      'position', new THREE.BufferAttribute(new Float32Array(d.path.length * 3), 3))
    this.path.visible = d.path.length > 1

    if (d.corpus) {
      this.dust.geometry.setAttribute('position', new THREE.BufferAttribute(d.corpus, 3))
      this.dust.geometry.computeBoundingSphere()
    }
    this.dust.visible = !!d.corpus

    this.buildBlob(d)
    this.buildAxes(d)
    this.writePositions()
    if (d.refit) this.fit(this.to, d.corpus)
  }

  setHighlight(index: number | null) {
    const a = this.nodes.geometry.getAttribute('highlight') as THREE.BufferAttribute | undefined
    if (!a) return
    ;(a.array as Float32Array).fill(0)
    if (index != null && index < a.count) (a.array as Float32Array)[index] = 1
    a.needsUpdate = true
  }

  dispose() {
    cancelAnimationFrame(this.frame)
    this.ro.disconnect()
    const el = this.renderer.domElement
    el.removeEventListener('pointermove', this.onMove)
    el.removeEventListener('pointerdown', this.onDown)
    el.removeEventListener('pointerup', this.onUp)
    el.removeEventListener('pointerleave', this.onLeave)
    this.controls.dispose()
    this.scene.traverse((o) => {
      const m = o as THREE.Mesh
      m.geometry?.dispose()
      const mats = Array.isArray(m.material) ? m.material : m.material ? [m.material] : []
      mats.forEach((mat) => {
        ;(mat as THREE.SpriteMaterial).map?.dispose()
        mat.dispose()
      })
    })
    this.renderer.dispose()
    el.remove()
  }

  // ------------------------------------------------------------ builders ---
  private stars(): THREE.Points {
    const n = 1800
    const p = new Float32Array(n * 3)
    const c = new Float32Array(n * 3)
    for (let i = 0; i < n; i++) {
      const u = Math.random() * 2 - 1
      const th = Math.random() * Math.PI * 2
      const r = 220 + Math.random() * 380
      const s = Math.sqrt(1 - u * u)
      p.set([r * s * Math.cos(th), r * u, r * s * Math.sin(th)], i * 3)
      const w = 0.55 + Math.random() * 0.45
      c.set([w, w * (0.82 + Math.random() * 0.12), w * (0.66 + Math.random() * 0.2)], i * 3)
    }
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.BufferAttribute(p, 3))
    g.setAttribute('color', new THREE.BufferAttribute(c, 3))
    return new THREE.Points(
      g,
      new THREE.PointsMaterial({
        size: 1.5, sizeAttenuation: false, vertexColors: true,
        transparent: true, opacity: 0.8, depthWrite: false,
      }),
    )
  }

  /**
   * Metaballs around the nodes at their DESTINATION; fades in with the tween.
   * One marching-cubes grid can't resolve a dense core and a far expedition at
   * once, so the hull covers the robust core (10-90th percentile box, widened)
   * and outliers stay as bare nodes + paths. The ball radius follows the
   * typical nearest-neighbour gap, so neighbours fuse into tubes and lobes
   * instead of reading as dust or one lump.
   */
  private buildBlob(d: SceneData) {
    const n = d.positions.length / 3
    this.blob.visible = d.showBlob && n > 0
    if (!this.blob.visible) return
    const p = d.positions
    const box = coreBox(p, 0.1, 0.9, 1.6)
    const inside: number[] = []
    for (let i = 0; i < n; i++)
      if ([0, 1, 2].every((k) => p[i * 3 + k] >= box.lo[k] && p[i * 3 + k] <= box.hi[k])) inside.push(i)
    const lo = [Infinity, Infinity, Infinity]
    const hi = [-Infinity, -Infinity, -Infinity]
    for (const i of inside)
      for (let k = 0; k < 3; k++) {
        lo[k] = Math.min(lo[k], p[i * 3 + k])
        hi[k] = Math.max(hi[k], p[i * 3 + k])
      }
    const extent = Math.max(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2], 1e-3)
    const radius = Math.min(extent * 0.12, Math.max(extent * 0.035, medianNN(p, inside) * 1.35))
    const half = extent / 2 + radius * 2.2 // cube half-size with margin
    const centre = lo.map((v, k) => (v + hi[k]) / 2)
    this.blob.position.set(centre[0], centre[1], centre[2])
    this.blob.scale.setScalar(half)
    const rn = radius / (2 * half)
    const strength = rn * rn * (ISO + SUBTRACT)
    this.blob.reset()
    for (const i of inside) {
      this.blob.addBall(
        0.5 + (p[i * 3] - centre[0]) / (2 * half),
        0.5 + (p[i * 3 + 1] - centre[1]) / (2 * half),
        0.5 + (p[i * 3 + 2] - centre[2]) / (2 * half),
        strength, SUBTRACT,
      )
    }
    this.blob.update()
  }

  private buildAxes(d: SceneData) {
    this.axes.traverse((o) => {
      const s = o as THREE.Sprite
      if (s.isSprite) {
        s.material.map?.dispose()
        s.material.dispose()
      }
    })
    this.axes.clear()
    const L = 3.2
    const dirs = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)]
    const tint = ['#f0a58a', '#b9e59c', '#9cc8f0']
    dirs.forEach((v, k) => {
      const g = new THREE.BufferGeometry().setFromPoints([v.clone().multiplyScalar(-L), v.clone().multiplyScalar(L)])
      this.axes.add(new THREE.Line(g, new THREE.LineBasicMaterial({
        color: tint[k], transparent: true, opacity: 0.22, depthWrite: false,
      })))
      const [neg, pos] = d.axisLabels[k] ?? ['', '']
      if (pos) {
        const s = textSprite(pos, tint[k])
        s.position.copy(v.clone().multiplyScalar(L + 0.25))
        s.center.set(0, 0.5)
        this.axes.add(s)
      }
      if (neg) {
        const s = textSprite(neg, tint[k])
        s.position.copy(v.clone().multiplyScalar(-L - 0.25))
        s.center.set(1, 0.5)
        this.axes.add(s)
      }
    })
  }

  /** Frame the robust core (nodes + corpus), not the furthest outlier. */
  private fit(p: Float32Array, corpus: Float32Array | null) {
    const all = corpus ? concat(p, corpus) : p
    const b = all.length ? coreBox(all, 0.05, 0.95, 1.15) : { lo: [-3, -3, -3], hi: [3, 3, 3] }
    const box = new THREE.Box3(new THREE.Vector3(...b.lo), new THREE.Vector3(...b.hi))
    const sphere = box.getBoundingSphere(new THREE.Sphere())
    const r = Math.max(sphere.radius, 2)
    const dir = this.camera.position.clone().sub(this.controls.target).normalize()
    const dist = r / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 0.95
    this.controls.target.copy(sphere.center)
    this.camera.position.copy(sphere.center).addScaledVector(dir, dist)
    this.camera.near = dist / 200
    this.camera.far = Math.max(2000, dist * 20)
    this.camera.updateProjectionMatrix()
  }

  // ---------------------------------------------------------------- loop ---
  private writePositions() {
    const d = this.data
    if (!d) return
    const pos = this.nodes.geometry.getAttribute('position') as THREE.BufferAttribute
    pos.needsUpdate = true
    this.nodes.geometry.computeBoundingSphere()
    const ep = this.edges.geometry.getAttribute('position') as THREE.BufferAttribute
    const ea = ep.array as Float32Array
    d.edges.forEach((e, i) => {
      ea.set(this.cur.subarray(e.a * 3, e.a * 3 + 3), i * 6)
      ea.set(this.cur.subarray(e.b * 3, e.b * 3 + 3), i * 6 + 3)
    })
    ep.needsUpdate = true
    const pp = this.path.geometry.getAttribute('position') as THREE.BufferAttribute
    const pa = pp.array as Float32Array
    d.path.forEach((ni, i) => pa.set(this.cur.subarray(ni * 3, ni * 3 + 3), i * 3))
    pp.needsUpdate = true
    if (d.path.length > 1) this.path.computeLineDistances()
  }

  private loop = () => {
    this.frame = requestAnimationFrame(this.loop)
    const t = Math.min(1, (performance.now() - this.t0) / TWEEN_MS)
    if (this.data && (t < 1 || this.lastT < 1)) {
      const k = ease(t)
      for (let i = 0; i < this.cur.length; i++)
        this.cur[i] = this.from[i] + (this.to[i] - this.from[i]) * k
      this.writePositions()
      this.blobMats.forEach((m, i) => (m.opacity = this.blobBase[i] * k))
    }
    this.lastT = t
    if (this.hoverQueued) {
      this.hover(this.hoverQueued)
      this.hoverQueued = null
    }
    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }
  private lastT = 1

  private resize() {
    const w = Math.max(1, this.host.clientWidth)
    const h = Math.max(1, this.host.clientHeight)
    this.renderer.setSize(w, h)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
    // world-size -> pixels at unit depth, for the node sprites
    this.nodeMat.uniforms.uProj.value =
      (h * this.renderer.getPixelRatio()) / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov / 2)))
  }

  // ------------------------------------------------------------- picking ---
  private ray(e: PointerEvent) {
    const r = this.renderer.domElement.getBoundingClientRect()
    const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
    this.raycaster.setFromCamera(ndc, this.camera)
    // threshold in world units, scaled with zoom so picking feels the same
    const dist = this.camera.position.distanceTo(this.controls.target)
    this.raycaster.params.Points = { threshold: Math.max(0.04, dist * 0.012) }
    return { x: e.clientX - r.left, y: e.clientY - r.top }
  }

  private nearest(obj: THREE.Points): number | null {
    if (!obj.visible || !obj.geometry.getAttribute('position')) return null
    const hits = this.raycaster.intersectObject(obj)
    if (!hits.length) return null
    let best = hits[0]
    for (const h of hits) if ((h.distanceToRay ?? 0) < (best.distanceToRay ?? 0)) best = h
    return best.index ?? null
  }

  private hover(e: PointerEvent) {
    const xy = this.ray(e)
    const i = this.nearest(this.nodes)
    if (i != null) return this.cb.hover({ kind: 'node', index: i, ...xy })
    const j = this.nearest(this.dust)
    if (j != null) return this.cb.hover({ kind: 'corpus', index: j, ...xy })
    this.cb.hover(null)
  }

  private onMove = (e: PointerEvent) => {
    this.hoverQueued = e
  }
  private onLeave = () => this.cb.hover(null)
  private onDown = (e: PointerEvent) => {
    this.down = { x: e.clientX, y: e.clientY }
  }
  private onUp = (e: PointerEvent) => {
    const d = this.down
    this.down = null
    if (!d || e.button !== 0 || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) return
    this.ray(e)
    const i = this.nearest(this.nodes)
    if (i != null) this.cb.pick(i, { shiftKey: e.shiftKey })
  }
}
