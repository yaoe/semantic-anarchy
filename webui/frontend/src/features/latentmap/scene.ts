/**
 * The 🌌 map's three.js scene, kept imperative and outside React: React hands
 * it typed arrays via `update()`, it hands back hover/pick events. Nothing in
 * here knows about the API or the store.
 *
 * Layers, back to front: a faint starfield, the corpus "dust", lineage edges,
 * the timeline path, the sampling probe, the selection halos, and the images
 * themselves -- point sprites that draw either a thumbnail out of one atlas
 * texture or a lit bead. Node positions tween when the layout changes.
 *
 * Selection is React's: it hands over scene indexes (`setSelection`) and asks
 * where every node is on screen (`projectAll`) to lasso or circle-select. The
 * halos are an ILLUSTRATION of a sampling spread around the selected images,
 * not a region of conditioning space (a UMAP has no inverse).
 */
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js'

export type EdgeKind = 'parent' | 'breed' | 'walk'

export interface SceneData {
  /** n*3 node positions (world units = z-scores) */
  positions: Float32Array
  /** n*3 linear RGB */
  colors: Float32Array
  /** n, world-unit diameter */
  sizes: Float32Array
  /** n, atlas cell of each node (-1 = no thumbnail) */
  tiles: Float32Array
  /** n, 1 = in focus, lower = faded back (region isolation) */
  fades: Float32Array
  edges: { a: number; b: number; kind: EdgeKind }[]
  /** m*3 corpus dust positions, or null when hidden */
  corpus: Float32Array | null
  /** node indexes of the 🎬 timeline, in order */
  path: number[]
  /** [neg, pos] end labels for the x, y, z axes; empty = this space has no axes */
  axisLabels: [string, string][]
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
  /** a click that hit no image: the point on the ray nearest what is around it */
  pickEmpty: (p: [number, number, number], e: { shiftKey: boolean }) => void
  open: (index: number) => void
}

const EDGE_COLOR: Record<EdgeKind, THREE.Color> = {
  parent: new THREE.Color('#f3c27a'),
  walk: new THREE.Color('#ffe2a8'),
  breed: new THREE.Color('#e07ab8'),
}
const TWEEN_MS = 750
const NODE_VERT = /* glsl */ `
  attribute float size;
  attribute float highlight;
  attribute float sel;
  attribute float tile;
  attribute float fade;
  attribute vec3 color;
  uniform float uProj;
  uniform float uScale;
  uniform float uDepth;
  varying vec3 vColor;
  varying float vHi;
  varying float vSel;
  varying float vTile;
  varying float vFade;
  void main() {
    vColor = color;
    vHi = highlight;
    vSel = sel;
    vTile = tile;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    // Depth cue: what is behind the orbit centre recedes.
    vFade = max(fade, sel) * clamp(1.25 - 0.08 * max(-mv.z - uDepth, 0.0), 0.4, 1.0);
    gl_PointSize = max(3.0, size * uScale * (1.0 + 0.8 * highlight + 0.3 * sel) * uProj / -mv.z);
  }
`
// uThumb = 1: the image, rounded, with its colour as a border. 0: a lit bead.
// A selected node (vSel) wears a thick mint border / ring instead.
const SEL_RGB = 'vec3(0.435, 1.0, 0.816)'
const NODE_FRAG = /* glsl */ `
  uniform sampler2D uAtlas;
  uniform float uCols;
  uniform float uThumb;
  varying vec3 vColor;
  varying float vHi;
  varying float vSel;
  varying float vTile;
  varying float vFade;
  void main() {
    vec2 uv = gl_PointCoord;
    if (uThumb > 0.5 && vTile >= 0.0) {
      vec2 q = abs(uv - 0.5);
      float corner = length(max(q - 0.40, 0.0));
      if (corner > 0.10) discard;
      float border = max(q.x, q.y) > (0.44 - 0.06 * vHi - 0.04 * vSel) ? 1.0 : 0.0;
      vec2 cell = vec2(mod(vTile, uCols), floor(vTile / uCols));
      vec3 img = texture2D(uAtlas, (cell + uv) / uCols).rgb;
      vec3 edge = mix(mix(vColor, vec3(1.0), vHi), ${SEL_RGB}, vSel);
      vec3 c = mix(img, edge, border);
      gl_FragColor = vec4(c * vFade, 1.0);
      return;
    }
    float d = length(uv - 0.5);
    if (d > 0.5) discard;
    float core = smoothstep(0.17, 0.10, d);
    float ring = smoothstep(0.035 + 0.04 * vSel, 0.0, abs(d - 0.30 - 0.1 * vSel)) * (0.55 + 0.45 * max(vHi, vSel));
    float halo = exp(-d * d * 26.0) * 0.55;
    vec3 c = mix(vColor, vec3(1.0), core * 0.55 + vHi * 0.25);
    c = mix(c, ${SEL_RGB}, ring * vSel);
    float a = clamp(core + ring + halo, 0.0, 1.0) * vFade;
    gl_FragColor = vec4(c * a, a);
  }
`

// Selection halos: soft discs whose world size tracks the spread.
const HALO_VERT = /* glsl */ `
  uniform float uProj;
  uniform float uSize;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    gl_PointSize = min(2048.0, uSize * uProj / -mv.z);
  }
`
const HALO_FRAG = /* glsl */ `
  uniform float uAlpha;
  void main() {
    float d = length(gl_PointCoord - 0.5) * 2.0;
    if (d > 1.0) discard;
    // A flat disc with a soft edge: under MAX blending the union of many reads
    // as one region rather than a stack of rings.
    float a = (1.0 - smoothstep(0.55, 1.0, d)) * uAlpha;
    gl_FragColor = vec4(${SEL_RGB} * a, a);
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

function ringTexture(): THREE.CanvasTexture {
  const c = document.createElement('canvas')
  c.width = c.height = 64
  const ctx = c.getContext('2d')!
  ctx.strokeStyle = '#fff'
  ctx.lineWidth = 5
  ctx.beginPath()
  ctx.arc(32, 32, 24, 0, Math.PI * 2)
  ctx.moveTo(32, 2)
  ctx.lineTo(32, 16)
  ctx.moveTo(32, 48)
  ctx.lineTo(32, 62)
  ctx.moveTo(2, 32)
  ctx.lineTo(16, 32)
  ctx.moveTo(48, 32)
  ctx.lineTo(62, 32)
  ctx.stroke()
  return new THREE.CanvasTexture(c)
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
  private probe: THREE.Group
  private probeLines: THREE.LineSegments
  private probeMark: THREE.Sprite
  private probeAt: [number, number, number] | null = null
  private probeTo: number[] = []
  private halo: THREE.Points
  private haloMat: THREE.ShaderMaterial
  private sel: number[] = []
  private atlasUrl: string | null = null
  private wantThumbs = false
  private selected: number | null = null
  private hovered: number | null = null
  private fly: { t0: number; fromT: THREE.Vector3; toT: THREE.Vector3; fromP: THREE.Vector3; toP: THREE.Vector3 } | null = null
  private axes = new THREE.Group()

  private from = new Float32Array(0)
  private to = new Float32Array(0)
  private cur = new Float32Array(0)
  private t0 = 0
  private data: SceneData | null = null
  /** Where the button went down, and whether the pointer has since wandered (a drag, not a click). */
  private down: { x: number; y: number; moved: boolean } | null = null
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
      uniforms: {
        uProj: { value: 500 },
        uScale: { value: 1 },
        uDepth: { value: 10 },
        uAtlas: { value: null },
        uCols: { value: 1 },
        uThumb: { value: 0 },
      },
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

    this.probe = new THREE.Group()
    this.probeLines = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0x6fffd0, transparent: true, opacity: 0.7, depthWrite: false }),
    )
    this.probeMark = new THREE.Sprite(
      new THREE.SpriteMaterial({
        map: ringTexture(), color: 0x6fffd0, depthTest: false, transparent: true, sizeAttenuation: false,
      }),
    )
    this.probeMark.scale.set(0.045, 0.045, 1)
    this.probe.add(this.probeLines, this.probeMark)
    this.probe.visible = false
    this.probe.renderOrder = 5
    this.scene.add(this.probe)

    this.haloMat = new THREE.ShaderMaterial({
      vertexShader: HALO_VERT,
      fragmentShader: HALO_FRAG,
      uniforms: { uProj: { value: 500 }, uSize: { value: 1 }, uAlpha: { value: 0.2 } },
      transparent: true,
      depthWrite: false,
      depthTest: false,
      // MAX, not additive: overlapping halos read as their union (one soft
      // region) instead of piling up into a white blob over a big selection.
      blending: THREE.CustomBlending,
      blendEquation: THREE.MaxEquation,
    })
    this.halo = new THREE.Points(new THREE.BufferGeometry(), this.haloMat)
    this.halo.renderOrder = 3
    this.halo.visible = false
    this.halo.frustumCulled = false
    this.scene.add(this.halo)

    const el = this.renderer.domElement
    el.addEventListener('pointermove', this.onMove)
    el.addEventListener('pointerdown', this.onDown)
    el.addEventListener('pointerup', this.onUp)
    el.addEventListener('pointerleave', this.onLeave)
    el.addEventListener('dblclick', this.onDouble)

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
    if (!sameCount || !g.getAttribute('sel'))
      g.setAttribute('sel', new THREE.BufferAttribute(new Float32Array(n), 1))
    g.setAttribute('color', new THREE.BufferAttribute(d.colors, 3))
    g.setAttribute('size', new THREE.BufferAttribute(d.sizes, 1))
    g.setAttribute('tile', new THREE.BufferAttribute(d.tiles, 1))
    g.setAttribute('fade', new THREE.BufferAttribute(d.fades, 1))

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

    this.buildAxes(d)
    this.writePositions()
    this.paintHighlight()
    this.paintSelection()
    if (d.refit) this.fit(this.to)
  }

  setHover(index: number | null) {
    this.hovered = index
    this.paintHighlight()
  }

  setSelected(index: number | null) {
    this.selected = index
    this.paintHighlight()
  }

  /** Scene indexes drawn as selected (mint border + halo). */
  setSelection(idx: number[]) {
    this.sel = idx
    this.paintSelection()
  }

  /**
   * Halo around each selected image, `radius` world units. Illustrative only:
   * it shows how far the spread reaches relative to the selection's own extent
   * ON THIS MAP, which is not where new images will land.
   */
  setHalo(radius: number, visible: boolean) {
    this.haloMat.uniforms.uSize.value = Math.max(0, radius * 2)
    this.halo.visible = visible && radius > 0 && this.sel.length > 0
  }

  /** While a selection tool is dragging, left-drag must not also orbit. */
  setOrbitEnabled(on: boolean) {
    this.controls.enabled = on
  }

  /**
   * Every node's position on screen (CSS px from the canvas' top-left), x,y
   * pairs; NaN for nodes behind the camera. What lasso / circle select test.
   */
  projectAll(): Float32Array {
    const n = this.cur.length / 3
    const out = new Float32Array(n * 2)
    const w = this.host.clientWidth
    const h = this.host.clientHeight
    this.camera.updateMatrixWorld()
    const m = new THREE.Matrix4().multiplyMatrices(this.camera.projectionMatrix, this.camera.matrixWorldInverse)
    const e = m.elements
    for (let i = 0; i < n; i++) {
      const x = this.cur[i * 3]
      const y = this.cur[i * 3 + 1]
      const z = this.cur[i * 3 + 2]
      const cw = e[3] * x + e[7] * y + e[11] * z + e[15]
      if (cw <= 1e-6) {
        out[i * 2] = NaN
        out[i * 2 + 1] = NaN
        continue
      }
      const cx = (e[0] * x + e[4] * y + e[8] * z + e[12]) / cw
      const cy = (e[1] * x + e[5] * y + e[9] * z + e[13]) / cw
      out[i * 2] = (cx * 0.5 + 0.5) * w
      out[i * 2 + 1] = (-cy * 0.5 + 0.5) * h
    }
    return out
  }

  /** The sampling marker, tied by lines to the images it would draw from. */
  setProbe(at: [number, number, number] | null, to: number[]) {
    this.probeAt = at
    this.probeTo = to
    this.probe.visible = !!at
    this.writeProbe()
  }

  /**
   * Draw thumbnails from this atlas, or beads for null. Images only switch on
   * once the texture has loaded: sampling an empty one paints every node black,
   * so a slow or failed download shows beads instead.
   */
  setAtlas(url: string | null, cols: number) {
    this.wantThumbs = !!url
    this.nodeMat.uniforms.uCols.value = cols
    if (url && url !== this.atlasUrl) {
      this.atlasUrl = url
      new THREE.TextureLoader().load(
        url,
        (tex) => {
          if (url !== this.atlasUrl) return tex.dispose()
          tex.flipY = false
          tex.needsUpdate = true
          ;(this.nodeMat.uniforms.uAtlas.value as THREE.Texture | null)?.dispose()
          this.nodeMat.uniforms.uAtlas.value = tex
          this.applyThumbMode()
        },
        undefined,
        () => {
          if (url === this.atlasUrl) this.atlasUrl = null // the next update retries
        },
      )
    }
    this.applyThumbMode()
  }

  /** Opaque with a depth test for images; soft glow for beads. */
  private applyThumbMode() {
    const on = this.wantThumbs && !!this.nodeMat.uniforms.uAtlas.value
    const m = this.nodeMat
    if ((m.uniforms.uThumb.value === 1) === on) return
    m.uniforms.uThumb.value = on ? 1 : 0
    m.depthWrite = on
    m.blending = on ? THREE.NormalBlending : THREE.CustomBlending
    m.needsUpdate = true
  }

  setSpin(on: boolean) {
    this.controls.autoRotate = on
    this.controls.autoRotateSpeed = 0.8
  }

  /** Where scene node `i` is heading (its layout position, not mid-tween). */
  positionOf(i: number): [number, number, number] | null {
    if ((i + 1) * 3 > this.to.length) return null
    return [this.to[i * 3], this.to[i * 3 + 1], this.to[i * 3 + 2]]
  }

  /** Glide to frame a point, at a viewing distance for `radius` world units. */
  flyTo(p: [number, number, number], radius: number) {
    const toT = new THREE.Vector3(...p)
    const dir = this.camera.position.clone().sub(this.controls.target).normalize()
    const dist = Math.max(radius, 0.15) / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2))
    this.fly = {
      t0: performance.now(),
      fromT: this.controls.target.clone(),
      toT,
      fromP: this.camera.position.clone(),
      toP: toT.clone().addScaledVector(dir, dist),
    }
  }

  resetView() {
    if (this.to.length) this.fit(this.to)
  }

  setNodeScale(v: number) {
    this.nodeMat.uniforms.uScale.value = v
  }

  dispose() {
    cancelAnimationFrame(this.frame)
    this.ro.disconnect()
    const el = this.renderer.domElement
    el.removeEventListener('pointermove', this.onMove)
    el.removeEventListener('pointerdown', this.onDown)
    el.removeEventListener('pointerup', this.onUp)
    el.removeEventListener('pointerleave', this.onLeave)
    el.removeEventListener('dblclick', this.onDouble)
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
        transparent: true, opacity: 0.3, depthWrite: false,
      }),
    )
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
    if (!d.axisLabels.length) return
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

  /** Frame the robust core of the images, not the furthest outlier. */
  private fit(p: Float32Array) {
    const b = p.length ? coreBox(p, 0.03, 0.97, 1.1) : { lo: [-3, -3, -3], hi: [3, 3, 3] }
    const box = new THREE.Box3(new THREE.Vector3(...b.lo), new THREE.Vector3(...b.hi))
    const sphere = box.getBoundingSphere(new THREE.Sphere())
    const r = Math.max(sphere.radius, 0.5)
    const dir = this.camera.position.clone().sub(this.controls.target).normalize()
    const dist = r / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2)) * 0.95
    this.fly = null
    this.controls.target.copy(sphere.center)
    this.camera.position.copy(sphere.center).addScaledVector(dir, dist)
    this.camera.near = dist / 200
    this.camera.far = Math.max(2000, dist * 20)
    this.camera.updateProjectionMatrix()
  }

  private paintHighlight() {
    const a = this.nodes.geometry.getAttribute('highlight') as THREE.BufferAttribute | undefined
    if (!a) return
    const v = a.array as Float32Array
    v.fill(0)
    for (const i of [this.selected, this.hovered]) if (i != null && i < a.count) v[i] = 1
    a.needsUpdate = true
  }

  private paintSelection() {
    const a = this.nodes.geometry.getAttribute('sel') as THREE.BufferAttribute | undefined
    if (a) {
      const v = a.array as Float32Array
      v.fill(0)
      for (const i of this.sel) if (i < a.count) v[i] = 1
      a.needsUpdate = true
    }
    this.writeHalo()
  }

  private writeHalo() {
    const n = this.cur.length / 3
    const idx = this.sel.filter((i) => i < n)
    const p = new Float32Array(idx.length * 3)
    idx.forEach((ni, k) => p.set(this.cur.subarray(ni * 3, ni * 3 + 3), k * 3))
    this.halo.geometry.setAttribute('position', new THREE.BufferAttribute(p, 3))
    if (!idx.length) this.halo.visible = false
  }

  private writeProbe() {
    if (!this.probeAt) return
    const [x, y, z] = this.probeAt
    this.probeMark.position.set(x, y, z)
    const seg = new Float32Array(this.probeTo.length * 6)
    this.probeTo.forEach((ni, i) => {
      seg.set([x, y, z], i * 6)
      seg.set(this.cur.subarray(ni * 3, ni * 3 + 3), i * 6 + 3)
    })
    this.probeLines.geometry.setAttribute('position', new THREE.BufferAttribute(seg, 3))
    this.probeLines.geometry.computeBoundingSphere()
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
    this.writeProbe()
    if (this.sel.length) this.writeHalo()
  }

  private loop = () => {
    this.frame = requestAnimationFrame(this.loop)
    const t = Math.min(1, (performance.now() - this.t0) / TWEEN_MS)
    if (this.data && (t < 1 || this.lastT < 1)) {
      const k = ease(t)
      for (let i = 0; i < this.cur.length; i++)
        this.cur[i] = this.from[i] + (this.to[i] - this.from[i]) * k
      this.writePositions()
    }
    if (this.fly) {
      const f = ease(Math.min(1, (performance.now() - this.fly.t0) / TWEEN_MS))
      this.controls.target.lerpVectors(this.fly.fromT, this.fly.toT, f)
      this.camera.position.lerpVectors(this.fly.fromP, this.fly.toP, f)
      if (f >= 1) this.fly = null
    }
    this.nodeMat.uniforms.uDepth.value = this.camera.position.distanceTo(this.controls.target)
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
    this.haloMat.uniforms.uProj.value = this.nodeMat.uniforms.uProj.value
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
    // A drag that comes back to where it started (a closed lasso, an orbit
    // there and back) is still a drag: judge by the furthest it went.
    const d = this.down
    if (d && !d.moved && Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) d.moved = true
  }
  private onLeave = () => this.cb.hover(null)
  private onDown = (e: PointerEvent) => {
    this.down = { x: e.clientX, y: e.clientY, moved: false }
  }
  private onUp = (e: PointerEvent) => {
    const d = this.down
    this.down = null
    if (!d || e.button !== 0 || d.moved || Math.hypot(e.clientX - d.x, e.clientY - d.y) > 4) return
    this.ray(e)
    const i = this.nearest(this.nodes)
    if (i != null) this.cb.pick(i, { shiftKey: e.shiftKey })
    else this.cb.pickEmpty(this.emptyPoint(), { shiftKey: e.shiftKey })
  }

  private onDouble = (e: MouseEvent) => {
    this.ray(e as PointerEvent)
    const i = this.nearest(this.nodes)
    if (i != null) this.cb.open(i)
  }

  /**
   * A click in empty space has no depth of its own. Take the depth of the
   * images passing closest to the ray, so the point lands among what you were
   * looking at rather than on some arbitrary plane.
   */
  private emptyPoint(): [number, number, number] {
    const ray = this.raycaster.ray
    const n = this.cur.length / 3
    const v = new THREE.Vector3()
    let best = Infinity
    for (let i = 0; i < n; i++) {
      v.set(this.cur[i * 3], this.cur[i * 3 + 1], this.cur[i * 3 + 2])
      best = Math.min(best, ray.distanceSqToPoint(v))
    }
    let sum = 0
    let weight = 0
    const scale = Math.max(best, 1e-4) * 2
    for (let i = 0; i < n; i++) {
      v.set(this.cur[i * 3], this.cur[i * 3 + 1], this.cur[i * 3 + 2])
      const w = Math.exp(-ray.distanceSqToPoint(v) / scale)
      sum += w * v.sub(ray.origin).dot(ray.direction)
      weight += w
    }
    const depth = weight > 0 ? sum / weight : this.camera.position.distanceTo(this.controls.target)
    const p = ray.at(depth, new THREE.Vector3())
    return [p.x, p.y, p.z]
  }
}
