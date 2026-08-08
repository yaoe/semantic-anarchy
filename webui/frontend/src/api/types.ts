/**
 * Wire types for webui/app.py. Every field here mirrors a pydantic model or a
 * JSONResponse literal in that file — keep them in lock-step.
 */

export type ActionId = 'generate' | 'temp_sweep' | 'sampler_sweep' | 'mine'
export type BackendId = 'sd15' | 'sd2' | 'sdxl' | 'flux2' | 'krea2'
export type SamplerId = 'diagonal' | 'pca' | 'blend' | 'hybrid'
export type SchedulerId = 'default' | 'ddim' | 'euler' | 'euler_a' | 'dpm'
export type NegModeId = 'mean' | 'empty' | 'zeros'
export type JobStatus = 'queued' | 'running' | 'done' | 'error' | 'cancelled'

/** Runner.snapshot() job rows. */
export interface JobSummary {
  id: number
  action: string
  label: string
  status: JobStatus
  rc: number | null
  started: number | null
  ended: number | null
  lines: number
  cmd: string
}

export interface StateSnapshot {
  running: number | null
  jobs: JobSummary[]
}

/** One row of /api/images (any bucket). */
export interface ImageItem {
  name: string
  rel: string
  url: string
  mtime: number
  size: number
  fav: boolean
  score: number | null
  dist: number | null
  nov: number | null
  res: number | null
}

/** Server-side buckets of /api/images (GALLERY_BUCKETS + the derived three). */
export type GalleryKey =
  | 'generated'
  | 'frontier'
  | 'top'
  | 'favorites'
  | 'temperature'
  | 'sampler'
  | 'marginals'

/** Gallery tabs = server buckets + the client-side films/timeline views. */
export type TabKey = GalleryKey | 'films' | 'timeline'

export type Images = Record<GalleryKey, ImageItem[]>

export interface InitFolder {
  name: string
  path: string
  count: number
}

export interface Config {
  python: string
  sd15_ckpt: string
  sd15_ckpt_exists: boolean
  sd2_ckpt: string
  sd2_ckpt_exists: boolean
  sdxl_models: Record<string, string>
  /** backend -> hand-picked checkpoint path (webui/model_config.json). */
  picked_models: Partial<Record<BackendId, string>>
  init_dir: string
  init_count: number
  init_folders: InitFolder[]
  repo: string
}

/* --------------------------------------------------------- model picker --- */

/** What a checkpoint path turned out to be. `repo` = a cached HF id. */
export type ModelKind = 'ckpt' | 'diffusers' | 'repo' | null

/** One row of /api/model — mirrors webui.app `_model_row`. */
export interface ModelRow {
  backend: BackendId
  /** The hand-picked path, or null when the env-var default is in force. */
  selected: string | null
  default: string
  effective: string
  /** Basename for a local path, the full id for an HF repo. */
  name: string
  kind: ModelKind
  /** null for HF repo ids — resolved out of the HF cache, so unknowable here. */
  exists: boolean | null
}

export interface FsRoot {
  name: string
  path: string
}

export interface ModelConfig {
  backends: Record<BackendId, ModelRow>
  config_file: string
  /** Which OS dialog the *server host* can drive ('zenity'|…), null if none. */
  native_picker: string | null
  roots: FsRoot[]
}

export interface NativePickResult {
  cancelled: boolean
  path: string | null
}

export interface FsEntry {
  name: string
  path: string
  dir: boolean
  kind: ModelKind
  size: number | null
}

export interface FsListing {
  path: string
  /** null when already at a browsable root. */
  parent: string | null
  kind: ModelKind
  entries: FsEntry[]
  roots: FsRoot[]
}

export interface Film {
  name: string
  dir: string
  rel: string
  mtime: number
  size: number
  frames: number | null
  fps: number | null
  keyframes: string[]
  refine: string | null
  interp: string | null
  easing: string | null
  loop: boolean | null
  backend: string | null
  duration: number | null
}

export type InterpId = 'slerp' | 'lerp'
export type EasingId = 'smooth' | 'smoother' | 'linear'

/** One row of POST /api/keyframes — what a timeline entry actually is. */
export interface KeyframeRow {
  rel: string
  /** The image that owns the conditioning (upscales point at their original). */
  source: string | null
  backend: BackendId | null
  height: number | null
  width: number | null
  filmable: boolean
  error: string | null
}

/** POST /api/film — mirrors webui.app.FilmRequest exactly. */
export interface FilmRequest {
  images: string[]
  name?: string | null
  height?: number | null
  width?: number | null
  fps: number
  frames_per: number
  interp: InterpId
  easing: EasingId
  loop: boolean
  refine: 'none' | 'flux'
  scale?: number
  fixed_noise: boolean
  noise_window: number
  film_seed?: number
  steps?: number | null
  guidance?: number | null
}

export interface FilmResponse extends RunResponse {
  name: string
  frames: number
}

export interface TasteBand {
  count: number
  mean?: number
  p25?: number
  p75?: number
}

/** POST /api/run — mirrors webui.app.RunRequest exactly. */
export interface RunRequest {
  action: ActionId
  backend: BackendId
  model?: string | null
  sampler: SamplerId
  temperature?: number | null
  n?: number | null
  seed?: number | null
  steps?: number | null
  guidance?: number | null
  coherence?: number | null
  components?: number | null
  truncation?: number | null
  neg_mode?: NegModeId | null
  temps?: string | null
  seeds?: string | null
  scheduler?: SchedulerId | null
  width?: number | null
  height?: number | null
  comp_lo?: number | null
  equalize?: boolean
  dist?: string
  target_distance?: number | null
  min_distance?: number | null
  init?: boolean
  init_mode?: string
  init_strength?: number
  ip_scale?: number
  init_folder?: string | null
}

export interface RefineRequest {
  src: string
  /** Upscale factor. hires snaps the resulting size to a multiple of 16 px. */
  scale: number
  /** hires: unset = replay the source's own step count. */
  steps?: number | null
  /** hires: fraction of the ORIGINAL schedule to re-run on the enlarged image. */
  strength: number
  scheduler?: string | null
  tiled: boolean
  overlap?: number
  engine: 'hires' | 'flux' | 'sd'
  prompt?: string | null
  interp?: 'lanczos' | 'bicubic' | 'bilinear' | 'nearest'
}

export interface ExploreRequest {
  src: string
  mode: 'neighborhood' | 'breed' | 'walk'
  b?: string | null
  radius?: number
  mutate?: number
  direction?: 'outward' | 'random' | 'axis'
  step?: number
  axis?: number | null
  n?: number
  steps?: number | null
  guidance?: number | null
}

export interface EvolveRequest {
  backend?: BackendId | null
  n?: number
  temperature?: number
  base_blend?: number
}

export interface InvertRequest {
  src: string
  tokens?: number
  space: 'clip' | 'native'
}

export interface GenPromptRequest {
  src: string
  which: 'inverted' | 'native'
}

export interface RunResponse {
  job_id: number
  label?: string
}

export interface WipePreview {
  count: number
}

export interface WipeResult {
  deleted: number
  files: number
}

/**
 * The `.json` sidecar of a generated image. Only the keys the UI reads are
 * named; the index signature keeps the rest addressable for the param dump.
 */
export interface ImageMeta {
  kind?: string
  mode?: string
  parent?: string
  parent_b?: string
  refined_from?: string
  /** refine sidecars: which upscaler made it (hires | flux2-klein | absent = sd img2img). */
  engine?: string
  /** hires: the ancestor whose .npz supplied the conditioning. */
  cond_from?: string
  factor?: number
  denoise?: number
  interp?: string
  backend?: string
  model?: string
  sampler?: string
  temperature?: number
  coherence?: number
  scheduler?: string
  steps?: number
  guidance?: number
  batch_seed?: number
  image_seed?: number
  radius?: number
  mutate?: number
  scale?: number
  strength?: number
  distance?: number
  inverted_prompt?: string
  inverted_tokens?: number
  inverted_sim?: number
  native_prompt?: string
  native_sim?: number
  native_from?: string
  [key: string]: unknown
}
