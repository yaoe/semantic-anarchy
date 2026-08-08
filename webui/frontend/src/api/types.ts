/**
 * Wire types for webui/app.py. Every field here mirrors a pydantic model or a
 * JSONResponse literal in that file — keep them in lock-step.
 */

export type ActionId = 'generate' | 'temp_sweep' | 'sampler_sweep' | 'mine'
export type BackendId = 'sd15' | 'sd2' | 'sdxl' | 'flux2' | 'krea2'
export type SamplerId = 'diagonal' | 'pca' | 'blend' | 'hybrid' | 'split'
export type LengthModeId = 'off' | 'corpus' | 'fixed'
export type SchedulerId = 'default' | 'ddim' | 'euler' | 'euler_a' | 'dpm'
export type NegModeId = 'text' | 'mean' | 'empty' | 'zeros'
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

/** Gallery tabs = server buckets + the client-side films/timeline/fit views.
 *  Labeling is NOT a tab — it is its own page at /label (see LabelApp). */
export type TabKey = GalleryKey | 'films' | 'timeline' | 'fit'

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
  /** backend -> hand-picked checkpoint path (config.json -> "models"). */
  picked_models: Partial<Record<BackendId, string>>
  /** The house sd15 CFG negative prompt (or an SA_SD15_NEGATIVE override). */
  sd15_negative: string
  init_dir: string
  init_count: number
  init_folders: InitFolder[]
  repo: string
  /** The fixed seed panel comparative batches render against. */
  seed_panel: { seed: number; n: number; seeds: number[] }
  /** Absolute path of the append-only labels dataset (git-tracked). */
  labels_file: string
  /** Absolute path of the gitignored config.json this install persists into. */
  config_file: string
}

/**
 * `GET /api/prefs` — the persisted UI store (config.json -> "ui"), verbatim as
 * zustand serialised it, or null when this install has never saved one.
 */
export interface PrefsPayload {
  ui: Record<string, unknown> | null
  config_file: string
}

/* ------------------------------------------------------------- labeling --- */

/** The dimensions the label queue can be sliced along (webui.app LABEL_FACETS). */
export type FacetDim =
  | 'experiment'
  | 'backend'
  | 'ckpt'
  | 'folder'
  | 'size'
  | 'kind'
  | 'sampler'

/** Server sentinel for "this image has no value on that dimension". */
export const UNSET = '__none__'

/** One image waiting for a score — mirrors webui.app `_label_index`. */
export interface LabelRow {
  rel: string
  url: string
  mtime: number
  fav: boolean
  /** The score that currently stands for it, or null when unlabeled. */
  score: number | null
  experiment: string | null
  backend: string | null
  /** Slug of the checkpoint that rendered it. */
  ckpt: string | null
  /** Directory under outputs/ it lives in. */
  folder: string | null
  /** "512x512" — from the sidecar, or read off the PNG when that predates it. */
  size: string | null
  kind: string | null
  sampler: string | null
  distance: number | null
  image_seed: number | null
  /** The sidecar knobs shown (collapsibly) beside the image. */
  knobs: Record<string, string | number | boolean>
}

/** What the queue may be narrowed to. */
export type LabelScope = 'unlabeled' | 'all' | 'labeled'
export type LabelBucket = 'generated' | 'favorites'
export type LabelOrder = 'shuffle' | 'new' | 'old'

export interface LabelQueue {
  queue: LabelRow[]
  /** Size of the whole selection, before `limit`. */
  total: number
  labeled: number
  filters: Partial<Record<FacetDim, string | null>>
  since: number | null
  until: number | null
  scope: LabelScope
  bucket: LabelBucket
  order: LabelOrder
}

/**
 * The query string of GET /api/label/queue — i.e. the label page's entire
 * selection, and also its TanStack Query key. Facet values are '' for "any" and
 * `UNSET` for "images missing a value on that dimension".
 */
export interface LabelQueryParams extends Partial<Record<FacetDim, string>> {
  scope: LabelScope
  bucket: LabelBucket
  order: LabelOrder
  /** Salts the server's stable shuffle — bump it to reshuffle on purpose. */
  seed: number
  /** Unix seconds; null/undefined = open-ended. */
  since?: number | null
  until?: number | null
  limit?: number
}

/** One selectable value of a facet, with how much of it is left to label. */
export interface FacetCell {
  value: string
  count: number
  unlabeled: number
}

/** GET /api/label/facets — what there is to choose from, over the whole set. */
export interface LabelFacets {
  total: number
  unlabeled: number
  favorites: number
  /** mtime bounds of the labelable set, for the time-window picker. */
  oldest: number | null
  newest: number | null
  facets: Record<FacetDim, FacetCell[]>
}

/** labels.summarize() — deliberately tail-weighted. */
export interface LabelSummary {
  n: number
  keeper_rate: number | null
  p90: number | null
  median: number | null
  mean: number | null
  max: number | null
  hist: number[]
}

export interface LabelStats {
  /** Distinct images labeled (latest record per image). */
  count: number
  /** Lines in the file — higher than `count` once anything was relabeled. */
  records: number
  file: string
  overall: LabelSummary
  experiments: (LabelSummary & { id: string })[]
}

/** One row of /api/experiments. `id: ''` is the untagged pile. */
export interface ExperimentRow {
  id: string
  hypothesis: string | null
  created: number | null
  runs: number
  seed_panel: boolean
  images: number
  labeled: number
}

/* --------------------------------------------------------- model picker --- */

/** What a checkpoint path turned out to be. `repo` = a cached HF id. */
export type ModelKind = 'ckpt' | 'diffusers' | 'repo' | null
/** What /api/fs?pick=dist lists: a prompt corpus (.txt) or a saved fit (.npz). */
export type FsKind = ModelKind | 'prompts' | 'npz'

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
  kind: FsKind
  size: number | null
  /** pick=dist only: this corpus is already encoded for the active checkpoint. */
  ready?: boolean
}

export interface FsListing {
  path: string
  /** null when already at a browsable root. */
  parent: string | null
  kind: FsKind
  entries: FsEntry[]
  roots: FsRoot[]
}

/** What the file browser is picking: a checkpoint, or a base distribution. */
export type PickMode = 'model' | 'dist'

export type DistKind = 'base' | 'evolved' | 'prompts' | 'file'

/**
 * One distribution choice — mirrors webui.app `describe_dist`. `base` is the
 * prefix `--dist` receives; `files` are the .npz it resolves to, and `ready`
 * says whether they exist for the checkpoint named in `model`.
 */
export interface DistRow {
  backend: BackendId
  kind: DistKind
  /** The picked .txt / .npz; null for the two built-in bases. */
  path: string | null
  base: string
  label: string
  ready: boolean
  files: { path: string; exists: boolean }[]
  /** The fit's .meta.json sidecar, once it exists. */
  meta: {
    feature_shape: number[]
    n_samples: number
    per_token: boolean
    has_pca: boolean
    has_corpus: boolean
    /** Absent on fits mined before the corpus-autopsy corrections existed. */
    has_length_stats?: boolean
    has_radius_band?: boolean
    noise_floor_axes?: number | null
  } | null
  /**
   * Set when this distribution was fitted from picked images (🧬 Fit) rather
   * than mined from prompts — the manifest that names them.
   */
  fit?: {
    name: string | null
    n_samples: number | null
    note: string | null
    created: number | null
    models: string[] | null
  } | null
  /** The checkpoint that encodes (or encoded) this corpus. */
  model: { path: string; name: string; slug: string }
}

export interface DistConfig extends DistRow {
  config_file: string
  default_prompts: string
}

/* ------------------------------------------------- fit from picked images -- */

/**
 * One candidate for a selection fit — the labeling index plus the two things
 * you select *by*: the label score that currently stands for it and the star.
 * `latents` is what makes it fittable at all (an upscale carries none).
 */
export interface FitCandidate {
  rel: string
  url: string
  mtime: number
  score: number | null
  fav: boolean
  latents: boolean
  experiment: string | null
  backend: BackendId | null
  ckpt: string | null
  folder: string | null
  size: string | null
  kind: string | null
  sampler: string | null
  distance: number | null
}

export type FitOrder = 'new' | 'old' | 'score' | 'distance'
export type FitScored = 'any' | 'labeled' | 'unlabeled'

/** GET /api/fit/candidates — the pool a selection is drawn from. */
export interface FitCandidates {
  /** The whole match; `rows` is capped by `limit`. */
  total: number
  shown: number
  rows: FitCandidate[]
  backends: BackendId[]
}

/** The query string of GET /api/fit/candidates, and its query key. */
export interface FitQueryParams extends Partial<Record<FacetDim, string>> {
  starred?: boolean
  scored?: FitScored
  min_score?: number | null
  max_score?: number | null
  since?: number | null
  until?: number | null
  order?: FitOrder
  limit?: number
}

/** One saved fit under outputs/dist_fits (GET /api/fit/list). */
export interface SavedFit {
  name: string
  base: string
  backend: BackendId | null
  created: number | null
  n_samples: number | null
  note: string | null
  models: string[]
  ready: boolean
  files: string[]
  meta: DistRow['meta']
}

export interface FitRequest {
  name: string
  rels: string[]
  backend?: BackendId | null
  components?: number | null
  note?: string | null
  overwrite?: boolean
}

export interface FitResponse extends RunResponse {
  name: string
  base: string
  backend: BackendId
  n: number
  /** Hand this to /api/dist as `{kind: 'file', path}` once the job lands. */
  file: string
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
  /** sd15/sd2 CFG negative text. null = keep the script's own default. */
  negative?: string | null
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
  /** Experiment id stamped into every sidecar of this batch. */
  experiment?: string | null
  hypothesis?: string | null
  /**
   * The corpus-autopsy corrections (see semantic_anarchy/distribution.py).
   * Every one is opt-in; leaving them alone reproduces the original samplers.
   */
  rho?: number | null
  length_mode?: LengthModeId | null
  length?: number | null
  empirical_head?: number | null
  temp_on?: number | null
  temp_off?: number | null
  radius_band?: boolean
  radius_scale?: number | null
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
  neg_mode?: string
  /** The CFG negative text this image was pushed away from (null = none). */
  negative?: string | null
  steps?: number
  guidance?: number
  batch_seed?: number
  image_seed?: number
  radius?: number
  mutate?: number
  scale?: number
  strength?: number
  distance?: number
  /** The corpus-autopsy knobs, recorded per image (length varies within a batch). */
  rho?: number | null
  length_mode?: string | null
  length?: number | null
  empirical_head?: number | null
  temp_on?: number | null
  temp_off?: number | null
  radius_band?: number | null
  radius_scale?: number | null
  inverted_prompt?: string
  inverted_tokens?: number
  inverted_sim?: number
  native_prompt?: string
  native_sim?: number
  native_from?: string
  [key: string]: unknown
}
