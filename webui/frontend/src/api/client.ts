/**
 * Every fetch in the app goes through here. FastAPI reports failures as
 * `{"detail": "..."}`, so ApiError.message is already the string the UI wants
 * to show.
 */
import type {
  BackendId,
  Config,
  DistConfig,
  DistKind,
  DistRow,
  EvolveRequest,
  ExploreRequest,
  Film,
  FilmRequest,
  FilmResponse,
  FitCandidates,
  FitQueryParams,
  FitRequest,
  FitResponse,
  FsListing,
  GenPromptRequest,
  ExperimentRow,
  ImageMeta,
  Images,
  InvertRequest,
  KeyframeRow,
  LabelFacets,
  LabelQueue,
  LabelQueryParams,
  LabelStats,
  ModelConfig,
  ModelRow,
  NativePickResult,
  PickMode,
  PrefsPayload,
  RefineRequest,
  RunRequest,
  RunResponse,
  SavedFit,
  StateSnapshot,
  TasteBand,
  WipePreview,
  WipeResult,
} from './types'

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, init)
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (body && typeof body.detail === 'string') detail = body.detail
    } catch {
      /* non-JSON error body — keep the status line */
    }
    throw new ApiError(res.status, detail)
  }
  return (await res.json()) as T
}

function post<T>(path: string, body?: unknown): Promise<T> {
  return req<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })
}

/** `?a=1&b=2` from the entries that actually have a value (never `?a=null`). */
function qs(params: Record<string, unknown>): string {
  const q = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v != null && v !== '') q.set(k, String(v))
  }
  const s = q.toString()
  return s ? `?${s}` : ''
}

export const api = {
  // ---- reads ----
  config: () => req<Config>('/api/config'),
  state: () => req<StateSnapshot>('/api/state'),
  images: () => req<Images>('/api/images'),
  films: () => req<Film[]>('/api/films'),
  tasteband: () => req<TasteBand>('/api/tasteband'),
  experiments: () => req<ExperimentRow[]>('/api/experiments'),
  wipePreview: () => req<WipePreview>('/api/wipe/preview'),
  meta: (path: string) =>
    req<ImageMeta>(`/api/meta?path=${encodeURIComponent(path)}`),
  /** Plain-text fallback for the SSE log stream. */
  log: async (jobId: number): Promise<string> => {
    const res = await fetch(`/api/log/${jobId}`)
    if (!res.ok) throw new ApiError(res.status, `log ${jobId}: ${res.status}`)
    return res.text()
  },
  logStreamUrl: (jobId: number) => `/api/log/${jobId}/stream`,

  // ---- persisted UI defaults (config.json -> "ui"; see lib/prefs.ts) ----
  prefs: () => req<PrefsPayload>('/api/prefs'),
  /** `null` forgets the stored blob and falls back to the schema defaults. */
  savePrefs: (ui: unknown | null) =>
    post<{ ok: boolean; stored: boolean }>('/api/prefs', { ui }),

  // ---- model picker ----
  model: () => req<ModelConfig>('/api/model'),
  setModel: (backend: BackendId, path: string | null) =>
    post<ModelRow>('/api/model', { backend, path }),
  /** Opens a real OS dialog on the *server host* — long-running by nature. */
  nativePick: (mode: 'file' | 'folder', start?: string | null) =>
    post<NativePickResult>('/api/model/native', { mode, start: start ?? null }),
  fs: (path?: string | null, opts?: { pick?: PickMode; backend?: string; model?: string | null }) =>
    req<FsListing>(`/api/fs${qs({ path, ...opts })}`),

  // ---- base-distribution picker ----
  dist: (backend: string, model?: string | null) =>
    req<DistConfig>(`/api/dist${qs({ backend, model })}`),
  /** What picking this .txt/.npz would mean, without committing to it. */
  probeDist: (backend: string, path: string, model?: string | null) =>
    req<DistRow>(`/api/dist/probe${qs({ backend, path, model })}`),
  setDist: (body: { backend: string; kind: DistKind; path?: string | null; model?: string | null }) =>
    post<DistRow>('/api/dist', body),
  encodeDist: (body: { backend: string; path: string; model?: string | null }) =>
    post<RunResponse & { out: string }>('/api/dist/encode', body),

  // ---- fit a distribution from picked images ----
  fitCandidates: (o: FitQueryParams) =>
    req<FitCandidates>(`/api/fit/candidates${qs({ ...o })}`),
  fitList: (backend?: string | null) => req<SavedFit[]>(`/api/fit/list${qs({ backend })}`),
  fit: (body: FitRequest) => post<FitResponse>('/api/fit', body),
  deleteFit: (name: string) => post<{ deleted: string[] }>('/api/fit/delete', { name }),

  // ---- job submissions ----
  run: (body: RunRequest) => post<RunResponse>('/api/run', body),
  refine: (body: RefineRequest) => post<RunResponse>('/api/refine', body),
  explore: (body: ExploreRequest) => post<RunResponse>('/api/explore', body),
  evolve: (body: EvolveRequest) => post<RunResponse>('/api/evolve', body),
  resonance: () => post<RunResponse>('/api/resonance'),
  score: () => post<RunResponse>('/api/score'),
  film: (body: FilmRequest) => post<FilmResponse>('/api/film', body),
  keyframes: (images: string[]) =>
    post<KeyframeRow[]>('/api/keyframes', { images }),
  invert: (body: InvertRequest) => post<RunResponse>('/api/invert', body),
  genprompt: (body: GenPromptRequest) => post<RunResponse>('/api/genprompt', body),
  cancel: (jobId: number) => post<{ ok: boolean }>(`/api/cancel/${jobId}`),

  // ---- labeling ----
  /** `qs` drops empty/null values, so "any" on a facet simply isn't sent. */
  labelQueue: (o: LabelQueryParams) => req<LabelQueue>(`/api/label/queue${qs({ ...o })}`),
  labelFacets: () => req<LabelFacets>('/api/label/facets'),
  labelStats: () => req<LabelStats>('/api/labels'),
  label: (rel: string, score: number) =>
    post<{ ok: boolean; rel: string; score: number; count: number }>('/api/label', {
      rel,
      score,
    }),

  // ---- mutations on the library ----
  favorite: (rel: string, on: boolean) =>
    post<{ ok: boolean; fav: boolean; count: number }>('/api/favorite', { rel, on }),
  wipe: () => post<WipeResult>('/api/wipe'),
  deleteFilm: (dir: string) => post<{ deleted: string }>('/api/films/delete', { dir }),
}
