/**
 * Every fetch in the app goes through here. FastAPI reports failures as
 * `{"detail": "..."}`, so ApiError.message is already the string the UI wants
 * to show.
 */
import type {
  BackendId,
  Config,
  EvolveRequest,
  ExploreRequest,
  Film,
  FilmRequest,
  FilmResponse,
  FsListing,
  GenPromptRequest,
  ImageMeta,
  Images,
  InvertRequest,
  KeyframeRow,
  ModelConfig,
  ModelRow,
  NativePickResult,
  RefineRequest,
  RunRequest,
  RunResponse,
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

export const api = {
  // ---- reads ----
  config: () => req<Config>('/api/config'),
  state: () => req<StateSnapshot>('/api/state'),
  images: () => req<Images>('/api/images'),
  films: () => req<Film[]>('/api/films'),
  tasteband: () => req<TasteBand>('/api/tasteband'),
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

  // ---- model picker ----
  model: () => req<ModelConfig>('/api/model'),
  setModel: (backend: BackendId, path: string | null) =>
    post<ModelRow>('/api/model', { backend, path }),
  /** Opens a real OS dialog on the *server host* — long-running by nature. */
  nativePick: (mode: 'file' | 'folder', start?: string | null) =>
    post<NativePickResult>('/api/model/native', { mode, start: start ?? null }),
  fs: (path?: string | null) =>
    req<FsListing>(path ? `/api/fs?path=${encodeURIComponent(path)}` : '/api/fs'),

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

  // ---- mutations on the library ----
  favorite: (rel: string, on: boolean) =>
    post<{ ok: boolean; fav: boolean; count: number }>('/api/favorite', { rel, on }),
  wipe: () => post<WipeResult>('/api/wipe'),
  deleteFilm: (dir: string) => post<{ deleted: string }>('/api/films/delete', { dir }),
}
