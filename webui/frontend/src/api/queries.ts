/**
 * All server state lives in TanStack Query. Nothing in the app calls
 * setInterval on a fetch — polling cadence is declared here, and every
 * job-submitting mutation invalidates the job snapshot so the queue updates
 * without a manual poke.
 */
import { useEffect, useRef } from 'react'
import { keepPreviousData, useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import { api } from './client'
import type {
  BackendId,
  EvolveRequest,
  ExploreRequest,
  FilmRequest,
  GenPromptRequest,
  InvertRequest,
  RefineRequest,
  RunRequest,
  RunResponse,
} from './types'
import { useUI } from '../store'

export const qk = {
  config: ['config'] as const,
  state: ['state'] as const,
  images: ['images'] as const,
  films: ['films'] as const,
  keyframes: (rels: string[]) => ['keyframes', rels.join('|')] as const,
  tasteband: ['tasteband'] as const,
  wipePreview: ['wipePreview'] as const,
  meta: (rel: string) => ['meta', rel] as const,
  model: ['model'] as const,
  fs: (path: string | null) => ['fs', path] as const,
}

/* --------------------------------------------------------------- reads --- */

/** Config changes when the user drops files into init_images/ — poll slowly. */
export function useConfig() {
  return useQuery({
    queryKey: qk.config,
    queryFn: api.config,
    refetchInterval: 8000,
  })
}

/** The job queue: the one genuinely live thing on the page. */
export function useJobState() {
  return useQuery({
    queryKey: qk.state,
    queryFn: api.state,
    refetchInterval: 1500,
  })
}

/** Gallery. Cheap when idle (no polling), 6s while a job is producing files. */
export function useImages(busy: boolean) {
  return useQuery({
    queryKey: qk.images,
    queryFn: api.images,
    refetchInterval: busy ? 6000 : false,
  })
}

export function useFilms(enabled: boolean) {
  return useQuery({ queryKey: qk.films, queryFn: api.films, enabled })
}

/**
 * What the timeline's keyframes actually are (backend, real resolution). The
 * server is the authority — filenames only hint at the backend, and nothing in
 * the gallery payload carries the pixel size. Re-probed on every edit.
 */
export function useKeyframes(timeline: string[]) {
  return useQuery({
    queryKey: qk.keyframes(timeline),
    queryFn: () => api.keyframes(timeline),
    enabled: timeline.length > 0,
    staleTime: 60_000,
  })
}

export function useTasteBand() {
  return useQuery({
    queryKey: qk.tasteband,
    queryFn: api.tasteband,
    refetchInterval: 20000,
  })
}

export function useMeta(rel: string | null) {
  return useQuery({
    queryKey: qk.meta(rel ?? ''),
    queryFn: () => api.meta(rel!),
    enabled: !!rel,
    staleTime: 5000,
    // Hold the previous image's params on screen while the next fetch is in
    // flight. Without this the lightbox panel empties out for a frame on every
    // arrow press, and the image — sized against the panel — visibly jumps.
    placeholderData: keepPreviousData,
  })
}

/**
 * Refresh the library the moment the worker goes idle — the legacy UI's
 * "lastBusy === true && !busy -> refreshImages()" edge, expressed once.
 */
export function useRefreshOnJobFinish(running: number | null | undefined) {
  const qc = useQueryClient()
  const prev = useRef<number | null | undefined>(undefined)
  useEffect(() => {
    if (prev.current != null && running == null) {
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.films })
      qc.invalidateQueries({ queryKey: qk.tasteband })
      qc.invalidateQueries({ queryKey: qk.wipePreview })
    }
    prev.current = running
  }, [running, qc])
}

/**
 * The hand-picked checkpoints. Only changes when someone uses the picker, so
 * no polling — the mutation below invalidates it.
 */
export function useModelConfig() {
  return useQuery({ queryKey: qk.model, queryFn: api.model, staleTime: 30_000 })
}

/** One directory listing for the fallback file browser. */
export function useFs(path: string | null, enabled: boolean) {
  return useQuery({
    queryKey: qk.fs(path),
    queryFn: () => api.fs(path),
    enabled,
    // Keep the current folder on screen while the next one loads, so the modal
    // doesn't collapse to zero height on every click.
    placeholderData: keepPreviousData,
  })
}

/* ----------------------------------------------------------- mutations --- */

/** Persist (or clear, with `path: null`) a backend's checkpoint. */
export function useSetModel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ backend, path }: { backend: BackendId; path: string | null }) =>
      api.setModel(backend, path),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.model })
      qc.invalidateQueries({ queryKey: qk.config })
    },
  })
}

/**
 * Ask the server to pop its OS file dialog. Nothing resolves until the dialog
 * is answered, hence no retry — a second call would open a second window.
 */
export function useNativePick() {
  return useMutation({
    mutationFn: ({ mode, start }: { mode: 'file' | 'folder'; start?: string | null }) =>
      api.nativePick(mode, start),
    retry: false,
  })
}

/**
 * Shared tail for every endpoint that queues a job: select it in the job list
 * and invalidate the snapshot so it shows up before the next poll tick.
 * Per-call callbacks still work — pass them to `mutate(vars, { onSuccess })`.
 */
function useJobMutation<TVars, TData extends RunResponse = RunResponse>(
  fn: (vars: TVars) => Promise<TData>,
) {
  const qc = useQueryClient()
  const setSelectedJob = useUI((s) => s.setSelectedJob)
  return useMutation<TData, Error, TVars>({
    mutationFn: fn,
    onSuccess: (data) => {
      setSelectedJob(data.job_id)
      qc.invalidateQueries({ queryKey: qk.state })
    },
  })
}

export const useRun = () => useJobMutation((b: RunRequest) => api.run(b))
export const useRefine = () => useJobMutation((b: RefineRequest) => api.refine(b))
export const useExplore = () => useJobMutation((b: ExploreRequest) => api.explore(b))
export const useEvolve = () => useJobMutation((b: EvolveRequest) => api.evolve(b))
export const useResonance = () => useJobMutation<void>(() => api.resonance())
export const useScore = () => useJobMutation<void>(() => api.score())
export const useInvert = () => useJobMutation((b: InvertRequest) => api.invert(b))
/** Latent-travel film through the timeline's keyframes (POST /api/film). */
export const useFilm = () => useJobMutation((b: FilmRequest) => api.film(b))
export const useGenPrompt = () =>
  useJobMutation((b: GenPromptRequest) => api.genprompt(b))

export function useCancel() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (jobId: number) => api.cancel(jobId),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.state }),
  })
}

/** Optimistic star toggle: the grid flips instantly, the server catches up. */
export function useFavorite() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ rel, on }: { rel: string; on: boolean }) => api.favorite(rel, on),
    onSettled: () => {
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.tasteband })
    },
  })
}

export function useWipe() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.wipe(),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.images })
      qc.invalidateQueries({ queryKey: qk.wipePreview })
    },
  })
}

export function useWipePreview(enabled: boolean) {
  return useQuery({
    queryKey: qk.wipePreview,
    queryFn: api.wipePreview,
    enabled,
    refetchInterval: 30000,
  })
}

export function useDeleteFilm() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (dir: string) => api.deleteFilm(dir),
    onSuccess: () => qc.invalidateQueries({ queryKey: qk.films }),
  })
}

/**
 * Wait for a queued job to reach a terminal state, then run `onDone`. Used by
 * the lightbox (invert / generate-from-prompt) where the panel must refresh
 * itself once the subprocess has written its sidecar.
 */
export function useJobWatcher() {
  const qc = useQueryClient()
  const timers = useRef<number[]>([])
  useEffect(
    () => () => {
      timers.current.forEach((t) => window.clearInterval(t))
    },
    [],
  )
  return (jobId: number, onDone: (status: string) => void) => {
    const t = window.setInterval(async () => {
      try {
        const st = await api.state()
        const j = st.jobs.find((x) => x.id === jobId)
        if (j && ['done', 'error', 'cancelled'].includes(j.status)) {
          window.clearInterval(t)
          qc.invalidateQueries({ queryKey: qk.state })
          onDone(j.status)
        }
      } catch {
        /* transient — try again on the next tick */
      }
    }, 4000)
    timers.current.push(t)
  }
}
