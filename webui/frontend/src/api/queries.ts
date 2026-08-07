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
  DistKind,
  EvolveRequest,
  ExploreRequest,
  FilmRequest,
  FitQueryParams,
  FitRequest,
  GenPromptRequest,
  InvertRequest,
  LabelQueryParams,
  LabelQueue,
  PickMode,
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
  experiments: ['experiments'] as const,
  labelStats: ['labelStats'] as const,
  labelFacets: ['labelFacets'] as const,
  labelQueue: (k: LabelQueryParams) => ['labelQueue', k] as const,
  wipePreview: ['wipePreview'] as const,
  meta: (rel: string) => ['meta', rel] as const,
  model: ['model'] as const,
  fs: (path: string | null, pick: PickMode, backend?: string, model?: string | null) =>
    ['fs', pick, backend ?? '', model ?? '', path] as const,
  dist: (backend: string, model?: string | null) => ['dist', backend, model ?? ''] as const,
  distProbe: (backend: string, path: string | null, model?: string | null) =>
    ['distProbe', backend, model ?? '', path] as const,
  fitCandidates: (k: FitQueryParams) => ['fitCandidates', k] as const,
  fitList: (backend?: string | null) => ['fitList', backend ?? ''] as const,
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

/**
 * Gallery. Cheap when idle (no polling), 2s while a job is producing files.
 *
 * generate.py writes each image the moment it renders rather than at the end of
 * the batch, so this poll is what turns a batch of 8 into eight arrivals — the
 * point being to judge an experiment without waiting for the whole run. A 6s
 * tick would hide a 3s image for twice its render time.
 */
export function useImages(busy: boolean) {
  return useQuery({
    queryKey: qk.images,
    queryFn: api.images,
    refetchInterval: busy ? 2000 : false,
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

/* ------------------------------------------------------------- labeling --- */

/**
 * The labeling queue.
 *
 * Deliberately frozen once fetched (`staleTime: Infinity`, no refetch on focus
 * or on a label being submitted): the page is a cursor walking a list, and a
 * background refetch would renumber "34/50" and slide a different image under
 * the keypress that was already on its way. New arrivals are picked up by the
 * explicit reload button, which invalidates this key.
 */
export function useLabelQueue(key: LabelQueryParams, enabled = true) {
  return useQuery({
    queryKey: qk.labelQueue(key),
    queryFn: () => api.labelQueue(key),
    enabled,
    staleTime: Infinity,
    refetchOnWindowFocus: false,
  })
}

/**
 * What the queue can be sliced by, with counts. Computed over the whole
 * labelable set, so an option never vanishes because of an earlier pick —
 * how many images a combination actually matches is what the queue reports.
 */
export function useLabelFacets(enabled = true) {
  return useQuery({
    queryKey: qk.labelFacets,
    queryFn: api.labelFacets,
    enabled,
    staleTime: 30_000,
  })
}

/** Experiment ids + how much of each is still unlabeled (the queue selector). */
export function useExperiments(enabled = true) {
  return useQuery({ queryKey: qk.experiments, queryFn: api.experiments, enabled, staleTime: 15_000 })
}

export function useLabelStats(enabled = true) {
  return useQuery({ queryKey: qk.labelStats, queryFn: api.labelStats, enabled, staleTime: 5_000 })
}

/**
 * Submit one label. The score is written straight into the cached queue row so
 * the UI never waits on the round trip — at one keypress per image, a 40ms
 * flicker per score is the difference between fluent and unusable.
 */
export function useSubmitLabel(key: LabelQueryParams) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: ({ rel, score }: { rel: string; score: number }) => api.label(rel, score),
    onMutate: async ({ rel, score }) => {
      qc.setQueryData<LabelQueue>(qk.labelQueue(key), (old) =>
        old
          ? { ...old, queue: old.queue.map((r) => (r.rel === rel ? { ...r, score } : r)) }
          : old,
      )
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: qk.labelStats })
      qc.invalidateQueries({ queryKey: qk.experiments })
      // Facet counts carry an "unlabeled" column, so they move on every label —
      // but the queue itself is deliberately NOT invalidated (see above).
      qc.invalidateQueries({ queryKey: qk.labelFacets })
    },
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

/**
 * One directory listing for the in-page file browsers. `pick` decides what the
 * server lists — checkpoints for the model picker, prompt corpora + saved fits
 * for the distribution picker (which also gets a per-corpus `ready` flag when
 * a backend is passed).
 */
export function useFs(
  path: string | null,
  enabled: boolean,
  opts?: { pick?: PickMode; backend?: string; model?: string | null },
) {
  const pick = opts?.pick ?? 'model'
  return useQuery({
    queryKey: qk.fs(path, pick, opts?.backend, opts?.model),
    queryFn: () => api.fs(path, { pick, backend: opts?.backend, model: opts?.model }),
    enabled,
    // Keep the current folder on screen while the next one loads, so the modal
    // doesn't collapse to zero height on every click.
    placeholderData: keepPreviousData,
  })
}

/** The distribution this backend samples from (webui/dist_config.json). */
export function useDistConfig(backend: BackendId, model?: string | null) {
  return useQuery({
    queryKey: qk.dist(backend, model),
    queryFn: () => api.dist(backend, model),
    staleTime: 10_000,
  })
}

/** Whether a candidate corpus/fit is already encoded for the active checkpoint. */
export function useDistProbe(
  backend: BackendId,
  path: string | null,
  model?: string | null,
) {
  return useQuery({
    queryKey: qk.distProbe(backend, path, model),
    queryFn: () => api.probeDist(backend, path!, model),
    enabled: !!path,
    retry: false,
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
      // A corpus is fitted PER checkpoint, so a new checkpoint means a
      // different fit — possibly one that hasn't been encoded yet.
      qc.invalidateQueries({ queryKey: ['dist'] })
      qc.invalidateQueries({ queryKey: ['fs'] })
    },
  })
}

/** Persist the base distribution this backend samples from. */
export function useSetDist() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: {
      backend: BackendId
      kind: DistKind
      path?: string | null
      model?: string | null
    }) => api.setDist(body),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['dist'] }),
  })
}

/**
 * Queue the encode pass for a prompt corpus. It goes through the same
 * single-worker queue as everything else, so the job list shows it running.
 */
export function useEncodeDist() {
  return useJobMutation(
    (b: { backend: BackendId; path: string; model?: string | null }) => api.encodeDist(b),
  )
}

/* ------------------------------------------------------------------ fit --- */

/**
 * The pool a selection fit is drawn from. Re-queried on every filter change —
 * the count it reports is what "n selected" is measured against, so it must not
 * lag behind the controls. Kept on screen while the next one loads so the grid
 * doesn't blink to empty between two adjacent filters.
 */
export function useFitCandidates(key: FitQueryParams, enabled = true) {
  return useQuery({
    queryKey: qk.fitCandidates(key),
    queryFn: () => api.fitCandidates(key),
    enabled,
    placeholderData: keepPreviousData,
    staleTime: 5_000,
  })
}

/** Saved selection fits, for the picker's third section and the fit page. */
export function useFitList(backend?: string | null, enabled = true) {
  return useQuery({
    queryKey: qk.fitList(backend),
    queryFn: () => api.fitList(backend),
    enabled,
    staleTime: 10_000,
  })
}

/** Queue the fit. Fast (no GPU) but still a job, so it lands in the job log. */
export const useCreateFit = () => useJobMutation((b: FitRequest) => api.fit(b))

export function useDeleteFit() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => api.deleteFit(name),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['fitList'] })
      // A deleted fit may have been the selected base distribution.
      qc.invalidateQueries({ queryKey: ['dist'] })
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
