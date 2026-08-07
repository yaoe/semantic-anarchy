/**
 * Client-only state. Anything that comes from the server lives in TanStack
 * Query instead (api/queries.ts) — this store holds the things the server has
 * no opinion about: which job is selected, which tab is open, the lightbox
 * cursor, the pending breed parent, and the raw text of every form control.
 *
 * Form values are kept as strings (exactly what an <input> holds); the cast to
 * the typed /api/run payload happens once, in params/schema.ts.
 */
import { create } from 'zustand'
import { persist } from 'zustand/middleware'

import { DEFAULT_VALUES, aspectDims, type Values } from './params/schema'
import type { TabKey } from './api/types'

export type SortKey = 'new' | 'score' | 'dist' | 'dist_asc' | 'nov' | 'res'

/** Refine + explore toolbar state (the legacy #refineBar inputs). */
export const DEFAULT_TOOLS: Record<string, string> = {
  rfScale: '1.5',
  rfSteps: '',
  rfEngine: 'hires',
  // hires keeps its own factor/denoise so switching engines doesn't clobber
  // the flux/sd numbers (and vice versa).
  rfFactor: '2.0',
  rfDenoise: '0.3',
  rfMode: 'tiled',
  rfStrength: '',
  rfSched: 'ddim',
  rfPromptSel: 'faithful',
  rfPrompt: '',
  exRadius: '',
  exN: '',
  wkStep: '',
}

/**
 * 🎬 Timeline knobs (POST /api/film). Kept as strings like every other form
 * value; Timeline.tsx casts them once when it submits.
 */
export const DEFAULT_FILM: Record<string, string> = {
  name: '',
  fps: '16',
  framesPer: '24',
  interp: 'slerp',
  easing: 'smooth',
  loop: '',
  size: '',            // '' = keyframe 1's own resolution, else "WxH"
  refine: 'none',
  scale: '1.5',
  noiseWindow: '1.0',
  fixedNoise: '',
  filmSeed: '42',
  steps: '',
  guidance: '',
}

/**
 * 🏷 Label page settings (the page at /label). Everything here except `knobs`
 * defines WHICH images are queued, and therefore the query key — the cursor
 * itself is `labelIndex`, which is per-session (a fresh queue starts at the top).
 *
 * Facet values: '' = any, '__none__' = images with no value on that dimension.
 * The time window is stored as ABSOLUTE unix seconds even when it came from a
 * "last 24h" button, so the query key doesn't drift with the clock.
 */
export const DEFAULT_LABEL: Record<string, string> = {
  experiment: '',
  backend: '',
  ckpt: '',
  folder: '',
  size: '',
  kind: '',
  sampler: '',
  since: '',           // unix seconds, '' = open-ended
  until: '',
  scope: 'unlabeled',
  bucket: 'generated',
  order: 'shuffle',
  seed: '0',           // salts the server's stable shuffle
  knobs: '',           // '1' = show the knob readout (blind labeling by default)
  queueOpen: '',       // collapsed: filters are set once, the image wants the room
}

/** Resetting the selection must not also reset how you like to look at it. */
export const LABEL_FILTER_KEYS = [
  'experiment', 'backend', 'ckpt', 'folder', 'size', 'kind', 'sampler',
  'since', 'until',
] as const

/**
 * 🧬 Fit page: which images the next distribution is fitted to.
 *
 * Same facet vocabulary as the labeling queue (the two answer the same "which
 * images do I mean" question), plus the two things you select *by* once you have
 * labeled: the star and the score band. Everything is a string here like every
 * other form value; FitPage casts once when it queries.
 */
export const DEFAULT_FIT: Record<string, string> = {
  experiment: '',
  backend: '',
  ckpt: '',
  folder: '',
  size: '',
  kind: '',
  sampler: '',
  since: '',            // unix seconds, '' = open-ended
  until: '',
  starred: '',          // '1' = ★ only
  scored: 'any',        // any | labeled | unlabeled
  minScore: '',         // label score floor, '' = no floor
  maxScore: '',
  order: 'new',
  name: '',             // what the fit will be called on disk
  note: '',
  components: '',       // '' = the selection's full N-1 rank
}

/** Cleared by "clear filters"; `name`/`note`/`order` are not filters. */
export const FIT_FILTER_KEYS = [
  'experiment', 'backend', 'ckpt', 'folder', 'size', 'kind', 'sampler',
  'since', 'until', 'starred', 'scored', 'minScore', 'maxScore',
] as const

interface UIState {
  params: Values
  setParam: (id: string, value: string) => void
  resetParams: () => void

  tools: Record<string, string>
  setTool: (id: string, value: string) => void

  /** Ordered keyframes (outputs-relative rel paths). Repeats are allowed —
   *  A → B → A is a perfectly good round trip. */
  timeline: string[]
  addKeyframe: (rel: string) => void
  removeKeyframe: (index: number) => void
  moveKeyframe: (from: number, to: number) => void
  clearTimeline: () => void

  film: Record<string, string>
  setFilm: (id: string, value: string) => void

  /** 🧬 Fit page filters + the new fit's name/note. */
  fit: Record<string, string>
  setFit: (id: string, value: string) => void
  resetFitFilters: () => void
  /**
   * The images the next fit is built from (outputs-relative rel paths).
   * Curation work, like the timeline — it survives a reload, and it is a SET:
   * fitting the same latent twice would just weight it twice.
   */
  fitSel: string[]
  toggleFitSel: (rel: string) => void
  addFitSel: (rels: string[]) => void
  removeFitSel: (rels: string[]) => void
  clearFitSel: () => void

  label: Record<string, string>
  setLabel: (id: string, value: string) => void
  /** Clear every facet + the time window, leaving scope/order/knobs alone. */
  resetLabelFilters: () => void
  /** Cursor into the fetched label queue. Reset whenever the queue changes. */
  labelIndex: number
  setLabelIndex: (i: number) => void

  selectedJob: number | null
  setSelectedJob: (id: number | null) => void

  /** Job log pane folded away to give the gallery the vertical space. */
  logCollapsed: boolean
  toggleLog: () => void

  tab: TabKey
  setTab: (t: TabKey) => void

  sortBy: SortKey
  setSortBy: (s: SortKey) => void

  /** -1 = closed. Indexes into the gallery's current (sorted) view. */
  lightboxIndex: number
  openLightbox: (i: number) => void
  closeLightbox: () => void
  moveLightbox: (delta: number, length: number) => void

  /** rel-path of the first-clicked 🧬 breed parent, or null. */
  breedParent: string | null
  setBreedParent: (rel: string | null) => void
}

/**
 * The slice worth surviving a reload. Job selection, the lightbox cursor and a
 * half-finished breed pairing are per-session; a half-built timeline is not —
 * that's curation work.
 */
const partialize = (s: UIState) => ({
  params: s.params,
  tools: s.tools,
  tab: s.tab,
  sortBy: s.sortBy,
  logCollapsed: s.logCollapsed,
  timeline: s.timeline,
  film: s.film,
  label: s.label,
  fit: s.fit,
  fitSel: s.fitSel,
})
type Persisted = ReturnType<typeof partialize>

export const useUI = create<UIState>()(
  persist(
    (set) => ({
      params: { ...DEFAULT_VALUES },
      setParam: (id, value) =>
        set((s) => {
          const params = { ...s.params, [id]: value }
          // Aspect presets are resolution-relative, so both the preset itself
          // and a backend switch recompute width/height (legacy applyAspect()).
          if (id === 'aspect' || id === 'backend') Object.assign(params, aspectDims(params))
          return { params }
        }),
      resetParams: () => set({ params: { ...DEFAULT_VALUES } }),

      tools: { ...DEFAULT_TOOLS },
      setTool: (id, value) => set((s) => ({ tools: { ...s.tools, [id]: value } })),

      timeline: [],
      addKeyframe: (rel) => set((s) => ({ timeline: [...s.timeline, rel] })),
      removeKeyframe: (index) =>
        set((s) => ({ timeline: s.timeline.filter((_, i) => i !== index) })),
      moveKeyframe: (from, to) =>
        set((s) => {
          if (from === to || from < 0 || from >= s.timeline.length) return s
          const next = [...s.timeline]
          const [moved] = next.splice(from, 1)
          next.splice(Math.max(0, Math.min(next.length, to)), 0, moved)
          return { timeline: next }
        }),
      clearTimeline: () => set({ timeline: [] }),

      film: { ...DEFAULT_FILM },
      setFilm: (id, value) => set((s) => ({ film: { ...s.film, [id]: value } })),

      fit: { ...DEFAULT_FIT },
      setFit: (id, value) => set((s) => ({ fit: { ...s.fit, [id]: value } })),
      resetFitFilters: () =>
        set((s) => ({
          fit: {
            ...s.fit,
            ...Object.fromEntries(FIT_FILTER_KEYS.map((k) => [k, ''])),
            scored: 'any',
          },
        })),

      fitSel: [],
      toggleFitSel: (rel) =>
        set((s) => ({
          fitSel: s.fitSel.includes(rel)
            ? s.fitSel.filter((r) => r !== rel)
            : [...s.fitSel, rel],
        })),
      // Adding the current filter's matches must not double up whatever is
      // already picked — hence a set union rather than a concat.
      addFitSel: (rels) =>
        set((s) => {
          const have = new Set(s.fitSel)
          return { fitSel: [...s.fitSel, ...rels.filter((r) => !have.has(r))] }
        }),
      removeFitSel: (rels) =>
        set((s) => {
          const drop = new Set(rels)
          return { fitSel: s.fitSel.filter((r) => !drop.has(r)) }
        }),
      clearFitSel: () => set({ fitSel: [] }),

      label: { ...DEFAULT_LABEL },
      // Any change to WHICH queue is being walked rewinds the cursor — the
      // fetched list is different, so the old index means nothing.
      setLabel: (id, value) =>
        set((s) => ({
          label: { ...s.label, [id]: value },
          labelIndex: id === 'knobs' ? s.labelIndex : 0,
        })),
      resetLabelFilters: () =>
        set((s) => ({
          label: {
            ...s.label,
            ...Object.fromEntries(LABEL_FILTER_KEYS.map((k) => [k, ''])),
          },
          labelIndex: 0,
        })),
      labelIndex: 0,
      setLabelIndex: (labelIndex) => set({ labelIndex }),

      selectedJob: null,
      setSelectedJob: (id) => set({ selectedJob: id }),

      logCollapsed: false,
      toggleLog: () => set((s) => ({ logCollapsed: !s.logCollapsed })),

      tab: 'generated',
      setTab: (tab) => set({ tab }),

      sortBy: 'new',
      setSortBy: (sortBy) => set({ sortBy }),

      lightboxIndex: -1,
      openLightbox: (i) => set({ lightboxIndex: i }),
      closeLightbox: () => set({ lightboxIndex: -1 }),
      moveLightbox: (delta, length) =>
        set((s) => {
          if (s.lightboxIndex < 0 || length === 0) return s
          const next = s.lightboxIndex + delta
          if (next < 0 || next >= length) return s
          return { lightboxIndex: next }
        }),

      breedParent: null,
      setBreedParent: (breedParent) => set({ breedParent }),
    }),
    {
      name: 'semantic-anarchy-ui',
      version: 5,
      // v1 -> v2: the same-latent hires pass became the upscale default. Anyone
      // still carrying the old 'flux' default gets moved onto it; an explicit
      // 'sd' pick is left alone. (A plain version bump would drop the whole
      // blob — including a half-built timeline — so migrate rather than reset.)
      // The cast is safe: whatever migrate returns goes straight through `merge`
      // below, which re-applies the defaults for every missing key.
      // v2 -> v3: labeling briefly lived as a gallery tab before moving to its
      // own page at /label. A blob still holding tab:'label' would select a tab
      // that no longer exists and show an empty grid.
      migrate: (persisted, from) => {
        const p = (persisted ?? {}) as Partial<Persisted>
        let out =
          from < 2 && p.tools?.rfEngine === 'flux'
            ? { ...p, tools: { ...p.tools, rfEngine: 'hires' } }
            : p
        if (from < 3 && (out.tab as string) === 'label') out = { ...out, tab: 'generated' }
        // v3 -> v4: the queue builder went from open-by-default to collapsed
        // (it was eating half the labeling page). Drop the stored flag so the
        // new default applies instead of a stale "open" outliving the change.
        if (from < 4 && out.label) {
          const { queueOpen: _drop, ...rest } = out.label
          out = { ...out, label: rest }
        }
        // v4 -> v5: neg-mode and prompt-length got new schema defaults (mean /
        // corpus). Every existing blob stores the *old* defaults explicitly, and
        // `merge` lets stored keys win, so drop those two and let the schema
        // supply them again. A value the user has since changed away from the
        // old default is kept.
        if (from < 5 && out.params) {
          const params = { ...out.params }
          if (params.neg_mode === '') delete params.neg_mode
          if (params.length_mode === 'off') delete params.length_mode
          out = { ...out, params }
        }
        return out as Persisted
      },
      partialize,
      // New knobs added to the schema must appear even for users with an old
      // blob in localStorage, so defaults always win on missing keys.
      merge: (persisted, current) => {
        const p = (persisted ?? {}) as Partial<UIState>
        return {
          ...current,
          ...p,
          params: { ...DEFAULT_VALUES, ...(p.params ?? {}) },
          tools: { ...DEFAULT_TOOLS, ...(p.tools ?? {}) },
          film: { ...DEFAULT_FILM, ...(p.film ?? {}) },
          label: { ...DEFAULT_LABEL, ...(p.label ?? {}) },
          fit: { ...DEFAULT_FIT, ...(p.fit ?? {}) },
          timeline: p.timeline ?? [],
          fitSel: p.fitSel ?? [],
        }
      },
    },
  ),
)
