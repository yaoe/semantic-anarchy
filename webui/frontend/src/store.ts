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
  rfEngine: 'flux',
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
      version: 1,
      // Only the settings worth surviving a reload. Job selection, the lightbox
      // cursor and a half-finished breed pairing are per-session.
      partialize: (s) => ({
        params: s.params,
        tools: s.tools,
        tab: s.tab,
        sortBy: s.sortBy,
        logCollapsed: s.logCollapsed,
        // A half-built timeline is worth a page reload — it's curation work.
        timeline: s.timeline,
        film: s.film,
      }),
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
          timeline: p.timeline ?? [],
        }
      },
    },
  ),
)
