/**
 * 🪐 Aesthetic universes (= 🏷 tags) as one shared model for every surface
 * that shows them: the gallery's universe bar, the 🌌 map's universe list and
 * selection panel, the card's "add to universe" popover.
 *
 * The vocabulary and its order come from GET /api/tags (commonest first — the
 * same order the viewer's and label page's shift+1…9 use, so a number means the
 * same universe everywhere). Members come from the cached gallery rows, which
 * already carry each image's tags.
 *
 * Also here: the record of 🪐 sampling jobs queued this session, so the new
 * images can be found again once they land (`useSampleBatches`).
 */
import { useMemo } from 'react'
import { create } from 'zustand'

import { useImages, useJobState, useTags } from '../../api/queries'
import type { ImageItem } from '../../api/types'

/** Well-separated universe colours (none of them the selection's mint). */
const PALETTE = [
  '#ff8a5c', '#7c8cff', '#ffd25c', '#e46fd6', '#62c9ff',
  '#a8e05f', '#ff5a6e', '#c6a2ff', '#d9a066', '#ffb3d9',
]

function hash(tag: string): number {
  let h = 2166136261
  for (const ch of tag.toLocaleLowerCase()) h = Math.imul(h ^ ch.charCodeAt(0), 16777619)
  return h >>> 0
}

/** Palette slots resolved for the current vocabulary (see `assignColors`). */
let assigned = new Map<string, string>()

/**
 * Colour by NAME, not by position (positions shift as counts change): each
 * universe hashes to a palette slot, and a clash within the current vocabulary
 * probes to the next free one, alphabetically, so two universes on screen never
 * share a colour while there are slots left.
 */
function assignColors(tags: string[]) {
  const out = new Map<string, string>()
  const taken = new Set<number>()
  for (const t of [...tags].sort((a, b) => a.localeCompare(b))) {
    let slot = hash(t) % PALETTE.length
    for (let k = 0; k < PALETTE.length && taken.has(slot); k++) slot = (slot + 1) % PALETTE.length
    if (taken.size >= PALETTE.length) {
      out.set(t.toLocaleLowerCase(), `hsl(${hash(t) % 360} 72% 64%)`)
      continue
    }
    taken.add(slot)
    out.set(t.toLocaleLowerCase(), PALETTE[slot])
  }
  assigned = out
}

/** A universe's colour (stable per name; see `assignColors`). */
export function universeColor(tag: string): string {
  return assigned.get(tag.toLocaleLowerCase()) ?? PALETTE[hash(tag) % PALETTE.length]
}

export interface Universe {
  tag: string
  /** images wearing it overall (the server's count) */
  count: number
  color: string
  /** its members among the gallery rows, newest first */
  rels: string[]
}

const same = (a: string, b: string) => a.toLocaleLowerCase() === b.toLocaleLowerCase()

/** Every universe in use, with its members, plus a rel -> row lookup. */
export function useUniverses(enabled = true) {
  const { data: vocab } = useTags(enabled)
  const { data: images } = useImages(false)
  return useMemo(() => {
    const byRel = new Map<string, ImageItem>()
    for (const list of Object.values(images ?? {})) for (const it of list) if (!byRel.has(it.rel)) byRel.set(it.rel, it)
    const members = new Map<string, string[]>()
    for (const it of byRel.values())
      for (const t of it.tags ?? []) {
        const k = t.toLocaleLowerCase()
        const arr = members.get(k)
        if (arr) arr.push(it.rel)
        else members.set(k, [it.rel])
      }
    assignColors((vocab?.tags ?? []).map((c) => c.tag))
    const list: Universe[] = (vocab?.tags ?? []).map((c) => ({
      tag: c.tag,
      count: c.count,
      color: universeColor(c.tag),
      rels: members.get(c.tag.toLocaleLowerCase()) ?? [],
    }))
    return { list, byRel }
  }, [vocab, images])
}

/** Does this row wear `tag`? */
export const wears = (im: ImageItem | undefined, tag: string) => !!im?.tags?.some((t) => same(t, tag))

/* ------------------------------------------------- 🪐 sampling batches --- */

export interface SampleBatch {
  jobId: number
  /** what it sampled: a universe name, or "N selected images" */
  name: string
  /** members it was given */
  members: number
  /** client clock (s) when it was queued: map freshness keys on it */
  queuedAt: number
  /** dismissed from the notice */
  seen?: boolean
}

interface Batches {
  batches: SampleBatch[]
  add: (b: SampleBatch) => void
  dismiss: (jobId: number) => void
}

/**
 * Session-only (not in the persisted `useUI`): a notice about a job from last
 * week is noise, and every set() on the persisted store is a prefs POST.
 */
export const useSampleBatches = create<Batches>()((set) => ({
  batches: [],
  add: (b) => set((s) => ({ batches: [b, ...s.batches.filter((x) => x.jobId !== b.jobId)].slice(0, 8) })),
  dismiss: (jobId) =>
    set((s) => ({ batches: s.batches.map((b) => (b.jobId === jobId ? { ...b, seen: true } : b)) })),
}))

/**
 * The newest un-dismissed batch, its job's status, and the images it made.
 *
 * "Made by that job" = a generated image written while the job ran. The job
 * queue runs one GPU job at a time per lane, so that window is the job's own
 * output; the sidecars also say `kind: "collection"`, but the gallery rows
 * don't carry the kind and this needs no extra request.
 */
export function useLatestBatch() {
  const batch = useSampleBatches((s) => s.batches.find((b) => !b.seen))
  const { data: state } = useJobState()
  const { data: images } = useImages(false)
  const job = batch ? state?.jobs.find((j) => j.id === batch.jobId) : undefined
  const made = useMemo(() => {
    if (!job?.started) return [] as ImageItem[]
    const lo = job.started - 1
    const hi = (job.ended ?? Infinity) + 2
    return (images?.generated ?? []).filter((im) => im.mtime >= lo && im.mtime <= hi)
  }, [job?.started, job?.ended, images])
  return { batch, job, made }
}

/** The backend an image's name says made it (anarchy_<backend>_…), or null. */
export function backendOfRel(rel: string): string | null {
  return /anarchy_(sd15|sd2|sdxl|flux2|krea2)_/.exec(rel)?.[1] ?? null
}

/**
 * A universe can hold several backends' images, but one sampling job renders
 * through one model: keep the commonest backend (unknown names ride along —
 * the server resolves upscales back to their source) and count the rest.
 */
export function oneBackend(rels: string[]): { backend: string | null; rels: string[]; skipped: number } {
  const counts = new Map<string, number>()
  for (const r of rels) {
    const b = backendOfRel(r)
    if (b) counts.set(b, (counts.get(b) ?? 0) + 1)
  }
  if (counts.size <= 1) return { backend: [...counts.keys()][0] ?? null, rels, skipped: 0 }
  const backend = [...counts.entries()].sort((a, b) => b[1] - a[1])[0][0]
  const keep = rels.filter((r) => {
    const b = backendOfRel(r)
    return !b || b === backend
  })
  return { backend, rels: keep, skipped: rels.length - keep.length }
}

/** How a spread number reads, in words (the slider's live readout). */
export function spreadWords(v: number): string {
  if (v <= 0.05) return 'on the members themselves'
  if (v < 0.4) return 'close to the members'
  if (v < 0.85) return 'inside the universe'
  if (v <= 1.15) return 'across its whole range'
  if (v <= 2) return 'past its edges'
  return 'far past its edges'
}
