import { clsx, type ClassValue } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

/** "1.4M" / "812K" — the legacy gallery's byte formatter. */
export function fmtSize(b: number): string {
  return b > 1e6 ? `${(b / 1e6).toFixed(1)}M` : `${Math.round(b / 1e3)}K`
}

/** Cache-bust gallery thumbnails on mtime, exactly as the old UI did. */
export function imgSrc(url: string, mtime: number): string {
  return `${url}&t=${Math.floor(mtime)}`
}

export function basename(rel: string): string {
  return rel.split('/').pop() ?? rel
}

export function fmtDuration(started?: number | null, ended?: number | null): string {
  if (!started) return ''
  const end = ended ?? Date.now() / 1000
  const s = Math.max(0, Math.round(end - started))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  return `${m}m ${s % 60}s`
}


/**
 * A store value (always a string) as a number, falling back when it is blank.
 * Shared by every surface that reads knob values out of `tools` — the gallery
 * card and the lightbox both submit the same actions from the same settings.
 */
export const numOr = (v: string | undefined, dflt: number): number =>
  (v ?? '').trim() === '' ? dflt : Number(v)


/** Backends whose images carry conditioning, in the order they are shown. */
export const IMAGE_BACKENDS = ['sd15', 'sd2', 'sdxl', 'flux2', 'krea2'] as const

/**
 * Which model made this image, read off its filename (`anarchy_<backend>_...`).
 *
 * Filename rather than sidecar because callers have a rel-path and nothing else
 * — the persisted fit selection, the timeline, a favorites entry. Returns null
 * for anything not named by a generator (an upscale keeps its ancestor's name,
 * so it still answers correctly).
 */
export function backendOf(rel: string): string | null {
  const name = rel.split('/').pop() ?? rel
  for (const b of IMAGE_BACKENDS) {
    if (name.includes(`anarchy_${b}_`)) return b
  }
  return null
}
