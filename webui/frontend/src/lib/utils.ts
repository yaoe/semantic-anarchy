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
