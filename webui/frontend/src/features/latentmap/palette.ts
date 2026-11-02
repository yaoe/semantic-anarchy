/** Colours and labels shared by the map scene's React shell and its controls. */
import type { MapNode } from '../../api/types'

/** What produced a node: the explore mode if it is a child, else its sampler. */
export function originOf(n: MapNode): string {
  if (n.kind === 'explore') return n.mode ?? 'explore'
  if (n.kind === 'evolve') return 'evolve'
  return n.sampler ?? 'other'
}

export const ORIGIN_COLORS: Record<string, string> = {
  diagonal: '#ff7a5c',
  pca: '#8fe08a',
  blend: '#e6c86e',
  split: '#f0a3e0',
  hybrid: '#6fc3f0',
  neighborhood: '#c39cf0',
  walk: '#ffe08a',
  breed: '#f08ac8',
  evolve: '#7ef0d8',
}

export const RAMPS = {
  taste: ['#3d4a6b', '#c9a56a', '#ffcf4d'],
  age: ['#3d4a6b', '#9aa7c6', '#fff1d6'],
}

export const LEGEND: Record<string, [string, string]> = {
  taste: ['unlike your ★', 'like your ★'],
  age: ['oldest', 'newest'],
}

export const UNCLAIMED = '#5a5f6e'

/** A stable, well-separated colour per region id (golden-angle hue steps). */
export function regionColor(id: number): string {
  if (id < 0) return UNCLAIMED
  return `hsl(${(id * 137.508 + 20) % 360} 80% 66%)`
}

/** A corpus prompt, cut to a label: first clause, no "#123" row tag. */
export function shortPrompt(p: string, max = 26): string {
  const s = p.replace(/\s*#\d+$/, '').split(',')[0].trim()
  return s.length > max ? s.slice(0, max - 1) + '…' : s
}
