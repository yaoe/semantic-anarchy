import { useEffect, useState } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { useQueryClient } from '@tanstack/react-query'

import { qk, useGenPrompt, useInvert, useJobWatcher, useMeta } from '../../api/queries'
import type { ImageItem, ImageMeta } from '../../api/types'
import { imgSrc } from '../../lib/utils'
import { useUI } from '../../store'

/** Field order of the param dump, copied from the legacy PARAM_ORDER. */
const PARAM_ORDER = [
  'kind', 'mode', 'parent', 'parent_b', 'distance', 'anchor_distance', 'radius',
  'mutate', 'direction', 'step', 'walk_frame', 'target_distance', 'elites',
  'base_blend', 'dist', 'backend', 'model', 'sampler', 'temperature', 'coherence',
  'components', 'comp_lo', 'equalize', 'truncation', 'steps', 'guidance',
  'scheduler', 'neg_mode', 'height', 'width', 'init_image', 'init_mode',
  'init_strength', 'ip_scale', 'batch_seed', 'image_seed', 'index', 'refined_from',
  'scale', 'strength', 'cond_reused', 'out_size', 'seed',
]

const LINKED = new Set(['parent', 'parent_b', 'refined_from'])

/** The "how would I reproduce this from the CLI" line. */
function buildCli(m: ImageMeta): string {
  if (m.kind === 'generate') {
    let s =
      `generate.py --backend ${m.backend} --sampler ${m.sampler} ` +
      `--temperature ${m.temperature} --seed ${m.batch_seed} ` +
      `--steps ${m.steps} --guidance ${m.guidance}`
    if (m.scheduler && m.scheduler !== 'default') s += ` --scheduler ${m.scheduler}`
    if (m.coherence != null) s += ` --coherence ${m.coherence}`
    return s
  }
  if (m.kind === 'refine')
    return (
      `refine.py --src <orig> --scale ${m.scale} --strength ${m.strength} ` +
      `--steps ${m.steps} --guidance ${m.guidance} --scheduler ${m.scheduler}`
    )
  if (m.kind === 'explore') {
    let s = `explore.py --mode ${m.mode} --src ${m.parent} --seed ${m.batch_seed}`
    s += m.mode === 'breed' ? ` --b ${m.parent_b} --mutate ${m.mutate}` : ` --radius ${m.radius}`
    return s
  }
  return ''
}

export function Lightbox({ items }: { items: ImageItem[] }) {
  const index = useUI((s) => s.lightboxIndex)
  const close = useUI((s) => s.closeLightbox)
  const move = useUI((s) => s.moveLightbox)
  const open = index >= 0

  // A parent link can point at an image outside the current view; when it does
  // we pin that rel until the user navigates again.
  const [pinned, setPinned] = useState<string | null>(null)
  useEffect(() => setPinned(null), [index])

  const item = items[index] as ImageItem | undefined
  const rel = pinned ?? item?.rel ?? null
  const src = pinned
    ? `/img?path=${encodeURIComponent(pinned)}`
    : item
      ? imgSrc(item.url, item.mtime)
      : ''

  // The param panel is fixed to the bottom and its height depends on how much
  // metadata a given image carries, so measure it and treat the viewport above
  // it as the area the image gets to fill.
  // A callback ref, not useRef: the portal mounts its children in a later commit
  // than the one that flips `open`, so an effect keyed on `open` would still see
  // a null ref and never attach the observer.
  const [panelEl, setPanelEl] = useState<HTMLDivElement | null>(null)
  const [panelH, setPanelH] = useState(0)
  useEffect(() => {
    if (!panelEl) return
    const ro = new ResizeObserver(() => setPanelH(panelEl.offsetHeight))
    ro.observe(panelEl)
    setPanelH(panelEl.offsetHeight)
    return () => ro.disconnect()
  }, [panelEl])

  const addKeyframe = useUI((s) => s.addKeyframe)
  const inTimeline = useUI((s) => s.timeline.filter((r) => r === rel).length)

  const { data: meta } = useMeta(open ? rel : null)
  const qc = useQueryClient()
  const invert = useInvert()
  const genprompt = useGenPrompt()
  const watch = useJobWatcher()

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        move(-1, items.length)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        move(1, items.length)
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open, items.length, move])

  const runInvert = (space: 'clip' | 'native') => {
    if (!rel) return
    invert.mutate(
      { src: rel, space },
      { onSuccess: (r) => watch(r.job_id, () => qc.invalidateQueries({ queryKey: qk.meta(rel) })) },
    )
  }

  const runGenPrompt = (which: 'inverted' | 'native') => {
    if (!rel) return
    genprompt.mutate(
      { src: rel, which },
      { onSuccess: (r) => watch(r.job_id, () => qc.invalidateQueries({ queryKey: qk.images })) },
    )
  }

  const score = item?.score ?? null
  const nativeable = rel ? /anarchy_(sd15|sdxl)_/.test(rel) : false
  const ordered = meta
    ? PARAM_ORDER.filter((k) => k in meta && meta[k] !== null)
    : []
  const cli = meta ? buildCli(meta) : ''

  return (
    <Dialog.Root open={open} onOpenChange={(v) => !v && close()}>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/[.92]" />
        <Dialog.Content
          className="fixed inset-0 z-50 flex items-center justify-center p-5 outline-none"
          style={{ paddingBottom: panelH }}
          onClick={(e) => {
            if (e.target === e.currentTarget) close()
          }}
        >
          <Dialog.Title className="sr-only">{rel ?? 'image'}</Dialog.Title>
          <Dialog.Description className="sr-only">
            Generation parameters and navigation for the selected image.
          </Dialog.Description>

          <button
            className="fixed left-3.5 top-1/2 z-[60] -translate-y-1/2 rounded-[10px] border border-line
                       bg-panel/75 px-4 py-3.5 text-[26px] leading-none text-ink"
            title="previous (←)"
            onClick={() => move(-1, items.length)}
          >
            ‹
          </button>
          <button
            className="fixed right-3.5 top-1/2 z-[60] -translate-y-1/2 rounded-[10px] border border-line
                       bg-panel/75 px-4 py-3.5 text-[26px] leading-none text-ink"
            title="next (→)"
            onClick={() => move(1, items.length)}
          >
            ›
          </button>

          {/* 80% of the free area (viewport minus the param panel): object-contain
              scales the image up or down until whichever axis hits the box first. */}
          {src ? (
            <img
              src={src}
              alt={rel ?? ''}
              className="w-[80vw] object-contain"
              style={{ height: `calc((100vh - ${panelH}px) * 0.8)` }}
            />
          ) : null}

          <div
            ref={setPanelEl}
            className="fixed inset-x-0 bottom-0 max-h-[38vh] overflow-auto border-t border-line
                       bg-logbg/[.92] px-4 py-2.5 font-mono text-[12px] leading-[1.6] text-[#cfd3dc]"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="mb-1 text-run">
              {rel}
              {score != null ? (
                <span className="ml-2.5 text-score">aesthetic {score.toFixed(2)}</span>
              ) : null}
            </div>

            {meta && ordered.length === 0 ? (
              <span className="text-dim">no params recorded (pre-dates param logging)</span>
            ) : null}

            {ordered.map((k) => (
              <span key={k} className="mr-3.5 inline-block">
                <span className="text-dim">{k}</span>{' '}
                {LINKED.has(k) && typeof meta?.[k] === 'string' ? (
                  <button
                    className="text-link hover:underline"
                    onClick={() => setPinned(`generated/${String(meta[k])}`)}
                  >
                    {String(meta[k])}
                  </button>
                ) : (
                  String(meta?.[k])
                )}
              </span>
            ))}

            {cli ? <div className="mt-1.5 text-score">{cli}</div> : null}

            {/* discovered hard prompts (PEZ / native inversion) + timeline */}
            <div className="mt-1.5 flex flex-wrap items-center gap-2">
              {rel && /anarchy_/.test(rel) ? (
                <button
                  className="sa-btn sa-btn-sm"
                  title="add as a keyframe to the 🎬 Timeline"
                  onClick={() => addKeyframe(rel)}
                >
                  🎬 add to timeline{inTimeline > 0 ? ` (${inTimeline})` : ''}
                </button>
              ) : null}

              {meta?.inverted_prompt !== undefined ? (
                <span className="text-pez">
                  🔤 “{meta.inverted_prompt}”{' '}
                  <span className="text-dim">
                    CLIP's eyes (PEZ, {meta.inverted_tokens} tok, sim {meta.inverted_sim})
                  </span>
                  <button
                    className="sa-btn sa-btn-sm ml-2"
                    disabled={genprompt.isPending}
                    onClick={() => runGenPrompt('inverted')}
                  >
                    🎨 generate from it
                  </button>
                </span>
              ) : (
                <button
                  className="sa-btn sa-btn-sm"
                  disabled={invert.isPending}
                  onClick={() => runInvert('clip')}
                >
                  🔤 reveal nearest prompt
                </button>
              )}

              {meta?.native_prompt !== undefined ? (
                <span className="text-native">
                  🔡 “{meta.native_prompt}”{' '}
                  <span className="text-dim">
                    model's own encoder (native, cond-cos {meta.native_sim})
                  </span>
                  <button
                    className="sa-btn sa-btn-sm ml-2"
                    disabled={genprompt.isPending}
                    onClick={() => runGenPrompt('native')}
                  >
                    🎨 generate from it
                  </button>
                </span>
              ) : nativeable ? (
                <button
                  className="sa-btn sa-btn-sm"
                  disabled={invert.isPending}
                  onClick={() => runInvert('native')}
                >
                  🔡 native prompt
                </button>
              ) : null}
            </div>

            {invert.isError ? (
              <div className="text-err">{(invert.error as Error).message}</div>
            ) : null}
            {genprompt.isError ? (
              <div className="text-err">{(genprompt.error as Error).message}</div>
            ) : null}
          </div>

          <Dialog.Close
            className="fixed right-3.5 top-3.5 z-[60] rounded-[10px] border border-line bg-panel/75
                       px-3 py-1.5 text-[14px] text-ink"
            title="close (Esc)"
          >
            ✕
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
