import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { useQueryClient } from '@tanstack/react-query'

import {
  qk,
  useAnimate,
  useDeleteImage,
  useFavorite,
  useGenPrompt,
  useInvert,
  useJobWatcher,
  useMeta,
  useScoreImage,
  useTagImage,
  useTags,
} from '../../api/queries'
import type { ImageItem, ImageMeta, MotionLora } from '../../api/types'
import { AnimateIcon, DeleteIcon, TimelineIcon } from '../../components/ui/actionIcons'
import { imgSrc, numOr } from '../../lib/utils'
import { ANIM_DENOISE, ANIM_MAX_FRAMES, WAN_FRAMES } from '../animate/AnimateBar'
import { useUI } from '../../store'
import { ScoreStrip } from '../labeling/ScoreStrip'
import { TagStrip } from '../labeling/TagStrip'

/** Field order of the param dump, copied from the legacy PARAM_ORDER. */
const PARAM_ORDER = [
  'kind', 'mode', 'parent', 'parent_b', 'distance', 'anchor_distance', 'radius',
  'mutate', 'direction', 'step', 'walk_frame', 'target_distance', 'elites',
  'base_blend', 'dist', 'backend', 'model', 'sampler', 'temperature', 'coherence',
  'components', 'comp_lo', 'equalize', 'truncation',
  'rho', 'length_mode', 'length', 'empirical_head', 'temp_on', 'temp_off',
  'radius_band', 'radius_scale', 'steps', 'guidance',
  'scheduler', 'neg_mode', 'negative', 'height', 'width', 'init_image', 'init_mode',
  'init_strength', 'ip_scale', 'batch_seed', 'image_seed', 'index', 'refined_from',
  'cond_from', 'engine', 'factor', 'denoise', 'denoise_steps', 'interp',
  'scale', 'strength', 'cond_reused', 'src_size', 'out_size', 'seed',
]

const LINKED = new Set(['parent', 'parent_b', 'refined_from', 'cond_from'])

/** The "how would I reproduce this from the CLI" line. */
function buildCli(m: ImageMeta): string {
  if (m.kind === 'generate') {
    let s =
      `generate.py --backend ${m.backend} --sampler ${m.sampler} ` +
      `--temperature ${m.temperature} --seed ${m.batch_seed} ` +
      `--steps ${m.steps} --guidance ${m.guidance}`
    if (m.scheduler && m.scheduler !== 'default') s += ` --scheduler ${m.scheduler}`
    if (m.coherence != null) s += ` --coherence ${m.coherence}`
    if (m.rho) s += ` --rho ${m.rho}`
    // `length` is per image, so a corpus-mode batch reproduces as a pin.
    if (m.length_mode) s += ` --length-mode fixed --length ${m.length}`
    if (m.empirical_head) s += ` --empirical-head ${m.empirical_head}`
    if (m.sampler === 'split') s += ` --temp-on ${m.temp_on} --temp-off ${m.temp_off}`
    if (m.radius_band != null) s += ` --target-distance ${m.radius_band}`
    return s
  }
  if (m.kind === 'refine' && m.engine === 'hires')
    return (
      `upscale.py --src ${m.refined_from ?? '<orig>'} --factor ${m.factor} ` +
      `--denoise ${m.denoise} --interp ${m.interp}`
    )
  if (m.kind === 'refine' && m.engine === 'flux2-klein')
    return `refine_flux.py --src ${m.refined_from ?? '<orig>'} --scale ${m.scale} --steps ${m.steps}`
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
  const cursor = useUI((s) => s.lightboxRel)
  const openLightbox = useUI((s) => s.openLightbox)
  const close = useUI((s) => s.closeLightbox)
  const open = cursor !== null

  // Where the open image sits in the view RIGHT NOW. The view reorders under us
  // (new renders land at the front), so the position is looked up, never stored.
  const found = useMemo(
    () => (cursor === null ? -1 : items.findIndex((im) => im.rel === cursor)),
    [items, cursor],
  )

  // An image can leave the view while open (deleted, or tagged out of the tag
  // filter). Show whatever took its place: remember where it last was.
  const lastAt = useRef(0)
  if (found >= 0) lastAt.current = found
  const at = found >= 0 ? found : Math.min(lastAt.current, items.length - 1)
  const item = items[at] as ImageItem | undefined

  // A parent link can point at an image outside the current view; when it does
  // we pin that rel until the user navigates again.
  const [pinned, setPinned] = useState<string | null>(null)
  useEffect(() => setPinned(null), [cursor])

  // Move the cursor onto the replacement, or close if nothing is left.
  const replacement = found < 0 ? item?.rel : undefined
  useEffect(() => {
    if (!open || found >= 0) return
    if (replacement) openLightbox(replacement)
    else close()
  }, [open, found, replacement, openLightbox, close])

  const move = useCallback(
    (delta: number) => {
      const next = items[at + delta]
      if (next) openLightbox(next.rel)
    },
    [items, at, openLightbox],
  )

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

  // The judgement comes from /api/meta rather than the gallery row, because a
  // parent link can pin an image that isn't in the current view at all -- and
  // toggling a universe needs the CURRENT set, not an empty fallback.
  const { data: vocab } = useTags(open)
  const tools = useUI((s) => s.tools)
  const animate = useAnimate()
  const favorite = useFavorite()
  const del = useDeleteImage()
  const scoreImage = useScoreImage()
  const tagImage = useTagImage()
  const tagInputRef = useRef<HTMLInputElement>(null)
  const labelScore = meta?.label_score ?? null
  const labelTags = meta?.label_tags ?? []
  const fav = item?.fav ?? false

  const pickScore = useCallback(
    (n: number) => {
      if (rel) scoreImage.mutate({ rel, score: n })
    },
    [rel, scoreImage],
  )
  const toggleStar = useCallback(() => {
    if (rel) favorite.mutate({ rel, on: !fav })
  }, [rel, fav, favorite])
  const applyTags = useCallback(
    (tags: string[]) => {
      if (rel) tagImage.mutate({ rel, tags })
    },
    [rel, tagImage],
  )

  // Held in refs for the same reason the label page does it: the listener is
  // attached once per open, not re-bound on every score or cursor move.
  const scoreRef = useRef(pickScore)
  const starRef = useRef(toggleStar)
  const tagRef = useRef(applyTags)
  const tagsRef = useRef(labelTags)
  const vocabRef = useRef(vocab?.tags)
  const moveRef = useRef(move)
  moveRef.current = move
  scoreRef.current = pickScore
  starRef.current = toggleStar
  tagRef.current = applyTags
  tagsRef.current = labelTags
  vocabRef.current = vocab?.tags

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => {
      // Typing a universe name must not score the image or flip to the next one.
      const t = e.target as HTMLElement | null
      if (t && /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName)) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      // Same bindings as the 🏷 label page, so the judgement is muscle memory
      // wherever it is made: shift+<n> is a universe, a bare digit is a score.
      if (e.shiftKey && /^Digit[1-9]$/.test(e.code)) {
        e.preventDefault()
        const cell = vocabRef.current?.[Number(e.code.slice(5)) - 1]
        const worn = tagsRef.current
        if (cell) {
          const on = worn.some((x) => x.toLocaleLowerCase() === cell.tag.toLocaleLowerCase())
          tagRef.current(
            on
              ? worn.filter((x) => x.toLocaleLowerCase() !== cell.tag.toLocaleLowerCase())
              : [...worn, cell.tag],
          )
        }
        return
      }
      if (e.shiftKey) return
      if (e.key === 'ArrowLeft') {
        e.preventDefault()
        moveRef.current(-1)
      } else if (e.key === 'ArrowRight') {
        e.preventDefault()
        moveRef.current(1)
      } else if (e.key >= '0' && e.key <= '9') {
        e.preventDefault()
        scoreRef.current(Number(e.key))
      } else if (e.key === 's') {
        e.preventDefault()
        starRef.current()
      } else if (e.key === 't') {
        e.preventDefault()      // else the 't' lands in the box we just focused
        tagInputRef.current?.focus()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

  const wanEngine = tools.anEngine === 'wan'
  const doAnimate = () => {
    if (!rel) return
    animate.mutate(
      wanEngine
        ? {
            src: rel,
            engine: 'wan',
            wan_model: (tools.wnModel || '2.2-a14b') as '2.2-a14b' | '2.1-14b',
            frames: numOr(tools.wnFrames, WAN_FRAMES),
            // blank = the engine's own default (6 with Lightning, 30 without)
            steps: (tools.wnSteps ?? '').trim() === '' ? null : Number(tools.wnSteps),
            lightning: tools.wnLightning === '1',
            size: (tools.wnSize ?? '').trim() === '' ? null : Number(tools.wnSize),
            quant: (tools.wnQuant || 'nf4') as 'nf4' | 'fp8',
            guidance: numOr(tools.wnGuidance, 1),
            prompt: (tools.wnPrompt ?? '').trim() || null,
            seed: (tools.wnSeed ?? '').trim() === '' ? null : Number(tools.wnSeed),
            smooth_fps: numOr(tools.anSmooth, 0),
          }
        : {
            src: rel,
            engine: 'animatediff',
            strength: tools.anSource === 'cond' ? 0 : numOr(tools.anDenoise, ANIM_DENOISE),
            frames: numOr(tools.anFrames, ANIM_MAX_FRAMES),
            smooth_fps: numOr(tools.anSmooth, 0),
            motion_lora: (tools.anLora || 'none') as MotionLora,
            prompt: (tools.anPrompt ?? '').trim() || null,
            // Pixels mode truncates the denoise schedule; FreeInit needs all of it.
            free_init: tools.anSource === 'cond' ? numOr(tools.anFreeInit, 0) : 0,
          },
    )
  }

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
            onClick={() => move(-1)}
          >
            ‹
          </button>
          <button
            className="fixed right-3.5 top-1/2 z-[60] -translate-y-1/2 rounded-[10px] border border-line
                       bg-panel/75 px-4 py-3.5 text-[26px] leading-none text-ink"
            title="next (→)"
            onClick={() => move(1)}
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
            {/* Judge without leaving the gallery: the same strips the 🏷 label
                page uses, on the same keys. */}
            <div className="mb-2 flex flex-col gap-2 font-sans">
              <ScoreStrip
                score={labelScore}
                fav={fav}
                onPick={pickScore}
                onStar={toggleStar}
                disabled={!rel}
              />
              <TagStrip
                tags={labelTags}
                vocabulary={vocab?.tags ?? []}
                onSet={applyTags}
                inputRef={tagInputRef}
                disabled={!rel}
              />
            </div>

            <div className="mb-1 text-run">
              {rel}
              {score != null ? (
                <span className="ml-2.5 text-score" title="LAION aesthetic model">
                  aesthetic {score.toFixed(2)}
                </span>
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
              {rel && (wanEngine ? /anarchy_/.test(rel) : /anarchy_(sd15|sd2)_/.test(rel)) ? (
                <button
                  className="sa-btn sa-btn-sm"
                  title={
                    wanEngine
                      ? 'animate with Wan 2.1 (minutes) — lands in the 🎞 Films tab'
                      : 'animate with AnimateDiff (~20s) — lands in the 🎞 Films tab'
                  }
                  disabled={animate.isPending}
                  onClick={doAnimate}
                >
                  <AnimateIcon size={12} className="mr-1 inline" />
                  animate{wanEngine ? ' (Wan)' : ''}
                </button>
              ) : null}
              {rel && /anarchy_/.test(rel) ? (
                <button
                  className="sa-btn sa-btn-sm"
                  title="add as a keyframe to the 🎬 Timeline"
                  onClick={() => addKeyframe(rel)}
                >
                  <TimelineIcon size={12} className="mr-1 inline" />
                  add to timeline{inTimeline > 0 ? ` (${inTimeline})` : ''}
                </button>
              ) : null}
              {rel ? (
                <button
                  className="sa-btn sa-btn-sm hover:border-err hover:text-err"
                  title="delete image + latents from disk (no undo)"
                  disabled={del.isPending}
                  onClick={() => del.mutate(rel, { onSuccess: () => setPinned(null) })}
                >
                  <DeleteIcon size={12} className="mr-1 inline" />
                  delete
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

            {del.isError ? (
              <div className="text-err">{(del.error as Error).message}</div>
            ) : null}
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
