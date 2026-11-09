import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as Dialog from '@radix-ui/react-dialog'
import { useQueryClient } from '@tanstack/react-query'

import {
  qk,
  useGenPrompt,
  useImages,
  useInvert,
  useJobWatcher,
  useMeta,
  useScoreImage,
  useTagImage,
  useTags,
} from '../../api/queries'
import type { ImageItem, ImageMeta } from '../../api/types'
import { BreedIcon } from '../../components/ui/actionIcons'
import { basename, cn, imgSrc } from '../../lib/utils'
import { useUI } from '../../store'
import { ScoreStrip } from '../labeling/ScoreStrip'
import { TagStrip } from '../labeling/TagStrip'
import { ImageActions } from './ImageActions'
import { useImageActions } from './useImageActions'
import { stubItem, useViewerLists } from './viewerLists'

/** Field order of the param dump, copied from the legacy PARAM_ORDER. */
const PARAM_ORDER = [
  'kind', 'mode', 'parent', 'parent_b', 'distance', 'anchor_distance', 'radius',
  'mutate', 'direction', 'step', 'turn', 'walk_frame', 'target_distance', 'elites',
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


const NO_ITEMS: ImageItem[] = []

/**
 * THE fullscreen image viewer — mounted once, in App, and opened from anywhere
 * with `openLightbox(rel, source)`. ←/→ walk the list registered under that
 * source (features/gallery/viewerLists.ts); the per-image actions are the same
 * `useImageActions` set the gallery card renders, with hotkeys.
 */
export function Lightbox() {
  const cursor = useUI((s) => s.lightboxRel)
  const source = useUI((s) => s.lightboxSource)
  const openLightbox = useUI((s) => s.openLightbox)
  const close = useUI((s) => s.closeLightbox)
  const open = cursor !== null
  const listed = useViewerLists((s) => (source ? s.lists[source] : undefined))

  // The live gallery rows, so whatever list the viewer was opened from (a map
  // node, a timeline keyframe) shows the current star / scores. No polling of
  // its own: the gallery's observer already refreshes this query.
  const { data: images } = useImages(false)
  const byRel = useMemo(() => {
    const m = new Map<string, ImageItem>()
    for (const list of Object.values(images ?? {})) for (const it of list) if (!m.has(it.rel)) m.set(it.rel, it)
    return m
  }, [images])
  const items = useMemo(() => {
    // Opened without a registered list: a one-image viewer.
    const base = listed ?? (cursor ? [stubItem(cursor)] : NO_ITEMS)
    return base.map((im) => byRel.get(im.rel) ?? im)
  }, [listed, cursor, byRel])

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

  // The image on screen: the pinned parent, else the cursor's row.
  const viewed = pinned ? (byRel.get(pinned) ?? stubItem(pinned)) : (item ?? null)
  const rel = viewed?.rel ?? null
  const src = viewed ? imgSrc(viewed.url, viewed.mtime) : ''

  // The param panel is fixed to the bottom and its height depends on how much
  // metadata a given image carries, so measure it and treat the viewport above
  // it as the area the image gets to fill.
  // A callback ref, not useRef: the portal mounts its children in a later commit
  // than the one that flips `open`, so an effect keyed on `open` would still see
  // a null ref and never attach the observer.
  const [panelEl, setPanelEl] = useState<HTMLDivElement | null>(null)
  // The params / CLI / prompts / universe editor fold away by default so the
  // image gets the screen. Lightbox is mounted once, so the choice holds for
  // the session across opens.
  const [details, setDetails] = useState(false)
  const [panelH, setPanelH] = useState(0)
  useEffect(() => {
    if (!panelEl) return
    const ro = new ResizeObserver(() => setPanelH(panelEl.offsetHeight))
    ro.observe(panelEl)
    setPanelH(panelEl.offsetHeight)
    return () => ro.disconnect()
  }, [panelEl])

  const { data: meta } = useMeta(open ? rel : null)
  const qc = useQueryClient()
  const invert = useInvert()
  const genprompt = useGenPrompt()
  const watch = useJobWatcher()

  // Same actions, same hook as the gallery card. Restore closes the viewer:
  // the sidebar it just changed is behind this overlay.
  // The viewer shows `feedback` whether it is a queued job or an error.
  const { actions, feedback, breedParent } = useImageActions(viewed, {
    onRestored: close,
    onDeleted: () => setPinned(null),
  })

  // The judgement comes from /api/meta rather than the gallery row, because a
  // parent link can pin an image that isn't in the current view at all -- and
  // toggling a universe needs the CURRENT set, not an empty fallback.
  const { data: vocab } = useTags(open)
  const scoreImage = useScoreImage()
  const tagImage = useTagImage()
  const tagInputRef = useRef<HTMLInputElement>(null)
  const labelScore = meta?.label_score ?? null
  const labelTags = meta?.label_tags ?? []
  const fav = viewed?.fav ?? false

  const pickScore = useCallback(
    (n: number) => {
      if (rel) scoreImage.mutate({ rel, score: n })
    },
    [rel, scoreImage],
  )
  const applyTags = useCallback(
    (tags: string[]) => {
      if (rel) tagImage.mutate({ rel, tags })
    },
    [rel, tagImage],
  )

  // Held in refs for the same reason the label page does it: the listener is
  // attached once per open, not re-bound on every score or cursor move.
  const scoreRef = useRef(pickScore)
  const tagRef = useRef(applyTags)
  const tagsRef = useRef(labelTags)
  const vocabRef = useRef(vocab?.tags)
  const moveRef = useRef(move)
  const actionsRef = useRef(actions)
  moveRef.current = move
  scoreRef.current = pickScore
  tagRef.current = applyTags
  tagsRef.current = labelTags
  vocabRef.current = vocab?.tags
  actionsRef.current = actions

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
      const key = e.key.length === 1 ? e.key.toLowerCase() : e.key
      if (key === 'ArrowLeft') {
        e.preventDefault()
        moveRef.current(-1)
      } else if (key === 'ArrowRight') {
        e.preventDefault()
        moveRef.current(1)
      } else if (key >= '0' && key <= '9' && key.length === 1) {
        e.preventDefault()
        scoreRef.current(Number(key))
      } else if (key === 't') {
        e.preventDefault()      // else the 't' lands in the box we just focused
        // The universe box lives in the folded panel: unfold, then focus it
        // once it has mounted.
        setDetails(true)
        requestAnimationFrame(() => tagInputRef.current?.focus())
      } else if (key === 'i') {
        e.preventDefault()
        setDetails((v) => !v)
      } else {
        // Every per-image action carries its own key (s star, e explore, …).
        // A held key must not queue the same job over and over.
        const a = actionsRef.current.find((x) => x.hotkey === key)
        if (a) {
          e.preventDefault()
          if (!a.disabled && !e.repeat) a.run()
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [open])

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

  const score = viewed?.score ?? null
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
          className="fixed inset-0 z-50 flex items-center justify-center p-1.5 outline-none"
          style={{ paddingBottom: panelH }}
          onClick={(e) => {
            if (e.target === e.currentTarget) close()
          }}
        >
          <Dialog.Title className="sr-only">{rel ?? 'image'}</Dialog.Title>
          <Dialog.Description className="sr-only">
            Generation parameters, per-image actions and navigation for the selected image.
          </Dialog.Description>

          {/* The arrows sit ON the image's edges rather than beside it, so the
              image can take the full width; faint until hovered. */}
          <button
            className="fixed left-1.5 top-1/2 z-[60] -translate-y-1/2 rounded-lg bg-black/30 px-2.5 py-3
                       text-[24px] leading-none text-ink opacity-40 transition-opacity hover:opacity-100
                       disabled:invisible"
            title="previous (←)"
            disabled={at <= 0}
            onClick={() => move(-1)}
          >
            ‹
          </button>
          <button
            className="fixed right-1.5 top-1/2 z-[60] -translate-y-1/2 rounded-lg bg-black/30 px-2.5 py-3
                       text-[24px] leading-none text-ink opacity-40 transition-opacity hover:opacity-100
                       disabled:invisible"
            title="next (→)"
            disabled={at >= items.length - 1}
            onClick={() => move(1)}
          >
            ›
          </button>

          {/* All of the free area (viewport minus the bottom bar, less the thin
              padding): object-contain scales the image until either axis fills. */}
          {src ? (
            <img
              src={src}
              alt={rel ?? ''}
              className="h-full w-full object-contain"
              style={{ maxHeight: `calc(100vh - ${panelH}px - 12px)` }}
            />
          ) : null}

          <div
            ref={setPanelEl}
            className="fixed inset-x-0 bottom-0 max-h-[50vh] overflow-auto border-t border-line
                       bg-logbg/[.92] px-3 py-1.5 font-mono text-[12px] leading-[1.6] text-[#cfd3dc]"
            onClick={(e) => e.stopPropagation()}
          >
            {/* The always-visible bar: every action, the 0–9 score, where you
                are in the list, and the fold toggle. One line on a wide screen. */}
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
              <div className="min-w-0 flex-1">
                <ImageActions variant="viewer" actions={actions} feedback={feedback} current={rel} />
              </div>
              <ScoreStrip compact score={labelScore} fav={fav} onPick={pickScore} disabled={!rel} />
              {items.length > 1 && !pinned ? (
                <span className="font-sans text-[12px] tabular-nums text-dim">
                  {at + 1} / {items.length}
                </span>
              ) : null}
              <button
                className={cn('sa-btn sa-btn-sm font-sans', details && 'sa-btn-sel')}
                onClick={() => setDetails((v) => !v)}
                title="params, CLI, prompts and the universe editor (i)"
                aria-expanded={details}
              >
                {details ? '▾' : '▸'} details <span className="text-[10px] text-dim">i</span>
              </button>
            </div>

            {breedParent ? (
              <div className="mt-1 font-sans text-[12px] text-dim">
                <BreedIcon size={12} className="mr-1 inline" />
                breed parent A = <b className="text-ink">{basename(breedParent)}</b>
                {breedParent === rel
                  ? ' — move to a second image (same backend) and press b, or b here again to cancel'
                  : ' — press b to cross it with this image'}
              </div>
            ) : null}

            {details ? (
              <div className="mt-2 border-t border-line pt-2">
                <div className="mb-2 font-sans">
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

                {/* discovered hard prompts (PEZ / native inversion) */}
                <div className="mt-1.5 flex flex-wrap items-center gap-2">
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
            ) : null}
          </div>

          <Dialog.Close
            className="fixed right-1.5 top-1.5 z-[60] rounded-lg bg-black/40 px-2.5 py-1 text-[14px]
                       text-ink opacity-50 transition-opacity hover:opacity-100"
            title="close (Esc)"
          >
            ✕
          </Dialog.Close>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
