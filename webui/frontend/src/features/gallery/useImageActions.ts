/**
 * Every per-image action, defined ONCE.
 *
 * The gallery card and the fullscreen viewer render the same list from the same
 * hook, so a new action (or a changed rule for when one applies) shows up in
 * both. The toolbars above the grid (ExploreBar / RefineBar / AnimateBar) hold
 * the *settings*; these buttons are where they get submitted.
 *
 *   useImageActions(im)  -> ordered action defs + the last job/error feedback
 *   <ImageActions ...>   -> (ImageActions.tsx) renders them, as a card's icon
 *                           row or the viewer's labelled toolbar
 *
 * The viewer also binds each action's `hotkey` (see Lightbox.tsx).
 */
import { useState, type ComponentType } from 'react'

import {
  useAnimate,
  useDeleteImage,
  useExplore,
  useFavorite,
  useRefine,
  useRestoreConditions,
} from '../../api/queries'
import type { ImageItem, MotionLora, RunResponse } from '../../api/types'
import {
  AnimateIcon,
  BreedIcon,
  DeleteIcon,
  ExploreIcon,
  FitAddIcon,
  FitSelectedIcon,
  RestoreIcon,
  StarIcon,
  TimelineIcon,
  UniverseIcon,
  UpscaleIcon,
  WalkIcon,
} from '../../components/ui/actionIcons'
import { basename, numOr } from '../../lib/utils'
import { useUI } from '../../store'
import { ANIM_DENOISE, ANIM_MAX_FRAMES, WAN_FRAMES } from '../animate/AnimateBar'
import { HIRES_DENOISE, HIRES_FACTOR, refinePromptFor } from '../refine/RefineBar'

export type ActionId =
  | 'fav' | 'restore' | 'explore' | 'walk' | 'animate'
  | 'timeline' | 'fit' | 'universe' | 'breed' | 'upscale' | 'delete'

export interface ImageAction {
  id: ActionId
  /** Short text for the viewer's labelled toolbar. */
  label: string
  /** What it does — the tooltip (the viewer appends the hotkey). */
  title: string
  /** Bare key the viewer binds; never a digit, `t`, or an arrow (taken). */
  hotkey?: string
  Icon: ComponentType<{ size?: number; className?: string; fill?: string }>
  run: () => void
  disabled: boolean
  /** Toggled on: starred / in the timeline / in the fit / breed parent A. */
  on: boolean
  /** How `on` (or a just-queued job) is drawn. */
  tone: 'star' | 'run' | 'sel' | 'danger'
  /** A job from THIS image was just queued by this button. */
  done: boolean
  /**
   * Opens a menu instead of running: the card renders the universe picker
   * popover for it. The viewer skips it — its tag strip is right below.
   */
  menu?: 'universe'
}

/** The last thing a button did: "queued as job #12" or why it failed. */
export interface ActionFeedback {
  ok: boolean
  text: string
  /** The image it was about (the viewer may have moved on since). */
  rel: string
  action: ActionId
}

export function useImageActions(
  im: ImageItem | null,
  opts: { onRestored?: () => void; onDeleted?: () => void } = {},
) {
  const tools = useUI((s) => s.tools)
  const breedParent = useUI((s) => s.breedParent)
  const setBreedParent = useUI((s) => s.setBreedParent)
  const addKeyframe = useUI((s) => s.addKeyframe)
  const rel = im?.rel ?? null
  const inTimeline = useUI((s) => (rel ? s.timeline.filter((r) => r === rel).length : 0))
  const toggleFitSel = useUI((s) => s.toggleFitSel)
  const inFitSel = useUI((s) => (rel ? s.fitSel.includes(rel) : false))

  const favorite = useFavorite()
  const refine = useRefine()
  const explore = useExplore()
  const animate = useAnimate()
  const del = useDeleteImage()
  const restore = useRestoreConditions()

  const [feedback, setFeedback] = useState<ActionFeedback | null>(null)

  if (!im || !rel) return { actions: [] as ImageAction[], feedback, error: null, breedParent }

  const name = im.name
  const isAnarchy = name.startsWith('anarchy_')
  const n = numOr(tools.exN, 6)

  /** mutate() callbacks that turn a queued job into a feedback line. */
  const queued = (action: ActionId, what: string, about = rel) => ({
    onSuccess: (r: RunResponse) =>
      setFeedback({ ok: true, action, rel: about, text: `${what} queued as job #${r.job_id}` }),
    onError: (e: Error) => setFeedback({ ok: false, action, rel: about, text: `${what}: ${e.message}` }),
  })
  const failed = (action: ActionId, what: string) => ({
    onError: (e: Error) => setFeedback({ ok: false, action, rel, text: `${what}: ${e.message}` }),
  })
  const done = (action: ActionId) => !!feedback?.ok && feedback.action === action && feedback.rel === rel

  const doRefine = () => {
    // hires owns its own two knobs and derives everything else (steps, guidance,
    // scheduler, seed) from the source image's sidecar — nothing else to send.
    if (tools.rfEngine === 'hires') {
      return refine.mutate(
        {
          src: rel,
          engine: 'hires',
          scale: numOr(tools.rfFactor, HIRES_FACTOR),
          strength: numOr(tools.rfDenoise, HIRES_DENOISE),
          tiled: false,
        },
        queued('upscale', 'upscale (hires)'),
      )
    }
    refine.mutate(
      {
        src: rel,
        scale: numOr(tools.rfScale, 1.5),
        steps: (tools.rfSteps ?? '').trim() === '' ? null : Number(tools.rfSteps),
        strength: numOr(tools.rfStrength, 0.45),
        scheduler: tools.rfSched || null,
        tiled: tools.rfMode === 'tiled',
        engine: tools.rfEngine === 'sd' ? 'sd' : 'flux',
        prompt: refinePromptFor(tools),
      },
      queued('upscale', `upscale (${tools.rfEngine === 'sd' ? 'sd' : 'flux'})`),
    )
  }

  // AnimateDiff rides the SD1.5 UNet, so it needs an sd15/sd2 source (an
  // upscale of one qualifies — the server follows the refine chain back to the
  // conditioning). Wan only reads pixels, so anything goes.
  const wan = tools.anEngine === 'wan'
  const animatable = wan || /anarchy_(sd15|sd2)_/.test(name)

  const doAnimate = () =>
    animate.mutate(
      wan
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
      queued('animate', `animate (${wan ? 'Wan' : 'AnimateDiff'}) — lands in 🎞 Films`),
    )

  const doExplore = () =>
    explore.mutate(
      {
        src: rel, mode: 'neighborhood', radius: numOr(tools.exRadius, 0.3), n,
        // The noise seed carries as much of an image's look as its conditioning
        // (measured: CLIP 0.67 with a fresh seed vs 0.99 with the parent's), so
        // 'keep composition' re-uses it: same layout, perturbed meaning.
        stay_local: tools.exKeep === '1',
      },
      queued('explore', `explore ×${n}`),
    )

  const walkDir = (['outward', 'random', 'drift'] as const).find((d) => d === tools.wkDir) ?? 'outward'
  const doWalk = () =>
    explore.mutate(
      {
        src: rel,
        mode: 'walk',
        direction: walkDir,
        step: numOr(tools.wkStep, 0.15),
        ...(walkDir === 'drift' ? { turn: numOr(tools.wkTurn, 0.35) } : {}),
        n,
      },
      queued('walk', `walk (${walkDir}) ×${n}`),
    )

  const doBreed = () => {
    if (breedParent === rel) return setBreedParent(null) // again on A = cancel
    if (breedParent === null) return setBreedParent(rel)
    explore.mutate(
      { src: breedParent, b: rel, mode: 'breed', mutate: 0.15, n },
      queued('breed', `breed ${basename(breedParent)} × ${name} ×${n}`),
    )
    setBreedParent(null)
  }

  const all: (ImageAction & { show: boolean })[] = [
    {
      id: 'fav', show: true, hotkey: 's', Icon: StarIcon, tone: 'star',
      label: im.fav ? 'starred' : 'star',
      title: im.fav ? 'remove from favorites' : 'add to favorites (feeds the taste loop)',
      on: im.fav, done: false, disabled: favorite.isPending,
      run: () => favorite.mutate({ rel, on: !im.fav }, failed('fav', 'star')),
    },
    {
      id: 'restore', show: isAnarchy, hotkey: 'r', Icon: RestoreIcon, tone: 'run',
      label: 'use settings',
      title: 'reset the sidebar to the conditions that generated this image',
      on: false, done: false, disabled: restore.isPending,
      run: () =>
        restore.mutate(rel, { ...failed('restore', 'restore'), onSuccess: () => opts.onRestored?.() }),
    },
    {
      id: 'explore', show: isAnarchy, hotkey: 'e', Icon: ExploreIcon, tone: 'run',
      label: 'explore',
      title: `explore: render ${n} variations around this image`,
      on: false, done: done('explore'), disabled: explore.isPending, run: doExplore,
    },
    {
      id: 'walk', show: isAnarchy, hotkey: 'w', Icon: WalkIcon, tone: 'run',
      label: 'walk',
      title:
        walkDir === 'drift'
          ? 'walk: a curving path from this image (the heading turns every frame)'
          : walkDir === 'random'
            ? 'walk: a straight line in a random on-manifold direction'
            : 'walk: step outward from this image, away from the corpus center',
      on: false, done: done('walk'), disabled: explore.isPending, run: doWalk,
    },
    {
      id: 'animate', show: isAnarchy && animatable, hotkey: 'a', Icon: AnimateIcon, tone: 'run',
      label: wan ? 'animate (Wan)' : 'animate',
      title: wan
        ? 'animate with Wan (minutes) — result lands in 🎞 Films'
        : 'animate with AnimateDiff (~20s) — result lands in 🎞 Films',
      on: false, done: done('animate'), disabled: animate.isPending, run: doAnimate,
    },
    {
      id: 'timeline', show: isAnarchy, hotkey: 'k', Icon: TimelineIcon, tone: 'run',
      label: inTimeline > 0 ? `keyframe ×${inTimeline}` : 'keyframe',
      title:
        inTimeline > 0
          ? `add as a keyframe to the 🎬 Timeline (already in it ×${inTimeline})`
          : 'add as a keyframe to the 🎬 Timeline',
      on: inTimeline > 0, done: false, disabled: false, run: () => addKeyframe(rel),
    },
    {
      // Hand-picking for a fit happens wherever you happen to be looking — the
      // 🧬 Fit tab shares this one selection.
      id: 'fit', show: isAnarchy, hotkey: 'f', Icon: inFitSel ? FitSelectedIcon : FitAddIcon, tone: 'run',
      label: inFitSel ? 'in fit' : 'fit',
      title: inFitSel
        ? 'remove from the 🧬 Fit selection'
        : 'add to the 🧬 Fit selection (fit a distribution on these latents)',
      on: inFitSel, done: false, disabled: false, run: () => toggleFitSel(rel),
    },
    {
      // Sorting into an aesthetic universe from wherever you are looking; the
      // card opens features/universes/UniversePicker in a popover.
      id: 'universe', show: true, Icon: UniverseIcon, tone: 'run', menu: 'universe',
      label: im.tags?.length ? im.tags.join(', ') : 'universe',
      title: im.tags?.length
        ? `universes: ${im.tags.join(', ')} — click to change`
        : 'add to an aesthetic universe (🏷 tag)',
      on: !!im.tags?.length, done: false, disabled: false, run: () => {},
    },
    {
      id: 'breed', show: isAnarchy, hotkey: 'b', Icon: BreedIcon, tone: 'sel',
      label: breedParent === rel ? 'parent A' : breedParent ? 'breed with A' : 'breed',
      title:
        breedParent === rel
          ? 'this is breed parent A — pick a second image to cross it with, or press again to cancel'
          : breedParent
            ? `breed: cross ${basename(breedParent)} (parent A) with this image`
            : 'breed: pick this, then another image (same backend), to cross the two',
      on: breedParent === rel, done: done('breed'), disabled: explore.isPending, run: doBreed,
    },
    {
      id: 'upscale', show: isAnarchy, hotkey: 'u', Icon: UpscaleIcon, tone: 'run',
      label: 'upscale',
      title: `upscale (${tools.rfEngine || 'hires'}): re-render at higher resolution`,
      on: false, done: done('upscale'), disabled: refine.isPending, run: doRefine,
    },
    {
      // No hotkey on purpose: there is no undo.
      id: 'delete', show: true, Icon: DeleteIcon, tone: 'danger',
      label: 'delete',
      title: 'delete image + latents from disk (no undo)',
      on: false, done: false, disabled: del.isPending,
      run: () =>
        del.mutate(rel, {
          onSuccess: () => {
            setFeedback({ ok: true, action: 'delete', rel, text: `deleted ${name}` })
            opts.onDeleted?.()
          },
          onError: failed('delete', 'delete').onError,
        }),
    },
  ]

  const actions: ImageAction[] = all.filter((a) => a.show)
  const error = feedback && !feedback.ok && feedback.rel === rel ? feedback.text : null
  return { actions, feedback, error, breedParent }
}
