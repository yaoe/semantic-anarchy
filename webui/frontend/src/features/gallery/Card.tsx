import { useAnimate, useDeleteImage, useExplore, useFavorite, useRefine } from '../../api/queries'
import type { ImageItem, MotionLora } from '../../api/types'
import {
  AnimateIcon,
  BreedIcon,
  DeleteIcon,
  ExploreIcon,
  FitAddIcon,
  FitSelectedIcon,
  StarIcon,
  TimelineIcon,
  UpscaleIcon,
  WalkIcon,
} from '../../components/ui/actionIcons'
import { cn, fmtSize, imgSrc, numOr } from '../../lib/utils'
import { useUI } from '../../store'
import { ANIM_DENOISE, ANIM_MAX_FRAMES, WAN_FRAMES } from '../animate/AnimateBar'
import { HIRES_DENOISE, HIRES_FACTOR, refinePromptFor } from '../refine/RefineBar'

const ICON = 14

/** A square-ish icon button that shrinks to share one line with its siblings. */
const ACTION =
  'sa-btn sa-btn-sm min-w-0 max-w-[34px] flex-1 overflow-hidden whitespace-nowrap px-0 py-0 ' +
  'inline-flex h-[22px] items-center justify-center'

/**
 * One gallery tile: the image, and one line of per-image actions (favorite /
 * upscale / explore / walk / breed / delete). Name, metrics and tags appear over
 * the image on hover, so the tile is pixels first. The toolbars above the grid
 * hold the *settings*; this is where they get submitted.
 */
export function Card({
  im,
  showNovRes,
  sheet = false,
}: {
  im: ImageItem
  showNovRes?: boolean
  sheet?: boolean
}) {
  const openLightbox = useUI((s) => s.openLightbox)
  const tools = useUI((s) => s.tools)
  const breedParent = useUI((s) => s.breedParent)
  const setBreedParent = useUI((s) => s.setBreedParent)
  const addKeyframe = useUI((s) => s.addKeyframe)
  const inTimeline = useUI((s) => s.timeline.filter((r) => r === im.rel).length)
  const toggleFitSel = useUI((s) => s.toggleFitSel)
  const inFitSel = useUI((s) => s.fitSel.includes(im.rel))

  const favorite = useFavorite()
  const refine = useRefine()
  const explore = useExplore()
  const animate = useAnimate()
  const del = useDeleteImage()

  const isAnarchy = im.name.startsWith('anarchy_')
  const n = numOr(tools.exN, 6)

  const doRefine = () => {
    // hires owns its own two knobs and derives everything else (steps, guidance,
    // scheduler, seed) from the source image's sidecar — nothing else to send.
    if (tools.rfEngine === 'hires') {
      return refine.mutate({
        src: im.rel,
        engine: 'hires',
        scale: numOr(tools.rfFactor, HIRES_FACTOR),
        strength: numOr(tools.rfDenoise, HIRES_DENOISE),
        tiled: false,
      })
    }
    refine.mutate({
      src: im.rel,
      scale: numOr(tools.rfScale, 1.5),
      steps: (tools.rfSteps ?? '').trim() === '' ? null : Number(tools.rfSteps),
      strength: numOr(tools.rfStrength, 0.45),
      scheduler: tools.rfSched || null,
      tiled: tools.rfMode === 'tiled',
      engine: tools.rfEngine === 'sd' ? 'sd' : 'flux',
      prompt: refinePromptFor(tools),
    })
  }

  // AnimateDiff rides the SD1.5 UNet, so it needs an sd15/sd2 source (an
  // upscale of one qualifies — the server follows the refine chain back to the
  // conditioning). Wan only reads pixels, so anything goes.
  const wan = tools.anEngine === 'wan'
  const animatable = wan || /anarchy_(sd15|sd2)_/.test(im.name)

  const doAnimate = () =>
    animate.mutate(
      wan
        ? {
            src: im.rel,
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
            src: im.rel,
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

  const doExplore = () =>
    explore.mutate({
      src: im.rel,
      mode: 'neighborhood',
      radius: numOr(tools.exRadius, 0.3),
      n,
    })

  const doWalk = () =>
    explore.mutate({
      src: im.rel,
      mode: 'walk',
      direction: 'outward',
      step: numOr(tools.wkStep, 0.15),
      n,
    })

  const doBreed = () => {
    if (breedParent === im.rel) return setBreedParent(null) // click again = cancel
    if (breedParent === null) return setBreedParent(im.rel)
    explore.mutate({ src: breedParent, b: im.rel, mode: 'breed', mutate: 0.15, n })
    setBreedParent(null)
  }

  const err =
    (refine.isError && (refine.error as Error).message) ||
    (explore.isError && (explore.error as Error).message) ||
    (del.isError && (del.error as Error).message) ||
    null

  return (
    <div className="sa-panel group relative flex h-full flex-col overflow-hidden">
      <button
        className="block w-full cursor-zoom-in bg-black p-0"
        onClick={() => openLightbox(im.rel)}
      >
        <img
          loading="lazy"
          src={imgSrc(im.url, im.mtime)}
          alt={im.name}
          className={cn(
            'w-full bg-black',
            sheet ? 'h-auto' : 'aspect-square object-contain',
          )}
        />
      </button>

      {/* Your own 0-9 call is the one number worth a few pixels at all times:
          it says whether this image still needs judging. */}
      {im.label != null ? (
        <span
          className="absolute right-1 top-1 rounded bg-black/70 px-1 text-[11px] tabular-nums
                     text-ink group-hover:opacity-0"
          title="your label (0–9) — open the image to change it"
        >
          {im.label}
        </span>
      ) : null}

      {/* Everything else about the image waits for hover. It lets clicks through
          to the image underneath, except on the numbers that carry a tooltip. */}
      <div
        className="pointer-events-none absolute inset-x-0 top-0 flex flex-col gap-0.5 bg-gradient-to-b
                   from-black/85 via-black/60 to-transparent px-1.5 pb-5 pt-1 text-[11px] text-ink
                   opacity-0 transition-opacity group-hover:opacity-100"
      >
        <span className="truncate">{im.name}</span>
        <div className="flex flex-wrap items-center gap-x-1.5 tabular-nums [&>span]:pointer-events-auto">
          {im.label != null ? (
            <span className="rounded border border-line2 px-1" title="your label (0–9) — open the image to change it">
              {im.label}
            </span>
          ) : null}
          {im.score != null ? (
            <span className="text-score" title="LAION aesthetic score (not your label)">
              {im.score.toFixed(2)}
            </span>
          ) : null}
          {im.dist != null ? (
            <span className="text-dist" title="distance from corpus center">
              d{im.dist.toFixed(2)}
            </span>
          ) : null}
          {showNovRes && im.nov != null ? (
            <span className="text-nov" title="novelty (NN distance in the gallery)">
              n{im.nov.toFixed(2)}
            </span>
          ) : null}
          {showNovRes && im.res != null ? (
            <span className="text-res" title="resonance P(you'd star it)">
              r{im.res.toFixed(2)}
            </span>
          ) : null}
          <span className="ml-auto text-dim">{fmtSize(im.size)}</span>
        </div>
        {/* Aesthetic universes, assigned on the 🏷 label page. */}
        {im.tags?.length ? (
          <div className="flex flex-wrap gap-1">
            {im.tags.map((t) => (
              <span
                key={t}
                className="rounded-full border border-line2 px-1.5 text-[10px] text-tag"
                title={`aesthetic universe: ${t}`}
              >
                {t}
              </span>
            ))}
          </div>
        ) : null}
      </div>

      {/* One line, whatever the column width: the buttons share it equally and
          are icon-only, with the name of each action in its tooltip. */}
      <div className="flex items-center gap-px p-px">
        <button
          className={cn(ACTION, im.fav && 'text-star border-star')}
          title={im.fav ? 'remove from favorites' : 'add to favorites'}
          disabled={favorite.isPending}
          onClick={() => favorite.mutate({ rel: im.rel, on: !im.fav })}
        >
          <StarIcon size={ICON} fill={im.fav ? 'currentColor' : 'none'} />
        </button>

        {isAnarchy ? (
          <>
            <button
              className={ACTION}
              title="explore: render variations around this image"
              disabled={explore.isPending}
              onClick={doExplore}
            >
              <ExploreIcon size={ICON} />
            </button>
            <button
              className={ACTION}
              title="walk: step outward from this image, away from the corpus center"
              disabled={explore.isPending}
              onClick={doWalk}
            >
              <WalkIcon size={ICON} />
            </button>
            {animatable ? (
              <button
                className={ACTION}
                title={
                  wan
                    ? 'animate with Wan (minutes) — result lands in 🎞 Films'
                    : 'animate with AnimateDiff (~20s) — result lands in 🎞 Films'
                }
                disabled={animate.isPending}
                onClick={doAnimate}
              >
                <AnimateIcon size={ICON} />
              </button>
            ) : null}
            <button
              className={cn(ACTION, inTimeline > 0 && 'border-run text-run')}
              title={
                inTimeline > 0
                  ? `add as a keyframe to the 🎬 Timeline (already in it ×${inTimeline})`
                  : 'add as a keyframe to the 🎬 Timeline'
              }
              onClick={() => addKeyframe(im.rel)}
            >
              <TimelineIcon size={ICON} />
            </button>
            {/* Hand-picking for a fit happens wherever you happen to be
                looking — the 🧬 Fit tab shares this one selection. */}
            <button
              className={cn(ACTION, inFitSel && 'border-run text-run')}
              title={
                inFitSel
                  ? 'remove from the 🧬 Fit selection'
                  : 'add to the 🧬 Fit selection (fit a distribution on these latents)'
              }
              onClick={() => toggleFitSel(im.rel)}
            >
              {inFitSel ? <FitSelectedIcon size={ICON} /> : <FitAddIcon size={ICON} />}
            </button>
            <button
              className={cn(ACTION, breedParent === im.rel && 'sa-btn-sel')}
              title="breed: click this, then another image, to cross the two"
              disabled={explore.isPending}
              onClick={doBreed}
            >
              <BreedIcon size={ICON} />
            </button>
            <button
              className={cn(ACTION, refine.isSuccess && 'border-run text-run')}
              title={
                refine.isSuccess
                  ? `upscale queued as job #${refine.data.job_id} — click to queue another`
                  : 'upscale: re-render at higher resolution'
              }
              disabled={refine.isPending}
              onClick={doRefine}
            >
              <UpscaleIcon size={ICON} />
            </button>
          </>
        ) : null}

        <button
          className={cn(ACTION, 'ml-auto hover:border-err hover:text-err')}
          title="delete image + latents from disk (no undo)"
          disabled={del.isPending}
          onClick={() => del.mutate(im.rel)}
        >
          <DeleteIcon size={ICON} />
        </button>
      </div>
      {err ? <span className="truncate px-1.5 text-[11px] text-err" title={err}>{err}</span> : null}
    </div>
  )
}
