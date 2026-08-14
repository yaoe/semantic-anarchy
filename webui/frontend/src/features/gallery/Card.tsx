import { useAnimate, useExplore, useFavorite, useRefine } from '../../api/queries'
import type { ImageItem, MotionLora } from '../../api/types'
import { cn, fmtSize, imgSrc, numOr } from '../../lib/utils'
import { useUI } from '../../store'
import { ANIM_DENOISE, ANIM_MAX_FRAMES, WAN_FRAMES } from '../animate/AnimateBar'
import { HIRES_DENOISE, HIRES_FACTOR, refinePromptFor } from '../refine/RefineBar'

/**
 * One gallery tile: the image, its metrics, and the per-image actions
 * (favorite / upscale / explore / walk / breed). The toolbars above the grid
 * hold the *settings*; this is where they get submitted.
 */
export function Card({
  im,
  index,
  showNovRes,
  sheet = false,
}: {
  im: ImageItem
  index: number
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
    null

  return (
    <div className="sa-panel flex h-full flex-col overflow-hidden">
      <button
        className="block w-full cursor-zoom-in bg-black p-0"
        onClick={() => openLightbox(index)}
        title={im.name}
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

      <div className="flex flex-col gap-1 px-2.5 py-1.5 text-[11px] text-dim">
        <span className="truncate" title={im.rel}>
          {im.name}
        </span>
        {/* Aesthetic universes, assigned on the 🏷 label page. Shown above the
            numbers because it says what KIND of thing this is, which frames
            every score under it. */}
        {im.tags?.length ? (
          <div className="flex flex-wrap gap-1">
            {im.tags.map((t) => (
              <span
                key={t}
                className="rounded-full border border-line2 px-1.5 py-px text-[10px] text-tag"
                title={`aesthetic universe: ${t}`}
              >
                {t}
              </span>
            ))}
          </div>
        ) : null}
        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1">
          {im.label != null ? (
            <span
              className="rounded border border-line2 px-1 tabular-nums text-ink"
              title="your label (0–9) — open the image to change it"
            >
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

          <button
            className={cn('sa-btn sa-btn-sm', im.fav && 'text-star border-star')}
            title="favorite"
            disabled={favorite.isPending}
            onClick={() => favorite.mutate({ rel: im.rel, on: !im.fav })}
          >
            {im.fav ? '★' : '☆'}
          </button>

          {isAnarchy ? (
            <>
              <button
                className="sa-btn sa-btn-sm"
                title="explore around this (neighborhood)"
                disabled={explore.isPending}
                onClick={doExplore}
              >
                🧭
              </button>
              <button
                className="sa-btn sa-btn-sm"
                title="walk outward toward the periphery from this point"
                disabled={explore.isPending}
                onClick={doWalk}
              >
                🚶
              </button>
              {animatable ? (
                <button
                  className="sa-btn sa-btn-sm"
                  title={
                    wan
                      ? 'animate with Wan 2.1 (minutes) — result lands in 🎞 Films'
                      : 'animate with AnimateDiff (~20s) — result lands in 🎞 Films'
                  }
                  disabled={animate.isPending}
                  onClick={doAnimate}
                >
                  🎥
                </button>
              ) : null}
              <button
                className={cn('sa-btn sa-btn-sm', inTimeline > 0 && 'border-run text-run')}
                title="add as a keyframe to the 🎬 Timeline"
                onClick={() => addKeyframe(im.rel)}
              >
                🎬{inTimeline > 1 ? `×${inTimeline}` : ''}
              </button>
              {/* Hand-picking for a fit happens wherever you happen to be
                  looking — the 🧬 Fit tab shares this one selection. */}
              <button
                className={cn('sa-btn sa-btn-sm', inFitSel && 'border-run text-run')}
                title={
                  inFitSel
                    ? 'remove from the 🧬 Fit selection'
                    : 'add to the 🧬 Fit selection (fit a distribution on these latents)'
                }
                onClick={() => toggleFitSel(im.rel)}
              >
                {inFitSel ? '⊖' : '⊕'}
              </button>
              <button
                className={cn('sa-btn sa-btn-sm', breedParent === im.rel && 'sa-btn-sel')}
                title="breed: click this then another image"
                disabled={explore.isPending}
                onClick={doBreed}
              >
                🧬
              </button>
              <button
                className="sa-btn sa-btn-sm whitespace-nowrap"
                title="upscale + more steps"
                disabled={refine.isPending}
                onClick={doRefine}
              >
                {refine.isSuccess ? `↑ #${refine.data.job_id}` : '⤴ Upscale'}
              </button>
            </>
          ) : null}

          <span className="ml-auto">{fmtSize(im.size)}</span>
        </div>
        {err ? <span className="text-err">{err}</span> : null}
      </div>
    </div>
  )
}
