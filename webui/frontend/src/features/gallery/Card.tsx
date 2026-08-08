import { useExplore, useFavorite, useRefine } from '../../api/queries'
import type { ImageItem } from '../../api/types'
import { cn, fmtSize, imgSrc } from '../../lib/utils'
import { useUI } from '../../store'
import { HIRES_DENOISE, HIRES_FACTOR, refinePromptFor } from '../refine/RefineBar'

const numOr = (v: string | undefined, dflt: number) =>
  (v ?? '').trim() === '' ? dflt : Number(v)

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

  const favorite = useFavorite()
  const refine = useRefine()
  const explore = useExplore()

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
        <div className="flex flex-wrap items-center gap-x-1.5 gap-y-1">
          {im.score != null ? (
            <span className="text-score" title="aesthetic score">
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
              <button
                className={cn('sa-btn sa-btn-sm', inTimeline > 0 && 'border-run text-run')}
                title="add as a keyframe to the 🎬 Timeline"
                onClick={() => addKeyframe(im.rel)}
              >
                🎬{inTimeline > 1 ? `×${inTimeline}` : ''}
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
