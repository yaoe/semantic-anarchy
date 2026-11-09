import type { ImageItem } from '../../api/types'
import { cn, fmtSize, imgSrc } from '../../lib/utils'
import { useUI } from '../../store'
import { ImageActions } from './ImageActions'
import { useImageActions } from './useImageActions'

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
  // The same action set, from the same hook, as the fullscreen viewer.
  const { actions, feedback, error } = useImageActions(im)

  return (
    <div className="sa-panel group relative flex h-full flex-col overflow-hidden">
      <button
        className="block w-full cursor-zoom-in bg-black p-0"
        onClick={() => openLightbox(im.rel, 'gallery')}
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
      <ImageActions variant="card" actions={actions} feedback={feedback} error={error} rel={im.rel} />
    </div>
  )
}
