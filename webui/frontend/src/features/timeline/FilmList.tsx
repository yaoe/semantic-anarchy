import { useState } from 'react'

import { useDeleteFilm, useFilms } from '../../api/queries'
import type { Film } from '../../api/types'
import { Confirm } from '../../components/ui/Confirm'

/** "24 frames · 16 fps · 1.5s · slerp/smooth · base" */
function summary(f: Film): string {
  const bits = [
    `${f.frames ?? '?'} frames`,
    `${f.fps ?? '?'} fps`,
    f.duration != null ? `${f.duration}s` : null,
    f.interp ? `${f.interp}${f.easing ? `/${f.easing}` : ''}` : null,
    f.loop ? 'loop' : null,
    f.refine === 'flux' ? 'flux-refined' : 'base',
  ]
  return bits.filter(Boolean).join(' · ')
}

/**
 * Rendered latent-travel films, newest first, each playable inline. Shared by
 * the 🎞 Films tab (all of them) and the 🎬 Timeline tab (`limit={1}` — the
 * render you just kicked off).
 */
export function FilmList({ limit }: { limit?: number }) {
  const { data: films, isLoading } = useFilms(true)
  const del = useDeleteFilm()
  const [pending, setPending] = useState<string | null>(null)

  if (isLoading) return <div className="p-8 text-center text-dim">loading films…</div>
  if (!films?.length)
    return (
      <div className="p-8 text-center text-dim">
        no films yet — add keyframes to the 🎬 Timeline and render one.
      </div>
    )

  return (
    <div className="flex flex-col gap-3.5">
      {(limit ? films.slice(0, limit) : films).map((f) => (
        <div key={f.rel} className="sa-panel max-w-[820px] p-3">
          <div className="mb-2 flex items-center gap-2.5">
            <span className="text-run">🎞 {f.name}</span>
            <span className="text-[11px] text-dim">{summary(f)}</span>
            <button
              className="sa-btn sa-btn-sm ml-auto text-danger"
              title="delete this film's whole folder (all its variants + frames)"
              onClick={() => setPending(f.dir)}
            >
              🗑 delete
            </button>
          </div>
          <video
            controls
            loop
            className="w-full rounded-lg bg-black"
            src={`/img?path=${encodeURIComponent(f.rel)}`}
          />
          {f.keyframes.length ? (
            <div className="mt-1.5 text-[11px] text-dim">
              keyframes: {f.keyframes.map((k) => k.split('/').pop()).join(' → ')}
            </div>
          ) : null}
        </div>
      ))}
      <Confirm
        open={pending !== null}
        onOpenChange={(v) => !v && setPending(null)}
        title={`Delete film folder “${pending}”?`}
        body="All its variants and frames go too. This cannot be undone."
        confirmLabel="Delete"
        danger
        onConfirm={() => {
          if (pending) del.mutate(pending)
          setPending(null)
        }}
      />
    </div>
  )
}
