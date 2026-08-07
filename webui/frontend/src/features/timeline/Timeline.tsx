import { useState, type DragEvent } from 'react'

import { useFilm, useKeyframes } from '../../api/queries'
import type { EasingId, InterpId, KeyframeRow } from '../../api/types'
import { Tool } from '../../components/ui/Tool'
import { cn } from '../../lib/utils'
import { useUI } from '../../store'
import { FilmList } from './FilmList'

const num = (v: string | undefined, dflt: number) =>
  (v ?? '').trim() === '' ? dflt : Number(v)

/** Backend of a generated image, read off its filename (as the server does). */
function backendOf(rel: string): string {
  const m = /anarchy_([a-z0-9]+)_/.exec(rel.split('/').pop() ?? '')
  return m ? m[1] : 'sd15'
}

/** "1280×768", or null when the probe hasn't answered for this keyframe. */
const sizeOf = (r?: KeyframeRow): string | null =>
  r?.width && r?.height ? `${r.width}×${r.height}` : null

/** Frames the render will produce — the same arithmetic app.py validates. */
function frameCount(keys: number, framesPer: number, loop: boolean): number {
  if (keys < 2) return 0
  return (loop ? keys : keys - 1) * framesPer + 1
}

/* ------------------------------------------------------------ keyframes --- */

/**
 * One draggable keyframe. Drag to reorder; the ‹ › buttons do the same thing
 * for touch (and for anyone who'd rather not drag).
 */
function Keyframe({
  rel,
  index,
  count,
  size,
  offSize,
  dragging,
  dropBefore,
  onDragStart,
  onDragEnd,
  onDragOver,
  onDrop,
}: {
  rel: string
  index: number
  count: number
  /** Rendered resolution, once the probe has answered. */
  size: string | null
  /** True when this keyframe won't survive the film's resolution intact. */
  offSize: boolean
  dragging: boolean
  dropBefore: boolean
  onDragStart: (e: DragEvent) => void
  onDragEnd: () => void
  onDragOver: (e: DragEvent) => void
  onDrop: (e: DragEvent) => void
}) {
  const remove = useUI((s) => s.removeKeyframe)
  const move = useUI((s) => s.moveKeyframe)
  const name = rel.split('/').pop() ?? rel

  return (
    <div
      draggable
      onDragStart={onDragStart}
      onDragEnd={onDragEnd}
      onDragOver={onDragOver}
      onDrop={onDrop}
      title={`${rel}\n(drag to reorder)`}
      className={cn(
        'relative w-[132px] shrink-0 cursor-grab rounded-lg border border-line bg-panel p-1.5',
        'active:cursor-grabbing',
        dragging && 'opacity-40',
        dropBefore && 'border-l-2 border-l-accent',
        offSize && 'border-run',
      )}
    >
      <img
        src={`/img?path=${encodeURIComponent(rel)}`}
        alt={name}
        draggable={false}
        className="aspect-square w-full rounded-md bg-black object-contain"
      />
      <span className="absolute left-2.5 top-2.5 rounded bg-black/70 px-1.5 text-[10px] text-ink">
        {index + 1}
      </span>
      <button
        className="absolute right-2.5 top-2.5 rounded bg-black/70 px-1.5 text-[11px] text-danger"
        title="remove from the timeline"
        onClick={() => remove(index)}
      >
        ✕
      </button>
      {size ? (
        <span
          className={cn(
            'absolute bottom-[34px] left-2.5 rounded bg-black/70 px-1.5 text-[10px]',
            offSize ? 'text-run' : 'text-dim',
          )}
          title={
            offSize
              ? 'made at a different resolution — it will be re-rendered at the ' +
                "film's size and won't come back exactly"
              : 'rendered resolution'
          }
        >
          {offSize ? '⚠ ' : ''}
          {size}
        </span>
      ) : null}
      <div className="mt-1 flex items-center gap-1">
        <button
          className="sa-btn sa-btn-sm"
          title="move left"
          disabled={index === 0}
          onClick={() => move(index, index - 1)}
        >
          ‹
        </button>
        <span className="min-w-0 flex-1 truncate text-center text-[10px] text-dim">
          {name.replace(/^anarchy_/, '').replace(/\.(jpe?g|png|webp)$/i, '')}
        </span>
        <button
          className="sa-btn sa-btn-sm"
          title="move right"
          disabled={index === count - 1}
          onClick={() => move(index, index + 1)}
        >
          ›
        </button>
      </div>
    </div>
  )
}

function Strip({
  framesPer,
  loop,
  rows,
  target,
}: {
  framesPer: number
  loop: boolean
  rows?: KeyframeRow[]
  /** The film's resolution as "W×H", or null while unknown. */
  target: string | null
}) {
  const timeline = useUI((s) => s.timeline)
  const move = useUI((s) => s.moveKeyframe)
  const [from, setFrom] = useState<number | null>(null)
  const [over, setOver] = useState<number | null>(null)

  // Splice semantics: the dragged keyframe takes the slot it was dropped on,
  // and everything between shuffles over — dropping past the last card appends.
  const drop = (to: number) => (e: DragEvent) => {
    e.preventDefault()
    // A drop on a card must NOT also reach the strip's own "append" handler,
    // or the keyframe gets moved twice.
    e.stopPropagation()
    const raw = from ?? e.dataTransfer.getData('text/plain')
    const src = Number(raw)
    // Ignore drags that didn't start on a keyframe (a stray image drop).
    if (raw !== '' && Number.isInteger(src)) move(src, to)
    setFrom(null)
    setOver(null)
  }
  const dragOver = (to: number) => (e: DragEvent) => {
    e.preventDefault()
    e.dataTransfer.dropEffect = 'move'
    setOver(to)
  }

  return (
    <div
      className="flex items-stretch gap-1 overflow-x-auto pb-1"
      onDragOver={(e) => e.preventDefault()}
      onDrop={drop(timeline.length)}
    >
      {timeline.map((rel, i) => (
        <div key={`${rel}#${i}`} className="flex items-center gap-1">
          <Keyframe
            rel={rel}
            index={i}
            count={timeline.length}
            size={sizeOf(rows?.[i])}
            offSize={!!target && !!sizeOf(rows?.[i]) && sizeOf(rows?.[i]) !== target}
            dragging={from === i}
            dropBefore={over === i && from !== null && from !== i}
            onDragStart={(e) => {
              e.dataTransfer.effectAllowed = 'move'
              e.dataTransfer.setData('text/plain', String(i))
              setFrom(i)
            }}
            onDragEnd={() => {
              setFrom(null)
              setOver(null)
            }}
            onDragOver={dragOver(i)}
            onDrop={drop(i)}
          />
          {i < timeline.length - 1 ? (
            <span
              className="shrink-0 text-[10px] text-dim"
              title={`${framesPer} interpolated frames`}
            >
              →<br />
              {framesPer}f
            </span>
          ) : null}
        </div>
      ))}
      {loop && timeline.length > 1 ? (
        <span
          className="flex shrink-0 items-center px-1 text-[10px] text-run"
          title={`${framesPer} frames back to keyframe 1`}
        >
          ↩ #1
        </span>
      ) : null}
    </div>
  )
}

/* -------------------------------------------------------------- the tab --- */

/**
 * 🎬 Timeline — pick keyframes anywhere in the gallery, order them here, and
 * render a latent-travel film that visits each one in turn. Every in-between
 * frame is a fresh render of interpolated conditioning (plus interpolated init
 * noise), so the keyframes come back pixel-exact and the space between them is
 * territory that never existed.
 */
export function Timeline() {
  const timeline = useUI((s) => s.timeline)
  const clear = useUI((s) => s.clearTimeline)
  const film = useUI((s) => s.film)
  const setFilm = useUI((s) => s.setFilm)
  const setTab = useUI((s) => s.setTab)
  const run = useFilm()

  const { data: rows } = useKeyframes(timeline)

  const fps = num(film.fps, 16)
  const framesPer = num(film.framesPer, 24)
  const loop = film.loop === 'on'
  const frames = frameCount(timeline.length, framesPer, loop)
  // The probe knows the truth; fall back to the filename while it's in flight.
  const backends = [...new Set(rows?.map((r) => r.backend ?? '?') ?? timeline.map(backendOf))]
  const mixed = backends.length > 1
  const ready = timeline.length >= 2 && !mixed

  // A video has ONE resolution. Default to keyframe 1's; anything else in the
  // timeline gets re-rendered at that size and can't come back as itself.
  const found = [...new Set((rows ?? []).map(sizeOf).filter(Boolean) as string[])]
  const target = film.size || sizeOf(rows?.[0]) || null
  const offSize = (rows ?? []).filter(
    (r) => sizeOf(r) && target && sizeOf(r) !== target,
  )
  const [tw, th] = target ? target.split('×').map(Number) : [null, null]

  const submit = () =>
    run.mutate({
      images: timeline,
      name: film.name.trim() || null,
      // Only pin the size when the user picked one — otherwise the script uses
      // keyframe 1's own, which is the same rule with one fewer round trip.
      height: film.size ? th : null,
      width: film.size ? tw : null,
      fps,
      frames_per: framesPer,
      interp: (film.interp || 'slerp') as InterpId,
      easing: (film.easing || 'smooth') as EasingId,
      loop,
      refine: film.refine === 'flux' ? 'flux' : 'none',
      scale: num(film.scale, 1.5),
      fixed_noise: film.fixedNoise === 'on',
      noise_window: num(film.noiseWindow, 1.0),
      film_seed: num(film.filmSeed, 42),
      steps: (film.steps ?? '').trim() === '' ? null : Number(film.steps),
      guidance: (film.guidance ?? '').trim() === '' ? null : Number(film.guidance),
    })

  return (
    <div className="flex flex-col gap-3">
      <div className="sa-panel p-3">
        <div className="mb-2 flex flex-wrap items-baseline gap-2.5">
          <span className="text-run">🎬 Keyframes</span>
          <span className="text-[11px] text-dim">
            {timeline.length === 0
              ? 'none yet — hit 🎬 on any generated image to add it here'
              : `${timeline.length} keyframe${timeline.length === 1 ? '' : 's'} · ` +
                `${frames} frames · ${(frames / fps).toFixed(1)}s @ ${fps}fps` +
                (backends.length ? ` · ${backends.join(', ')}` : '') +
                (target ? ` · ${target}` : '')}
          </span>
          {timeline.length ? (
            <button
              className="sa-btn sa-btn-sm ml-auto text-danger"
              title="empty the timeline"
              onClick={clear}
            >
              clear
            </button>
          ) : null}
        </div>

        {timeline.length ? (
          <>
            <Strip framesPer={framesPer} loop={loop} rows={rows} target={target} />
            {offSize.length ? (
              <div className="mt-2 rounded-md border border-run/50 bg-run/10 px-2.5 py-2 text-[11px] text-run">
                ⚠ {offSize.length} of {timeline.length} keyframes were made at a
                different resolution. A video has one size, so they're re-rendered
                at {target} from a differently-shaped noise draw — those keyframes
                will <b>not</b> come back exactly; the film only passes near them.
                <div className="mt-1 text-dim">
                  {offSize
                    .map((r) => `${r.rel.split('/').pop()} (${sizeOf(r)})`)
                    .join(', ')}
                </div>
                <div className="mt-1 text-dim">
                  Drop them, or pick their size under Advanced → Render size.
                </div>
              </div>
            ) : null}
          </>
        ) : (
          <div className="py-6 text-center text-[12px] text-dim">
            Browse{' '}
            <button className="text-link underline" onClick={() => setTab('generated')}>
              Generated
            </button>{' '}
            or{' '}
            <button className="text-link underline" onClick={() => setTab('favorites')}>
              ★ Favorites
            </button>{' '}
            and click 🎬 on the images you want to travel through. Order them here,
            then render.
          </div>
        )}
      </div>

      <div className="sa-panel flex flex-wrap items-end gap-2.5 px-3 py-2.5">
        <Tool
          id="filmFps"
          label="Video fps"
          tooltip="playback rate of the mp4"
          type="number"
          width="w-[90px]"
          value={film.fps}
          onChange={(v) => setFilm('fps', v)}
        />
        <Tool
          id="filmFramesPer"
          label="Frames / hop"
          tooltip="frames rendered between each pair of keyframes — each one is a full diffusion render"
          type="number"
          width="w-[110px]"
          value={film.framesPer}
          onChange={(v) => setFilm('framesPer', v)}
        />
        <Tool
          id="filmInterp"
          label="Interpolation"
          tooltip="how the conditioning blends between keyframes"
          width="w-[170px]"
          value={film.interp}
          onChange={(v) => setFilm('interp', v)}
          options={[
            { value: 'slerp', label: 'slerp (spherical)', title: 'great-circle blend — keeps the magnitude of the endpoints' },
            { value: 'lerp', label: 'lerp (linear)', title: 'straight line — dips through the washed-out interior' },
          ]}
        />
        <Tool
          id="filmEasing"
          label="Easing"
          tooltip="pacing inside each hop"
          width="w-[160px]"
          value={film.easing}
          onChange={(v) => setFilm('easing', v)}
          options={[
            { value: 'smooth', label: 'smooth (rests on keys)' },
            { value: 'smoother', label: 'smoother' },
            { value: 'linear', label: 'linear (constant)' },
          ]}
        />
        <Tool
          id="filmLoop"
          label="Loop"
          tooltip="also travel from the last keyframe back to the first"
          width="w-[110px]"
          value={film.loop}
          onChange={(v) => setFilm('loop', v)}
          options={[
            { value: '', label: 'off' },
            { value: 'on', label: 'on (seamless)' },
          ]}
        />
        <Tool
          id="filmRefine"
          label="Refine pass"
          tooltip="run every frame through FLUX.2-klein upscaling with one locked seed — much slower"
          width="w-[150px]"
          value={film.refine}
          onChange={(v) => setFilm('refine', v)}
          options={[
            { value: 'none', label: 'none (fast)' },
            { value: 'flux', label: 'FLUX klein ×1.5' },
          ]}
        />
      </div>

      <details className="sa-panel px-3 py-2">
        <summary className="cursor-pointer text-[12px] text-dim">Advanced</summary>
        <div className="flex flex-wrap items-end gap-2.5 pb-1 pt-2">
          <Tool
            id="filmName"
            label="Name"
            placeholder="auto"
            width="w-[160px]"
            value={film.name}
            onChange={(v) => setFilm('name', v)}
          />
          <Tool
            id="filmSize"
            label="Render size"
            tooltip="one resolution for the whole video. Keyframes made at another size are re-rendered here and won't be reproduced exactly."
            width="w-[180px]"
            value={film.size}
            onChange={(v) => setFilm('size', v)}
            options={[
              { value: '', label: `auto — keyframe 1 (${sizeOf(rows?.[0]) ?? '?'})` },
              ...found.map((s) => ({ value: s, label: s })),
            ]}
          />
          <Tool
            id="filmNoiseWindow"
            label="Noise window"
            tooltip="fraction of each hop the init noise travels in (centered). 1.0 = alongside the conditioning; 0.4 = composition locked at both ends, drifting only mid-hop"
            type="number"
            step={0.1}
            width="w-[120px]"
            value={film.noiseWindow}
            onChange={(v) => setFilm('noiseWindow', v)}
          />
          <Tool
            id="filmFixedNoise"
            label="Noise mode"
            tooltip="hold keyframe 1's init latent for the whole film: pure conditioning travel, composition stays put (later keyframes then won't match exactly)"
            width="w-[170px]"
            value={film.fixedNoise}
            onChange={(v) => setFilm('fixedNoise', v)}
            options={[
              { value: '', label: 'travel with keys' },
              { value: 'on', label: 'fixed (cond only)' },
            ]}
          />
          <Tool
            id="filmSteps"
            label="Steps"
            placeholder="as keyframe"
            type="number"
            width="w-[110px]"
            value={film.steps}
            onChange={(v) => setFilm('steps', v)}
          />
          <Tool
            id="filmGuidance"
            label="Guidance"
            placeholder="as keyframe"
            type="number"
            step={0.5}
            width="w-[110px]"
            value={film.guidance}
            onChange={(v) => setFilm('guidance', v)}
          />
          <Tool
            id="filmSeed"
            label="Flux seed"
            tooltip="one seed for every refined frame, so the reinterpretation doesn't flicker"
            type="number"
            width="w-[100px]"
            value={film.filmSeed}
            onChange={(v) => setFilm('filmSeed', v)}
          />
          {film.refine === 'flux' ? (
            <Tool
              id="filmScale"
              label="Refine ×"
              type="number"
              step={0.25}
              width="w-[90px]"
              value={film.scale}
              onChange={(v) => setFilm('scale', v)}
            />
          ) : null}
        </div>
      </details>

      <div className="flex flex-wrap items-center gap-2.5">
        <button
          className={cn('sa-btn', ready && 'sa-btn-sel')}
          disabled={!ready || run.isPending}
          title="render the travel and mux it to an x264 mp4"
          onClick={submit}
        >
          🎬 Render travel{frames ? ` (${frames} frames)` : ''}
        </button>
        {mixed ? (
          <span className="text-[12px] text-err">
            keyframes mix backends ({backends.join(', ')}) — their conditioning
            tensors don't share a space
          </span>
        ) : timeline.length < 2 ? (
          <span className="text-[12px] text-dim">add at least 2 keyframes</span>
        ) : (
          <span className="text-[12px] text-dim">
            every frame is a full render — {frames} of them
          </span>
        )}
        {run.isError ? (
          <span className="text-[12px] text-err">{(run.error as Error).message}</span>
        ) : null}
        {run.isSuccess ? (
          <span className="text-[12px] text-score">
            queued job #{run.data.job_id} → {run.data.name}
          </span>
        ) : null}
      </div>

      <div>
        <h2 className="sa-h2">Latest film</h2>
        <FilmList limit={1} />
      </div>
    </div>
  )
}
