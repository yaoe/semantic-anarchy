import { useState } from 'react'

import { useDistConfig } from '../../api/queries'
import type { BackendId } from '../../api/types'
import { Tip } from '../../components/ui/Tip'
import { cn } from '../../lib/utils'
import { useUI } from '../../store'
import { DistModal } from './DistModal'

const ICON: Record<string, string> = {
  base: '📚',
  evolved: '🧪',
  prompts: '📝',
  file: '🎯',
}

/**
 * Which distribution every sample is drawn from, shown in the sidebar's Sampler
 * group. The trigger opens the picker; the two lines under it are the current
 * choice and how big a fit it is.
 *
 * The choice is server-side state (webui/dist_config.json, per backend), not a
 * form value — it survives a restart and applies to every action, so it lives
 * here rather than in params/schema.ts.
 */
export function DistPicker() {
  const backend = useUI((s) => s.params.backend) as BackendId
  const modelKey = useUI((s) => s.params.model)
  const { data } = useDistConfig(backend, backend === 'sdxl' ? modelKey : null)
  const [open, setOpen] = useState(false)

  const meta = data?.meta
  return (
    <section className="sa-panel mt-2 px-3 py-2.5">
      <div className="flex items-baseline gap-2">
        <h2 className="m-0 text-[12px] uppercase tracking-[0.8px] text-dim">Distribution</h2>
        <span className="text-[11px] text-dim">{backend}</span>
        {data && !data.ready ? (
          <span className="ml-auto text-[10px] uppercase tracking-wide text-danger">
            not encoded
          </span>
        ) : null}
      </div>

      <Tip content={data?.base} side="bottom">
        <div className="mt-1.5 flex min-w-0 items-center gap-1.5">
          <span className="shrink-0">{data?.fit ? '🧬' : ICON[data?.kind ?? 'base'] ?? '📚'}</span>
          <span className={cn('min-w-0 flex-1 truncate text-[13px]', !data?.ready && 'text-danger')}>
            {data?.label ?? '—'}
          </span>
        </div>
      </Tip>
      <p className="sa-hint">
        {!meta
          ? 'no fit on disk yet for this checkpoint'
          : data?.fit
            // A selection fit's samples are images, not prompts, and the
            // checkpoint that encoded them is recorded per image.
            ? `fitted from ${data.fit.n_samples ?? meta.n_samples} images · ` +
              `${meta.feature_shape.join('×')}` +
              (data.fit.models?.length ? ` · ${data.fit.models.join(', ')}` : '')
            : `${meta.n_samples} prompts · ${meta.feature_shape.join('×')} · encoded with ${data?.model.name}`}
      </p>
      {data?.fit?.note ? <p className="sa-hint">{data.fit.note}</p> : null}
      {/* A fit mined before the length split / radius band exists happily, but
          the knobs that read them silently do nothing — say so here rather than
          letting a run quietly ignore the setting you came for. A selection fit
          never has them (there are no prompts to measure an EOS against), so it
          gets the standing fact rather than a "re-encode to fix" it can't. */}
      {meta && !meta.has_length_stats ? (
        <p className={cn('sa-hint', data?.fit ? 'text-dim' : 'text-err')}>
          {data?.fit
            ? 'fitted from latents, not prompts — Prompt length has nothing to read here'
            : 'mined before the length split — Prompt length and Radius band do nothing until this corpus is re-encoded'}
        </p>
      ) : null}

      <button className="sa-btn sa-btn-sm mt-2 w-full" onClick={() => setOpen(true)}>
        📚 Select base distribution…
      </button>

      <DistModal
        open={open}
        onOpenChange={setOpen}
        backend={backend}
        modelKey={backend === 'sdxl' ? modelKey : null}
        current={data}
      />
    </section>
  )
}
