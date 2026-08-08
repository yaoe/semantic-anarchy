import { useUI } from '../../store'
import { Tip } from './Tip'

/**
 * A labelled range control bound to `tools` in the store — same string-valued
 * contract as `Tool`, so the submit sites cast it identically.
 *
 * A slider hides its own number, so the label row carries a live readout;
 * `format` is where a knob spells out what its value *means* (a denoise of 0.3
 * reads "0.30 · last 30%").
 */
export function Slider({
  id,
  label,
  tooltip,
  min,
  max,
  step,
  fallback,
  format = (v: number) => v.toFixed(2),
  width = 'w-[150px]',
}: {
  id: string
  label: string
  tooltip?: string
  min: number
  max: number
  step: number
  /** Used when the stored value is blank or unparseable. */
  fallback: number
  format?: (v: number) => string
  width?: string
}) {
  const raw = useUI((s) => s.tools[id])
  const setTool = useUI((s) => s.setTool)
  const parsed = Number(raw)
  const value = (raw ?? '').trim() !== '' && Number.isFinite(parsed) ? parsed : fallback

  return (
    <div className={width}>
      <div className="flex items-baseline justify-between gap-1">
        <Tip content={tooltip} side="top">
          <label className="sa-label !mt-0 truncate" htmlFor={`t-${id}`}>
            {label}
          </label>
        </Tip>
        <span className="mb-[3px] shrink-0 font-mono text-[11px] tabular-nums text-ink">
          {format(value)}
        </span>
      </div>
      <input
        id={`t-${id}`}
        className="sa-range"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        aria-label={label}
        onChange={(e) => setTool(id, e.target.value)}
      />
    </div>
  )
}
