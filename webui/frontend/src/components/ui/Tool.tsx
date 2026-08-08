import { useUI } from '../../store'
import type { Option } from '../../params/schema'
import { Select } from './Select'
import { Tip } from './Tip'

/**
 * One labelled control in a horizontal toolbar, bound to `tools` in the store.
 * Used by the refine/explore bars, which are settings-only (the actions
 * themselves live on the gallery cards).
 */
export function Tool({
  id,
  label,
  tooltip,
  width = 'w-[90px]',
  options,
  type = 'text',
  step,
  placeholder,
  value: valueProp,
  onChange,
}: {
  id: string
  label: string
  tooltip?: string
  width?: string
  options?: Option[]
  type?: 'text' | 'number'
  step?: number
  placeholder?: string
  /** Bind to something other than `tools` (the timeline's film knobs do). */
  value?: string
  onChange?: (v: string) => void
}) {
  const stored = useUI((s) => s.tools[id] ?? '')
  const setTool = useUI((s) => s.setTool)
  const value = valueProp ?? stored
  const set = onChange ?? ((v: string) => setTool(id, v))

  return (
    <div className={width}>
      <Tip content={tooltip} side="top">
        <label className="sa-label !mt-0" htmlFor={`t-${id}`}>
          {label}
        </label>
      </Tip>
      {options ? (
        <Select value={value} onChange={set} options={options} ariaLabel={label} />
      ) : (
        <input
          id={`t-${id}`}
          className="sa-input"
          type={type}
          step={step}
          placeholder={placeholder}
          value={value}
          onChange={(e) => set(e.target.value)}
        />
      )}
    </div>
  )
}
