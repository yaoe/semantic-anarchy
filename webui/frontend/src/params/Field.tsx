import { Select } from '../components/ui/Select'
import { Segmented } from '../components/ui/Segmented'
import { Tip } from '../components/ui/Tip'
import { evalField } from '../lib/calc'
import { cn } from '../lib/utils'
import { resolve, type Ctx, type Hint, type Option, type ParamField } from './schema'

function HintLine({ hint }: { hint: Hint | string }) {
  const h: Hint = typeof hint === 'string' ? { text: hint } : hint
  return (
    <div className={cn('sa-hint', h.tone === 'warn' && 'text-err')}>{h.text}</div>
  )
}

/** Renders exactly one schema entry. Nothing here knows what a knob means. */
export function Field({
  field,
  ctx,
  value,
  onChange,
}: {
  field: ParamField
  ctx: Ctx
  value: string
  onChange: (v: string) => void
}) {
  const hint = resolve(field.hint, ctx)

  if (field.type === 'note') return hint ? <HintLine hint={hint} /> : null

  const options = (resolve(field.options, ctx) ?? []) as Option[]
  const placeholder = resolve(field.placeholder, ctx)

  const label = field.label ? (
    <Tip content={field.tooltip} side="top">
      <label
        className={cn('sa-label w-fit', field.tooltip && 'cursor-help decoration-dotted')}
        htmlFor={`f-${field.id}`}
      >
        {field.label}
      </label>
    </Tip>
  ) : null

  let control
  if (field.type === 'segmented') {
    control = <Segmented value={value} onChange={onChange} options={options} />
  } else if (field.type === 'select') {
    control = (
      <Select
        value={value}
        onChange={onChange}
        options={options}
        ariaLabel={field.label ?? field.id}
        title={field.tooltip}
      />
    )
  } else if (field.expr) {
    // Arithmetic-capable box: a text input (type=number can't hold "1024+"
    // mid-typing), resolved to a plain number on Enter or blur.
    const collapse = () => {
      const out = evalField(value, Number.isInteger(field.step ?? 1))
      if (out !== null) onChange(out)
    }
    control = (
      <input
        id={`f-${field.id}`}
        className="sa-input"
        type="text"
        inputMode="text"
        autoComplete="off"
        spellCheck={false}
        value={value}
        placeholder={placeholder}
        title={field.tooltip}
        onChange={(e) => onChange(e.target.value)}
        onBlur={collapse}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            e.preventDefault()
            collapse()
          }
        }}
      />
    )
  } else {
    control = (
      <input
        id={`f-${field.id}`}
        className="sa-input"
        type={field.type === 'number' ? 'number' : 'text'}
        step={field.step}
        min={field.min}
        value={value}
        placeholder={placeholder}
        title={field.tooltip}
        onChange={(e) => onChange(e.target.value)}
      />
    )
  }

  return (
    <div className="min-w-0">
      {label}
      {control}
      {hint ? <HintLine hint={hint} /> : null}
    </div>
  )
}
