import { useMemo, type ReactNode } from 'react'

import { useConfig, useTasteBand } from '../api/queries'
import { cn } from '../lib/utils'
import { useUI } from '../store'
import { Field } from './Field'
import {
  GROUPS,
  PARAM_SCHEMA,
  isVisible,
  type Ctx,
  type GroupId,
  type ParamField,
} from './schema'

function GroupBody({ fields, ctx }: { fields: ParamField[]; ctx: Ctx }) {
  const setParam = useUI((s) => s.setParam)
  return (
    <div className="grid grid-cols-2 gap-x-2 gap-y-0 items-end">
      {fields.map((f) => (
        <div key={f.id} className={cn('min-w-0', (f.span ?? 1) === 2 && 'col-span-2')}>
          <Field
            field={f}
            ctx={ctx}
            value={ctx.values[f.id] ?? ''}
            onChange={(v) => setParam(f.id, v)}
          />
        </div>
      ))}
    </div>
  )
}

/**
 * The whole left sidebar form. It renders PARAM_SCHEMA and nothing else — the
 * only per-knob knowledge in this file is which group headers exist.
 *
 * `after` slots non-schema widgets in below a group (the checkpoint picker
 * lives under Model), so App still owns *what* goes where.
 */
export function ParamPanel({ after }: { after?: Partial<Record<GroupId, ReactNode>> }) {
  const values = useUI((s) => s.params)
  const { data: config } = useConfig()
  const { data: tasteband } = useTasteBand()

  const ctx: Ctx = useMemo(
    () => ({ values, config, tasteband }),
    [values, config, tasteband],
  )

  const byGroup = useMemo(() => {
    const m = new Map<string, ParamField[]>()
    for (const f of PARAM_SCHEMA) {
      if (!isVisible(f, values)) continue
      const list = m.get(f.group) ?? []
      list.push(f)
      m.set(f.group, list)
    }
    return m
  }, [values])

  return (
    <div>
      {GROUPS.map((g) => {
        const fields = byGroup.get(g.id)
        const slot = after?.[g.id]
        if (!fields?.length && !slot) return null
        if (g.collapsible)
          return (
            <details key={g.id} className="mt-4 group">
              <summary className="cursor-pointer text-[12px] text-dim marker:text-dim select-none">
                {g.title}
              </summary>
              {fields?.length ? <GroupBody fields={fields} ctx={ctx} /> : null}
              {slot}
            </details>
          )
        return (
          <section key={g.id}>
            {g.title ? <h2 className="sa-h2">{g.title}</h2> : null}
            {fields?.length ? <GroupBody fields={fields} ctx={ctx} /> : null}
            {slot}
          </section>
        )
      })}
    </div>
  )
}
