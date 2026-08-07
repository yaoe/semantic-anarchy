/**
 * The label page's selection, as the exact query the server takes.
 *
 * One place derives it from the store so the queue builder and the labeling
 * page ask for the *same* thing — TanStack then dedupes them onto one request,
 * and the "N images match" count in the header is by construction the queue the
 * keypresses are walking.
 */
import { useMemo } from 'react'

import type { LabelBucket, LabelOrder, LabelQueryParams, LabelScope } from '../../api/types'
import { useUI } from '../../store'

const num = (v: string): number | null => (v ? Number(v) : null)

export function useLabelSelection(): LabelQueryParams {
  const label = useUI((s) => s.label)
  return useMemo(
    () => ({
      experiment: label.experiment,
      backend: label.backend,
      ckpt: label.ckpt,
      folder: label.folder,
      size: label.size,
      kind: label.kind,
      sampler: label.sampler,
      scope: (label.scope as LabelScope) || 'unlabeled',
      bucket: (label.bucket as LabelBucket) || 'generated',
      order: (label.order as LabelOrder) || 'shuffle',
      seed: Number(label.seed) || 0,
      since: num(label.since),
      until: num(label.until),
    }),
    [label],
  )
}
