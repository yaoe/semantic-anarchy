import { useState } from 'react'

import { useWipe, useWipePreview } from '../../api/queries'
import { Confirm } from '../../components/ui/Confirm'

/**
 * 🧹 Wipe <5 — delete every non-starred image the aesthetic model scored below
 * 5. Favorites and their whole ancestry are protected server-side.
 */
export function WipeDialog() {
  const [open, setOpen] = useState(false)
  const { data: preview } = useWipePreview(true)
  const wipe = useWipe()
  const count = preview?.count ?? 0

  return (
    <>
      <button
        className="sa-btn sa-btn-sm text-danger"
        title="delete all non-starred images scored below 5 (unscored images are kept)"
        disabled={wipe.isPending}
        onClick={() => setOpen(true)}
      >
        {wipe.isPending ? '🧹 wiping…' : `🧹 Wipe <5${count ? ` (${count})` : ''}`}
      </button>
      <Confirm
        open={open}
        onOpenChange={setOpen}
        title={count ? `Delete ${count} images?` : 'Nothing to wipe'}
        danger={count > 0}
        confirmLabel={count ? `Delete ${count}` : 'OK'}
        body={
          count ? (
            <>
              Every non-starred image scored below 5, plus its <code>.json</code>/
              <code>.npz</code> sidecars. Favorites and their originals are protected.
              This cannot be undone.
            </>
          ) : (
            <>No non-starred images are scored below 5. Run “Score all ▶” first.</>
          )
        }
        onConfirm={() => {
          if (count) wipe.mutate()
        }}
      />
      {wipe.isSuccess ? (
        <span className="sa-hint self-center">
          deleted {wipe.data.deleted} images ({wipe.data.files} files)
        </span>
      ) : null}
    </>
  )
}
