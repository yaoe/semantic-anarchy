import { useEffect, useState } from 'react'

import { api } from '../../api/client'

/**
 * Job log over SSE (`/api/log/{id}/stream`), which pushes only newly appended
 * lines. Reconnects resume from `Last-Event-ID`, so a dropped connection never
 * duplicates output. If the stream can't be established at all (older server,
 * buffering proxy) this silently falls back to polling the plain-text endpoint.
 */
export function useJobLog(jobId: number | null) {
  const [lines, setLines] = useState<string[]>([])
  const [streaming, setStreaming] = useState(false)

  useEffect(() => {
    setLines([])
    setStreaming(false)
    if (jobId == null) return

    let cancelled = false
    let es: EventSource | null = null
    let pollTimer: number | undefined

    const startPolling = () => {
      if (pollTimer !== undefined) return
      const tick = async () => {
        try {
          const txt = await api.log(jobId)
          if (!cancelled) setLines(txt ? txt.split('\n') : [])
        } catch {
          /* server restarting — retry on the next tick */
        }
      }
      void tick()
      pollTimer = window.setInterval(tick, 2000)
    }

    try {
      es = new EventSource(api.logStreamUrl(jobId))
      es.addEventListener('lines', (e) => {
        try {
          const chunk = JSON.parse((e as MessageEvent).data) as string[]
          if (!cancelled && chunk.length) setLines((prev) => prev.concat(chunk))
        } catch {
          /* ignore a malformed frame rather than kill the stream */
        }
      })
      // The runner trims its ring buffer at 5000 lines; the server then tells
      // us to start over rather than splice a hole into the transcript.
      es.addEventListener('reset', () => {
        if (!cancelled) setLines([])
      })
      es.addEventListener('status', () => {
        es?.close()
        if (!cancelled) setStreaming(false)
      })
      es.onopen = () => {
        if (!cancelled) setStreaming(true)
      }
      es.onerror = () => {
        if (cancelled) return
        setStreaming(false)
        if (es && es.readyState === EventSource.CLOSED) startPolling()
      }
    } catch {
      startPolling()
    }

    return () => {
      cancelled = true
      es?.close()
      if (pollTimer !== undefined) window.clearInterval(pollTimer)
    }
  }, [jobId])

  return { lines, streaming }
}
