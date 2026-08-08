/**
 * Where the UI store persists itself: the server's `config.json` (section
 * "ui"), not the browser.
 *
 * The dashboard is opened from several devices against one server, and the
 * sidebar's settings are setup, not per-browser taste — so the machine that
 * owns the GPU owns the defaults, and a phone that has never seen the app
 * opens it configured. localStorage is kept as a MIRROR, for two jobs only:
 *
 *   1. the offline/oops fallback — if `GET /api/prefs` fails we read the
 *      mirror and stop writing, so a hiccup can never upload schema defaults
 *      over a good server-side blob;
 *   2. adoption — the first load after this switch finds nothing on the server
 *      and inherits whatever that browser had, instead of resetting.
 *
 * Writes are debounced (zustand persists on *every* keystroke in a form field)
 * and flushed with sendBeacon when the page goes away, so the last edit before
 * a tab close still lands. Two tabs open = last write wins; they are the same
 * person, and the loser is one sidebar value, not the file.
 */
import type { StateStorage } from 'zustand/middleware'

import { api } from '../api/client'

const SAVE_DEBOUNCE_MS = 600

/** Cleared once a read fails: from then on this session is mirror-only. */
let serverOk = true
let timer: ReturnType<typeof setTimeout> | null = null
let pending: string | null = null

/** Send whatever is queued. `beacon` is the page-is-unloading path. */
function flush(beacon = false): void {
  if (timer) {
    clearTimeout(timer)
    timer = null
  }
  const value = pending
  pending = null
  if (value == null || !serverOk) return

  let body: string
  try {
    body = JSON.stringify({ ui: JSON.parse(value) })
  } catch (err) {
    console.warn('[prefs] not saving unparseable state:', err)
    return
  }
  if (beacon && typeof navigator.sendBeacon === 'function') {
    navigator.sendBeacon('/api/prefs', new Blob([body], { type: 'application/json' }))
    return
  }
  fetch('/api/prefs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body,
    keepalive: true,
  }).catch((err) => console.warn('[prefs] save failed:', err))
}

if (typeof window !== 'undefined') {
  // pagehide covers the bfcache/mobile-background cases that beforeunload misses.
  window.addEventListener('pagehide', () => flush(true))
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flush(true)
  })
}

export const serverPrefsStorage: StateStorage = {
  getItem: async (name) => {
    const mirror = localStorage.getItem(name)
    try {
      const { ui } = await api.prefs()
      if (ui && Object.keys(ui).length) return JSON.stringify(ui)
      return mirror
    } catch (err) {
      serverOk = false
      console.warn('[prefs] server unreachable — using this browser\'s copy:', err)
      return mirror
    }
  },

  setItem: async (name, value) => {
    try {
      localStorage.setItem(name, value)
    } catch {
      /* quota / private mode — the server copy is the one that matters */
    }
    if (!serverOk) return
    pending = value
    if (timer) clearTimeout(timer)
    timer = setTimeout(() => flush(), SAVE_DEBOUNCE_MS)
  },

  removeItem: async (name) => {
    localStorage.removeItem(name)
    pending = null
    if (timer) {
      clearTimeout(timer)
      timer = null
    }
    if (serverOk) await api.savePrefs(null).catch(() => undefined)
  },
}
