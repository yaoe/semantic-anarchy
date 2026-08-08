import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// The FastAPI dashboard (webui/app.py) owns /api and /img; in dev we proxy both
// to it so `npm run dev` talks to a real server. In production this bundle is
// served by that same app out of webui/frontend/dist.
const BACKEND = process.env.SA_BACKEND_URL ?? 'http://127.0.0.1:8800'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': new URL('./src/', import.meta.url).pathname },
  },
  server: {
    proxy: {
      '/api': { target: BACKEND, changeOrigin: true },
      '/img': { target: BACKEND, changeOrigin: true },
      '/legacy': { target: BACKEND, changeOrigin: true },
    },
  },
  build: { outDir: 'dist', emptyOutDir: true },
})
