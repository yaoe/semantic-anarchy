import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import * as Tooltip from '@radix-ui/react-tooltip'

import App from './App'
import { LabelApp } from './features/labeling/LabelApp'
import './index.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // The dashboard is a single-user local tool: refetch aggressively on
      // focus, never retry forever against a server that may be restarting.
      retry: 1,
      refetchOnWindowFocus: true,
      staleTime: 0,
    },
  },
})

/**
 * Routing, such as it is: two pages, no router. `webui/app.py` serves the same
 * bundle at "/" and "/label"; the labeling page is a separate browser tab
 * rather than a dashboard tab because it wants the whole window and stays open
 * beside the dashboard while a batch renders.
 */
const path = window.location.pathname.replace(/\/+$/, '')
const Page = path === '/label' ? LabelApp : App

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <Tooltip.Provider delayDuration={250} skipDelayDuration={200}>
        <Page />
      </Tooltip.Provider>
    </QueryClientProvider>
  </StrictMode>,
)
