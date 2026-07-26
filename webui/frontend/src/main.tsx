import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import * as Tooltip from '@radix-ui/react-tooltip'

import App from './App'
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

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <Tooltip.Provider delayDuration={250} skipDelayDuration={200}>
        <App />
      </Tooltip.Provider>
    </QueryClientProvider>
  </StrictMode>,
)
