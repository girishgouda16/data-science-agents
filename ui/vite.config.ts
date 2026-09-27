import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Dev server proxies the API and charts to the gateway, so the UI is same-origin
// in dev exactly as in production (the gateway serves ui/dist) — no CORS.
const gateway = process.env.VITE_GATEWAY ?? 'http://localhost:9001'

export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': gateway, '/charts': gateway } },
})
