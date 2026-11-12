import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: the UI runs on :5173 and proxies /api to the FastAPI server. Production: same origin, no proxy needed.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { '/api': 'http://localhost:8000' } },
})
