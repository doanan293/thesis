import { reactRouter } from "@react-router/dev/vite"
import tailwindcss from "@tailwindcss/vite"
import { defineConfig } from "vite"

export default defineConfig({
  resolve: { tsconfigPaths: true },
  plugins: [tailwindcss(), reactRouter()],
  server: {
    port: 3000,
    strictPort: true,
    // Same origin in development: the browser calls /api on :3000 and Vite forwards it.
    proxy: {
      "/api": {
        target: process.env.API_INTERNAL_URL ?? "http://127.0.0.1:8000",
      },
    },
  },
})
