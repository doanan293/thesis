// Test-only stand-in for devops/nginx/conf.d/default.conf: one origin, /api/ to FastAPI,
// everything else to react-router-serve. Run with `node tests/e2e/proxy.ts`.
import { createServer } from "node:http"

import { createProxyServer } from "http-proxy-3"

const port = Number(process.env.E2E_PROXY_PORT ?? "3200")
const apiTarget = process.env.E2E_API_TARGET ?? "http://127.0.0.1:8001"
const webTarget = process.env.E2E_WEB_TARGET ?? "http://127.0.0.1:3100"

// http-proxy pipes response bodies as they arrive, so SSE is not buffered
// (the nginx equivalent is proxy_buffering off).
const proxy = createProxyServer({ xfwd: true })

proxy.on("error", (error, _request, response) => {
  if ("writeHead" in response) {
    if (!response.headersSent) {
      response.writeHead(502, { "content-type": "text/plain; charset=utf-8" })
    }
    response.end(`E2E proxy error: ${error.message}`)
    return
  }
  response.destroy()
})

createServer((request, response) => {
  const target = request.url?.startsWith("/api/") ? apiTarget : webTarget
  proxy.web(request, response, { target })
}).listen(port, () => {
  console.log(
    `E2E proxy on http://localhost:${port} (api -> ${apiTarget}, web -> ${webTarget})`
  )
})
