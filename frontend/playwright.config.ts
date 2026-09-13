import { defineConfig, devices } from "@playwright/test"

const API_PORT = 8001
const WEB_PORT = 3100
const PROXY_PORT = 3200
const baseURL = `http://localhost:${PROXY_PORT}`
const isCI = Boolean(process.env.CI)

export default defineConfig({
  testDir: "./tests/e2e",
  testMatch: "**/*.e2e.ts",
  // One backend and one database: run tests one at a time; each still uses its own user.
  workers: 1,
  fullyParallel: false,
  forbidOnly: isCI,
  retries: 0,
  timeout: 60_000,
  expect: { timeout: 10_000 },
  reporter: [["list"], ["html", { open: "never" }]],
  use: {
    baseURL,
    locale: "vi-VN",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
  webServer: [
    {
      // P7: recreates the `pharma_e2e` database and the `e2e_chunks_current` alias on
      // start. E2E_POSTGRES_DSN / E2E_QDRANT_URL pass through from the environment.
      name: "backend",
      command: `uv run python -m tests.e2e.server --port ${API_PORT}`,
      cwd: "../backend",
      url: `http://127.0.0.1:${API_PORT}/api/v1/health`,
      timeout: 180_000,
      reuseExistingServer: !isCI,
      stdout: "pipe",
      stderr: "pipe",
      gracefulShutdown: { signal: "SIGTERM", timeout: 10_000 },
    },
    {
      name: "frontend",
      command: "npm run build && npm start",
      url: `http://127.0.0.1:${WEB_PORT}/`,
      env: { PORT: String(WEB_PORT), HOST: "127.0.0.1" },
      timeout: 300_000,
      reuseExistingServer: !isCI,
      stdout: "pipe",
      stderr: "pipe",
    },
    {
      name: "proxy",
      command: "node tests/e2e/proxy.ts",
      // Answers 502 until the frontend is up, so Playwright keeps waiting.
      url: `${baseURL}/`,
      env: {
        E2E_PROXY_PORT: String(PROXY_PORT),
        E2E_API_TARGET: `http://127.0.0.1:${API_PORT}`,
        E2E_WEB_TARGET: `http://127.0.0.1:${WEB_PORT}`,
      },
      timeout: 300_000,
      reuseExistingServer: !isCI,
    },
  ],
})
