import { playwright } from "@vitest/browser-playwright"
import { defineConfig } from "vitest/config"

export default defineConfig({
  resolve: { tsconfigPaths: true },
  test: {
    projects: [
      {
        extends: true,
        test: {
          name: "unit",
          environment: "node",
          include: ["app/**/*.test.{ts,tsx}"],
          exclude: ["app/**/*.browser.test.tsx"],
          setupFiles: [
            "./tests/setup/fail-on-console.ts",
            "./tests/setup/msw-node.ts",
          ],
        },
      },
      {
        extends: true,
        test: {
          name: "browser",
          include: ["app/**/*.browser.test.tsx"],
          setupFiles: [
            "./tests/setup/fail-on-console.ts",
            "./tests/setup/msw-browser.ts",
          ],
          browser: {
            enabled: true,
            headless: true,
            provider: playwright(),
            instances: [{ browser: "chromium" }],
          },
        },
      },
    ],
  },
})
