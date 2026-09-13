import tailwindcss from "@tailwindcss/vite"
import { playwright } from "@vitest/browser-playwright"
import { searchForWorkspaceRoot } from "vite"
import { defineConfig } from "vitest/config"

export default defineConfig({
  resolve: { tsconfigPaths: true },
  server: {
    fs: {
      allow: [
        searchForWorkspaceRoot(process.cwd()),
        "../backend/tests/contract/fixtures",
      ],
    },
  },
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
        plugins: [tailwindcss()],
        // Vite reuses its optimizer cache without scanning imports, so a dependency a browser
        // test imports for the first time is optimized mid-run and reloads the page. Force a
        // fresh pre-bundle that scans every browser test and setup file before tests start.
        optimizeDeps: {
          force: true,
          entries: ["app/**/*.browser.test.tsx", "tests/setup/*.ts"],
        },
        test: {
          name: "browser",
          include: ["app/**/*.browser.test.tsx"],
          setupFiles: [
            "./tests/setup/fail-on-console.ts",
            "./tests/setup/msw-browser.ts",
            "./tests/setup/styles.ts",
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
