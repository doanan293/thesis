# Frontend Foundation Implementation Plan (Plan 8 of 10)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `frontend/` React Router 8 app up to the point where Plans 9 and 10 only add features: scaffold and tooling, shadcn base components, i18n, theme, the API layer (CSRF, problem+json, fetcher, QueryClient, orval client), auth pages and the private-route guard, the app shell, the SSR landing page, the 404 page, and Docker/nginx/compose.

**Architecture:** React Router 8 framework mode with SSR. Public pages (`/`, `/en`, 404) render on the server and `/`, `/en` are prerendered. Private pages (`/chat`, `/skills`, `/settings`) are client-only under `app/routes/app/layout.tsx`, whose `clientMiddleware` asks the backend for the current user through TanStack Query and redirects to `/login?next=` on 401. All HTTP goes through one orval mutator, `app/api/fetcher.ts`, which adds the CSRF header, forwards cookies on the server and turns `application/problem+json` into `ApiError`. remix-i18next (server middleware) and remix-themes (cookie session) run in `root.tsx`.

**Tech Stack:** Node 24.15 / npm 11.12; react-router, @react-router/node, @react-router/serve, @react-router/dev 8.3.1; react, react-dom 19.3.0; vite 8.3.0; tailwindcss, @tailwindcss/vite 4.3.3; shadcn 4.21.0 (CLI and `shadcn/tailwind.css`), cn 0.3.0, @base-ui/react 1.8.0, lucide-react 1.45.0, class-variance-authority 0.7.1, tw-animate-css 1.4.0, @fontsource-variable/geist 5.3.0; @tanstack/react-query 5.102.8; react-hook-form 7.88.0; @hookform/resolvers 5.9.1; zod 4.6.4; i18next 26.4.2; react-i18next 17.0.13; remix-i18next 8.0.0; remix-themes 2.0.4; isbot 5.2.2; cookie 2.0.1; typescript 7.0.2; @types/node 24.13.4; @types/react, @types/react-dom 19.3.0; oxlint 1.82.0; oxlint-tsgolint 7.0.2001; prettier 3.9.6; prettier-plugin-tailwindcss 0.8.1; orval 8.32.0; @faker-js/faker 10.6.0 (orval MSW mocks); msw 2.15.0; vitest 5.0.0; @vitest/browser-playwright 5.0.0; vitest-browser-react 2.3.0; vitest-fail-on-console 0.10.1; @playwright/test 1.63.0. Docker: `node:24.21.0-alpine3.24`, `nginx:1.30.4-alpine3.24`.

**Spec:** `frontend/docs/superpowers/specs/2026-09-13-frontend-v1-design.md` (§2–§9, §12, §14, §13 unit rows for fetcher, problem and CSRF). Backend contract: `backend/docs/superpowers/specs/2026-09-13-web-client-contract-design.md` (§6 auth, §7 errors, §8 OpenAPI). Cross-plan names: `backend/docs/superpowers/plans/2026-09-13-plans-overview.md` §4.

## Global Constraints

- The environment is development only; nothing needs backward compatibility.
- TypeScript 7 strict (`strict`, `noUncheckedIndexedAccess`), Oxlint type-aware with `--deny-warnings`, Prettier. Never write `oxlint-disable`, `eslint-disable`, `@ts-ignore`, `@ts-expect-error` or `@ts-nocheck`; fix the code instead.
- npm bundled with Node 24 (`"engines": {"node": ">=24.15.0"}`). Every dependency is installed with `--save-exact` at the versions in **Tech Stack**.
- Warnings are failures: `tsc` errors, Oxlint warnings (`--deny-warnings`), any `console.error` or `console.warn` during a Vitest test (`tests/setup/fail-on-console.ts`), and any request MSW does not handle (`onUnhandledRequest: "error"`).
- Every task ends green, run inside `frontend/`: `npm run lint && npm run format && npm run typecheck && npm test`.
- Test naming (overview §4): unit tests are `app/**/*.test.{ts,tsx}` and run in the Vitest project `unit` (Node); browser tests are `app/**/*.browser.test.tsx` (always `.tsx`, even without JSX) and run in the project `browser` (Browser Mode, Chromium through `@vitest/browser-playwright`, rendering with `vitest-browser-react`).
- Prefer established libraries over custom code. No feature flag or fake mode in production code; fakes and MSW handlers live under `tests/` or come from the generated `app/api/gen/endpoints.msw.ts`.
- Product copy is Vietnamese by default (`vi`, second language `en`) and always comes from i18n keys; components hold no literal UI strings. The agent never shows medical disclaimers.
- Names pinned by the overview §4 are used exactly: `frontend/openapi.json`; npm scripts `api:openapi`, `api:generate`; orval outputs `app/api/gen/endpoints.ts`, `app/api/gen/schemas/` with `index.ts` (imported as `~/api/gen/schemas`), `app/api/gen/zod.ts`, `app/api/gen/endpoints.msw.ts` (imported as `~/api/gen/endpoints.msw`); test support `tests/msw/node.ts` (`server`), `tests/msw/browser.ts` (`worker`), `tests/setup/fail-on-console.ts`, `tests/setup/msw-node.ts`, `tests/setup/msw-browser.ts`, `tests/utils/providers.tsx` (`TestProviders`, `createTestQueryClient`), `tests/utils/i18n.ts` (`createTestI18n`); `~/components/ui/toast` exports `Toaster` and the shared manager `toast`; actions `POST /actions/locale` (`routes/actions/locale.ts`) and `POST /actions/theme` (`routes/actions/theme.ts`); landing paths `/` (vi) and `/en/` (en); passwords entered in the UI need at least 8 characters; `fetcher<T>(url: string, init?: RequestInit): Promise<T>` in `app/api/fetcher.ts`; `ApiError` and `isApiError` in `app/api/problem.ts`; `readCsrfToken()` and `CSRF_HEADER = "x-csrftoken"` in `app/lib/csrf.ts`; `queryClient` in `app/api/query-client.ts`; i18n namespaces `common`, `auth`, `chat`, `citations`, `skills`, `settings`, `landing`, `errors`; `app/components/elements/`; `frontend/playwright.config.ts`; `frontend/tests/e2e/`.
- HTTP contract: one origin; API prefix `/api/v1`; session cookie `pharma_session` (HttpOnly, set by the backend); CSRF cookie `csrftoken`, header `x-csrftoken` on POST, PUT, PATCH and DELETE; errors are `application/problem+json` `{type, title, status, detail, code, errors}`; locale cookie `lng`; theme session cookie `theme`.
- Ports: `react-router dev` listens on 3000 and proxies `/api` to `http://127.0.0.1:8000`. Compose reads `FRONTEND_PORT` (default 3000), `WEB_PORT` (default 8080) and `BACKEND_PORT` (default 8000), and every service uses `network_mode: host`.
- Backend dependencies: Tasks 1–8 need no backend. Task 9 needs Plan 5 (`pharma-agent export-openapi`, problem+json, operation ids). Tasks 10 and 11 need Plan 7 (cookie login, CSRF, Google OAuth on the cookie backend).
- Commits: one commit per task, conventional message, ending with the session attribution trailer of the executing session. This plan never hard-codes it: before the first commit, export `COMMIT_TRAILER` with the exact trailer lines that the executing session's attribution reminder gives (for example `export COMMIT_TRAILER="$(printf '%s\n%s' '<first trailer line>' '<second trailer line>')"`, filled in by the executor). Every commit step passes it as the last paragraph with `-m "$COMMIT_TRAILER"`.
- Oxlint must ignore `node_modules/**` (Task 1 `.oxlintrc.json`). In the planning spike, `oxlint --type-aware` without that ignore linted every installed package and was killed for memory twice on a 20 GB WSL machine; with the ignore it finished in under a second. Task 1 Step 9 runs lint on its own once so a regression shows up early; never drop `--type-aware` to make it pass.

---

## File Structure

```text
.pre-commit-config.yaml                          (+ frontend hooks: lint, format, typecheck)               # Task 1; api drift hook Task 9
.env.example                                     (+ FRONTEND_PORT, WEB_PORT)                                # Task 8
docker-compose.yml                               (+ frontend, nginx; backend PHARMA_AUTH__FRONTEND_URL)      # Task 8
docker/nginx/nginx.conf                          one origin: /api/ -> backend (SSE safe), / -> frontend      # Task 8
frontend/
  package.json, package-lock.json                scripts, exact pins for every dependency                   # Task 1; api scripts Task 9
  tsconfig.json                                  TS 7 strict, noUncheckedIndexedAccess, typegen rootDirs, ~ # Task 1
  react-router.config.ts                         ssr: true                                                  # Task 1; prerender Task 7
  vite.config.ts                                 tailwind + react-router plugins, port 3000, /api proxy     # Task 1
  vitest.config.ts                               projects unit (Node) and browser (Chromium)                # Task 1
  playwright.config.ts                           E2E skeleton, testDir tests/e2e (Plan 10 fills it)         # Task 1
  .oxlintrc.json, .prettierrc, .prettierignore, .gitignore                                                   # Task 1
  components.json                                shadcn base-nova                                           # Task 1
  Dockerfile, .dockerignore, .env.example        node:24.21.0-alpine3.24 image, react-router-serve          # Task 8
  orval.config.ts, openapi.json                  client generation input and config                        # Task 9
  public/favicon.ico, public/mockServiceWorker.js                                                            # Task 1
  tests/setup/fail-on-console.ts                 vitest-fail-on-console: console.error/warn fail a test     # Task 1
  tests/setup/msw-node.ts, tests/setup/msw-browser.ts   MSW lifecycle per Vitest project                    # Task 1
  tests/msw/node.ts, tests/msw/browser.ts        shared `server` and `worker`                               # Task 1
  tests/utils/i18n.ts                            createTestI18n                                             # Task 3
  tests/utils/providers.tsx                      TestProviders, createTestQueryClient                       # Task 10
  tests/e2e/.gitkeep                                                                                         # Task 1
  app/
    app.css                                      Tailwind 4 + shadcn tokens, linked through `?url`          # Task 1
    root.tsx                                     links, i18n middleware, loader {locale, theme}, providers, ErrorBoundary # Tasks 1–4, 6
    routes.ts                                    route table                                                # Tasks 1, 3, 4, 7, 10, 11
    entry.server.tsx, entry.client.tsx           revealed entries wrapped in I18nextProvider               # Task 3
    lib/utils.ts (+ utils.test.ts)               cn                                                         # Task 1
    lib/csrf.ts (+ csrf.test.ts, csrf.browser.test.tsx)   readCsrfToken, CSRF_HEADER, CSRF_COOKIE           # Task 5
    lib/theme.server.ts (+ theme.server.test.ts) themeSessionResolver                                       # Task 4
    lib/form-errors.ts (+ form-errors.test.ts)   applyApiError                                              # Task 10
    components/document.tsx (+ document.test.tsx)   <html lang dir class>                                   # Tasks 3, 4
    components/ui/*.tsx                          shadcn CLI output, lint fixes, translated labels           # Tasks 1, 2, 11
    components/ui/button.browser.test.tsx, components/ui/toast.browser.test.tsx                               # Tasks 1, 2, 11
    components/elements/.gitkeep                 registry components of Plan 9                              # Task 2
    hooks/use-mobile.ts                          media query through useSyncExternalStore                  # Task 2
    i18n/config.ts (+ config.test.ts, resources.test.ts, public-page-locale.test.ts)   languages, resources, URL locale # Tasks 3, 7
    i18n/types.d.ts                              CustomTypeOptions                                          # Task 3
    i18n/locale.server.ts (+ locale.server.test.ts)   lng cookie, setLocale                                 # Task 3
    i18n/middleware.server.ts                    remix-i18next middleware, getLocale, getInstance           # Tasks 3, 7
    i18n/zod-locale.ts (+ zod-locale.test.ts)    applyZodLocale                                             # Task 3
    i18n/error-message.ts (+ error-message.test.ts)   isErrorCode, apiErrorMessage                          # Task 5
    i18n/resources/{vi,en}/{common,auth,chat,citations,skills,settings,landing,errors}.json                 # Task 3; keys in 5, 7, 10, 11
    api/problem.ts (+ problem.test.ts)           ApiError, isApiError, toApiError                           # Task 5
    api/fetcher.ts (+ fetcher.test.ts, fetcher.browser.test.tsx)   orval mutator                            # Task 6
    api/query-client.ts (+ query-client.test.ts) queryClient, createQueryClient, shouldRetry, redirectToLogin # Task 6
    api/gen.test.ts                              contract of the generated client                          # Task 9
    api/gen/endpoints.ts, api/gen/endpoints.msw.ts, api/gen/schemas/*.ts, api/gen/zod.ts   orval output     # Task 9
    features/landing/LandingPage.tsx (+ test), features/landing/meta.ts (+ test)                             # Task 7
    features/auth/AuthCard.tsx, LoginForm.tsx, RegisterForm.tsx, GoogleButton.tsx (+ browser tests)          # Task 10
    features/auth/lib/next-path.ts (+ test), lib/current-user.ts, lib/google.ts (+ browser test)             # Task 10
    features/auth/hooks/use-redirect-if-authenticated.ts                                                     # Task 10
    features/auth/lib/require-user.ts (+ browser test), features/auth/hooks/use-logout.ts                   # Task 11
    features/shell/AppSidebar.tsx (+ browser test), features/shell/UserMenu.tsx                             # Task 11
    routes/public/landing.tsx                    `/` and `/en`, prerendered                                 # Tasks 1, 7
    routes/not-found.tsx (+ not-found.test.tsx)  `*`, status 404                                            # Task 7
    routes/actions/locale.ts, routes/actions/theme.ts                                                        # Tasks 3, 4
    routes/auth/login.tsx, routes/auth/register.tsx, routes/auth/google-callback.tsx                         # Task 10
    routes/app/layout.tsx                        clientMiddleware guard and shell                           # Task 11
    routes/app/chat.tsx, routes/app/skills.tsx, routes/app/settings.tsx   placeholders for Plans 9 and 10    # Task 11
```

---

### Task 1: Scaffold React Router 8 with shadcn, Oxlint, Prettier, Vitest and pre-commit hooks

No backend needed.

**Files:**
- Create: `frontend/package.json`, `frontend/package-lock.json`, `frontend/tsconfig.json`, `frontend/react-router.config.ts`, `frontend/vite.config.ts`, `frontend/vitest.config.ts`, `frontend/playwright.config.ts`, `frontend/.oxlintrc.json`, `frontend/.prettierrc`, `frontend/.prettierignore`, `frontend/.gitignore`
- Create (copied from the shadcn template): `frontend/components.json`, `frontend/app/app.css`, `frontend/app/lib/utils.ts`, `frontend/app/components/ui/button.tsx`, `frontend/public/favicon.ico`
- Create: `frontend/app/root.tsx`, `frontend/app/routes.ts`, `frontend/app/routes/public/landing.tsx`, `frontend/public/mockServiceWorker.js` (MSW CLI)
- Create: `frontend/tests/setup/fail-on-console.ts`, `frontend/tests/setup/msw-node.ts`, `frontend/tests/setup/msw-browser.ts`, `frontend/tests/msw/node.ts`, `frontend/tests/msw/browser.ts`, `frontend/tests/e2e/.gitkeep`
- Modify: `.pre-commit-config.yaml` (repo root)
- Test: `frontend/app/lib/utils.test.ts`, `frontend/app/components/ui/button.browser.test.tsx`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `cn(...inputs: ClassValue[]): string` from `~/lib/utils` (re-export of the `cn` package, clsx + tailwind-merge semantics).
  - `server: SetupServerApi` from `tests/msw/node.ts`; `worker: SetupWorker` from `tests/msw/browser.ts`. Tests add handlers with `server.use(...)` / `worker.use(...)`; handlers reset after each test.
  - npm scripts: `dev`, `build`, `start`, `typecheck`, `lint`, `format`, `format:write`, `test`, `test:unit`, `test:browser`, `e2e` (Task 9 adds `api:openapi`, `api:generate`, `api:check`).
  - Vitest projects `unit` and `browser`; path alias `~/*` → `app/*`.
  - All runtime and dev dependencies of this plan, pinned exactly (later tasks do not install packages).

- [ ] **Step 1: Generate the shadcn React Router template and copy the files this task keeps**

The template pins React Router 7 and TypeScript 6, so only the files that do not depend on those versions are copied; `package.json`, `tsconfig.json` and the configs are written in later steps.

```bash
cd /home/andv/personal/thesis
SCAFFOLD_DIR=$(mktemp -d)
(cd "$SCAFFOLD_DIR" && npx -y shadcn@4.21.0 init -t react-router -b base -p nova -n web --yes --no-monorepo)
mkdir -p frontend/app/lib frontend/app/components/ui frontend/public
cp "$SCAFFOLD_DIR/web/components.json" frontend/components.json
cp "$SCAFFOLD_DIR/web/app/app.css" frontend/app/app.css
cp "$SCAFFOLD_DIR/web/app/lib/utils.ts" frontend/app/lib/utils.ts
cp "$SCAFFOLD_DIR/web/app/components/ui/button.tsx" frontend/app/components/ui/button.tsx
cp "$SCAFFOLD_DIR/web/public/favicon.ico" frontend/public/favicon.ico
grep '"style"' frontend/components.json
cat frontend/app/lib/utils.ts
```

Expected: `"style": "base-nova",` and `export { cn } from "cn"`. `components.json` has `"baseColor": "neutral"`, `"iconLibrary": "lucide"` and aliases `~/components`, `~/lib/utils`, `~/components/ui`, `~/lib`, `~/hooks`.

- [ ] **Step 2: Write `frontend/package.json` and install**

```json
{
  "name": "pharma-agent-web",
  "private": true,
  "type": "module",
  "engines": {
    "node": ">=24.15.0"
  },
  "scripts": {
    "dev": "react-router dev",
    "build": "react-router build",
    "start": "react-router-serve ./build/server/index.js",
    "typecheck": "react-router typegen && tsc",
    "lint": "oxlint --type-aware --deny-warnings",
    "format": "prettier --check .",
    "format:write": "prettier --write .",
    "test": "vitest run",
    "test:unit": "vitest run --project unit",
    "test:browser": "vitest run --project browser",
    "e2e": "playwright test"
  },
  "msw": {
    "workerDirectory": [
      "public"
    ]
  },
  "dependencies": {
    "@base-ui/react": "1.8.0",
    "@fontsource-variable/geist": "5.3.0",
    "@hookform/resolvers": "5.9.1",
    "@react-router/node": "8.3.1",
    "@react-router/serve": "8.3.1",
    "@tanstack/react-query": "5.102.8",
    "class-variance-authority": "0.7.1",
    "cn": "0.3.0",
    "cookie": "2.0.1",
    "i18next": "26.4.2",
    "isbot": "5.2.2",
    "lucide-react": "1.45.0",
    "react": "19.3.0",
    "react-dom": "19.3.0",
    "react-hook-form": "7.88.0",
    "react-i18next": "17.0.13",
    "react-router": "8.3.1",
    "remix-i18next": "8.0.0",
    "remix-themes": "2.0.4",
    "shadcn": "4.21.0",
    "tw-animate-css": "1.4.0",
    "zod": "4.6.4"
  },
  "devDependencies": {
    "@faker-js/faker": "10.6.0",
    "@playwright/test": "1.63.0",
    "@react-router/dev": "8.3.1",
    "@tailwindcss/vite": "4.3.3",
    "@types/node": "24.13.4",
    "@types/react": "19.3.0",
    "@types/react-dom": "19.3.0",
    "@vitest/browser-playwright": "5.0.0",
    "msw": "2.15.0",
    "orval": "8.32.0",
    "oxlint": "1.82.0",
    "oxlint-tsgolint": "7.0.2001",
    "prettier": "3.9.6",
    "prettier-plugin-tailwindcss": "0.8.1",
    "tailwindcss": "4.3.3",
    "typescript": "7.0.2",
    "vite": "8.3.0",
    "vitest": "5.0.0",
    "vitest-browser-react": "2.3.0",
    "vitest-fail-on-console": "0.10.1"
  }
}
```

`shadcn` is a runtime dependency because `app/app.css` imports `shadcn/tailwind.css`. `vitest-fail-on-console` is the maintained Vitest port of `jest-fail-on-console`; it replaces a hand-written console spy.

```bash
cd /home/andv/personal/thesis/frontend
npm install
npx playwright install chromium
npx msw init public
```

Expected: `added N packages`, `0 vulnerabilities` or a list to review; Chromium downloaded; `public/mockServiceWorker.js` created. If Chromium fails to start later because of missing system libraries on WSL, run `sudo npx playwright install-deps chromium` once.

- [ ] **Step 3: Write the failing tests**

`frontend/app/lib/utils.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { cn } from "~/lib/utils"

describe("cn", () => {
  test("keeps the last conflicting Tailwind class", () => {
    expect(cn("px-2 py-1", "px-4")).toBe("py-1 px-4")
  })

  test("drops falsy values", () => {
    expect(cn("block", false, undefined, "text-sm")).toBe("block text-sm")
  })
})
```

`frontend/app/components/ui/button.browser.test.tsx`:

```tsx
import { expect, test, vi } from "vitest"
import { render } from "vitest-browser-react"

import { Button } from "~/components/ui/button"

test("Button renders a native button and reports clicks", async () => {
  const onClick = vi.fn<() => void>()
  const screen = await render(<Button onClick={onClick}>Gửi</Button>)

  await screen.getByRole("button", { name: "Gửi" }).click()

  expect(onClick).toHaveBeenCalledOnce()
})
```

- [ ] **Step 4: Run the tests without any Vitest config**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run`
Expected: FAIL. `utils.test.ts` cannot resolve `~/lib/utils` (no alias yet) and `button.browser.test.tsx` fails because it runs in Node instead of Browser Mode.

- [ ] **Step 5: Write the TypeScript, React Router, Vite and Vitest configs, the MSW setup and the minimal app**

`frontend/tsconfig.json`:

```json
{
  "include": [
    "**/*",
    "**/.server/**/*",
    "**/.client/**/*",
    ".react-router/types/**/*"
  ],
  "exclude": ["node_modules", "build"],
  "compilerOptions": {
    "lib": ["DOM", "DOM.Iterable", "ES2023"],
    "types": ["node", "vite/client"],
    "target": "ES2022",
    "module": "ES2022",
    "moduleResolution": "bundler",
    "jsx": "react-jsx",
    "rootDirs": [".", "./.react-router/types"],
    "paths": {
      "~/*": ["./app/*"]
    },
    "esModuleInterop": true,
    "verbatimModuleSyntax": true,
    "noEmit": true,
    "resolveJsonModule": true,
    "skipLibCheck": true,
    "strict": true,
    "noUncheckedIndexedAccess": true,
    "noImplicitOverride": true,
    "noFallthroughCasesInSwitch": true
  }
}
```

`frontend/react-router.config.ts`:

```ts
import type { Config } from "@react-router/dev/config"

export default {
  ssr: true,
} satisfies Config
```

`frontend/vite.config.ts`:

```ts
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
      "/api": { target: process.env.API_INTERNAL_URL ?? "http://127.0.0.1:8000" },
    },
  },
})
```

`frontend/vitest.config.ts` (the React Router Vite plugin is not loaded in tests; Vite compiles TSX with the automatic runtime from `tsconfig.json`):

```ts
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
```

`frontend/tests/setup/fail-on-console.ts`:

```ts
import failOnConsole from "vitest-fail-on-console"

failOnConsole({ shouldFailOnError: true, shouldFailOnWarn: true })
```

`frontend/tests/msw/node.ts`:

```ts
import { setupServer } from "msw/node"

export const server = setupServer()
```

`frontend/tests/msw/browser.ts`:

```ts
import { setupWorker } from "msw/browser"

export const worker = setupWorker()
```

`frontend/tests/setup/msw-node.ts`:

```ts
import { afterAll, afterEach, beforeAll } from "vitest"

import { server } from "../msw/node"

beforeAll(() => {
  server.listen({ onUnhandledRequest: "error" })
})

afterEach(() => {
  server.resetHandlers()
})

afterAll(() => {
  server.close()
})
```

`frontend/tests/setup/msw-browser.ts` (the service worker also sees Vite's module requests, so only API calls must have a handler):

```ts
import { afterAll, afterEach, beforeAll } from "vitest"

import { worker } from "../msw/browser"

beforeAll(async () => {
  await worker.start({
    quiet: true,
    onUnhandledRequest(request, print) {
      if (new URL(request.url).pathname.startsWith("/api/")) {
        print.error()
      }
    },
  })
})

afterEach(() => {
  worker.resetHandlers()
})

afterAll(() => {
  worker.stop()
})
```

`frontend/app/root.tsx`:

```tsx
import type { ReactNode } from "react"
import { Links, Meta, Outlet, Scripts, ScrollRestoration } from "react-router"

import type { Route } from "./+types/root"
import stylesheet from "./app.css?url"

export const links: Route.LinksFunction = () => [
  { rel: "stylesheet", href: stylesheet },
]

export function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="vi">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        {children}
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  )
}

export default function App() {
  return <Outlet />
}
```

`frontend/app/routes.ts`:

```ts
import { type RouteConfig, index } from "@react-router/dev/routes"

export default [
  index("routes/public/landing.tsx", { id: "landing-vi" }),
] satisfies RouteConfig
```

`frontend/app/routes/public/landing.tsx` (Task 7 replaces the body with the real landing page):

```tsx
export default function Landing() {
  return <main className="min-h-svh bg-background text-foreground" />
}
```

- [ ] **Step 6: Run the tests**

Run: `cd /home/andv/personal/thesis/frontend && npm test`
Expected: PASS, `Test Files 2 passed (2)`, `Tests 3 passed (3)`, with `|unit|` and `|browser| chromium` prefixes.

- [ ] **Step 7: Write the lint, format, git-ignore and Playwright configs**

`frontend/.oxlintrc.json`:

```json
{
  "$schema": "./node_modules/oxlint/configuration_schema.json",
  "plugins": ["typescript", "react", "jsx-a11y", "import", "vitest", "promise"],
  "categories": {
    "correctness": "error",
    "suspicious": "error"
  },
  "env": {
    "builtin": true,
    "browser": true,
    "node": true
  },
  "rules": {
    "react/rules-of-hooks": "error",
    "react/exhaustive-deps": "error",
    "react/react-in-jsx-scope": "off"
  },
  "ignorePatterns": [
    "node_modules/**",
    "build/**",
    ".react-router/**",
    "app/api/gen/**",
    "public/mockServiceWorker.js"
  ]
}
```

Why each non-default entry is there:

- `react/react-in-jsx-scope` is in the `suspicious` category but only applies to the classic JSX runtime. This project compiles JSX with the automatic runtime (`"jsx": "react-jsx"`), where `React` is never needed in scope. That makes it a genuine false positive; the only "fix" in code would be importing React into every file for nothing. It is the only rule turned off.
- `node_modules/**`: without it Oxlint 1.82 lints every package when no ignore file covers the folder, and with `--type-aware` that used up all memory in the planning spike (killed twice). With the folder ignored, type-aware lint of the scratch copy peaked at a few MB.
- `app/api/gen/**` is orval output (Task 9). It is regenerated, never edited, and still type-checked by `tsc`.

Rules in the enabled categories that shape the code in this plan: `typescript/no-unsafe-type-assertion` (no `as T` or narrowing casts), `typescript/no-floating-promises` (use `void` or `await`), `vitest/require-mock-type-parameters` (`vi.fn<() => void>()`), `import/no-unassigned-import` (CSS is linked through `?url`, not a bare `import "./app.css"`), `eslint/no-shadow`, and the React Compiler rules such as `react/set-state-in-effect`.

`frontend/.prettierrc` (the shadcn template's style, with the stylesheet path from spec §12):

```json
{
  "endOfLine": "lf",
  "semi": false,
  "singleQuote": false,
  "tabWidth": 2,
  "trailingComma": "es5",
  "printWidth": 80,
  "plugins": ["prettier-plugin-tailwindcss"],
  "tailwindStylesheet": "./app/app.css",
  "tailwindFunctions": ["cn", "cva"]
}
```

`frontend/.prettierignore`:

```text
node_modules/
build/
.react-router/
coverage/
playwright-report/
test-results/
package-lock.json
openapi.json
public/mockServiceWorker.js
```

`frontend/.gitignore`:

```text
node_modules/
build/
.react-router/
*.tsbuildinfo
.env
.env.*
!.env.example
playwright-report/
test-results/
__screenshots__/
```

`frontend/playwright.config.ts` (Plan 10 adds `webServer` and the specs):

```ts
import { defineConfig, devices } from "@playwright/test"

export default defineConfig({
  testDir: "./tests/e2e",
  forbidOnly: Boolean(process.env.CI),
  retries: 0,
  reporter: "list",
  use: {
    baseURL: process.env.E2E_BASE_URL ?? "http://localhost:3000",
    trace: "retain-on-failure",
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
})
```

Create the empty `frontend/tests/e2e/.gitkeep`.

- [ ] **Step 8: Format, type-check, test and build**

```bash
cd /home/andv/personal/thesis/frontend
npm run format:write
npm run format && npm run typecheck && npm test
npm run build
ls build/server/index.js build/client
```

Expected: `All matched files use Prettier code style!`; `tsc` prints nothing and exits 0; tests pass as in Step 6; the build writes `build/server/index.js` and `build/client/assets/`.

- [ ] **Step 9: Run type-aware lint on its own**

Run: `cd /home/andv/personal/thesis/frontend && npm run lint`
Expected: `Found 0 warnings and 0 errors.` Fix any finding in the code (including shadcn files, which are ours once copied). If the process is killed for memory, stop and report; do not remove `--type-aware`.

- [ ] **Step 10: Add the frontend hooks to the root pre-commit config**

In `/home/andv/personal/thesis/.pre-commit-config.yaml`, replace the first two lines

```yaml
# Shared hooks for every project in this repo (corpus-pipeline, backend).
# Install once from the repo root: `uv run --project backend pre-commit install`.
exclude: '.*(data/|\.venv/|ai-models/|\.agents/)'
```

with

```yaml
# Shared hooks for every project in this repo (corpus-pipeline, backend, frontend).
# Install once from the repo root: `uv run --project backend pre-commit install`.
# Frontend hooks need `npm ci` in frontend/ first.
exclude: '.*(data/|\.venv/|ai-models/|\.agents/|node_modules/)'
```

and insert this block right after the `validate-agent-skills` local repo block (before the `gitleaks` repo):

```yaml
  - repo: local
    hooks:
      - id: frontend-lint
        name: oxlint --type-aware (frontend)
        entry: npm --prefix frontend run lint
        language: system
        files: ^frontend/.*\.(ts|tsx|js|mjs|json)$
        pass_filenames: false
        require_serial: true
      - id: frontend-format
        name: prettier --check (frontend)
        entry: npm --prefix frontend run format
        language: system
        files: ^frontend/
        pass_filenames: false
        require_serial: true
      - id: frontend-typecheck
        name: typecheck (frontend)
        entry: npm --prefix frontend run typecheck
        language: system
        files: ^frontend/.*\.(ts|tsx|json)$
        pass_filenames: false
        require_serial: true
```

Run: `cd /home/andv/personal/thesis && uv run --project backend pre-commit run --files frontend/package.json frontend/app/root.tsx .pre-commit-config.yaml`
Expected: every hook `Passed` or `(no files to check)Skipped`, including `oxlint --type-aware (frontend)`, `prettier --check (frontend)` and `typecheck (frontend)`.

- [ ] **Step 11: Full check**

Run: `cd /home/andv/personal/thesis/frontend && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all four succeed.

- [ ] **Step 12: Commit**

```bash
cd /home/andv/personal/thesis
git add .pre-commit-config.yaml frontend/package.json frontend/package-lock.json frontend/tsconfig.json \
  frontend/react-router.config.ts frontend/vite.config.ts frontend/vitest.config.ts frontend/playwright.config.ts \
  frontend/.oxlintrc.json frontend/.prettierrc frontend/.prettierignore frontend/.gitignore frontend/components.json \
  frontend/app/app.css frontend/app/root.tsx frontend/app/routes.ts frontend/app/routes/public/landing.tsx \
  frontend/app/lib/utils.ts frontend/app/lib/utils.test.ts frontend/app/components/ui/button.tsx \
  frontend/app/components/ui/button.browser.test.tsx frontend/public/favicon.ico frontend/public/mockServiceWorker.js \
  frontend/tests/setup frontend/tests/msw frontend/tests/e2e/.gitkeep
git commit -m "feat(frontend): scaffold React Router 8 app with shadcn, Oxlint, Prettier and Vitest" \
  -m "$COMMIT_TRAILER"
```

---

### Task 2: shadcn base components, Toaster and TooltipProvider

No backend needed.

**Files:**
- Create (shadcn CLI): `frontend/app/components/ui/{sidebar,toast,sheet,drawer,dialog,alert-dialog,card,switch,badge,input-group,textarea,field,input,label,separator,dropdown-menu,avatar,skeleton,spinner,tooltip}.tsx`, `frontend/app/hooks/use-mobile.ts`
- Create: `frontend/app/components/elements/.gitkeep`
- Modify: `frontend/app/root.tsx`
- Test: `frontend/app/components/ui/toast.browser.test.tsx`

**Interfaces:**
- Consumes: `cn` from `~/lib/utils` (Task 1).
- Produces (all generated by the CLI, Base UI flavour, style `base-nova`):
  - `~/components/ui/toast`: `Toaster`, `toast` (the shared Base UI toast manager: `toast.add({ title, description, type })`), `useToastManager`, `createToastManager`, `Toast*` parts.
  - `~/components/ui/sidebar`: `SidebarProvider`, `Sidebar`, `SidebarHeader`, `SidebarContent`, `SidebarFooter`, `SidebarGroup`, `SidebarGroupLabel`, `SidebarGroupContent`, `SidebarMenu`, `SidebarMenuItem`, `SidebarMenuButton`, `SidebarInset`, `SidebarTrigger`, `SidebarRail`, `SidebarSeparator`, `useSidebar`.
  - `~/components/ui/field`: `Field`, `FieldLabel`, `FieldDescription`, `FieldError`, `FieldGroup`, `FieldLegend`, `FieldSeparator`, `FieldSet`, `FieldContent`, `FieldTitle`.
  - `~/components/ui/tooltip`: `TooltipProvider` and tooltip parts; `~/hooks/use-mobile`: `useIsMobile(): boolean`.
  - `app/root.tsx` default export wraps every route in `<Toaster>` and `<TooltipProvider>`.

- [ ] **Step 1: Write the failing test**

`frontend/app/components/ui/toast.browser.test.tsx`:

```tsx
import { expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"

import { Toaster, useToastManager } from "~/components/ui/toast"

function SaveButton() {
  const manager = useToastManager()
  return (
    <button
      type="button"
      onClick={() => {
        manager.add({ title: "Không lưu được lượt này", type: "error" })
      }}
    >
      Lưu
    </button>
  )
}

test("Toaster shows a toast added through its manager", async () => {
  await render(
    <Toaster>
      <SaveButton />
    </Toaster>
  )

  await page.getByRole("button", { name: "Lưu" }).click()

  await expect
    .element(page.getByText("Không lưu được lượt này"))
    .toBeVisible()
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project browser app/components/ui/toast.browser.test.tsx`
Expected: FAIL with `Failed to resolve import "~/components/ui/toast"`.

- [ ] **Step 3: Add the components with the shadcn CLI**

```bash
cd /home/andv/personal/thesis/frontend
npx shadcn@4.21.0 add sidebar toast sheet drawer dialog alert-dialog card switch badge input-group textarea field input label separator dropdown-menu avatar skeleton spinner --yes
mkdir -p app/components/elements && touch app/components/elements/.gitkeep
git diff --stat package.json
```

Expected: `Created 21 files` listing the 20 `app/components/ui/*.tsx` files above plus `app/hooks/use-mobile.ts` (`tooltip` comes with `sidebar`), and `Skipped 1 file` for `button.tsx`. `git diff --stat package.json` is empty because every dependency the components import (`@base-ui/react`, `cn`, `class-variance-authority`, `lucide-react`) is already pinned; if the CLI added or re-ranged a dependency, set it back to the exact version from Task 1 and run `npm install`.

The generated code breaks six lint rules. The planning spike ran Oxlint 1.82 with this plan's config, with and without `--type-aware`, over the base-nova registry sources; after the edits below it exits 0. The files are ours once copied, so fix them in place:

`frontend/app/hooks/use-mobile.ts` (`react/set-state-in-effect`; a media query is an external store):

```ts
import * as React from "react"

const MOBILE_BREAKPOINT = 768
const MOBILE_QUERY = `(max-width: ${MOBILE_BREAKPOINT - 1}px)`

function subscribe(onChange: () => void): () => void {
  const mediaQuery = window.matchMedia(MOBILE_QUERY)
  mediaQuery.addEventListener("change", onChange)
  return () => {
    mediaQuery.removeEventListener("change", onChange)
  }
}

function getSnapshot(): boolean {
  return window.matchMedia(MOBILE_QUERY).matches
}

function getServerSnapshot(): boolean {
  return false
}

export function useIsMobile(): boolean {
  return React.useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot)
}
```

`frontend/app/components/ui/label.tsx` (`jsx-a11y/label-has-associated-control`; `htmlFor` is passed through explicitly):

```tsx
import * as React from "react"
import { cn } from "cn"

function Label({ className, htmlFor, ...props }: React.ComponentProps<"label">) {
  return (
    <label
      data-slot="label"
      htmlFor={htmlFor}
      className={cn(
        "flex items-center gap-2 text-sm leading-none font-medium select-none group-data-[disabled=true]:pointer-events-none group-data-[disabled=true]:opacity-50 peer-disabled:cursor-not-allowed peer-disabled:opacity-50",
        className
      )}
      {...props}
    />
  )
}

export { Label }
```

`frontend/app/components/ui/spinner.tsx` (`jsx-a11y/prefer-tag-over-role` on `role="status"`, plus a hard-coded English `aria-label`). The icon is decorative; whoever shows a spinner also renders the translated text, visible or `sr-only`:

```tsx
import { Loader2Icon } from "lucide-react"
import { cn } from "cn"

function Spinner({ className, ...props }: React.ComponentProps<"svg">) {
  return (
    <Loader2Icon
      data-slot="spinner"
      aria-hidden="true"
      className={cn("size-4 animate-spin", className)}
      {...props}
    />
  )
}

export { Spinner }
```

`frontend/app/components/ui/field.tsx` (`jsx-a11y/prefer-tag-over-role`): in `function Field`, delete the line `role="group"` so the element starts with `<div` then `data-slot="field"`.

`frontend/app/components/ui/input-group.tsx`:
- In `function InputGroup`, delete the line `role="group"` (`jsx-a11y/prefer-tag-over-role`).
- In `function InputGroupAddon`, replace

  ```tsx
      <div
        role="group"
        data-slot="input-group-addon"
        data-align={align}
        className={cn(inputGroupAddonVariants({ align }), className)}
        onClick={(e) => {
          if ((e.target as HTMLElement).closest("button")) {
            return
          }
          e.currentTarget.parentElement?.querySelector("input")?.focus()
        }}
        {...props}
      />
  ```

  with

  ```tsx
      <div
        data-slot="input-group-addon"
        data-align={align}
        className={cn(inputGroupAddonVariants({ align }), className)}
        {...props}
      />
  ```

  This removes `prefer-tag-over-role`, `click-events-have-key-events`, `no-noninteractive-element-interactions` and `no-unsafe-type-assertion` together. Clicking the padding of an addon no longer focuses the input; the control itself stays the click and keyboard target.

`frontend/app/components/ui/sidebar.tsx`:
- Above `const SIDEBAR_WIDTH = "16rem"` add

  ```tsx
  type CssVariables = Record<`--${string}`, string>

  ```

- In `toggleSidebar` (`eslint/no-shadow`), replace `setOpenMobile((open) => !open) : setOpen((open) => !open)` with `setOpenMobile((value) => !value) : setOpen((value) => !value)`.
- In `function Sidebar` (`typescript/no-unsafe-type-assertion`), replace

  ```tsx
    if (isMobile) {
      return (
  ```

  with

  ```tsx
    if (isMobile) {
      const mobileStyle: React.CSSProperties & CssVariables = {
        "--sidebar-width": SIDEBAR_WIDTH_MOBILE,
      }
      return (
  ```

  and replace

  ```tsx
            style={
              {
                "--sidebar-width": SIDEBAR_WIDTH_MOBILE,
              } as React.CSSProperties
            }
  ```

  with `style={mobileStyle}`.
- In `function SidebarMenuSkeleton`, after the `React.useState` that computes `width`, add

  ```tsx
    const skeletonStyle: React.CSSProperties & CssVariables = {
      "--skeleton-width": width,
    }
  ```

  and replace

  ```tsx
          style={
            {
              "--skeleton-width": width,
            } as React.CSSProperties
          }
  ```

  with `style={skeletonStyle}`.

The provider style in `SidebarProvider` spreads the caller's `style` into the object and is not flagged; leave it as generated.

- [ ] **Step 4: Mount the Toaster and TooltipProvider in the root route**

`frontend/app/root.tsx`:

```tsx
import type { ReactNode } from "react"
import { Links, Meta, Outlet, Scripts, ScrollRestoration } from "react-router"

import { Toaster } from "~/components/ui/toast"
import { TooltipProvider } from "~/components/ui/tooltip"

import type { Route } from "./+types/root"
import stylesheet from "./app.css?url"

export const links: Route.LinksFunction = () => [
  { rel: "stylesheet", href: stylesheet },
]

export function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="vi">
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        {children}
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  )
}

export default function App() {
  return (
    <Toaster>
      <TooltipProvider>
        <Outlet />
      </TooltipProvider>
    </Toaster>
  )
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project browser app/components/ui/toast.browser.test.tsx`
Expected: PASS, `Tests 1 passed (1)`.

- [ ] **Step 6: Full check**

```bash
cd /home/andv/personal/thesis/frontend
npm run format:write
npm run lint && npm run format && npm run typecheck && npm test
```

Expected: all succeed. Lint findings inside the generated components are fixed in those files (they are part of this repo now), never silenced.

- [ ] **Step 7: Commit**

```bash
cd /home/andv/personal/thesis
git add frontend/app/components/ui frontend/app/components/elements/.gitkeep frontend/app/hooks/use-mobile.ts \
  frontend/app/root.tsx frontend/package.json frontend/package-lock.json
git commit -m "feat(frontend): add shadcn base components with toaster and tooltip provider" \
  -m "$COMMIT_TRAILER"
```

---

### Task 3: i18n with remix-i18next, locale cookie action and `<html lang>`

No backend needed.

**Files:**
- Create: `frontend/app/i18n/config.ts`, `frontend/app/i18n/types.d.ts`, `frontend/app/i18n/locale.server.ts`, `frontend/app/i18n/middleware.server.ts`, `frontend/app/i18n/zod-locale.ts`
- Create: `frontend/app/i18n/resources/vi/{common,auth,chat,citations,skills,settings,landing,errors}.json`, `frontend/app/i18n/resources/en/{common,auth,chat,citations,skills,settings,landing,errors}.json`
- Create: `frontend/app/components/document.tsx`, `frontend/app/entry.server.tsx`, `frontend/app/entry.client.tsx`, `frontend/app/routes/actions/locale.ts`, `frontend/tests/utils/i18n.ts`
- Modify: `frontend/app/root.tsx`, `frontend/app/routes.ts`
- Test: `frontend/app/i18n/resources.test.ts`, `frontend/app/i18n/config.test.ts`, `frontend/app/i18n/locale.server.test.ts`, `frontend/app/i18n/zod-locale.test.ts`, `frontend/app/components/document.test.tsx`

**Interfaces:**
- Consumes: `Toaster`, `TooltipProvider` (Task 2).
- Produces:
  - `~/i18n/config`: `SUPPORTED_LANGUAGES = ["vi", "en"] as const`; `type Language = "vi" | "en"`; `FALLBACK_LANGUAGE: Language = "vi"`; `NAMESPACES = ["common", "auth", "chat", "citations", "skills", "settings", "landing", "errors"] as const`; `type Namespace`; `resources` (bundled JSON, `vi` is the key source of truth); `i18nextOptions` (i18next `InitOptions` without `lng`); `isLanguage(value: unknown): value is Language`; `localeFromPath(pathname: string): Language | null` (`/en` and `/en/...` → `"en"`, anything else → `null`).
  - `~/i18n/locale.server`: `localeCookie: Cookie` (name `lng`); `setLocale(request: Request): Promise<Response>` (form field `locale`; 200 with `Set-Cookie`, or 400).
  - `~/i18n/middleware.server`: `i18nextMiddleware`, `getLocale(context): string`, `getInstance(context): i18n` (detection order: URL prefix → cookie `lng` → `Accept-Language` → `vi`).
  - `~/i18n/zod-locale`: `applyZodLocale(language: string): void`.
  - `~/components/document`: `Document({ children }: { children: ReactNode })` renders `<html lang dir>`.
  - `tests/utils/i18n.ts`: `createTestI18n(language?: Language): i18n` (initialised synchronously with `initReactI18next`).
  - Route `POST /actions/locale` (id `routes/actions/locale`). Root loader data: `{ locale: string }`.
  - `common` keys: `appName`, `actions.{save,cancel,retry,close,loading}`, `nav.{chat,skills,settings}`, `errorPage.{title,description,notFoundTitle,notFoundDescription,backHome}`; `errors` key `UNKNOWN`. Other namespaces start as `{}`; later tasks and plans add keys to `vi` and `en` together.

- [ ] **Step 1: Write the failing tests**

`frontend/app/i18n/resources.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { NAMESPACES, resources } from "~/i18n/config"

function keysOf(value: unknown, prefix = ""): string[] {
  if (typeof value !== "object" || value === null) {
    return [prefix]
  }
  return Object.entries(value).flatMap(([key, child]) =>
    keysOf(child, prefix === "" ? key : `${prefix}.${key}`)
  )
}

describe("translation resources", () => {
  test("both languages register every namespace", () => {
    const expected = [...NAMESPACES].toSorted()
    expect(Object.keys(resources.vi).toSorted()).toEqual(expected)
    expect(Object.keys(resources.en).toSorted()).toEqual(expected)
  })

  test.each(NAMESPACES)("vi and en define the same keys in %s", (namespace) => {
    expect(keysOf(resources.en[namespace]).toSorted()).toEqual(
      keysOf(resources.vi[namespace]).toSorted()
    )
  })
})
```

`frontend/app/i18n/config.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { isLanguage, localeFromPath } from "~/i18n/config"

describe("localeFromPath", () => {
  test.each([
    ["/en", "en"],
    ["/en/", "en"],
    ["/en/features", "en"],
    ["/", null],
    ["/vi", null],
    ["/chat", null],
    ["/english", null],
  ] as const)("%s -> %s", (pathname, expected) => {
    expect(localeFromPath(pathname)).toBe(expected)
  })
})

describe("isLanguage", () => {
  test("accepts only supported languages", () => {
    expect(isLanguage("vi")).toBe(true)
    expect(isLanguage("en")).toBe(true)
    expect(isLanguage("fr")).toBe(false)
    expect(isLanguage(undefined)).toBe(false)
  })
})
```

`frontend/app/i18n/locale.server.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { localeCookie, setLocale } from "~/i18n/locale.server"

function localeRequest(locale: string): Request {
  return new Request("http://localhost/actions/locale", {
    method: "POST",
    body: new URLSearchParams({ locale }),
  })
}

describe("setLocale", () => {
  test("stores a supported language in the lng cookie", async () => {
    const response = await setLocale(localeRequest("en"))

    expect(response.status).toBe(200)
    const setCookie = response.headers.get("Set-Cookie")
    expect(setCookie).toContain("lng=")
    expect(await localeCookie.parse(setCookie)).toBe("en")
  })

  test("rejects an unsupported language and sets no cookie", async () => {
    const response = await setLocale(localeRequest("fr"))

    expect(response.status).toBe(400)
    expect(response.headers.get("Set-Cookie")).toBeNull()
  })
})
```

`frontend/app/i18n/zod-locale.test.ts`:

```ts
import { afterEach, describe, expect, test } from "vitest"
import { z } from "zod"

import { applyZodLocale } from "~/i18n/zod-locale"

function tooShortMessage(): string | undefined {
  return z.string().min(3).safeParse("a").error?.issues[0]?.message
}

describe("applyZodLocale", () => {
  afterEach(() => {
    applyZodLocale("en")
  })

  test("uses Vietnamese validation messages for vi", () => {
    applyZodLocale("vi")
    expect(tooShortMessage()).toBe("Quá nhỏ: mong đợi string có >=3 ký tự")
  })

  test("uses English validation messages for en", () => {
    applyZodLocale("en")
    expect(tooShortMessage()).toBe(
      "Too small: expected string to have >=3 characters"
    )
  })
})
```

`frontend/app/components/document.test.tsx`:

```tsx
import { renderToString } from "react-dom/server"
import { I18nextProvider } from "react-i18next"
import { describe, expect, test } from "vitest"

import { Document } from "~/components/document"

import { createTestI18n } from "../../tests/utils/i18n"

describe("Document", () => {
  test.each(["vi", "en"] as const)("sets lang and dir for %s", (language) => {
    const html = renderToString(
      <I18nextProvider i18n={createTestI18n(language)}>
        <Document>
          <body />
        </Document>
      </I18nextProvider>
    )

    expect(html).toContain(`<html lang="${language}" dir="ltr">`)
  })
})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project unit app/i18n app/components/document.test.tsx`
Expected: FAIL, 5 files, each with `Failed to resolve import` for `~/i18n/config`, `~/i18n/locale.server`, `~/i18n/zod-locale` or `~/components/document`.

- [ ] **Step 3: Write the translation resources**

`frontend/app/i18n/resources/vi/common.json`:

```json
{
  "appName": "Pharma Agent",
  "actions": {
    "save": "Lưu",
    "cancel": "Huỷ",
    "retry": "Thử lại",
    "close": "Đóng",
    "loading": "Đang tải…"
  },
  "nav": {
    "chat": "Trò chuyện",
    "skills": "Kỹ năng",
    "settings": "Cài đặt"
  },
  "errorPage": {
    "title": "Đã xảy ra lỗi",
    "description": "Không tải được trang này. Vui lòng thử lại.",
    "notFoundTitle": "Không tìm thấy trang",
    "notFoundDescription": "Trang bạn tìm không tồn tại hoặc đã bị xoá.",
    "backHome": "Về trang chủ"
  }
}
```

`frontend/app/i18n/resources/en/common.json`:

```json
{
  "appName": "Pharma Agent",
  "actions": {
    "save": "Save",
    "cancel": "Cancel",
    "retry": "Try again",
    "close": "Close",
    "loading": "Loading…"
  },
  "nav": {
    "chat": "Chat",
    "skills": "Skills",
    "settings": "Settings"
  },
  "errorPage": {
    "title": "Something went wrong",
    "description": "This page could not be loaded. Please try again.",
    "notFoundTitle": "Page not found",
    "notFoundDescription": "The page you are looking for does not exist or was removed.",
    "backHome": "Back to home"
  }
}
```

`frontend/app/i18n/resources/vi/errors.json`:

```json
{
  "UNKNOWN": "Đã xảy ra lỗi. Vui lòng thử lại."
}
```

`frontend/app/i18n/resources/en/errors.json`:

```json
{
  "UNKNOWN": "Something went wrong. Please try again."
}
```

Each of these twelve files contains exactly `{}` followed by a newline: `frontend/app/i18n/resources/vi/{auth,chat,citations,skills,settings,landing}.json` and `frontend/app/i18n/resources/en/{auth,chat,citations,skills,settings,landing}.json`.

- [ ] **Step 4: Write the i18n modules, Document and the locale action**

`frontend/app/i18n/config.ts`:

```ts
import type { InitOptions, Resource } from "i18next"

import enAuth from "./resources/en/auth.json"
import enChat from "./resources/en/chat.json"
import enCitations from "./resources/en/citations.json"
import enCommon from "./resources/en/common.json"
import enErrors from "./resources/en/errors.json"
import enLanding from "./resources/en/landing.json"
import enSettings from "./resources/en/settings.json"
import enSkills from "./resources/en/skills.json"
import viAuth from "./resources/vi/auth.json"
import viChat from "./resources/vi/chat.json"
import viCitations from "./resources/vi/citations.json"
import viCommon from "./resources/vi/common.json"
import viErrors from "./resources/vi/errors.json"
import viLanding from "./resources/vi/landing.json"
import viSettings from "./resources/vi/settings.json"
import viSkills from "./resources/vi/skills.json"

export const SUPPORTED_LANGUAGES = ["vi", "en"] as const
export type Language = (typeof SUPPORTED_LANGUAGES)[number]
export const FALLBACK_LANGUAGE: Language = "vi"

export const NAMESPACES = [
  "common",
  "auth",
  "chat",
  "citations",
  "skills",
  "settings",
  "landing",
  "errors",
] as const
export type Namespace = (typeof NAMESPACES)[number]

export const resources = {
  vi: {
    common: viCommon,
    auth: viAuth,
    chat: viChat,
    citations: viCitations,
    skills: viSkills,
    settings: viSettings,
    landing: viLanding,
    errors: viErrors,
  },
  en: {
    common: enCommon,
    auth: enAuth,
    chat: enChat,
    citations: enCitations,
    skills: enSkills,
    settings: enSettings,
    landing: enLanding,
    errors: enErrors,
  },
} satisfies Resource

export const i18nextOptions = {
  fallbackLng: FALLBACK_LANGUAGE,
  supportedLngs: SUPPORTED_LANGUAGES,
  ns: [...NAMESPACES],
  defaultNS: "common",
  resources,
  interpolation: { escapeValue: false },
} satisfies InitOptions

export function isLanguage(value: unknown): value is Language {
  return SUPPORTED_LANGUAGES.some((language) => language === value)
}

/** Only public pages carry a language prefix, and only for non-default languages. */
export function localeFromPath(pathname: string): Language | null {
  const firstSegment = pathname.split("/")[1]
  return SUPPORTED_LANGUAGES.find(
    (language) => language !== FALLBACK_LANGUAGE && language === firstSegment
  ) ?? null
}
```

`frontend/app/i18n/types.d.ts` (the type import makes this file a module, so `declare module` augments i18next instead of replacing it; a bare `import "i18next"` would trip `import/no-unassigned-import`):

```ts
import type { resources } from "./config"

declare module "i18next" {
  interface CustomTypeOptions {
    defaultNS: "common"
    resources: (typeof resources)["vi"]
  }
}
```

`frontend/app/i18n/locale.server.ts`:

```ts
import { createCookie } from "react-router"
import { z } from "zod"

import { SUPPORTED_LANGUAGES } from "~/i18n/config"

const ONE_YEAR_SECONDS = 60 * 60 * 24 * 365

export const localeCookie = createCookie("lng", {
  path: "/",
  sameSite: "lax",
  httpOnly: true,
  secure: process.env.NODE_ENV === "production",
  maxAge: ONE_YEAR_SECONDS,
})

const localeForm = z.object({ locale: z.enum(SUPPORTED_LANGUAGES) })

export async function setLocale(request: Request): Promise<Response> {
  const parsed = localeForm.safeParse(
    Object.fromEntries(await request.formData())
  )
  if (!parsed.success) {
    return Response.json({ ok: false }, { status: 400 })
  }
  return Response.json(
    { ok: true },
    {
      headers: {
        "Set-Cookie": await localeCookie.serialize(parsed.data.locale),
      },
    }
  )
}
```

`frontend/app/i18n/middleware.server.ts`:

```ts
import { initReactI18next } from "react-i18next"
import { createI18nextMiddleware } from "remix-i18next"

import {
  FALLBACK_LANGUAGE,
  SUPPORTED_LANGUAGES,
  i18nextOptions,
  localeFromPath,
} from "~/i18n/config"
import { localeCookie } from "~/i18n/locale.server"

export const [i18nextMiddleware, getLocale, getInstance] =
  createI18nextMiddleware({
    detection: {
      supportedLanguages: [...SUPPORTED_LANGUAGES],
      fallbackLanguage: FALLBACK_LANGUAGE,
      cookie: localeCookie,
      order: ["custom", "cookie", "header"],
      findLocale: ({ request }) =>
        Promise.resolve(localeFromPath(new URL(request.url).pathname)),
    },
    i18next: i18nextOptions,
    plugins: [initReactI18next],
  })
```

`frontend/app/i18n/zod-locale.ts`:

```ts
import { z } from "zod"

/** Form validation messages follow the UI language. */
export function applyZodLocale(language: string): void {
  z.config(language === "en" ? z.locales.en() : z.locales.vi())
}
```

`frontend/app/components/document.tsx`:

```tsx
import type { ReactNode } from "react"
import { useTranslation } from "react-i18next"

export function Document({ children }: { children: ReactNode }) {
  const { i18n } = useTranslation()
  return (
    <html lang={i18n.language} dir={i18n.dir(i18n.language)}>
      {children}
    </html>
  )
}
```

`frontend/tests/utils/i18n.ts`:

```ts
import { createInstance, type i18n } from "i18next"
import { initReactI18next } from "react-i18next"

import { i18nextOptions, type Language } from "~/i18n/config"

export function createTestI18n(language: Language = "vi"): i18n {
  const instance = createInstance()
  void instance
    .use(initReactI18next)
    .init({ ...i18nextOptions, lng: language, initAsync: false })
  return instance
}
```

`frontend/app/routes/actions/locale.ts`:

```ts
import { setLocale } from "~/i18n/locale.server"

import type { Route } from "./+types/locale"

export function action({ request }: Route.ActionArgs) {
  return setLocale(request)
}
```

- [ ] **Step 5: Run the unit tests to verify they pass**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project unit app/i18n app/components/document.test.tsx`
Expected: PASS, `Test Files 5 passed (5)`.

- [ ] **Step 6: Wire i18n into the server entry, client entry, root route and route table**

`frontend/app/entry.server.tsx` (the file `react-router reveal entry.server` prints for 8.3.1, with `I18nextProvider` added and the rejection typed as `Error`):

```tsx
import { PassThrough } from "node:stream"

import { createReadableStreamFromReadable } from "@react-router/node"
import { isbot } from "isbot"
import type { RenderToPipeableStreamOptions } from "react-dom/server"
import { renderToPipeableStream } from "react-dom/server"
import { I18nextProvider } from "react-i18next"
import type { EntryContext, RouterContextProvider } from "react-router"
import { ServerRouter } from "react-router"

import { getInstance } from "~/i18n/middleware.server"

export const streamTimeout = 5_000

export default function handleRequest(
  request: Request,
  responseStatusCode: number,
  responseHeaders: Headers,
  routerContext: EntryContext,
  loadContext: RouterContextProvider
) {
  // https://httpwg.org/specs/rfc9110.html#HEAD
  if (request.method.toUpperCase() === "HEAD") {
    return new Response(null, {
      status: responseStatusCode,
      headers: responseHeaders,
    })
  }

  return new Promise<Response>((resolve, reject) => {
    let shellRendered = false
    let statusCode = responseStatusCode
    const userAgent = request.headers.get("user-agent")

    // Bots and SPA mode wait for all content so crawlers get the full page.
    const readyOption: keyof RenderToPipeableStreamOptions =
      (userAgent !== null && isbot(userAgent)) || routerContext.isSpaMode
        ? "onAllReady"
        : "onShellReady"

    let timeoutId: ReturnType<typeof setTimeout> | undefined = setTimeout(
      () => {
        abort()
      },
      streamTimeout + 1000
    )

    const { pipe, abort } = renderToPipeableStream(
      <I18nextProvider i18n={getInstance(loadContext)}>
        <ServerRouter context={routerContext} url={request.url} />
      </I18nextProvider>,
      {
        [readyOption]() {
          shellRendered = true
          const body = new PassThrough({
            final(callback) {
              clearTimeout(timeoutId)
              timeoutId = undefined
              callback()
            },
          })
          const stream = createReadableStreamFromReadable(body)

          responseHeaders.set("Content-Type", "text/html")

          pipe(body)

          resolve(
            new Response(stream, {
              headers: responseHeaders,
              status: statusCode,
            })
          )
        },
        onShellError(error: unknown) {
          reject(error instanceof Error ? error : new Error(String(error)))
        },
        onError(error: unknown) {
          statusCode = 500
          // Errors before the shell reject above and are logged by React Router.
          if (shellRendered) {
            console.error(error)
          }
        },
      }
    )
  })
}
```

`frontend/app/entry.client.tsx` (bundled resources, language from the server-rendered `<html lang>`):

```tsx
import i18next from "i18next"
import { StrictMode, startTransition } from "react"
import { hydrateRoot } from "react-dom/client"
import { I18nextProvider, initReactI18next } from "react-i18next"
import { HydratedRouter } from "react-router/dom"

import { FALLBACK_LANGUAGE, i18nextOptions } from "~/i18n/config"
import { applyZodLocale } from "~/i18n/zod-locale"

async function hydrate() {
  await i18next.use(initReactI18next).init({
    ...i18nextOptions,
    lng: document.documentElement.lang || FALLBACK_LANGUAGE,
  })
  applyZodLocale(i18next.language)
  i18next.on("languageChanged", applyZodLocale)

  startTransition(() => {
    hydrateRoot(
      document,
      <I18nextProvider i18n={i18next}>
        <StrictMode>
          <HydratedRouter />
        </StrictMode>
      </I18nextProvider>
    )
  })
}

hydrate().catch((error: unknown) => {
  console.error(error)
})
```

`frontend/app/root.tsx`:

```tsx
import { useEffect, type ReactNode } from "react"
import { useTranslation } from "react-i18next"
import {
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  isRouteErrorResponse,
} from "react-router"

import { Document } from "~/components/document"
import { Toaster } from "~/components/ui/toast"
import { TooltipProvider } from "~/components/ui/tooltip"
import { getLocale, i18nextMiddleware } from "~/i18n/middleware.server"

import type { Route } from "./+types/root"
import stylesheet from "./app.css?url"

export const links: Route.LinksFunction = () => [
  { rel: "stylesheet", href: stylesheet },
]

export const middleware = [i18nextMiddleware]

export function loader({ context }: Route.LoaderArgs) {
  return { locale: getLocale(context) }
}

export function Layout({ children }: { children: ReactNode }) {
  return (
    <Document>
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        {children}
        <ScrollRestoration />
        <Scripts />
      </body>
    </Document>
  )
}

export default function App({ loaderData }: Route.ComponentProps) {
  const { i18n } = useTranslation()

  useEffect(() => {
    if (i18n.language !== loaderData.locale) {
      void i18n.changeLanguage(loaderData.locale)
    }
  }, [i18n, loaderData.locale])

  return (
    <Toaster>
      <TooltipProvider>
        <Outlet />
      </TooltipProvider>
    </Toaster>
  )
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  const { t } = useTranslation("common")
  const notFound = isRouteErrorResponse(error) && error.status === 404

  return (
    <main className="mx-auto flex min-h-svh max-w-md flex-col justify-center gap-2 p-6">
      <h1 className="text-2xl font-semibold">
        {notFound ? t("errorPage.notFoundTitle") : t("errorPage.title")}
      </h1>
      <p className="text-muted-foreground">
        {notFound
          ? t("errorPage.notFoundDescription")
          : t("errorPage.description")}
      </p>
      {import.meta.env.DEV && error instanceof Error ? (
        <pre className="overflow-x-auto text-xs">{error.stack}</pre>
      ) : null}
    </main>
  )
}
```

`frontend/app/routes.ts`:

```ts
import { type RouteConfig, index, route } from "@react-router/dev/routes"

export default [
  index("routes/public/landing.tsx", { id: "landing-vi" }),
  route("actions/locale", "routes/actions/locale.ts"),
] satisfies RouteConfig
```

- [ ] **Step 7: Verify detection and the locale cookie against the built server**

```bash
cd /home/andv/personal/thesis/frontend
npm run build
PORT=3100 npm start > /tmp/frontend-start.log 2>&1 &
SERVER_PID=$!
timeout 30 bash -c 'until curl -sf http://localhost:3100/ > /dev/null; do sleep 0.5; done'
curl -s http://localhost:3100/ | grep -o '<html[^>]*>'
curl -s -H 'Accept-Language: en-US,en;q=0.9' http://localhost:3100/ | grep -o '<html[^>]*>'
curl -s -i -X POST -d 'locale=en' http://localhost:3100/actions/locale | grep -i '^set-cookie'
kill "$SERVER_PID"
```

Expected, in order: `<html lang="vi" dir="ltr">`, `<html lang="en" dir="ltr">`, `set-cookie: lng=...; Max-Age=31536000; Path=/; HttpOnly; SameSite=Lax`.

- [ ] **Step 8: Full check**

```bash
cd /home/andv/personal/thesis/frontend
npm run format:write
npm run lint && npm run format && npm run typecheck && npm test
```

Expected: all succeed; `tsc` accepts `t("errorPage.title")` against the typed `common` namespace.

- [ ] **Step 9: Commit**

```bash
cd /home/andv/personal/thesis
git add frontend/app/i18n frontend/app/components/document.tsx frontend/app/components/document.test.tsx \
  frontend/app/entry.server.tsx frontend/app/entry.client.tsx frontend/app/routes/actions/locale.ts \
  frontend/app/root.tsx frontend/app/routes.ts frontend/tests/utils/i18n.ts
git commit -m "feat(frontend): add i18n with remix-i18next, locale cookie action and Vietnamese defaults" \
  -m "$COMMIT_TRAILER"
```

---

### Task 4: Light, dark and system theme with remix-themes

No backend needed. This is the SSR theme check on React Router 8 that spec §15 asks for.

**Files:**
- Create: `frontend/app/lib/theme.server.ts`, `frontend/app/routes/actions/theme.ts`
- Modify: `frontend/app/components/document.tsx`, `frontend/app/components/document.test.tsx`, `frontend/app/root.tsx`, `frontend/app/routes.ts`
- Test: `frontend/app/lib/theme.server.test.ts`, `frontend/app/components/document.test.tsx`

**Interfaces:**
- Consumes: `Document`, `getLocale`, `i18nextMiddleware` (Task 3); `createTestI18n` (Task 3).
- Produces:
  - `~/lib/theme.server`: `themeSessionResolver: ThemeSessionResolver` (remix-themes, cookie session named `theme`, signed).
  - Route `POST /actions/theme` (remix-themes action; JSON body `{"theme": "light" | "dark" | null}`, `null` means follow the system).
  - Root loader data becomes `{ locale: string; theme: Theme | null }`. `ThemeProvider` wraps the document with `themeAction="/actions/theme"`; components change the theme with `const [theme, setTheme, { definedBy }] = useTheme()` from `remix-themes` (Plan 10 settings page).
  - `Document` adds `class="light" | "dark"` on `<html>`; prerendered pages have no stored theme, so `PreventFlashOnWrongTheme` applies the system theme before hydration.

- [ ] **Step 1: Write the failing tests**

`frontend/app/lib/theme.server.test.ts`:

```ts
import { Theme } from "remix-themes"
import { describe, expect, test } from "vitest"

import { themeSessionResolver } from "~/lib/theme.server"

function requestWithCookie(cookie: string | null): Request {
  return new Request("http://localhost/", {
    headers: cookie === null ? {} : { Cookie: cookie },
  })
}

describe("themeSessionResolver", () => {
  test("has no theme until the user picks one", async () => {
    const session = await themeSessionResolver(requestWithCookie(null))
    expect(session.getTheme()).toBeNull()
  })

  test("round-trips the chosen theme through the signed cookie", async () => {
    const session = await themeSessionResolver(requestWithCookie(null))
    session.setTheme(Theme.DARK)
    const setCookie = await session.commit()

    expect(setCookie).toMatch(/^theme=/)
    const cookiePair = setCookie.split(";")[0] ?? ""
    const next = await themeSessionResolver(requestWithCookie(cookiePair))
    expect(next.getTheme()).toBe(Theme.DARK)
  })

  test("ignores a cookie that was not signed by the server", async () => {
    const session = await themeSessionResolver(
      requestWithCookie("theme=eyJ0aGVtZSI6ImRhcmsifQ%3D%3D")
    )
    expect(session.getTheme()).toBeNull()
  })
})
```

`frontend/app/components/document.test.tsx` (replaces the Task 3 version; `Document` now needs a `ThemeProvider`):

```tsx
import { renderToString } from "react-dom/server"
import { I18nextProvider } from "react-i18next"
import { Theme, ThemeProvider } from "remix-themes"
import { describe, expect, test } from "vitest"

import { Document } from "~/components/document"
import type { Language } from "~/i18n/config"

import { createTestI18n } from "../../tests/utils/i18n"

function renderDocument(language: Language, theme: Theme | null): string {
  return renderToString(
    <I18nextProvider i18n={createTestI18n(language)}>
      <ThemeProvider specifiedTheme={theme} themeAction="/actions/theme">
        <Document>
          <body />
        </Document>
      </ThemeProvider>
    </I18nextProvider>
  )
}

describe("Document", () => {
  test.each(["vi", "en"] as const)("sets lang and dir for %s", (language) => {
    expect(renderDocument(language, null)).toContain(
      `<html lang="${language}" dir="ltr">`
    )
  })

  test("renders the stored theme as the html class on the server", () => {
    expect(renderDocument("vi", Theme.DARK)).toContain(
      '<html lang="vi" dir="ltr" class="dark">'
    )
    expect(renderDocument("vi", Theme.LIGHT)).toContain(
      '<html lang="vi" dir="ltr" class="light">'
    )
  })
})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project unit app/lib/theme.server.test.ts app/components/document.test.tsx`
Expected: FAIL. `theme.server.test.ts`: `Failed to resolve import "~/lib/theme.server"`. `document.test.tsx`: the two theme assertions fail because `<html>` has no `class`.

- [ ] **Step 3: Implement the theme session, action and Document class**

`frontend/app/lib/theme.server.ts`:

```ts
import { createCookieSessionStorage } from "react-router"
import { createThemeSessionResolver } from "remix-themes"

const ONE_YEAR_SECONDS = 60 * 60 * 24 * 365

export const themeSessionResolver = createThemeSessionResolver(
  createCookieSessionStorage({
    cookie: {
      name: "theme",
      path: "/",
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
      maxAge: ONE_YEAR_SECONDS,
      // The cookie only holds "light" or "dark". React Router warns about unsigned
      // session cookies, so it is signed; the value is not sensitive.
      secrets: [process.env.THEME_COOKIE_SECRET ?? "pharma-agent-theme"],
    },
  })
)
```

`frontend/app/routes/actions/theme.ts`:

```ts
import { createThemeAction } from "remix-themes"

import { themeSessionResolver } from "~/lib/theme.server"

export const action = createThemeAction(themeSessionResolver)
```

`frontend/app/components/document.tsx`:

```tsx
import type { ReactNode } from "react"
import { useTranslation } from "react-i18next"
import { useTheme } from "remix-themes"

export function Document({ children }: { children: ReactNode }) {
  const { i18n } = useTranslation()
  const [theme] = useTheme()

  return (
    <html
      lang={i18n.language}
      dir={i18n.dir(i18n.language)}
      className={theme ?? undefined}
    >
      {children}
    </html>
  )
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project unit app/lib/theme.server.test.ts app/components/document.test.tsx`
Expected: PASS, `Test Files 2 passed (2)`, `Tests 6 passed (6)`.

- [ ] **Step 5: Load the theme in the root route and register the action**

`frontend/app/root.tsx`:

```tsx
import { useEffect, type ReactNode } from "react"
import { useTranslation } from "react-i18next"
import {
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  isRouteErrorResponse,
  useRouteLoaderData,
} from "react-router"
import { PreventFlashOnWrongTheme, ThemeProvider } from "remix-themes"

import { Document } from "~/components/document"
import { Toaster } from "~/components/ui/toast"
import { TooltipProvider } from "~/components/ui/tooltip"
import { getLocale, i18nextMiddleware } from "~/i18n/middleware.server"
import { themeSessionResolver } from "~/lib/theme.server"

import type { Route } from "./+types/root"
import stylesheet from "./app.css?url"

export const links: Route.LinksFunction = () => [
  { rel: "stylesheet", href: stylesheet },
]

export const middleware = [i18nextMiddleware]

export async function loader({ request, context }: Route.LoaderArgs) {
  const { getTheme } = await themeSessionResolver(request)
  return { locale: getLocale(context), theme: getTheme() }
}

export function Layout({ children }: { children: ReactNode }) {
  const data = useRouteLoaderData<typeof loader>("root")
  const theme = data?.theme ?? null

  return (
    <ThemeProvider
      specifiedTheme={theme}
      themeAction="/actions/theme"
      disableTransitionOnThemeChange
    >
      <Document>
        <head>
          <meta charSet="utf-8" />
          <meta
            name="viewport"
            content="width=device-width, initial-scale=1"
          />
          <Meta />
          <PreventFlashOnWrongTheme ssrTheme={theme !== null} />
          <Links />
        </head>
        <body>
          {children}
          <ScrollRestoration />
          <Scripts />
        </body>
      </Document>
    </ThemeProvider>
  )
}

export default function App({ loaderData }: Route.ComponentProps) {
  const { i18n } = useTranslation()

  useEffect(() => {
    if (i18n.language !== loaderData.locale) {
      void i18n.changeLanguage(loaderData.locale)
    }
  }, [i18n, loaderData.locale])

  return (
    <Toaster>
      <TooltipProvider>
        <Outlet />
      </TooltipProvider>
    </Toaster>
  )
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  const { t } = useTranslation("common")
  const notFound = isRouteErrorResponse(error) && error.status === 404

  return (
    <main className="mx-auto flex min-h-svh max-w-md flex-col justify-center gap-2 p-6">
      <h1 className="text-2xl font-semibold">
        {notFound ? t("errorPage.notFoundTitle") : t("errorPage.title")}
      </h1>
      <p className="text-muted-foreground">
        {notFound
          ? t("errorPage.notFoundDescription")
          : t("errorPage.description")}
      </p>
      {import.meta.env.DEV && error instanceof Error ? (
        <pre className="overflow-x-auto text-xs">{error.stack}</pre>
      ) : null}
    </main>
  )
}
```

`frontend/app/routes.ts`:

```ts
import { type RouteConfig, index, route } from "@react-router/dev/routes"

export default [
  index("routes/public/landing.tsx", { id: "landing-vi" }),
  route("actions/locale", "routes/actions/locale.ts"),
  route("actions/theme", "routes/actions/theme.ts"),
] satisfies RouteConfig
```

- [ ] **Step 6: Verify the theme cookie and SSR class against the built server**

```bash
cd /home/andv/personal/thesis/frontend
npm run build
PORT=3100 npm start > /tmp/frontend-start.log 2>&1 &
SERVER_PID=$!
timeout 30 bash -c 'until curl -sf http://localhost:3100/ > /dev/null; do sleep 0.5; done'
curl -s http://localhost:3100/ | grep -o '<html[^>]*>\|<script>[^<]*prefers-color-scheme' | head -2
THEME_COOKIE=$(curl -s -i -X POST -H 'Content-Type: application/json' -d '{"theme":"dark"}' \
  http://localhost:3100/actions/theme | grep -i '^set-cookie' | sed -E 's/^[Ss]et-[Cc]ookie: ([^;]+);.*/\1/')
curl -s -H "Cookie: $THEME_COOKIE" http://localhost:3100/ | grep -o '<html[^>]*>'
kill "$SERVER_PID"
cat /tmp/frontend-start.log
```

Expected: first `<html lang="vi" dir="ltr">` followed by the inline flash-guard script (it contains `prefers-color-scheme`); then `<html lang="vi" dir="ltr" class="dark">`; the server log has no warning about unsigned session cookies.

- [ ] **Step 7: Full check**

```bash
cd /home/andv/personal/thesis/frontend
npm run format:write
npm run lint && npm run format && npm run typecheck && npm test
```

Expected: all succeed.

- [ ] **Step 8: Commit**

```bash
cd /home/andv/personal/thesis
git add frontend/app/lib/theme.server.ts frontend/app/lib/theme.server.test.ts frontend/app/routes/actions/theme.ts \
  frontend/app/components/document.tsx frontend/app/components/document.test.tsx frontend/app/root.tsx frontend/app/routes.ts
git commit -m "feat(frontend): add light, dark and system theme with remix-themes" \
  -m "$COMMIT_TRAILER"
```

---

### Task 5: CSRF cookie reader, problem+json errors and translated error messages

No backend needed. Covers the spec §13 unit rows "chuyển problem+json thành `ApiError`" and "đọc cookie CSRF".

**Files:**
- Create: `frontend/app/lib/csrf.ts`, `frontend/app/api/problem.ts`, `frontend/app/i18n/error-message.ts`
- Modify: `frontend/app/i18n/resources/vi/errors.json`, `frontend/app/i18n/resources/en/errors.json`
- Test: `frontend/app/lib/csrf.test.ts`, `frontend/app/lib/csrf.browser.test.tsx`, `frontend/app/api/problem.test.ts`, `frontend/app/i18n/error-message.test.ts`

**Interfaces:**
- Consumes: `resources` (Task 3), `createTestI18n` (Task 3).
- Produces:
  - `~/lib/csrf`: `CSRF_COOKIE = "csrftoken"`, `CSRF_HEADER = "x-csrftoken"`, `readCsrfToken(): string | undefined` (`undefined` on the server or before the backend set the cookie).
  - `~/api/problem`: `type ProblemItem = { loc: (string | number)[]; message: string; type: string }`; `UNKNOWN_ERROR_CODE = "UNKNOWN"`; `class ApiError extends Error { readonly status: number; readonly code: string; readonly detail: string | undefined; readonly errors: ProblemItem[] }` with `constructor(init: { status: number; code: string; title: string; detail?: string | undefined; errors?: ProblemItem[] })` (`message` is the problem `title`); `isApiError(value: unknown): value is ApiError`; `toApiError(response: Response): Promise<ApiError>`.
  - `~/i18n/error-message`: `type ErrorCode` (keys of the `errors` namespace); `isErrorCode(code: string): code is ErrorCode`; `apiErrorMessage(error: unknown, t: TFunction<"errors">): string` (translated by `code`, else the problem `title`, else `errors:UNKNOWN`).
  - `errors` namespace keys: `UNKNOWN`, `UNAUTHORIZED`, `CSRF_FAILED`, `VALIDATION_ERROR`, `INVALID_INPUT`, `INVALID_CURSOR`, `CONVERSATION_NOT_FOUND`, `MESSAGE_NOT_FOUND`, `CITATION_NOT_FOUND`, `SKILL_NOT_FOUND`, `SKILL_NAME_TAKEN`, `SKILL_PARSE_ERROR`, `PAYLOAD_TOO_LARGE`, `AGENT_UNAVAILABLE`, `REGISTER_USER_ALREADY_EXISTS`, `REGISTER_INVALID_PASSWORD`, `LOGIN_BAD_CREDENTIALS`, `LOGIN_USER_NOT_VERIFIED`, `OAUTH_NOT_AVAILABLE_EMAIL`, `OAUTH_USER_ALREADY_EXISTS`, `OAUTH_INVALID_STATE`, `ACCESS_TOKEN_DECODE_ERROR`, `ACCESS_TOKEN_ALREADY_EXPIRED`, `UPDATE_USER_EMAIL_ALREADY_EXISTS`, `UPDATE_USER_INVALID_PASSWORD`. Plans 9 and 10 add codes they display to both languages.

- [ ] **Step 1: Write the failing tests**

`frontend/app/lib/csrf.test.ts`:

```ts
import { expect, test } from "vitest"

import { readCsrfToken } from "~/lib/csrf"

test("readCsrfToken returns undefined on the server", () => {
  expect(readCsrfToken()).toBeUndefined()
})
```

`frontend/app/lib/csrf.browser.test.tsx`:

```ts
import { afterEach, describe, expect, test } from "vitest"

import { CSRF_COOKIE, CSRF_HEADER, readCsrfToken } from "~/lib/csrf"

function expireCookie(name: string): void {
  document.cookie = `${name}=; path=/; max-age=0`
}

describe("readCsrfToken", () => {
  afterEach(() => {
    expireCookie(CSRF_COOKIE)
    expireCookie("theme")
  })

  test("returns undefined before the backend sets the cookie", () => {
    expect(readCsrfToken()).toBeUndefined()
  })

  test("reads csrftoken among other cookies", () => {
    document.cookie = "theme=dark; path=/"
    document.cookie = "csrftoken=abc123; path=/"
    expect(readCsrfToken()).toBe("abc123")
  })

  test("matches the backend CSRF middleware names", () => {
    expect(CSRF_COOKIE).toBe("csrftoken")
    expect(CSRF_HEADER).toBe("x-csrftoken")
  })
})
```

`frontend/app/api/problem.test.ts` (Vitest compares `Error` objects specially, so the tests compare plain fields):

```ts
import { describe, expect, test } from "vitest"

import { ApiError, isApiError, toApiError } from "~/api/problem"

function jsonResponse(
  status: number,
  body: unknown,
  contentType = "application/problem+json"
): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": contentType },
  })
}

function fieldsOf(error: ApiError) {
  return {
    status: error.status,
    code: error.code,
    message: error.message,
    detail: error.detail,
    errors: error.errors,
  }
}

describe("toApiError", () => {
  test("maps an RFC 9457 problem", async () => {
    const error = await toApiError(
      jsonResponse(404, {
        type: "urn:pharma-agent:problem:conversation-not-found",
        title: "Conversation not found",
        status: 404,
        detail: "No conversation 7f0c",
        code: "CONVERSATION_NOT_FOUND",
      })
    )

    expect(error).toBeInstanceOf(ApiError)
    expect(fieldsOf(error)).toEqual({
      status: 404,
      code: "CONVERSATION_NOT_FOUND",
      message: "Conversation not found",
      detail: "No conversation 7f0c",
      errors: [],
    })
  })

  test("keeps the validation items of a 422 problem", async () => {
    const items = [
      {
        loc: ["body", "message"],
        message: "String should have at most 4000 characters",
        type: "string_too_long",
      },
    ]
    const error = await toApiError(
      jsonResponse(422, {
        type: "urn:pharma-agent:problem:validation-error",
        title: "Validation error",
        status: 422,
        code: "VALIDATION_ERROR",
        errors: items,
      })
    )

    expect(fieldsOf(error)).toEqual({
      status: 422,
      code: "VALIDATION_ERROR",
      message: "Validation error",
      detail: undefined,
      errors: items,
    })
  })

  test("reads a fastapi-users error code from a plain detail body", async () => {
    const error = await toApiError(
      jsonResponse(400, { detail: "LOGIN_BAD_CREDENTIALS" }, "application/json")
    )

    expect(fieldsOf(error)).toEqual({
      status: 400,
      code: "LOGIN_BAD_CREDENTIALS",
      message: "HTTP 400",
      detail: "LOGIN_BAD_CREDENTIALS",
      errors: [],
    })
  })

  test("uses UNKNOWN when a detail is prose instead of a code", async () => {
    const error = await toApiError(
      jsonResponse(404, { detail: "Not Found" }, "application/json")
    )

    expect(fieldsOf(error)).toEqual({
      status: 404,
      code: "UNKNOWN",
      message: "HTTP 404",
      detail: "Not Found",
      errors: [],
    })
  })

  test("uses UNKNOWN when the body is not JSON", async () => {
    const error = await toApiError(
      new Response("<html>Bad Gateway</html>", {
        status: 502,
        statusText: "Bad Gateway",
      })
    )

    expect(fieldsOf(error)).toEqual({
      status: 502,
      code: "UNKNOWN",
      message: "Bad Gateway",
      detail: undefined,
      errors: [],
    })
  })
})

describe("isApiError", () => {
  test("recognises only ApiError instances", () => {
    expect(
      isApiError(
        new ApiError({ status: 401, code: "UNAUTHORIZED", title: "Unauthorized" })
      )
    ).toBe(true)
    expect(isApiError(new Error("boom"))).toBe(false)
    expect(isApiError({ status: 401, code: "UNAUTHORIZED" })).toBe(false)
  })
})
```

`frontend/app/i18n/error-message.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { ApiError } from "~/api/problem"
import { apiErrorMessage, isErrorCode } from "~/i18n/error-message"

import { createTestI18n } from "../../tests/utils/i18n"

const t = createTestI18n("vi").getFixedT("vi", "errors")

describe("apiErrorMessage", () => {
  test("translates a known problem code", () => {
    const error = new ApiError({
      status: 400,
      code: "LOGIN_BAD_CREDENTIALS",
      title: "Bad credentials",
    })
    expect(apiErrorMessage(error, t)).toBe("Email hoặc mật khẩu không đúng.")
  })

  test("falls back to the problem title for an untranslated code", () => {
    const error = new ApiError({
      status: 409,
      code: "SOMETHING_NEW",
      title: "Something new happened",
    })
    expect(apiErrorMessage(error, t)).toBe("Something new happened")
  })

  test("uses the generic message for errors that are not ApiError", () => {
    expect(apiErrorMessage(new TypeError("Failed to fetch"), t)).toBe(
      "Đã xảy ra lỗi. Vui lòng thử lại."
    )
  })
})

describe("isErrorCode", () => {
  test("knows the codes of the errors namespace only", () => {
    expect(isErrorCode("AGENT_UNAVAILABLE")).toBe(true)
    expect(isErrorCode("toString")).toBe(false)
    expect(isErrorCode("SOMETHING_NEW")).toBe(false)
  })
})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run app/lib/csrf.test.ts app/lib/csrf.browser.test.tsx app/api/problem.test.ts app/i18n/error-message.test.ts`
Expected: FAIL, 4 files, `Failed to resolve import` for `~/lib/csrf`, `~/api/problem` and `~/i18n/error-message`.

- [ ] **Step 3: Implement the CSRF reader and the problem mapping**

`frontend/app/lib/csrf.ts`:

```ts
import { parseCookie } from "cookie"

/** Cookie set by the backend CSRF middleware; readable by JavaScript on purpose. */
export const CSRF_COOKIE = "csrftoken"
export const CSRF_HEADER = "x-csrftoken"

export function readCsrfToken(): string | undefined {
  if (typeof document === "undefined") {
    return undefined
  }
  const token = parseCookie(document.cookie)[CSRF_COOKIE]
  return token === "" ? undefined : token
}
```

`frontend/app/api/problem.ts`:

```ts
import { z } from "zod"

const problemItemSchema = z.object({
  loc: z.array(z.union([z.string(), z.number()])),
  message: z.string(),
  type: z.string(),
})

export type ProblemItem = z.infer<typeof problemItemSchema>

const problemSchema = z.object({
  title: z.string().optional(),
  detail: z.string().nullish(),
  code: z.string(),
  errors: z.array(problemItemSchema).nullish(),
})

/** Bodies such as {"detail": "LOGIN_BAD_CREDENTIALS"} or {"detail": "Not Found"}. */
const detailSchema = z.object({ detail: z.string() })

const ERROR_CODE_PATTERN = /^[A-Z][A-Z0-9_]*$/

export const UNKNOWN_ERROR_CODE = "UNKNOWN"

export class ApiError extends Error {
  override readonly name = "ApiError"
  readonly status: number
  readonly code: string
  readonly detail: string | undefined
  readonly errors: ProblemItem[]

  constructor(init: {
    status: number
    code: string
    title: string
    detail?: string | undefined
    errors?: ProblemItem[]
  }) {
    super(init.title)
    this.status = init.status
    this.code = init.code
    this.detail = init.detail
    this.errors = init.errors ?? []
  }
}

export function isApiError(value: unknown): value is ApiError {
  return value instanceof ApiError
}

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json()
  } catch {
    return undefined
  }
}

export async function toApiError(response: Response): Promise<ApiError> {
  const status = response.status
  const fallbackTitle = response.statusText || `HTTP ${status}`
  const body = await readJson(response)

  const problem = problemSchema.safeParse(body)
  if (problem.success) {
    return new ApiError({
      status,
      code: problem.data.code,
      title: problem.data.title ?? fallbackTitle,
      detail: problem.data.detail ?? undefined,
      errors: problem.data.errors ?? [],
    })
  }

  const detail = detailSchema.safeParse(body)
  if (detail.success) {
    return new ApiError({
      status,
      code: ERROR_CODE_PATTERN.test(detail.data.detail)
        ? detail.data.detail
        : UNKNOWN_ERROR_CODE,
      title: fallbackTitle,
      detail: detail.data.detail,
    })
  }

  return new ApiError({ status, code: UNKNOWN_ERROR_CODE, title: fallbackTitle })
}
```

`frontend/app/i18n/error-message.ts`:

```ts
import type { TFunction } from "i18next"

import { isApiError } from "~/api/problem"
import { resources } from "~/i18n/config"

export type ErrorCode = keyof (typeof resources)["vi"]["errors"]

export function isErrorCode(code: string): code is ErrorCode {
  return Object.hasOwn(resources.vi.errors, code)
}

/** Message for a failed request: translated by problem code, else the problem title. */
export function apiErrorMessage(error: unknown, t: TFunction<"errors">): string {
  if (isApiError(error)) {
    if (isErrorCode(error.code)) {
      return t(error.code)
    }
    if (error.message !== "") {
      return error.message
    }
  }
  return t("UNKNOWN")
}
```

- [ ] **Step 4: Write the error translations**

`frontend/app/i18n/resources/vi/errors.json`:

```json
{
  "UNKNOWN": "Đã xảy ra lỗi. Vui lòng thử lại.",
  "UNAUTHORIZED": "Phiên đăng nhập đã hết hạn. Vui lòng đăng nhập lại.",
  "CSRF_FAILED": "Phiên làm việc không hợp lệ. Vui lòng tải lại trang.",
  "VALIDATION_ERROR": "Dữ liệu gửi lên không hợp lệ.",
  "INVALID_INPUT": "Dữ liệu gửi lên không hợp lệ.",
  "INVALID_CURSOR": "Không tải được trang tiếp theo. Vui lòng tải lại.",
  "CONVERSATION_NOT_FOUND": "Không tìm thấy cuộc trò chuyện.",
  "MESSAGE_NOT_FOUND": "Không tìm thấy tin nhắn.",
  "CITATION_NOT_FOUND": "Không tìm thấy nguồn trích dẫn.",
  "SKILL_NOT_FOUND": "Không tìm thấy kỹ năng.",
  "SKILL_NAME_TAKEN": "Tên kỹ năng đã được dùng.",
  "SKILL_PARSE_ERROR": "Tệp SKILL.md không đúng định dạng.",
  "PAYLOAD_TOO_LARGE": "Tệp quá lớn.",
  "AGENT_UNAVAILABLE": "Agent tạm thời không sẵn sàng.",
  "REGISTER_USER_ALREADY_EXISTS": "Email này đã được đăng ký.",
  "REGISTER_INVALID_PASSWORD": "Mật khẩu không đạt yêu cầu.",
  "LOGIN_BAD_CREDENTIALS": "Email hoặc mật khẩu không đúng.",
  "LOGIN_USER_NOT_VERIFIED": "Tài khoản chưa được xác thực.",
  "OAUTH_NOT_AVAILABLE_EMAIL": "Tài khoản Google không cung cấp email.",
  "OAUTH_USER_ALREADY_EXISTS": "Email này đã liên kết với một tài khoản khác.",
  "OAUTH_INVALID_STATE": "Phiên đăng nhập Google không hợp lệ. Vui lòng thử lại.",
  "ACCESS_TOKEN_DECODE_ERROR": "Phiên đăng nhập Google không hợp lệ. Vui lòng thử lại.",
  "ACCESS_TOKEN_ALREADY_EXPIRED": "Phiên đăng nhập Google đã hết hạn. Vui lòng thử lại.",
  "UPDATE_USER_EMAIL_ALREADY_EXISTS": "Email này đã được dùng.",
  "UPDATE_USER_INVALID_PASSWORD": "Mật khẩu mới không đạt yêu cầu."
}
```

`frontend/app/i18n/resources/en/errors.json`:

```json
{
  "UNKNOWN": "Something went wrong. Please try again.",
  "UNAUTHORIZED": "Your session has expired. Please sign in again.",
  "CSRF_FAILED": "Your session is no longer valid. Please reload the page.",
  "VALIDATION_ERROR": "The submitted data is not valid.",
  "INVALID_INPUT": "The submitted data is not valid.",
  "INVALID_CURSOR": "The next page could not be loaded. Please reload.",
  "CONVERSATION_NOT_FOUND": "Conversation not found.",
  "MESSAGE_NOT_FOUND": "Message not found.",
  "CITATION_NOT_FOUND": "Citation source not found.",
  "SKILL_NOT_FOUND": "Skill not found.",
  "SKILL_NAME_TAKEN": "This skill name is already used.",
  "SKILL_PARSE_ERROR": "The SKILL.md file is not valid.",
  "PAYLOAD_TOO_LARGE": "The file is too large.",
  "AGENT_UNAVAILABLE": "The agent is temporarily unavailable.",
  "REGISTER_USER_ALREADY_EXISTS": "This email is already registered.",
  "REGISTER_INVALID_PASSWORD": "The password does not meet the requirements.",
  "LOGIN_BAD_CREDENTIALS": "Incorrect email or password.",
  "LOGIN_USER_NOT_VERIFIED": "This account is not verified.",
  "OAUTH_NOT_AVAILABLE_EMAIL": "The Google account did not share an email address.",
  "OAUTH_USER_ALREADY_EXISTS": "This email is already linked to another account.",
  "OAUTH_INVALID_STATE": "The Google sign-in session is not valid. Please try again.",
  "ACCESS_TOKEN_DECODE_ERROR": "The Google sign-in session is not valid. Please try again.",
  "ACCESS_TOKEN_ALREADY_EXPIRED": "The Google sign-in session has expired. Please try again.",
  "UPDATE_USER_EMAIL_ALREADY_EXISTS": "This email is already used.",
  "UPDATE_USER_INVALID_PASSWORD": "The new password does not meet the requirements."
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run app/lib/csrf.test.ts app/lib/csrf.browser.test.tsx app/api/problem.test.ts app/i18n/error-message.test.ts app/i18n/resources.test.ts`
Expected: PASS, `Test Files 5 passed (5)`; the resources parity test still passes with the new `errors` keys.

- [ ] **Step 6: Full check**

```bash
cd /home/andv/personal/thesis/frontend
npm run format:write
npm run lint && npm run format && npm run typecheck && npm test
```

Expected: all succeed.

- [ ] **Step 7: Commit**

```bash
cd /home/andv/personal/thesis
git add frontend/app/lib/csrf.ts frontend/app/lib/csrf.test.ts frontend/app/lib/csrf.browser.test.tsx \
  frontend/app/api/problem.ts frontend/app/api/problem.test.ts frontend/app/i18n/error-message.ts \
  frontend/app/i18n/error-message.test.ts frontend/app/i18n/resources/vi/errors.json frontend/app/i18n/resources/en/errors.json
git commit -m "feat(frontend): add CSRF cookie reader, problem+json ApiError and translated error messages" \
  -m "$COMMIT_TRAILER"
```

---

### Task 6: orval mutator `fetcher` and the QueryClient

No backend needed (MSW stands in). Covers the spec §13 unit row for the fetcher and spec §6 "Hết phiên" and §7.2–§7.3.

**Files:**
- Create: `frontend/app/api/fetcher.ts`, `frontend/app/api/query-client.ts`
- Modify: `frontend/app/root.tsx`
- Test: `frontend/app/api/fetcher.test.ts`, `frontend/app/api/fetcher.browser.test.tsx`, `frontend/app/api/query-client.test.ts`

**Interfaces:**
- Consumes: `toApiError`, `isApiError`, `ApiError` (Task 5); `readCsrfToken`, `CSRF_HEADER` (Task 5); `server` (`tests/msw/node.ts`), `worker` (`tests/msw/browser.ts`) (Task 1).
- Produces:
  - `~/api/fetcher`: `fetcher<T>(url: string, init?: RequestInit): Promise<T>` (the orval mutator). `url` is the API path orval generates (`/api/v1/...`). In the browser it calls the same origin with `credentials: "same-origin"` and adds `x-csrftoken` to POST, PUT, PATCH and DELETE when the cookie exists. On the server it prefixes `API_INTERNAL_URL` (default `http://127.0.0.1:8000`); a loader forwards the user's cookie through `init.headers`. A non-2xx response throws `ApiError`; an empty body resolves to `null`. `apiBaseUrl(): string`.
  - `~/api/query-client`: `queryClient: QueryClient` (single browser instance); `createQueryClient(options: { onUnauthorized: () => void }): QueryClient`; `shouldRetry(failureCount: number, error: Error): boolean` (no retry for 4xx, at most 2 retries otherwise); `redirectToLogin(): void` (`window.location.assign("/login?next=<path+search>")`, no-op on `/login` and `/register`).
  - TanStack Query meta: `Register` declares `queryMeta` and `mutationMeta` as `{ skipAuthRedirect?: boolean }`. A 401 from any query or mutation calls `onUnauthorized` unless its `meta.skipAuthRedirect` is `true` (the auth guard and the login page handle 401 themselves).
  - `app/root.tsx` wraps routes in `<QueryClientProvider client={queryClient}>`.

- [ ] **Step 1: Write the failing tests**

`frontend/app/api/fetcher.test.ts` (Node, so this is the server path):

```ts
import { http, HttpResponse } from "msw"
import { afterEach, describe, expect, test, vi } from "vitest"

import { apiBaseUrl, fetcher } from "~/api/fetcher"
import { ApiError } from "~/api/problem"

import { server } from "../../tests/msw/node"

const API = "http://api.test"

describe("fetcher on the server", () => {
  afterEach(() => {
    vi.unstubAllEnvs()
  })

  test("defaults to the backend on localhost", () => {
    vi.stubEnv("API_INTERNAL_URL", undefined)
    expect(apiBaseUrl()).toBe("http://127.0.0.1:8000")
  })

  test("prefixes API_INTERNAL_URL and forwards the caller's cookie", async () => {
    vi.stubEnv("API_INTERNAL_URL", API)
    server.use(
      http.get(`${API}/api/v1/users/me`, ({ request }) =>
        HttpResponse.json({ cookie: request.headers.get("cookie") })
      )
    )

    const body = await fetcher<{ cookie: string | null }>("/api/v1/users/me", {
      headers: { Cookie: "pharma_session=s1" },
    })

    expect(body).toEqual({ cookie: "pharma_session=s1" })
  })

  test("throws ApiError built from problem+json", async () => {
    vi.stubEnv("API_INTERNAL_URL", API)
    server.use(
      http.get(`${API}/api/v1/conversations/7f0c`, () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:conversation-not-found",
            title: "Conversation not found",
            status: 404,
            code: "CONVERSATION_NOT_FOUND",
          },
          {
            status: 404,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )

    const request = fetcher("/api/v1/conversations/7f0c")

    await expect(request).rejects.toBeInstanceOf(ApiError)
    await expect(request).rejects.toHaveProperty("status", 404)
    await expect(request).rejects.toHaveProperty("code", "CONVERSATION_NOT_FOUND")
  })

  test("resolves null for an empty 204 response", async () => {
    vi.stubEnv("API_INTERNAL_URL", API)
    server.use(
      http.post(
        `${API}/api/v1/auth/cookie/logout`,
        () => new HttpResponse(null, { status: 204 })
      )
    )

    await expect(
      fetcher("/api/v1/auth/cookie/logout", { method: "POST" })
    ).resolves.toBeNull()
  })

  test("sends no CSRF header because there is no browser cookie jar", async () => {
    vi.stubEnv("API_INTERNAL_URL", API)
    server.use(
      http.post(`${API}/api/v1/conversations`, ({ request }) =>
        HttpResponse.json(
          { csrf: request.headers.get("x-csrftoken") },
          { status: 201 }
        )
      )
    )

    await expect(
      fetcher("/api/v1/conversations", { method: "POST" })
    ).resolves.toEqual({ csrf: null })
  })
})
```

`frontend/app/api/fetcher.browser.test.tsx`:

```ts
import { http, HttpResponse } from "msw"
import { afterEach, describe, expect, test } from "vitest"

import { fetcher } from "~/api/fetcher"

import { worker } from "../../tests/msw/browser"

describe("fetcher in the browser", () => {
  afterEach(() => {
    document.cookie = "csrftoken=; path=/; max-age=0"
  })

  test("calls the same origin and sends the CSRF token on unsafe methods", async () => {
    document.cookie = "csrftoken=token-123; path=/"
    worker.use(
      http.post("/api/v1/conversations", ({ request }) =>
        HttpResponse.json(
          { url: request.url, csrf: request.headers.get("x-csrftoken") },
          { status: 201 }
        )
      )
    )

    const body = await fetcher<{ url: string; csrf: string | null }>(
      "/api/v1/conversations",
      { method: "POST" }
    )

    expect(body).toEqual({
      url: `${window.location.origin}/api/v1/conversations`,
      csrf: "token-123",
    })
  })

  test("does not send the CSRF token on safe methods", async () => {
    document.cookie = "csrftoken=token-123; path=/"
    worker.use(
      http.get("/api/v1/users/me", ({ request }) =>
        HttpResponse.json({ csrf: request.headers.get("x-csrftoken") })
      )
    )

    await expect(fetcher("/api/v1/users/me")).resolves.toEqual({ csrf: null })
  })
})
```

`frontend/app/api/query-client.test.ts`:

```ts
import { describe, expect, test, vi } from "vitest"

import { ApiError } from "~/api/problem"
import { createQueryClient, shouldRetry } from "~/api/query-client"

function unauthorized(): ApiError {
  return new ApiError({ status: 401, code: "UNAUTHORIZED", title: "Unauthorized" })
}

describe("shouldRetry", () => {
  test("never retries client errors", () => {
    const notFound = new ApiError({
      status: 404,
      code: "CONVERSATION_NOT_FOUND",
      title: "Conversation not found",
    })
    expect(shouldRetry(0, notFound)).toBe(false)
    expect(shouldRetry(0, unauthorized())).toBe(false)
  })

  test("retries server and network errors at most twice", () => {
    const unavailable = new ApiError({
      status: 503,
      code: "AGENT_UNAVAILABLE",
      title: "Agent unavailable",
    })
    expect(shouldRetry(0, unavailable)).toBe(true)
    expect(shouldRetry(1, new TypeError("Failed to fetch"))).toBe(true)
    expect(shouldRetry(2, new TypeError("Failed to fetch"))).toBe(false)
  })
})

describe("createQueryClient", () => {
  test("reports a 401 from a query", async () => {
    const onUnauthorized = vi.fn<() => void>()
    const client = createQueryClient({ onUnauthorized })

    await expect(
      client.fetchQuery({
        queryKey: ["/api/v1/users/me"],
        queryFn: () => Promise.reject(unauthorized()),
      })
    ).rejects.toBeInstanceOf(ApiError)

    expect(onUnauthorized).toHaveBeenCalledOnce()
  })

  test("leaves a 401 alone when the query handles it itself", async () => {
    const onUnauthorized = vi.fn<() => void>()
    const client = createQueryClient({ onUnauthorized })

    await expect(
      client.fetchQuery({
        queryKey: ["/api/v1/users/me"],
        queryFn: () => Promise.reject(unauthorized()),
        meta: { skipAuthRedirect: true },
      })
    ).rejects.toBeInstanceOf(ApiError)

    expect(onUnauthorized).not.toHaveBeenCalled()
  })

  test("reports a 401 from a mutation", async () => {
    const onUnauthorized = vi.fn<() => void>()
    const client = createQueryClient({ onUnauthorized })
    const mutation = client.getMutationCache().build(client, {
      mutationFn: () => Promise.reject(unauthorized()),
    })

    await expect(mutation.execute(undefined)).rejects.toBeInstanceOf(ApiError)

    expect(onUnauthorized).toHaveBeenCalledOnce()
  })

  test("ignores errors other than 401", async () => {
    const onUnauthorized = vi.fn<() => void>()
    const client = createQueryClient({ onUnauthorized })

    await expect(
      client.fetchQuery({
        queryKey: ["/api/v1/skills"],
        queryFn: () =>
          Promise.reject(
            new ApiError({ status: 403, code: "CSRF_FAILED", title: "Forbidden" })
          ),
      })
    ).rejects.toBeInstanceOf(ApiError)

    expect(onUnauthorized).not.toHaveBeenCalled()
  })
})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run app/api/fetcher.test.ts app/api/fetcher.browser.test.tsx app/api/query-client.test.ts`
Expected: FAIL, 3 files, `Failed to resolve import "~/api/fetcher"` and `"~/api/query-client"`.

- [ ] **Step 3: Implement the fetcher and the QueryClient**

`frontend/app/api/fetcher.ts`:

```ts
import { toApiError } from "~/api/problem"
import { CSRF_HEADER, readCsrfToken } from "~/lib/csrf"

const UNSAFE_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"])
const DEFAULT_API_INTERNAL_URL = "http://127.0.0.1:8000"

/** Same origin in the browser; the backend's internal URL when a loader calls the API. */
export function apiBaseUrl(): string {
  if (typeof document !== "undefined") {
    return ""
  }
  return process.env.API_INTERNAL_URL ?? DEFAULT_API_INTERNAL_URL
}

/** orval mutator for every generated request. */
export async function fetcher<T>(url: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? "GET").toUpperCase()
  const headers = new Headers(init?.headers)
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json, application/problem+json")
  }
  if (UNSAFE_METHODS.has(method)) {
    const token = readCsrfToken()
    if (token !== undefined) {
      headers.set(CSRF_HEADER, token)
    }
  }

  const response = await fetch(`${apiBaseUrl()}${url}`, {
    ...init,
    method,
    headers,
    credentials: "same-origin",
  })
  if (!response.ok) {
    throw await toApiError(response)
  }

  // The generated request type decides T; JSON.parse returns `any`, so no type assertion is needed.
  const text = await response.text()
  return JSON.parse(text === "" ? "null" : text)
}
```

`frontend/app/api/query-client.ts`:

```ts
import { MutationCache, QueryCache, QueryClient } from "@tanstack/react-query"

import { isApiError } from "~/api/problem"

type AuthRedirectMeta = { skipAuthRedirect?: boolean }

declare module "@tanstack/react-query" {
  interface Register {
    queryMeta: AuthRedirectMeta
    mutationMeta: AuthRedirectMeta
  }
}

const MAX_RETRIES = 2

export function shouldRetry(failureCount: number, error: Error): boolean {
  if (isApiError(error) && error.status < 500) {
    return false
  }
  return failureCount < MAX_RETRIES
}

export function redirectToLogin(): void {
  const { pathname, search } = window.location
  if (pathname === "/login" || pathname === "/register") {
    return
  }
  window.location.assign(`/login?next=${encodeURIComponent(pathname + search)}`)
}

export function createQueryClient({
  onUnauthorized,
}: {
  onUnauthorized: () => void
}): QueryClient {
  function reportUnauthorized(
    error: Error,
    meta: AuthRedirectMeta | undefined
  ): void {
    if (
      isApiError(error) &&
      error.status === 401 &&
      meta?.skipAuthRedirect !== true
    ) {
      onUnauthorized()
    }
  }

  return new QueryClient({
    queryCache: new QueryCache({
      onError: (error, query) => {
        reportUnauthorized(error, query.meta)
      },
    }),
    mutationCache: new MutationCache({
      onError: (error, _variables, _onMutateResult, mutation) => {
        reportUnauthorized(error, mutation.meta)
      },
    }),
    defaultOptions: {
      queries: { retry: shouldRetry },
    },
  })
}

/** Private routes run only in the browser, so one client lives for the whole tab. */
export const queryClient = createQueryClient({
  onUnauthorized: redirectToLogin,
})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run app/api/fetcher.test.ts app/api/fetcher.browser.test.tsx app/api/query-client.test.ts`
Expected: PASS, `Test Files 3 passed (3)`, `Tests 11 passed (11)`.

- [ ] **Step 5: Provide the QueryClient to every route**

In `frontend/app/root.tsx`, add these imports next to the other `~/` imports:

```tsx
import { QueryClientProvider } from "@tanstack/react-query"

import { queryClient } from "~/api/query-client"
```

and replace the `return` of the default export `App`

```tsx
  return (
    <Toaster>
      <TooltipProvider>
        <Outlet />
      </TooltipProvider>
    </Toaster>
  )
```

with

```tsx
  return (
    <QueryClientProvider client={queryClient}>
      <Toaster>
        <TooltipProvider>
          <Outlet />
        </TooltipProvider>
      </Toaster>
    </QueryClientProvider>
  )
```

(`@tanstack/react-query` goes with the other package imports at the top; Prettier does not reorder imports, so keep packages first, then `~/`, then relative imports.)

- [ ] **Step 6: Full check**

```bash
cd /home/andv/personal/thesis/frontend
npm run format:write
npm run lint && npm run format && npm run typecheck && npm test
npm run build
```

Expected: all succeed; the build still prerenders nothing yet and bundles the provider without errors.

- [ ] **Step 7: Commit**

```bash
cd /home/andv/personal/thesis
git add frontend/app/api/fetcher.ts frontend/app/api/fetcher.test.ts frontend/app/api/fetcher.browser.test.tsx \
  frontend/app/api/query-client.ts frontend/app/api/query-client.test.ts frontend/app/root.tsx
git commit -m "feat(frontend): add orval fetcher mutator and QueryClient with session-expiry redirect" \
  -m "$COMMIT_TRAILER"
```

---

### Task 7: Prerendered landing page with SEO meta and hreflang, and the 404 route

No backend needed; the landing page calls no API.

Two facts shape the URLs. React Router writes the prerendered English page to `build/client/en/index.html`, and `react-router-serve` serves `build/client` through `express.static`, which answers `/en` with `301` to `/en/`. So the English page's canonical and hreflang URL is `/en/`. And `/` is always Vietnamese: `publicPageLocale` fixes the language of public pages from the URL, so SSR, the prerendered file and hreflang agree even when a visitor's `lng` cookie says otherwise.

**Files:**
- Create: `frontend/app/features/landing/LandingPage.tsx`, `frontend/app/features/landing/meta.ts`, `frontend/app/routes/not-found.tsx`
- Modify: `frontend/app/routes/public/landing.tsx`, `frontend/app/routes.ts`, `frontend/react-router.config.ts`, `frontend/app/i18n/config.ts`, `frontend/app/i18n/middleware.server.ts`, `frontend/app/i18n/resources/vi/landing.json`, `frontend/app/i18n/resources/en/landing.json`
- Test: `frontend/app/i18n/public-page-locale.test.ts`, `frontend/app/features/landing/meta.test.ts`, `frontend/app/features/landing/LandingPage.test.tsx`, `frontend/app/routes/not-found.test.tsx`

**Interfaces:**
- Consumes: `getLocale`, `getInstance`, `localeFromPath`, `isLanguage`, `FALLBACK_LANGUAGE`, `createTestI18n` (Task 3); `buttonVariants`, `Card*` (Tasks 1–2).
- Produces:
  - `~/i18n/config`: `publicPageLocale(pathname: string): Language | null` (`/` → `"vi"`, `/en` and `/en/...` → `"en"`, anything else → `null`); the i18n middleware uses it as `findLocale`.
  - `~/features/landing/meta`: `LANDING_PATHS: Record<Language, string>` (`{ vi: "/", en: "/en/" }`); `type LandingMetaData = { locale: Language; title: string; description: string; siteUrl: string }`; `landingMeta(data: LandingMetaData): MetaDescriptor[]` (title, description, Open Graph, canonical, `alternate` links for `vi`, `en`, `x-default`).
  - `~/features/landing/LandingPage`: `LandingPage({ locale }: { locale: Language })`.
  - Landing paths (overview §4): `/` (vi, route id `landing-vi`) and `/en/` (en, route id `landing-en`, route path `en`), both served by `routes/public/landing.tsx`; `react-router-serve` redirects `/en` to `/en/`. `prerender: ["/", "/en"]` writes `build/client/index.html` and `build/client/en/index.html`. `*` uses `routes/not-found.tsx` (status 404).
  - Server env `PUBLIC_SITE_URL` (default `http://localhost:8080`, the nginx origin) builds absolute URLs; it is read at build time for prerendered pages.
  - `landing` keys: `meta.{title,description}`, `hero.{title,subtitle,signIn,register}`, `features.title`, `features.{citations,streaming,skills}.{title,description}`, `sources.{title,formulary,leaflets}`, `language.switch`.

- [ ] **Step 1: Write the failing tests**

`frontend/app/i18n/public-page-locale.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { publicPageLocale } from "~/i18n/config"

describe("publicPageLocale", () => {
  test.each([
    ["/", "vi"],
    ["/en", "en"],
    ["/en/", "en"],
    ["/login", null],
    ["/chat/7f0c", null],
  ] as const)("%s -> %s", (pathname, expected) => {
    expect(publicPageLocale(pathname)).toBe(expected)
  })
})
```

`frontend/app/features/landing/meta.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { landingMeta } from "~/features/landing/meta"

const SITE = "https://pharma.example"

describe("landingMeta", () => {
  test("describes the Vietnamese page with Open Graph tags and a canonical URL", () => {
    const tags = landingMeta({
      locale: "vi",
      title: "Tiêu đề",
      description: "Mô tả",
      siteUrl: SITE,
    })

    expect(tags).toEqual(
      expect.arrayContaining([
        { title: "Tiêu đề" },
        { name: "description", content: "Mô tả" },
        { property: "og:type", content: "website" },
        { property: "og:title", content: "Tiêu đề" },
        { property: "og:description", content: "Mô tả" },
        { property: "og:url", content: "https://pharma.example/" },
        { property: "og:locale", content: "vi_VN" },
        { tagName: "link", rel: "canonical", href: "https://pharma.example/" },
      ])
    )
  })

  test("points the English page at /en/", () => {
    const tags = landingMeta({
      locale: "en",
      title: "Title",
      description: "Description",
      siteUrl: SITE,
    })

    expect(tags).toEqual(
      expect.arrayContaining([
        { property: "og:locale", content: "en_US" },
        { property: "og:url", content: "https://pharma.example/en/" },
        { tagName: "link", rel: "canonical", href: "https://pharma.example/en/" },
      ])
    )
  })

  test.each(["vi", "en"] as const)(
    "links both languages and x-default from the %s page",
    (locale) => {
      const tags = landingMeta({
        locale,
        title: "T",
        description: "D",
        siteUrl: SITE,
      })

      expect(tags).toEqual(
        expect.arrayContaining([
          { tagName: "link", rel: "alternate", hrefLang: "vi", href: "https://pharma.example/" },
          { tagName: "link", rel: "alternate", hrefLang: "en", href: "https://pharma.example/en/" },
          { tagName: "link", rel: "alternate", hrefLang: "x-default", href: "https://pharma.example/" },
        ])
      )
    }
  )
})
```

`frontend/app/features/landing/LandingPage.test.tsx`:

```tsx
import { renderToString } from "react-dom/server"
import { I18nextProvider } from "react-i18next"
import { MemoryRouter } from "react-router"
import { describe, expect, test } from "vitest"

import { LandingPage } from "~/features/landing/LandingPage"
import type { Language } from "~/i18n/config"

import { createTestI18n } from "../../../tests/utils/i18n"

function renderLanding(locale: Language): string {
  return renderToString(
    <I18nextProvider i18n={createTestI18n(locale)}>
      <MemoryRouter>
        <LandingPage locale={locale} />
      </MemoryRouter>
    </I18nextProvider>
  )
}

describe("LandingPage", () => {
  test("renders the Vietnamese page with sign-in links, sources and a switch to English", () => {
    const html = renderLanding("vi")

    expect(html).toContain("Tra cứu thông tin thuốc, có trích dẫn nguồn")
    expect(html).toContain("Dược thư Quốc gia Việt Nam")
    expect(html).toContain('href="/login"')
    expect(html).toContain('href="/register"')
    expect(html).toContain('href="/en/"')
  })

  test("renders the English page with a switch back to Vietnamese", () => {
    const html = renderLanding("en")

    expect(html).toContain("Drug information, with cited sources")
    expect(html).toContain("Vietnamese National Drug Formulary")
    expect(html).toContain('hreflang="vi"')
  })
})
```

`frontend/app/routes/not-found.test.tsx`:

```tsx
import { renderToString } from "react-dom/server"
import { I18nextProvider } from "react-i18next"
import { MemoryRouter } from "react-router"
import { describe, expect, test } from "vitest"

import NotFound, { loader } from "~/routes/not-found"

import { createTestI18n } from "../../tests/utils/i18n"

describe("not-found route", () => {
  test("answers with HTTP 404", () => {
    expect(loader().init?.status).toBe(404)
  })

  test("renders a translated message with a link home", () => {
    const html = renderToString(
      <I18nextProvider i18n={createTestI18n("vi")}>
        <MemoryRouter>
          <NotFound />
        </MemoryRouter>
      </I18nextProvider>
    )

    expect(html).toContain("Không tìm thấy trang")
    expect(html).toContain('href="/"')
  })
})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project unit app/i18n/public-page-locale.test.ts app/features/landing app/routes/not-found.test.tsx`
Expected: FAIL. `public-page-locale.test.ts`: `publicPageLocale is not a function` (no such export yet); the other three: `Failed to resolve import` for `~/features/landing/meta`, `~/features/landing/LandingPage` and `~/routes/not-found`.

- [ ] **Step 3: Fix the language of public pages from the URL**

Append to `frontend/app/i18n/config.ts`:

```ts
/** Public pages get their language from the URL so SSR, prerendered files and hreflang agree. */
export function publicPageLocale(pathname: string): Language | null {
  if (pathname === "/") {
    return FALLBACK_LANGUAGE
  }
  return localeFromPath(pathname)
}
```

Replace `frontend/app/i18n/middleware.server.ts` with:

```ts
import { initReactI18next } from "react-i18next"
import { createI18nextMiddleware } from "remix-i18next"

import {
  FALLBACK_LANGUAGE,
  SUPPORTED_LANGUAGES,
  i18nextOptions,
  publicPageLocale,
} from "~/i18n/config"
import { localeCookie } from "~/i18n/locale.server"

export const [i18nextMiddleware, getLocale, getInstance] =
  createI18nextMiddleware({
    detection: {
      supportedLanguages: [...SUPPORTED_LANGUAGES],
      fallbackLanguage: FALLBACK_LANGUAGE,
      cookie: localeCookie,
      order: ["custom", "cookie", "header"],
      findLocale: ({ request }) =>
        Promise.resolve(publicPageLocale(new URL(request.url).pathname)),
    },
    i18next: i18nextOptions,
    plugins: [initReactI18next],
  })
```

- [ ] **Step 4: Write the landing translations**

`frontend/app/i18n/resources/vi/landing.json`:

```json
{
  "meta": {
    "title": "Pharma Agent – Tra cứu thông tin thuốc có trích dẫn nguồn",
    "description": "Hỏi đáp về thuốc bằng tiếng Việt. Mỗi câu trả lời dẫn tới đoạn gốc trong Dược thư Quốc gia Việt Nam và tờ hướng dẫn sử dụng."
  },
  "hero": {
    "title": "Tra cứu thông tin thuốc, có trích dẫn nguồn",
    "subtitle": "Hỏi về liều dùng, chống chỉ định, tác dụng phụ hay tương tác thuốc. Agent tìm trong tài liệu chính thống và chỉ rõ đoạn đã dùng để trả lời.",
    "signIn": "Đăng nhập",
    "register": "Tạo tài khoản"
  },
  "features": {
    "title": "Tính năng",
    "citations": {
      "title": "Trích dẫn từng ý",
      "description": "Bấm vào [1], [2] để xem đúng đoạn tài liệu mà câu trả lời dựa vào."
    },
    "streaming": {
      "title": "Thấy agent đang làm gì",
      "description": "Theo dõi các bước tìm kiếm, đọc tài liệu và soạn câu trả lời theo thời gian thực."
    },
    "skills": {
      "title": "Kỹ năng tuỳ chỉnh",
      "description": "Tải lên tệp SKILL.md để agent trả lời theo cách bạn cần."
    }
  },
  "sources": {
    "title": "Nguồn dữ liệu",
    "formulary": "Dược thư Quốc gia Việt Nam",
    "leaflets": "Tờ hướng dẫn sử dụng thuốc"
  },
  "language": {
    "switch": "English"
  }
}
```

`frontend/app/i18n/resources/en/landing.json`:

```json
{
  "meta": {
    "title": "Pharma Agent – Drug information with cited sources",
    "description": "Ask about medicines and get answers that link to the exact passage in the Vietnamese National Drug Formulary and package leaflets."
  },
  "hero": {
    "title": "Drug information, with cited sources",
    "subtitle": "Ask about dosage, contraindications, side effects or interactions. The agent searches official documents and shows which passage each answer uses.",
    "signIn": "Sign in",
    "register": "Create account"
  },
  "features": {
    "title": "Features",
    "citations": {
      "title": "Every point cited",
      "description": "Click [1], [2] to open the exact passage an answer relies on."
    },
    "streaming": {
      "title": "See what the agent is doing",
      "description": "Follow the search, reading and writing steps in real time."
    },
    "skills": {
      "title": "Custom skills",
      "description": "Upload a SKILL.md file so the agent answers the way you need."
    }
  },
  "sources": {
    "title": "Data sources",
    "formulary": "Vietnamese National Drug Formulary",
    "leaflets": "Medicine package leaflets"
  },
  "language": {
    "switch": "Tiếng Việt"
  }
}
```

- [ ] **Step 5: Implement the meta builder, the landing page and the 404 route**

`frontend/app/features/landing/meta.ts`:

```ts
import type { MetaDescriptor } from "react-router"

import type { Language } from "~/i18n/config"

export const LANDING_PATHS: Record<Language, string> = {
  vi: "/",
  en: "/en/",
}

const OPEN_GRAPH_LOCALES: Record<Language, string> = {
  vi: "vi_VN",
  en: "en_US",
}

export type LandingMetaData = {
  locale: Language
  title: string
  description: string
  siteUrl: string
}

export function landingMeta({
  locale,
  title,
  description,
  siteUrl,
}: LandingMetaData): MetaDescriptor[] {
  const absolute = (language: Language) =>
    new URL(LANDING_PATHS[language], siteUrl).toString()

  return [
    { title },
    { name: "description", content: description },
    { property: "og:type", content: "website" },
    { property: "og:title", content: title },
    { property: "og:description", content: description },
    { property: "og:url", content: absolute(locale) },
    { property: "og:locale", content: OPEN_GRAPH_LOCALES[locale] },
    { tagName: "link", rel: "canonical", href: absolute(locale) },
    { tagName: "link", rel: "alternate", hrefLang: "vi", href: absolute("vi") },
    { tagName: "link", rel: "alternate", hrefLang: "en", href: absolute("en") },
    {
      tagName: "link",
      rel: "alternate",
      hrefLang: "x-default",
      href: absolute("vi"),
    },
  ]
}
```

`frontend/app/features/landing/LandingPage.tsx`:

```tsx
import { useTranslation } from "react-i18next"
import { Link } from "react-router"

import { buttonVariants } from "~/components/ui/button"
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "~/components/ui/card"
import { LANDING_PATHS } from "~/features/landing/meta"
import type { Language } from "~/i18n/config"

const FEATURES = ["citations", "streaming", "skills"] as const

export function LandingPage({ locale }: { locale: Language }) {
  const { t } = useTranslation("landing")
  const { t: tCommon } = useTranslation("common")
  const otherLocale: Language = locale === "vi" ? "en" : "vi"

  return (
    <div className="min-h-svh bg-background text-foreground">
      <header className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
        <span className="font-semibold">{tCommon("appName")}</span>
        <Link
          to={LANDING_PATHS[otherLocale]}
          hrefLang={otherLocale}
          lang={otherLocale}
          className={buttonVariants({ variant: "ghost", size: "sm" })}
        >
          {t("language.switch")}
        </Link>
      </header>

      <main className="mx-auto flex max-w-5xl flex-col gap-16 px-6 pt-12 pb-16">
        <section className="flex flex-col items-start gap-4">
          <h1 className="text-4xl font-semibold tracking-tight text-balance">
            {t("hero.title")}
          </h1>
          <p className="max-w-2xl text-lg text-muted-foreground">
            {t("hero.subtitle")}
          </p>
          <div className="flex flex-wrap gap-3">
            <Link to="/login" className={buttonVariants({ size: "lg" })}>
              {t("hero.signIn")}
            </Link>
            <Link
              to="/register"
              className={buttonVariants({ variant: "outline", size: "lg" })}
            >
              {t("hero.register")}
            </Link>
          </div>
        </section>

        <section aria-labelledby="features-title" className="flex flex-col gap-6">
          <h2 id="features-title" className="text-2xl font-semibold">
            {t("features.title")}
          </h2>
          <div className="grid gap-4 md:grid-cols-3">
            {FEATURES.map((feature) => (
              <Card key={feature}>
                <CardHeader>
                  <CardTitle>{t(`features.${feature}.title`)}</CardTitle>
                  <CardDescription>
                    {t(`features.${feature}.description`)}
                  </CardDescription>
                </CardHeader>
              </Card>
            ))}
          </div>
        </section>

        <section aria-labelledby="sources-title" className="flex flex-col gap-4">
          <h2 id="sources-title" className="text-2xl font-semibold">
            {t("sources.title")}
          </h2>
          <ul className="list-disc pl-6 text-muted-foreground">
            <li>{t("sources.formulary")}</li>
            <li>{t("sources.leaflets")}</li>
          </ul>
        </section>
      </main>
    </div>
  )
}
```

`frontend/app/routes/public/landing.tsx`:

```tsx
import { LandingPage } from "~/features/landing/LandingPage"
import { landingMeta } from "~/features/landing/meta"
import { FALLBACK_LANGUAGE, isLanguage } from "~/i18n/config"
import { getInstance, getLocale } from "~/i18n/middleware.server"

import type { Route } from "./+types/landing"

const DEFAULT_SITE_URL = "http://localhost:8080"

export function loader({ context }: Route.LoaderArgs) {
  const detected = getLocale(context)
  const locale = isLanguage(detected) ? detected : FALLBACK_LANGUAGE
  const t = getInstance(context).getFixedT(locale, "landing")

  return {
    locale,
    title: t("meta.title"),
    description: t("meta.description"),
    siteUrl: process.env.PUBLIC_SITE_URL ?? DEFAULT_SITE_URL,
  }
}

export const meta: Route.MetaFunction = ({ loaderData }) =>
  loaderData === undefined ? [] : landingMeta(loaderData)

export default function Landing({ loaderData }: Route.ComponentProps) {
  return <LandingPage locale={loaderData.locale} />
}
```

`frontend/app/routes/not-found.tsx`:

```tsx
import { useTranslation } from "react-i18next"
import { Link, data } from "react-router"

import { buttonVariants } from "~/components/ui/button"

export function loader() {
  return data(null, { status: 404 })
}

export default function NotFound() {
  const { t } = useTranslation("common")

  return (
    <main className="mx-auto flex min-h-svh max-w-md flex-col items-start justify-center gap-3 p-6">
      <h1 className="text-2xl font-semibold">{t("errorPage.notFoundTitle")}</h1>
      <p className="text-muted-foreground">
        {t("errorPage.notFoundDescription")}
      </p>
      <Link to="/" className={buttonVariants({ variant: "outline" })}>
        {t("errorPage.backHome")}
      </Link>
    </main>
  )
}
```

`frontend/app/routes.ts`:

```ts
import { type RouteConfig, index, route } from "@react-router/dev/routes"

export default [
  index("routes/public/landing.tsx", { id: "landing-vi" }),
  route("en", "routes/public/landing.tsx", { id: "landing-en" }),
  route("actions/locale", "routes/actions/locale.ts"),
  route("actions/theme", "routes/actions/theme.ts"),
  route("*", "routes/not-found.tsx"),
] satisfies RouteConfig
```

`frontend/react-router.config.ts`:

```ts
import type { Config } from "@react-router/dev/config"

export default {
  ssr: true,
  prerender: ["/", "/en"],
} satisfies Config
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project unit app/i18n app/features/landing app/routes/not-found.test.tsx`
Expected: PASS, including the unchanged Task 3 i18n tests and the resources parity test with the new `landing` keys.

- [ ] **Step 7: Verify the prerendered files, hreflang and status codes**

```bash
cd /home/andv/personal/thesis/frontend
npm run build
ls build/client/index.html build/client/en/index.html
grep -o '<html[^>]*>' build/client/index.html build/client/en/index.html
grep -o '<title>[^<]*</title>' build/client/en/index.html
grep -io '<link[^>]*hreflang[^>]*>' build/client/en/index.html
PORT=3100 npm start > /tmp/frontend-start.log 2>&1 &
SERVER_PID=$!
timeout 30 bash -c 'until curl -sf http://localhost:3100/ > /dev/null; do sleep 0.5; done'
curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' http://localhost:3100/en
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3100/en/
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:3100/khong-ton-tai
kill "$SERVER_PID"
```

Expected:
- both files exist;
- `build/client/index.html:<html lang="vi" dir="ltr">` and `build/client/en/index.html:<html lang="en" dir="ltr">`;
- `<title>Pharma Agent – Drug information with cited sources</title>`;
- three `<link rel="alternate" ...>` tags with `hreflang="vi" href="http://localhost:8080/"`, `hreflang="en" href="http://localhost:8080/en/"` and `hreflang="x-default" href="http://localhost:8080/"`;
- `301 http://localhost:3100/en/`, then `200`, then `404`.

- [ ] **Step 8: Full check**

```bash
cd /home/andv/personal/thesis/frontend
npm run format:write
npm run lint && npm run format && npm run typecheck && npm test
```

Expected: all succeed; `tsc` accepts the template keys `features.${feature}.title` because every member of `FEATURES` exists in `landing.json`.

- [ ] **Step 9: Commit**

```bash
cd /home/andv/personal/thesis
git add frontend/app/features/landing frontend/app/routes/public/landing.tsx frontend/app/routes/not-found.tsx \
  frontend/app/routes/not-found.test.tsx frontend/app/routes.ts frontend/react-router.config.ts \
  frontend/app/i18n/config.ts frontend/app/i18n/middleware.server.ts frontend/app/i18n/public-page-locale.test.ts \
  frontend/app/i18n/resources/vi/landing.json frontend/app/i18n/resources/en/landing.json
git commit -m "feat(frontend): add prerendered landing page with SEO meta, hreflang and 404 route" \
  -m "$COMMIT_TRAILER"
```

---

### Task 8: Frontend image, nginx single-origin proxy and compose services

No backend needed to build and route; the full stack check with SSE is Plan 10's E2E.

**Files:**
- Create: `frontend/Dockerfile`, `frontend/.dockerignore`, `frontend/.env.example`, `docker/nginx/nginx.conf`
- Modify: `docker-compose.yml`, `.env.example` (repo root)

**Interfaces:**
- Consumes: `npm run build`, `react-router-serve` (Task 1); `PUBLIC_SITE_URL`, `API_INTERNAL_URL`, `THEME_COOKIE_SECRET` env (Tasks 4, 6, 7).
- Produces:
  - Image `thesis-pharma-web:local`; container `pharma_agent_frontend` listening on `FRONTEND_PORT` (default 3000).
  - Container `pharma_agent_nginx` on `WEB_PORT` (default 8080): `/api/` → `127.0.0.1:${BACKEND_PORT}` with SSE-safe settings, `/` → `127.0.0.1:${FRONTEND_PORT}`. The browser origin is `http://localhost:8080`.
  - Backend setting `PHARMA_AUTH__FRONTEND_URL=http://localhost:${WEB_PORT}` so the Google OAuth redirect lands on the nginx origin.

- [ ] **Step 1: Write the failing check**

Run: `cd /home/andv/personal/thesis && docker compose config --services | grep -x -e frontend -e nginx`
Expected: FAIL (exit 1, no output): neither service exists yet.

- [ ] **Step 2: Write the image files**

`frontend/Dockerfile` (multi-stage like the React Router template, pinned Node, non-root user; `--ignore-scripts` because MSW's `postinstall` would look for `public/` before it is copied, and no runtime dependency needs an install script):

```dockerfile
# syntax=docker/dockerfile:1

FROM node:24.21.0-alpine3.24 AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --ignore-scripts

FROM node:24.21.0-alpine3.24 AS build
WORKDIR /app
# Prerendered pages embed absolute canonical and hreflang URLs.
ARG PUBLIC_SITE_URL=http://localhost:8080
ENV PUBLIC_SITE_URL=${PUBLIC_SITE_URL}
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:24.21.0-alpine3.24 AS production-deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci --omit=dev --ignore-scripts

FROM node:24.21.0-alpine3.24
WORKDIR /app
ENV NODE_ENV=production \
    PORT=3000
COPY --chown=node:node package.json package-lock.json ./
COPY --from=production-deps --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/build ./build
USER node
EXPOSE 3000
CMD ["node_modules/.bin/react-router-serve", "./build/server/index.js"]
```

`frontend/.dockerignore`:

```text
node_modules
build
.react-router
tests
docs
coverage
playwright-report
test-results
.env
.env.*
```

`frontend/.env.example`:

```text
# Optional overrides for `npm run dev`, `npm start` and the frontend container.
# Backend URL that server-side loaders call.
# API_INTERNAL_URL=http://127.0.0.1:8000
# Public origin for canonical and hreflang URLs; read at build time for prerendered pages.
# PUBLIC_SITE_URL=http://localhost:8080
# Signs the theme cookie (it only stores light/dark); any random string.
# THEME_COOKIE_SECRET=
```

`docker/nginx/nginx.conf` (spec §14; checked with `nginx -t` on `nginx:1.30.4-alpine3.24` during planning):

```nginx
# Template for the official nginx image: it is mounted as
# /etc/nginx/templates/default.conf.template and the entrypoint substitutes
# WEB_PORT, FRONTEND_PORT and BACKEND_PORT from the container environment.
upstream frontend {
    server 127.0.0.1:${FRONTEND_PORT};
    keepalive 16;
}

upstream backend {
    server 127.0.0.1:${BACKEND_PORT};
    keepalive 16;
}

server {
    listen ${WEB_PORT};
    server_name _;

    location /api/ {
        proxy_pass http://backend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # AI SDK UI Message Stream (SSE): pass each chunk through immediately.
        proxy_buffering off;
        proxy_cache off;
        gzip off;
        proxy_read_timeout 1h;
    }

    location / {
        proxy_pass http://frontend;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

- [ ] **Step 3: Add the compose services and ports**

In `/home/andv/personal/thesis/docker-compose.yml`, inside the `x-backend` anchor's `environment`, replace

```yaml
    PHARMA_API__PORT: "${BACKEND_PORT:-8000}"
```

with

```yaml
    PHARMA_API__PORT: "${BACKEND_PORT:-8000}"
    # Google OAuth returns to the single nginx origin.
    PHARMA_AUTH__FRONTEND_URL: "http://localhost:${WEB_PORT:-8080}"
```

and insert these services after the `backend` service, right before the top-level `volumes:` key:

```yaml
  frontend:
    build:
      context: ./frontend
      args:
        PUBLIC_SITE_URL: "http://localhost:${WEB_PORT:-8080}"
    image: thesis-pharma-web:local
    container_name: pharma_agent_frontend
    # Same reason as the backend: host networking keeps every service on 127.0.0.1.
    network_mode: host
    env_file:
      - path: ./frontend/.env
        required: false
    environment:
      HOST: "0.0.0.0"
      PORT: "${FRONTEND_PORT:-3000}"
      API_INTERNAL_URL: "http://127.0.0.1:${BACKEND_PORT:-8000}"
      PUBLIC_SITE_URL: "http://localhost:${WEB_PORT:-8080}"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://127.0.0.1:${FRONTEND_PORT:-3000}/"]
      interval: 15s
      timeout: 5s
      retries: 5
      start_period: 20s

  nginx:
    image: nginx:1.30.4-alpine3.24
    container_name: pharma_agent_nginx
    network_mode: host
    environment:
      WEB_PORT: "${WEB_PORT:-8080}"
      FRONTEND_PORT: "${FRONTEND_PORT:-3000}"
      BACKEND_PORT: "${BACKEND_PORT:-8000}"
    volumes:
      - "./docker/nginx/nginx.conf:/etc/nginx/templates/default.conf.template:ro"
    depends_on:
      frontend:
        condition: service_healthy
      backend:
        condition: service_healthy
    restart: unless-stopped
```

In `/home/andv/personal/thesis/.env.example`, replace

```text
# BACKEND_PORT=8000
```

with

```text
# BACKEND_PORT=8000
# FRONTEND_PORT=3000
# WEB_PORT=8080
```

- [ ] **Step 4: Run the check to verify it passes**

```bash
cd /home/andv/personal/thesis
docker compose config --quiet && echo "compose ok"
docker compose config --services | grep -x -e frontend -e nginx
```

Expected: `compose ok`, then `frontend` and `nginx`.

- [ ] **Step 5: Build the image and route through nginx**

```bash
cd /home/andv/personal/thesis
docker compose build frontend
docker compose up -d --no-deps frontend nginx
timeout 90 bash -c 'until curl -sf http://localhost:8080/ > /dev/null; do sleep 1; done'
curl -s http://localhost:8080/ | grep -o '<html[^>]*>'
curl -s -o /dev/null -w '%{http_code} %{redirect_url}\n' http://localhost:8080/en
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8080/api/v1/health
docker compose exec nginx nginx -T 2>/dev/null | grep -c 'proxy_buffering off'
docker compose stop frontend nginx && docker compose rm -f frontend nginx
```

Expected: `<html lang="vi" dir="ltr">`; `301 http://localhost:8080/en/`; `/api/v1/health` returns `200` if the backend is running or `502` if it is not (either proves nginx forwards `/api/` to the backend port instead of the frontend); `1`.

- [ ] **Step 6: Full check**

Run: `cd /home/andv/personal/thesis/frontend && npm run lint && npm run format && npm run typecheck && npm test`
Expected: all succeed (the Dockerfile, nginx and compose changes do not affect them; `.dockerignore` and `.env.example` are not Prettier-parsed files).

- [ ] **Step 7: Commit**

```bash
cd /home/andv/personal/thesis
git add frontend/Dockerfile frontend/.dockerignore frontend/.env.example docker/nginx/nginx.conf docker-compose.yml .env.example
git commit -m "feat(infra): add frontend image, nginx single-origin proxy and compose services" \
  -m "$COMMIT_TRAILER"
```

---

### Task 9: Generated API client with orval (needs Plan 5)

**Needs Plan 5** merged: `pharma-agent export-openapi`, route names as operation ids, problem+json responses in OpenAPI. Plan 7 changes the auth routes to the cookie backend; after Plan 7 lands, rerun Steps 3–4 so names such as `authCookieLogin` and `oauthGoogleCookieAuthorize` exist (Tasks 10 and 11 need Plan 7 anyway).

orval 8.32 cannot write TypeScript types into one `schemas.ts` (`schemas.mode: "single"` is Zod-only) and names the MSW file after the target. The generated layout is therefore `app/api/gen/endpoints.ts`, `app/api/gen/endpoints.msw.ts`, `app/api/gen/schemas/` (with `index.ts`) and `app/api/gen/zod.ts`. Imports stay `~/api/gen/endpoints`, `~/api/gen/schemas`, `~/api/gen/zod`, and `~/api/gen/endpoints.msw` for mocks. The planning spike generated this layout from a fastapi-users OpenAPI document with the route-name operation ids, type-checked it under TypeScript 7 strict with `noUncheckedIndexedAccess`, and it passed `prettier --check`.

**Files:**
- Create: `frontend/orval.config.ts`, `frontend/openapi.json` (exported), `frontend/app/api/gen/**` (generated, committed)
- Modify: `frontend/package.json` (scripts), `.pre-commit-config.yaml` (drift hook)
- Test: `frontend/app/api/gen.test.ts`

**Interfaces:**
- Consumes: `fetcher` (Task 6) as the mutator; `pharma-agent export-openapi --output PATH` (Plan 5).
- Produces (operation id `a:b.c_d` becomes camelCase `aBCD`; examples use the ids pinned in the overview §3.5):
  - `~/api/gen/endpoints`: for each operation `x`: `x(...)` (returns the response body type, not a `{data, status}` wrapper), `getXUrl(...)`, and for GET `getXQueryKey(...)` (relative path plus params), `getXQueryOptions(params?, options?: { query?, request? })`, `useX(...)`; for mutations `useX(options?)` with variables `{ data }` or `{ id, data }`. Operations with a `cursor` parameter also get `getXInfiniteQueryOptions` and `useXInfinite`. Examples: `usersCurrentUser`, `getUsersCurrentUserQueryKey`, `getUsersCurrentUserQueryOptions`, `useAuthCookieLogin`, `authCookieLogout`, `useAuthCookieLogout`, `useRegisterRegister`, `oauthGoogleCookieAuthorize(params?) → OAuth2AuthorizeResponse`, `oauthGoogleCookieCallback(params?)`, `useUsersPatchCurrentUser`, `listConversations`, `getListConversationsInfiniteQueryOptions`.
  - `~/api/gen/schemas`: model types, e.g. `UserRead`, `UserCreate`, `UserUpdate`, `Problem`, `ProblemItem`, `OAuth2AuthorizeResponse`, `ConversationView`, `ConversationPage`, `UIMessage`, `MessagePage`, `CitationDetail`, `SkillView`.
  - `~/api/gen/zod`: `XBody`, `XParams`, `XQueryParams`, `XResponse` Zod 4 schemas, e.g. `AuthCookieLoginBody`, `RegisterRegisterBody`, `UsersPatchCurrentUserBody`.
  - `~/api/gen/endpoints.msw`: `getXMockHandler(override?)` (paths like `*/api/v1/users/me`), `getXResponseMock()`, `getPharmaAgentAPIMock()` (all handlers).
  - npm scripts `api:openapi` (export from the backend), `api:generate` (orval), `api:check` (regenerate and fail if `app/api/gen` differs from git).

- [ ] **Step 1: Write the failing contract test**

`frontend/app/api/gen.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import {
  getUsersCurrentUserQueryKey,
  getUsersCurrentUserUrl,
} from "~/api/gen/endpoints"
import {
  getPharmaAgentAPIMock,
  getUsersCurrentUserMockHandler,
} from "~/api/gen/endpoints.msw"
import type { UserRead } from "~/api/gen/schemas"
import { AuthCookieLoginBody, RegisterRegisterBody } from "~/api/gen/zod"

describe("generated API client", () => {
  test("query keys are relative API paths, the same on server and client", () => {
    expect(getUsersCurrentUserUrl()).toBe("/api/v1/users/me")
    expect(getUsersCurrentUserQueryKey()).toEqual(["/api/v1/users/me"])
  })

  test("zod schemas validate the auth forms", () => {
    expect(
      AuthCookieLoginBody.safeParse({
        username: "an@example.com",
        password: "correct horse",
      }).success
    ).toBe(true)
    expect(
      RegisterRegisterBody.safeParse({
        email: "not-an-email",
        password: "correct horse",
      }).success
    ).toBe(false)
  })

  test("MSW handlers cover the operations", () => {
    const user: UserRead = {
      id: "0b8f7a52-5d4e-4c7b-9a51-3f7d1c2b9e10",
      email: "an@example.com",
    }
    expect(getUsersCurrentUserMockHandler(user).info.path).toBe(
      "*/api/v1/users/me"
    )
    expect(getPharmaAgentAPIMock().length).toBeGreaterThan(0)
  })
})
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project unit app/api/gen.test.ts`
Expected: FAIL with `Failed to resolve import "~/api/gen/endpoints"`.

- [ ] **Step 3: Configure orval and export the OpenAPI document**

`frontend/orval.config.ts`:

```ts
import { defineConfig } from "orval"

export default defineConfig({
  api: {
    input: { target: "./openapi.json" },
    output: {
      mode: "split",
      target: "./app/api/gen/endpoints.ts",
      schemas: "./app/api/gen/schemas",
      client: "react-query",
      httpClient: "fetch",
      mock: { generators: [{ type: "msw" }] },
      clean: true,
      formatter: "prettier",
      override: {
        mutator: { path: "./app/api/fetcher.ts", name: "fetcher" },
        // Generated functions resolve to the response body; errors are thrown by the fetcher.
        fetch: { includeHttpResponseReturnType: false },
        // Without an explicit version orval guesses TanStack Query v4.
        query: {
          version: 5,
          useInfinite: true,
          useInfiniteQueryParam: "cursor",
        },
      },
    },
  },
  zod: {
    input: { target: "./openapi.json" },
    output: {
      mode: "single",
      target: "./app/api/gen/zod.ts",
      client: "zod",
      formatter: "prettier",
      override: { zod: { version: 4 } },
    },
  },
})
```

In `frontend/package.json`, add these entries at the end of `"scripts"` (after `"e2e"`):

```json
    "api:openapi": "uv run --directory ../backend pharma-agent export-openapi --output ../frontend/openapi.json",
    "api:generate": "orval",
    "api:check": "orval && git diff --exit-code -- app/api/gen && test -z \"$(git status --porcelain -- app/api/gen)\""
```

(`"e2e": "playwright test"` needs a trailing comma now.)

Run:

```bash
cd /home/andv/personal/thesis/frontend
npm run api:openapi
grep -o '"operationId": "[^"]*"' openapi.json | sort | head -40
```

Expected: `openapi.json` written; the operation ids include `"users:current_user"`, `"auth:cookie.login"`, `"auth:cookie.logout"`, `"register:register"`, `"oauth:google.cookie.authorize"` (after Plan 7), `"list_conversations"`, `"create_conversation"`, `"chat_stream"`.

- [ ] **Step 4: Generate the client**

Run:

```bash
cd /home/andv/personal/thesis/frontend
npm run api:generate
find app/api/gen -maxdepth 1 | sort
ls app/api/gen/schemas | head -5
```

Expected: orval prints `🎉 Pharma Agent API - Your OpenAPI spec has been converted into ready to use orval!` twice (api and zod); `app/api/gen` contains `endpoints.msw.ts`, `endpoints.ts`, `schemas`, `zod.ts`; `schemas/` contains `index.ts` and one file per model.

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project unit app/api/gen.test.ts`
Expected: PASS, `Tests 3 passed (3)`.

- [ ] **Step 6: Add the drift hook**

In `/home/andv/personal/thesis/.pre-commit-config.yaml`, append this hook to the frontend local repo block added in Task 1 (after `frontend-typecheck`):

```yaml
      - id: frontend-api-drift
        name: orval output matches openapi.json (frontend)
        entry: npm --prefix frontend run api:check
        language: system
        files: ^frontend/(openapi\.json|orval\.config\.ts|app/api/gen/)
        pass_filenames: false
        require_serial: true
```

Run:

```bash
cd /home/andv/personal/thesis/frontend
git add openapi.json orval.config.ts app/api/gen
npm run api:check && echo "no drift"
```

Expected: orval regenerates identical files and prints `no drift`.

- [ ] **Step 7: Full check**

```bash
cd /home/andv/personal/thesis/frontend
npm run lint && npm run format && npm run typecheck && npm test
```

Expected: all succeed. `tsc` type-checks the generated code (Oxlint ignores `app/api/gen/**`); `prettier --check` passes because orval formats its output with the project's Prettier config.

- [ ] **Step 8: Commit**

```bash
cd /home/andv/personal/thesis
git add .pre-commit-config.yaml frontend/package.json frontend/orval.config.ts frontend/openapi.json \
  frontend/app/api/gen frontend/app/api/gen.test.ts
git commit -m "feat(frontend): generate the API client, zod schemas and MSW mocks with orval" \
  -m "$COMMIT_TRAILER"
```

---

### Task 10: Login, registration and Google sign-in pages (needs Plan 7)

**Needs Plan 7** (cookie login and logout, Google OAuth on the cookie backend) and Task 9 regenerated after it. The planning spike type-checked this task's components, helpers and tests under TypeScript 7 and ran type-aware Oxlint over them with this plan's config: zero findings.

**Files:**
- Create: `frontend/app/lib/form-errors.ts`, `frontend/app/features/auth/lib/next-path.ts`, `frontend/app/features/auth/lib/password.ts`, `frontend/app/features/auth/lib/current-user.ts`, `frontend/app/features/auth/lib/google.ts`, `frontend/app/features/auth/hooks/use-redirect-if-authenticated.ts`
- Create: `frontend/app/features/auth/AuthCard.tsx`, `frontend/app/features/auth/LoginForm.tsx`, `frontend/app/features/auth/RegisterForm.tsx`, `frontend/app/features/auth/GoogleButton.tsx`
- Create: `frontend/app/routes/auth/login.tsx`, `frontend/app/routes/auth/register.tsx`, `frontend/app/routes/auth/google-callback.tsx`, `frontend/tests/utils/providers.tsx`
- Modify: `frontend/app/routes.ts`, `frontend/app/i18n/resources/vi/auth.json`, `frontend/app/i18n/resources/en/auth.json`
- Test: `frontend/app/lib/form-errors.test.ts`, `frontend/app/features/auth/lib/next-path.test.ts`, `frontend/app/features/auth/lib/password.test.ts`, `frontend/app/features/auth/lib/google.browser.test.tsx`, `frontend/app/features/auth/LoginForm.browser.test.tsx`, `frontend/app/features/auth/RegisterForm.browser.test.tsx`

**Interfaces:**
- Consumes: `useAuthCookieLogin`, `useRegisterRegister`, `oauthGoogleCookieAuthorize`, `oauthGoogleCookieCallback`, `getUsersCurrentUserQueryOptions`, `getUsersCurrentUserQueryKey`, `getOauthGoogleCookieAuthorizeMockHandler`, `AuthCookieLoginBody`, `RegisterRegisterBody` (Task 9); `ApiError`, `isApiError`, `UNKNOWN_ERROR_CODE`, `apiErrorMessage`, `isErrorCode` (Task 5); `queryClient`, `Register` meta `skipAuthRedirect` (Task 6); `applyZodLocale`, `createTestI18n`, `Language` (Task 3); `Field*`, `Input`, `Button`, `Card*`, `Spinner` (Task 2); `worker` (Task 1).
- Produces:
  - `~/lib/form-errors`: `applyApiError<TFieldValues extends FieldValues>(error: unknown, setError: UseFormSetError<TFieldValues>, fields: readonly NoInfer<Path<TFieldValues>>[], fallbackMessage: string): void` (422 items go onto matching fields; otherwise `root.server`). Plan 10 reuses it for settings and skill upload.
  - `~/features/auth/lib/next-path`: `DEFAULT_NEXT_PATH = "/chat"`, `safeNextPath(value: string | null | undefined): string`.
  - `~/features/auth/lib/password`: `PASSWORD_MIN_LENGTH = 8`, `passwordSchema(tooShortMessage: string): z.ZodString` (overview §4: passwords entered in the UI need at least 8 characters). Registration uses it with `t("auth:passwordTooShort")`; Plan 10's change-password form uses the same function. Login keeps `z.string().min(1)` so existing accounts can always sign in.
  - `~/features/auth/lib/current-user`: `currentUserQueryOptions()` (generated options for `GET /users/me` with `meta.skipAuthRedirect`, `staleTime` 60 s, `retry: false`).
  - `~/features/auth/lib/google`: `NEXT_PATH_STORAGE_KEY`, `startGoogleLogin(nextPath: string, assign?: (url: string) => void): Promise<void>`, `completeGoogleLogin(callbackUrl: URL): Promise<string>` (returns the safe next path).
  - `useRedirectIfAuthenticated(nextPath: string): void`; components `AuthCard({ title, description, children })`, `LoginForm({ nextPath })`, `RegisterForm({ nextPath })`, `GoogleButton({ nextPath })`.
  - `tests/utils/providers.tsx`: `createTestQueryClient(): QueryClient`, `TestProviders({ children, client?, language? })`.
  - Routes `/login`, `/register` (SSR shell, form hydrates), `/auth/google/callback` (client-only; redirects to the next path or `/login?error=<code>`).
  - `auth` keys: `or`, `email`, `password`, `passwordTooShort`, `displayName`, `google`, `googleCompleting`, `login.{title,description,submit,noAccount,registerLink}`, `register.{title,description,submit,hasAccount,loginLink}`.

- [ ] **Step 1: Write the failing unit tests**

`frontend/app/lib/form-errors.test.ts` (the explicit `<Values>` is needed because TypeScript cannot infer the form type through a mocked setter):

```ts
import type { UseFormSetError } from "react-hook-form"
import { describe, expect, test, vi } from "vitest"

import { ApiError } from "~/api/problem"
import { applyApiError } from "~/lib/form-errors"

type Values = { email: string; password: string }

const FIELDS = ["email", "password"] as const

describe("applyApiError", () => {
  test("puts 422 validation items on their fields", () => {
    const setError = vi.fn<UseFormSetError<Values>>()
    const error = new ApiError({
      status: 422,
      code: "VALIDATION_ERROR",
      title: "Validation error",
      errors: [
        {
          loc: ["body", "password"],
          message: "String should have at least 8 characters",
          type: "string_too_short",
        },
      ],
    })

    applyApiError<Values>(error, setError, FIELDS, "fallback")

    expect(setError).toHaveBeenCalledExactlyOnceWith("password", {
      type: "server",
      message: "String should have at least 8 characters",
    })
  })

  test("uses root.server when no field matches", () => {
    const setError = vi.fn<UseFormSetError<Values>>()
    const error = new ApiError({
      status: 400,
      code: "REGISTER_USER_ALREADY_EXISTS",
      title: "User already exists",
    })

    applyApiError<Values>(error, setError, FIELDS, "Email này đã được đăng ký.")

    expect(setError).toHaveBeenCalledExactlyOnceWith("root.server", {
      type: "server",
      message: "Email này đã được đăng ký.",
    })
  })
})
```

`frontend/app/features/auth/lib/next-path.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import { safeNextPath } from "~/features/auth/lib/next-path"

describe("safeNextPath", () => {
  test.each([
    ["/chat/7f0c?draft=1", "/chat/7f0c?draft=1"],
    ["/skills", "/skills"],
    [null, "/chat"],
    [undefined, "/chat"],
    ["", "/chat"],
    ["chat", "/chat"],
    ["https://evil.example/", "/chat"],
    ["//evil.example/", "/chat"],
    ["/\\evil.example", "/chat"],
  ] as const)("%s -> %s", (value, expected) => {
    expect(safeNextPath(value)).toBe(expected)
  })
})
```

`frontend/app/features/auth/lib/password.test.ts`:

```ts
import { describe, expect, test } from "vitest"

import {
  PASSWORD_MIN_LENGTH,
  passwordSchema,
} from "~/features/auth/lib/password"

describe("passwordSchema", () => {
  test("requires at least 8 characters and uses the given message", () => {
    const schema = passwordSchema("Mật khẩu cần ít nhất 8 ký tự.")

    expect(PASSWORD_MIN_LENGTH).toBe(8)
    expect(schema.safeParse("1234567").error?.issues[0]?.message).toBe(
      "Mật khẩu cần ít nhất 8 ký tự."
    )
    expect(schema.safeParse("12345678").success).toBe(true)
  })
})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project unit app/lib/form-errors.test.ts app/features/auth/lib/next-path.test.ts app/features/auth/lib/password.test.ts`
Expected: FAIL, `Failed to resolve import` for `"~/lib/form-errors"`, `"~/features/auth/lib/next-path"` and `"~/features/auth/lib/password"`.

- [ ] **Step 3: Implement the helpers**

`frontend/app/lib/form-errors.ts`:

```ts
import type { FieldValues, Path, UseFormSetError } from "react-hook-form"

import { isApiError } from "~/api/problem"

/** 422 items go onto their fields; anything else becomes the form-level `root.server` error. */
export function applyApiError<TFieldValues extends FieldValues>(
  error: unknown,
  setError: UseFormSetError<TFieldValues>,
  fields: readonly NoInfer<Path<TFieldValues>>[],
  fallbackMessage: string
): void {
  if (isApiError(error) && error.status === 422) {
    let placed = false
    for (const item of error.errors) {
      const field = fields.find((name) => name === item.loc.at(-1))
      if (field !== undefined) {
        setError(field, { type: "server", message: item.message })
        placed = true
      }
    }
    if (placed) {
      return
    }
  }
  setError("root.server", { type: "server", message: fallbackMessage })
}
```

`frontend/app/features/auth/lib/next-path.ts`:

```ts
export const DEFAULT_NEXT_PATH = "/chat"

/** Accept only same-origin paths so `next` cannot send the user to another site. */
export function safeNextPath(value: string | null | undefined): string {
  if (
    value === null ||
    value === undefined ||
    !value.startsWith("/") ||
    value.startsWith("//") ||
    value.startsWith("/\\")
  ) {
    return DEFAULT_NEXT_PATH
  }
  return value
}
```

`frontend/app/features/auth/lib/password.ts`:

```ts
import { z } from "zod"

/** Passwords entered in the UI; shared by registration and Plan 10's change-password form. */
export const PASSWORD_MIN_LENGTH = 8

export function passwordSchema(tooShortMessage: string) {
  return z.string().min(PASSWORD_MIN_LENGTH, tooShortMessage)
}
```

`frontend/app/features/auth/lib/current-user.ts`:

```ts
import { getUsersCurrentUserQueryOptions } from "~/api/gen/endpoints"

/** The auth guard and the auth pages handle 401 themselves, so no global redirect. */
export function currentUserQueryOptions() {
  return getUsersCurrentUserQueryOptions({
    query: {
      meta: { skipAuthRedirect: true },
      staleTime: 60_000,
      retry: false,
    },
  })
}
```

`frontend/app/features/auth/lib/google.ts`:

```ts
import {
  oauthGoogleCookieAuthorize,
  oauthGoogleCookieCallback,
} from "~/api/gen/endpoints"
import { safeNextPath } from "~/features/auth/lib/next-path"

/** fastapi-users' OAuth state does not carry our `next`, so it waits in sessionStorage. */
export const NEXT_PATH_STORAGE_KEY = "pharma-agent:auth-next"

function assignLocation(url: string): void {
  window.location.assign(url)
}

export async function startGoogleLogin(
  nextPath: string,
  assign: (url: string) => void = assignLocation
): Promise<void> {
  const { authorization_url: authorizationUrl } =
    await oauthGoogleCookieAuthorize()
  window.sessionStorage.setItem(NEXT_PATH_STORAGE_KEY, nextPath)
  assign(authorizationUrl)
}

export async function completeGoogleLogin(callbackUrl: URL): Promise<string> {
  const { searchParams } = callbackUrl
  // Same-origin call; the backend reads its OAuth state cookie and sets pharma_session.
  await oauthGoogleCookieCallback({
    code: searchParams.get("code") ?? undefined,
    state: searchParams.get("state") ?? undefined,
  })
  const next = window.sessionStorage.getItem(NEXT_PATH_STORAGE_KEY)
  window.sessionStorage.removeItem(NEXT_PATH_STORAGE_KEY)
  return safeNextPath(next)
}
```

`frontend/app/features/auth/hooks/use-redirect-if-authenticated.ts`:

```ts
import { useQuery } from "@tanstack/react-query"
import { useEffect } from "react"
import { useNavigate } from "react-router"

import { currentUserQueryOptions } from "~/features/auth/lib/current-user"

/** Signed-in users who open /login or /register go straight to where they were heading. */
export function useRedirectIfAuthenticated(nextPath: string): void {
  const navigate = useNavigate()
  const { data: user } = useQuery(currentUserQueryOptions())

  useEffect(() => {
    if (user !== undefined) {
      void navigate(nextPath, { replace: true })
    }
  }, [navigate, nextPath, user])
}
```

- [ ] **Step 4: Run the unit tests to verify they pass**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project unit app/lib/form-errors.test.ts app/features/auth/lib/next-path.test.ts app/features/auth/lib/password.test.ts`
Expected: PASS, `Tests 12 passed (12)`.

- [ ] **Step 5: Write the failing browser tests**

`frontend/tests/utils/providers.tsx`:

```tsx
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useState, type ReactNode } from "react"
import { I18nextProvider } from "react-i18next"

import type { Language } from "~/i18n/config"

import { createTestI18n } from "./i18n"

export function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
}

export function TestProviders({
  children,
  client,
  language = "vi",
}: {
  children: ReactNode
  client?: QueryClient
  language?: Language
}) {
  const [queryClient] = useState(() => client ?? createTestQueryClient())
  const [i18n] = useState(() => createTestI18n(language))

  return (
    <QueryClientProvider client={queryClient}>
      <I18nextProvider i18n={i18n}>{children}</I18nextProvider>
    </QueryClientProvider>
  )
}
```

`frontend/app/features/auth/lib/google.browser.test.tsx`:

```ts
import { http, HttpResponse } from "msw"
import { afterEach, describe, expect, test, vi } from "vitest"

import { getOauthGoogleCookieAuthorizeMockHandler } from "~/api/gen/endpoints.msw"
import {
  NEXT_PATH_STORAGE_KEY,
  completeGoogleLogin,
  startGoogleLogin,
} from "~/features/auth/lib/google"

import { worker } from "../../../../tests/msw/browser"

const PROBLEM_HEADERS = { "Content-Type": "application/problem+json" }

describe("Google sign-in", () => {
  afterEach(() => {
    window.sessionStorage.clear()
  })

  test("start stores the next path and opens the authorization URL", async () => {
    worker.use(
      getOauthGoogleCookieAuthorizeMockHandler({
        authorization_url: "https://accounts.google.com/o/oauth2/v2/auth?state=s1",
      })
    )
    const assign = vi.fn<(url: string) => void>()

    await startGoogleLogin("/skills", assign)

    expect(assign).toHaveBeenCalledExactlyOnceWith(
      "https://accounts.google.com/o/oauth2/v2/auth?state=s1"
    )
    expect(window.sessionStorage.getItem(NEXT_PATH_STORAGE_KEY)).toBe("/skills")
  })

  test("complete forwards code and state and returns the stored next path", async () => {
    let search = ""
    worker.use(
      http.get("/api/v1/auth/google/callback", ({ request }) => {
        search = new URL(request.url).search
        return new HttpResponse(null, { status: 204 })
      })
    )
    window.sessionStorage.setItem(NEXT_PATH_STORAGE_KEY, "/skills")

    const next = await completeGoogleLogin(
      new URL("http://localhost/auth/google/callback?code=abc&state=xyz")
    )

    expect(next).toBe("/skills")
    const params = new URLSearchParams(search)
    expect(params.get("code")).toBe("abc")
    expect(params.get("state")).toBe("xyz")
    expect(window.sessionStorage.getItem(NEXT_PATH_STORAGE_KEY)).toBeNull()
  })

  test("complete rejects with the problem code when the backend refuses", async () => {
    worker.use(
      http.get("/api/v1/auth/google/callback", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:oauth-invalid-state",
            title: "Invalid state",
            status: 400,
            code: "OAUTH_INVALID_STATE",
          },
          { status: 400, headers: PROBLEM_HEADERS }
        )
      )
    )

    await expect(
      completeGoogleLogin(
        new URL("http://localhost/auth/google/callback?code=abc&state=bad")
      )
    ).rejects.toHaveProperty("code", "OAUTH_INVALID_STATE")
  })
})
```

`frontend/app/features/auth/LoginForm.browser.test.tsx`:

```tsx
import { http, HttpResponse } from "msw"
import { createRoutesStub } from "react-router"
import { beforeEach, describe, expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"

import { LoginForm } from "~/features/auth/LoginForm"
import { applyZodLocale } from "~/i18n/zod-locale"

import { worker } from "../../../tests/msw/browser"
import { TestProviders } from "../../../tests/utils/providers"

function LoginRoute() {
  return <LoginForm nextPath="/chat" />
}

function ChatRoute() {
  return <p>chat-page</p>
}

const Stub = createRoutesStub([
  { path: "/login", Component: LoginRoute },
  { path: "/chat", Component: ChatRoute },
])

async function renderLogin() {
  await render(
    <TestProviders>
      <Stub initialEntries={["/login"]} />
    </TestProviders>
  )
}

async function submit(email: string, password: string) {
  await page.getByLabelText("Email").fill(email)
  await page.getByLabelText("Mật khẩu").fill(password)
  await page.getByRole("button", { name: "Đăng nhập" }).click()
}

describe("LoginForm", () => {
  beforeEach(() => {
    applyZodLocale("vi")
  })

  test("signs in through the cookie endpoint and opens the next path", async () => {
    let body = ""
    worker.use(
      http.post("/api/v1/auth/cookie/login", async ({ request }) => {
        body = await request.text()
        return new HttpResponse(null, { status: 204 })
      }),
      http.get("/api/v1/users/me", () =>
        HttpResponse.json({ id: "u1", email: "an@example.com" })
      )
    )
    await renderLogin()

    await submit("an@example.com", "correct horse")

    await expect.element(page.getByText("chat-page")).toBeVisible()
    const form = new URLSearchParams(body)
    expect(form.get("username")).toBe("an@example.com")
    expect(form.get("password")).toBe("correct horse")
  })

  test("shows the translated message for bad credentials", async () => {
    worker.use(
      http.post("/api/v1/auth/cookie/login", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:login-bad-credentials",
            title: "Bad credentials",
            status: 400,
            code: "LOGIN_BAD_CREDENTIALS",
          },
          {
            status: 400,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )
    await renderLogin()

    await submit("an@example.com", "wrong")

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Email hoặc mật khẩu không đúng.")
  })

  test("validates the email before calling the API", async () => {
    await renderLogin()

    await submit("not-an-email", "x")

    await expect
      .element(page.getByText("địa chỉ email không hợp lệ"))
      .toBeVisible()
  })
})
```

`frontend/app/features/auth/RegisterForm.browser.test.tsx`:

```tsx
import { http, HttpResponse } from "msw"
import { createRoutesStub } from "react-router"
import { beforeEach, describe, expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"

import { RegisterForm } from "~/features/auth/RegisterForm"
import { applyZodLocale } from "~/i18n/zod-locale"

import { worker } from "../../../tests/msw/browser"
import { TestProviders } from "../../../tests/utils/providers"

function RegisterRoute() {
  return <RegisterForm nextPath="/chat" />
}

function ChatRoute() {
  return <p>chat-page</p>
}

const Stub = createRoutesStub([
  { path: "/register", Component: RegisterRoute },
  { path: "/chat", Component: ChatRoute },
])

async function renderAndSubmit(password = "correct horse") {
  await render(
    <TestProviders>
      <Stub initialEntries={["/register"]} />
    </TestProviders>
  )
  await page.getByLabelText("Tên hiển thị").fill("An")
  await page.getByLabelText("Email").fill("an@example.com")
  await page.getByLabelText("Mật khẩu").fill(password)
  await page.getByRole("button", { name: "Tạo tài khoản" }).click()
}

describe("RegisterForm", () => {
  beforeEach(() => {
    applyZodLocale("vi")
  })

  test("registers, signs in with the same credentials and opens the next path", async () => {
    let registered: unknown = null
    let loginBody = ""
    worker.use(
      http.post("/api/v1/auth/register", async ({ request }) => {
        registered = await request.json()
        return HttpResponse.json(
          { id: "u1", email: "an@example.com", display_name: "An" },
          { status: 201 }
        )
      }),
      http.post("/api/v1/auth/cookie/login", async ({ request }) => {
        loginBody = await request.text()
        return new HttpResponse(null, { status: 204 })
      }),
      http.get("/api/v1/users/me", () =>
        HttpResponse.json({ id: "u1", email: "an@example.com" })
      )
    )

    await renderAndSubmit()

    await expect.element(page.getByText("chat-page")).toBeVisible()
    expect(registered).toEqual({
      email: "an@example.com",
      password: "correct horse",
      display_name: "An",
    })
    expect(new URLSearchParams(loginBody).get("username")).toBe("an@example.com")
  })

  test("requires a password of at least 8 characters before calling the API", async () => {
    // No handler is registered: MSW's onUnhandledRequest "error" fails the test if a request is sent.
    await renderAndSubmit("1234567")

    await expect
      .element(page.getByText("Mật khẩu cần ít nhất 8 ký tự."))
      .toBeVisible()
  })

  test("shows the translated message when the email is taken", async () => {
    worker.use(
      http.post("/api/v1/auth/register", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:register-user-already-exists",
            title: "User already exists",
            status: 400,
            code: "REGISTER_USER_ALREADY_EXISTS",
          },
          {
            status: 400,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )

    await renderAndSubmit()

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Email này đã được đăng ký.")
  })
})
```

- [ ] **Step 6: Run them to verify they fail**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project browser app/features/auth`
Expected: FAIL, 3 files, `Failed to resolve import` for `~/features/auth/lib/google`, `~/features/auth/LoginForm` and `~/features/auth/RegisterForm`.

- [ ] **Step 7: Write the auth translations**

`frontend/app/i18n/resources/vi/auth.json`:

```json
{
  "or": "hoặc",
  "email": "Email",
  "password": "Mật khẩu",
  "passwordTooShort": "Mật khẩu cần ít nhất 8 ký tự.",
  "displayName": "Tên hiển thị",
  "google": "Tiếp tục với Google",
  "googleCompleting": "Đang hoàn tất đăng nhập Google…",
  "login": {
    "title": "Đăng nhập",
    "description": "Đăng nhập để tra cứu thông tin thuốc.",
    "submit": "Đăng nhập",
    "noAccount": "Chưa có tài khoản?",
    "registerLink": "Đăng ký"
  },
  "register": {
    "title": "Tạo tài khoản",
    "description": "Tạo tài khoản bằng email hoặc Google.",
    "submit": "Tạo tài khoản",
    "hasAccount": "Đã có tài khoản?",
    "loginLink": "Đăng nhập"
  }
}
```

`frontend/app/i18n/resources/en/auth.json`:

```json
{
  "or": "or",
  "email": "Email",
  "password": "Password",
  "passwordTooShort": "Password must be at least 8 characters.",
  "displayName": "Display name",
  "google": "Continue with Google",
  "googleCompleting": "Finishing Google sign-in…",
  "login": {
    "title": "Sign in",
    "description": "Sign in to look up drug information.",
    "submit": "Sign in",
    "noAccount": "No account yet?",
    "registerLink": "Register"
  },
  "register": {
    "title": "Create account",
    "description": "Create an account with email or Google.",
    "submit": "Create account",
    "hasAccount": "Already have an account?",
    "loginLink": "Sign in"
  }
}
```

- [ ] **Step 8: Implement the forms and the Google button**

`frontend/app/features/auth/AuthCard.tsx`:

```tsx
import type { ReactNode } from "react"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "~/components/ui/card"

export function AuthCard({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children: ReactNode
}) {
  return (
    <main className="flex min-h-svh items-center justify-center bg-muted p-6">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>
            <h1 className="text-xl font-semibold">{title}</h1>
          </CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-6">{children}</CardContent>
      </Card>
    </main>
  )
}
```

`frontend/app/features/auth/LoginForm.tsx` (fastapi-users calls the email field `username` in the form-encoded login body):

```tsx
import { zodResolver } from "@hookform/resolvers/zod"
import { useQueryClient } from "@tanstack/react-query"
import { Controller, useForm } from "react-hook-form"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router"
import { z } from "zod"

import {
  getUsersCurrentUserQueryKey,
  useAuthCookieLogin,
} from "~/api/gen/endpoints"
import { AuthCookieLoginBody } from "~/api/gen/zod"
import { Button } from "~/components/ui/button"
import {
  Field,
  FieldError,
  FieldGroup,
  FieldLabel,
} from "~/components/ui/field"
import { Input } from "~/components/ui/input"
import { Spinner } from "~/components/ui/spinner"
import { apiErrorMessage } from "~/i18n/error-message"
import { applyApiError } from "~/lib/form-errors"

const loginSchema = AuthCookieLoginBody.pick({
  username: true,
  password: true,
}).extend({
  username: z.email(),
  password: z.string().min(1),
})

type LoginValues = z.infer<typeof loginSchema>

const LOGIN_FIELDS = ["username", "password"] as const

export function LoginForm({ nextPath }: { nextPath: string }) {
  const { t } = useTranslation("auth")
  const { t: tErrors } = useTranslation("errors")
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const login = useAuthCookieLogin({
    mutation: { meta: { skipAuthRedirect: true } },
  })
  const form = useForm<LoginValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: { username: "", password: "" },
  })

  async function onSubmit(values: LoginValues) {
    try {
      await login.mutateAsync({ data: values })
    } catch (error) {
      applyApiError(
        error,
        form.setError,
        LOGIN_FIELDS,
        apiErrorMessage(error, tErrors)
      )
      return
    }
    await queryClient.invalidateQueries({
      queryKey: getUsersCurrentUserQueryKey(),
    })
    await navigate(nextPath, { replace: true })
  }

  const serverError = form.formState.errors.root?.server?.message

  return (
    <form
      noValidate
      onSubmit={form.handleSubmit(onSubmit)}
      className="flex flex-col gap-6"
    >
      <FieldGroup>
        <Controller
          name="username"
          control={form.control}
          render={({ field, fieldState }) => (
            <Field data-invalid={fieldState.invalid}>
              <FieldLabel htmlFor="login-email">{t("email")}</FieldLabel>
              <Input
                {...field}
                id="login-email"
                type="email"
                autoComplete="email"
                aria-invalid={fieldState.invalid}
              />
              <FieldError errors={[fieldState.error]} />
            </Field>
          )}
        />
        <Controller
          name="password"
          control={form.control}
          render={({ field, fieldState }) => (
            <Field data-invalid={fieldState.invalid}>
              <FieldLabel htmlFor="login-password">{t("password")}</FieldLabel>
              <Input
                {...field}
                id="login-password"
                type="password"
                autoComplete="current-password"
                aria-invalid={fieldState.invalid}
              />
              <FieldError errors={[fieldState.error]} />
            </Field>
          )}
        />
      </FieldGroup>
      {serverError === undefined ? null : (
        <p role="alert" className="text-sm text-destructive">
          {serverError}
        </p>
      )}
      <Button type="submit" disabled={form.formState.isSubmitting}>
        {form.formState.isSubmitting ? <Spinner /> : null}
        {t("login.submit")}
      </Button>
    </form>
  )
}
```

`frontend/app/features/auth/RegisterForm.tsx` (fastapi-users registration does not sign in, so the form logs in with the same credentials right after):

```tsx
import { zodResolver } from "@hookform/resolvers/zod"
import { useQueryClient } from "@tanstack/react-query"
import { Controller, useForm } from "react-hook-form"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router"
import { useMemo } from "react"
import { z } from "zod"

import {
  getUsersCurrentUserQueryKey,
  useAuthCookieLogin,
  useRegisterRegister,
} from "~/api/gen/endpoints"
import { RegisterRegisterBody } from "~/api/gen/zod"
import { Button } from "~/components/ui/button"
import {
  Field,
  FieldError,
  FieldGroup,
  FieldLabel,
} from "~/components/ui/field"
import { Input } from "~/components/ui/input"
import { Spinner } from "~/components/ui/spinner"
import { passwordSchema } from "~/features/auth/lib/password"
import { apiErrorMessage } from "~/i18n/error-message"
import { applyApiError } from "~/lib/form-errors"

/** Built per render language so the password message is translated. */
function createRegisterSchema(passwordTooShort: string) {
  return RegisterRegisterBody.pick({
    email: true,
    password: true,
    display_name: true,
  }).extend({
    password: passwordSchema(passwordTooShort),
    display_name: z.string().trim().max(100),
  })
}

type RegisterValues = z.infer<ReturnType<typeof createRegisterSchema>>

const REGISTER_FIELDS = ["email", "password", "display_name"] as const

export function RegisterForm({ nextPath }: { nextPath: string }) {
  const { t } = useTranslation("auth")
  const { t: tErrors } = useTranslation("errors")
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const register = useRegisterRegister({
    mutation: { meta: { skipAuthRedirect: true } },
  })
  const login = useAuthCookieLogin({
    mutation: { meta: { skipAuthRedirect: true } },
  })
  const registerSchema = useMemo(
    () => createRegisterSchema(t("passwordTooShort")),
    [t]
  )
  const form = useForm<RegisterValues>({
    resolver: zodResolver(registerSchema),
    defaultValues: { email: "", password: "", display_name: "" },
  })

  async function onSubmit(values: RegisterValues) {
    try {
      await register.mutateAsync({ data: values })
      await login.mutateAsync({
        data: { username: values.email, password: values.password },
      })
    } catch (error) {
      applyApiError(
        error,
        form.setError,
        REGISTER_FIELDS,
        apiErrorMessage(error, tErrors)
      )
      return
    }
    await queryClient.invalidateQueries({
      queryKey: getUsersCurrentUserQueryKey(),
    })
    await navigate(nextPath, { replace: true })
  }

  const serverError = form.formState.errors.root?.server?.message

  return (
    <form
      noValidate
      onSubmit={form.handleSubmit(onSubmit)}
      className="flex flex-col gap-6"
    >
      <FieldGroup>
        <Controller
          name="display_name"
          control={form.control}
          render={({ field, fieldState }) => (
            <Field data-invalid={fieldState.invalid}>
              <FieldLabel htmlFor="register-display-name">
                {t("displayName")}
              </FieldLabel>
              <Input
                {...field}
                id="register-display-name"
                autoComplete="name"
                aria-invalid={fieldState.invalid}
              />
              <FieldError errors={[fieldState.error]} />
            </Field>
          )}
        />
        <Controller
          name="email"
          control={form.control}
          render={({ field, fieldState }) => (
            <Field data-invalid={fieldState.invalid}>
              <FieldLabel htmlFor="register-email">{t("email")}</FieldLabel>
              <Input
                {...field}
                id="register-email"
                type="email"
                autoComplete="email"
                aria-invalid={fieldState.invalid}
              />
              <FieldError errors={[fieldState.error]} />
            </Field>
          )}
        />
        <Controller
          name="password"
          control={form.control}
          render={({ field, fieldState }) => (
            <Field data-invalid={fieldState.invalid}>
              <FieldLabel htmlFor="register-password">
                {t("password")}
              </FieldLabel>
              <Input
                {...field}
                id="register-password"
                type="password"
                autoComplete="new-password"
                aria-invalid={fieldState.invalid}
              />
              <FieldError errors={[fieldState.error]} />
            </Field>
          )}
        />
      </FieldGroup>
      {serverError === undefined ? null : (
        <p role="alert" className="text-sm text-destructive">
          {serverError}
        </p>
      )}
      <Button type="submit" disabled={form.formState.isSubmitting}>
        {form.formState.isSubmitting ? <Spinner /> : null}
        {t("register.submit")}
      </Button>
    </form>
  )
}
```

`frontend/app/features/auth/GoogleButton.tsx`:

```tsx
import { useState } from "react"
import { useTranslation } from "react-i18next"

import { Button } from "~/components/ui/button"
import { startGoogleLogin } from "~/features/auth/lib/google"
import { apiErrorMessage } from "~/i18n/error-message"

export function GoogleButton({ nextPath }: { nextPath: string }) {
  const { t } = useTranslation("auth")
  const { t: tErrors } = useTranslation("errors")
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function signIn() {
    setPending(true)
    setError(null)
    try {
      await startGoogleLogin(nextPath)
    } catch (caught) {
      setError(apiErrorMessage(caught, tErrors))
      setPending(false)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <Button
        type="button"
        variant="outline"
        disabled={pending}
        onClick={() => {
          void signIn()
        }}
      >
        {t("google")}
      </Button>
      {error === null ? null : (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </div>
  )
}
```

- [ ] **Step 9: Run the browser tests to verify they pass**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project browser app/features/auth`
Expected: PASS, `Test Files 3 passed (3)`, `Tests 9 passed (9)`.

- [ ] **Step 10: Add the route modules**

`frontend/app/routes/auth/login.tsx`:

```tsx
import { useTranslation } from "react-i18next"
import { Link, useSearchParams } from "react-router"

import { FieldSeparator } from "~/components/ui/field"
import { AuthCard } from "~/features/auth/AuthCard"
import { GoogleButton } from "~/features/auth/GoogleButton"
import { LoginForm } from "~/features/auth/LoginForm"
import { useRedirectIfAuthenticated } from "~/features/auth/hooks/use-redirect-if-authenticated"
import { safeNextPath } from "~/features/auth/lib/next-path"
import { isErrorCode } from "~/i18n/error-message"

export default function LoginPage() {
  const { t } = useTranslation("auth")
  const { t: tErrors } = useTranslation("errors")
  const [searchParams] = useSearchParams()
  const nextPath = safeNextPath(searchParams.get("next"))
  const errorCode = searchParams.get("error")
  useRedirectIfAuthenticated(nextPath)

  return (
    <AuthCard title={t("login.title")} description={t("login.description")}>
      <title>{t("login.title")}</title>
      {errorCode === null ? null : (
        <p role="alert" className="text-sm text-destructive">
          {tErrors(isErrorCode(errorCode) ? errorCode : "UNKNOWN")}
        </p>
      )}
      <LoginForm nextPath={nextPath} />
      <FieldSeparator>{t("or")}</FieldSeparator>
      <GoogleButton nextPath={nextPath} />
      <p className="text-center text-sm text-muted-foreground">
        {t("login.noAccount")}{" "}
        <Link
          to={`/register?next=${encodeURIComponent(nextPath)}`}
          className="underline underline-offset-4"
        >
          {t("login.registerLink")}
        </Link>
      </p>
    </AuthCard>
  )
}
```

`frontend/app/routes/auth/register.tsx`:

```tsx
import { useTranslation } from "react-i18next"
import { Link, useSearchParams } from "react-router"

import { FieldSeparator } from "~/components/ui/field"
import { AuthCard } from "~/features/auth/AuthCard"
import { GoogleButton } from "~/features/auth/GoogleButton"
import { RegisterForm } from "~/features/auth/RegisterForm"
import { useRedirectIfAuthenticated } from "~/features/auth/hooks/use-redirect-if-authenticated"
import { safeNextPath } from "~/features/auth/lib/next-path"

export default function RegisterPage() {
  const { t } = useTranslation("auth")
  const [searchParams] = useSearchParams()
  const nextPath = safeNextPath(searchParams.get("next"))
  useRedirectIfAuthenticated(nextPath)

  return (
    <AuthCard
      title={t("register.title")}
      description={t("register.description")}
    >
      <title>{t("register.title")}</title>
      <RegisterForm nextPath={nextPath} />
      <FieldSeparator>{t("or")}</FieldSeparator>
      <GoogleButton nextPath={nextPath} />
      <p className="text-center text-sm text-muted-foreground">
        {t("register.hasAccount")}{" "}
        <Link
          to={`/login?next=${encodeURIComponent(nextPath)}`}
          className="underline underline-offset-4"
        >
          {t("register.loginLink")}
        </Link>
      </p>
    </AuthCard>
  )
}
```

`frontend/app/routes/auth/google-callback.tsx` (client only: with a `clientLoader` and no server `loader`, the server renders `HydrateFallback` and the browser runs the loader on hydration):

```tsx
import { useTranslation } from "react-i18next"
import { redirect } from "react-router"

import { getUsersCurrentUserQueryKey } from "~/api/gen/endpoints"
import { UNKNOWN_ERROR_CODE, isApiError } from "~/api/problem"
import { queryClient } from "~/api/query-client"
import { Spinner } from "~/components/ui/spinner"
import { completeGoogleLogin } from "~/features/auth/lib/google"

import type { Route } from "./+types/google-callback"

export async function clientLoader({ request }: Route.ClientLoaderArgs) {
  let nextPath: string
  try {
    nextPath = await completeGoogleLogin(new URL(request.url))
  } catch (error) {
    const code = isApiError(error) ? error.code : UNKNOWN_ERROR_CODE
    throw redirect(`/login?error=${encodeURIComponent(code)}`)
  }
  await queryClient.invalidateQueries({ queryKey: getUsersCurrentUserQueryKey() })
  throw redirect(nextPath)
}

export function HydrateFallback() {
  const { t } = useTranslation("auth")
  return (
    <main className="flex min-h-svh items-center justify-center gap-2 text-sm text-muted-foreground">
      <Spinner />
      <span>{t("googleCompleting")}</span>
    </main>
  )
}

export default function GoogleCallback() {
  return <HydrateFallback />
}
```

`frontend/app/routes.ts`:

```ts
import { type RouteConfig, index, route } from "@react-router/dev/routes"

export default [
  index("routes/public/landing.tsx", { id: "landing-vi" }),
  route("en", "routes/public/landing.tsx", { id: "landing-en" }),
  route("login", "routes/auth/login.tsx"),
  route("register", "routes/auth/register.tsx"),
  route("auth/google/callback", "routes/auth/google-callback.tsx"),
  route("actions/locale", "routes/actions/locale.ts"),
  route("actions/theme", "routes/actions/theme.ts"),
  route("*", "routes/not-found.tsx"),
] satisfies RouteConfig
```

- [ ] **Step 11: Full check and SSR smoke test**

```bash
cd /home/andv/personal/thesis/frontend
npm run format:write
npm run lint && npm run format && npm run typecheck && npm test
npm run build
PORT=3100 npm start > /tmp/frontend-start.log 2>&1 &
SERVER_PID=$!
timeout 30 bash -c 'until curl -sf http://localhost:3100/login > /dev/null; do sleep 0.5; done'
curl -s http://localhost:3100/login | grep -o '<title>[^<]*</title>\|id="login-email"'
curl -s http://localhost:3100/auth/google/callback | grep -o 'Đang hoàn tất đăng nhập Google…'
kill "$SERVER_PID"
```

Expected: all checks succeed; `/login` is server-rendered with `<title>Đăng nhập</title>` and the `id="login-email"` input; the callback page server-renders its fallback text. The full browser flow against the real backend (register → login → logout) is Plan 10's Playwright scenario.

- [ ] **Step 12: Commit**

```bash
cd /home/andv/personal/thesis
git add frontend/app/lib/form-errors.ts frontend/app/lib/form-errors.test.ts frontend/app/features/auth \
  frontend/app/routes/auth frontend/app/routes.ts frontend/tests/utils/providers.tsx \
  frontend/app/i18n/resources/vi/auth.json frontend/app/i18n/resources/en/auth.json
git commit -m "feat(frontend): add login, registration and Google sign-in pages" \
  -m "$COMMIT_TRAILER"
```

---

### Task 11: Private-route guard, app shell, logout and placeholder pages (needs Plan 7)

**Needs Plan 7** (cookie session, `POST /auth/cookie/logout`) and Tasks 9–10. The planning spike type-checked `requireUser`, `useLogout`, `UserMenu`, `AppSidebar` and their tests under TypeScript 7 and ran type-aware Oxlint over them: zero findings.

Server middleware does not run on client navigations, so the guard is a `clientMiddleware` on the private layout (spec §6). The layout also has a `clientLoader` with `hydrate = true` so the first page load of `/chat` runs the guard before rendering; both call `requireUser`, which reads the TanStack Query cache, so `/users/me` is fetched once.

**Files:**
- Create: `frontend/app/features/auth/lib/require-user.ts`, `frontend/app/features/auth/hooks/use-logout.ts`, `frontend/app/features/shell/AppSidebar.tsx`, `frontend/app/features/shell/UserMenu.tsx`
- Create: `frontend/app/routes/app/layout.tsx`, `frontend/app/routes/app/chat.tsx`, `frontend/app/routes/app/skills.tsx`, `frontend/app/routes/app/settings.tsx`
- Modify: `frontend/app/routes.ts`, `frontend/app/i18n/resources/vi/common.json`, `frontend/app/i18n/resources/en/common.json`, `frontend/app/components/ui/sidebar.tsx`, `frontend/app/components/ui/toast.tsx`, `frontend/app/components/ui/toast.browser.test.tsx`
- Test: `frontend/app/features/auth/lib/require-user.browser.test.tsx`, `frontend/app/features/shell/AppSidebar.browser.test.tsx`

**Interfaces:**
- Consumes: `currentUserQueryOptions` (Task 10); `useAuthCookieLogout`, `getUsersCurrentUserMockHandler`, `UserRead` (Task 9); `isApiError` (Task 5); `queryClient` (Task 6); `TestProviders` (Task 10); `Sidebar*`, `DropdownMenu*`, `TooltipProvider`, `Spinner` (Task 2); `worker` (Task 1).
- Produces:
  - `~/features/auth/lib/require-user`: `requireUser(request: Request): Promise<UserRead>` (throws `redirect("/login?next=<path+search>")` on 401).
  - `~/features/auth/hooks/use-logout`: `useLogout(): () => Promise<void>` (logout request, `queryClient.clear()`, navigate to `/login`; runs the last two even if the request fails).
  - `~/features/shell/AppSidebar`: `AppSidebar({ user }: { user: UserRead })` (links `/chat`, `/skills`, `/settings`, active state, `UserMenu` in the footer); `~/features/shell/UserMenu`: `UserMenu({ user })`.
  - Route module `routes/app/layout.tsx` (pathless layout): `clientMiddleware`, `clientLoader` returning `{ user: UserRead }`, `HydrateFallback`, default shell with `SidebarProvider`, `SidebarInset`, `SidebarTrigger`, `<Outlet />`. Child routes `chat/:conversationId?` → `routes/app/chat.tsx` (Plan 9 replaces), `skills` → `routes/app/skills.tsx` and `settings` → `routes/app/settings.tsx` (Plan 10 replaces). Child modules read the user with `useRouteLoaderData<typeof clientLoader>("routes/app/layout")`.
  - `common` keys added: `userMenu.{open,logout}`, `shell.{toggleSidebar,sidebarTitle,sidebarDescription}`.

- [ ] **Step 1: Write the failing tests**

`frontend/app/features/auth/lib/require-user.browser.test.tsx`:

```ts
import { http, HttpResponse } from "msw"
import { afterEach, describe, expect, test } from "vitest"

import { getUsersCurrentUserMockHandler } from "~/api/gen/endpoints.msw"
import { queryClient } from "~/api/query-client"
import { requireUser } from "~/features/auth/lib/require-user"

import { worker } from "../../../../tests/msw/browser"

describe("requireUser", () => {
  afterEach(() => {
    queryClient.clear()
  })

  test("returns the signed-in user", async () => {
    worker.use(
      getUsersCurrentUserMockHandler({ id: "u1", email: "an@example.com" })
    )

    await expect(
      requireUser(new Request("http://localhost/chat"))
    ).resolves.toEqual({ id: "u1", email: "an@example.com" })
  })

  test("throws a redirect to /login with the requested path on 401", async () => {
    worker.use(
      http.get("/api/v1/users/me", () =>
        HttpResponse.json(
          {
            type: "urn:pharma-agent:problem:unauthorized",
            title: "Unauthorized",
            status: 401,
            code: "UNAUTHORIZED",
          },
          {
            status: 401,
            headers: { "Content-Type": "application/problem+json" },
          }
        )
      )
    )

    const thrown = await requireUser(
      new Request("http://localhost/chat/7f0c?draft=1")
    ).catch((caught: unknown) => caught)

    expect(thrown).toBeInstanceOf(Response)
    expect(
      thrown instanceof Response
        ? [thrown.status, thrown.headers.get("Location")]
        : []
    ).toEqual([302, "/login?next=%2Fchat%2F7f0c%3Fdraft%3D1"])
  })
})
```

`frontend/app/features/shell/AppSidebar.browser.test.tsx` (the viewport is set to desktop because the sidebar turns into a closed sheet below 768 px):

```tsx
import { http, HttpResponse } from "msw"
import { createRoutesStub } from "react-router"
import { beforeEach, describe, expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"

import type { UserRead } from "~/api/gen/schemas"
import { SidebarProvider } from "~/components/ui/sidebar"
import { TooltipProvider } from "~/components/ui/tooltip"
import { AppSidebar } from "~/features/shell/AppSidebar"

import { worker } from "../../../tests/msw/browser"
import { TestProviders } from "../../../tests/utils/providers"

const USER: UserRead = {
  id: "u1",
  email: "an@example.com",
  display_name: "An",
}

function ShellRoute() {
  return (
    <SidebarProvider>
      <AppSidebar user={USER} />
    </SidebarProvider>
  )
}

function LoginRoute() {
  return <p>login-page</p>
}

const Stub = createRoutesStub([
  { path: "/chat", Component: ShellRoute },
  { path: "/login", Component: LoginRoute },
])

async function renderShell() {
  await render(
    <TestProviders>
      <TooltipProvider>
        <Stub initialEntries={["/chat"]} />
      </TooltipProvider>
    </TestProviders>
  )
}

describe("AppSidebar", () => {
  beforeEach(async () => {
    await page.viewport(1280, 800)
  })

  test("links the private pages and marks the current one", async () => {
    await renderShell()

    await expect
      .element(page.getByRole("link", { name: "Trò chuyện" }))
      .toHaveAttribute("data-active", "true")
    await expect
      .element(page.getByRole("link", { name: "Kỹ năng" }))
      .toHaveAttribute("href", "/skills")
    await expect
      .element(page.getByRole("link", { name: "Cài đặt" }))
      .toHaveAttribute("href", "/settings")
  })

  test("logs out through the cookie endpoint and returns to the login page", async () => {
    let loggedOut = false
    worker.use(
      http.post("/api/v1/auth/cookie/logout", () => {
        loggedOut = true
        return new HttpResponse(null, { status: 204 })
      })
    )
    await renderShell()

    await page.getByRole("button", { name: "Menu tài khoản" }).click()
    await page.getByRole("menuitem", { name: "Đăng xuất" }).click()

    await expect.element(page.getByText("login-page")).toBeVisible()
    expect(loggedOut).toBe(true)
  })
})
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project browser app/features/auth/lib/require-user.browser.test.tsx app/features/shell`
Expected: FAIL, 2 files, `Failed to resolve import "~/features/auth/lib/require-user"` and `"~/features/shell/AppSidebar"`.

- [ ] **Step 3: Add the shell translations**

`frontend/app/i18n/resources/vi/common.json`:

```json
{
  "appName": "Pharma Agent",
  "actions": {
    "save": "Lưu",
    "cancel": "Huỷ",
    "retry": "Thử lại",
    "close": "Đóng",
    "loading": "Đang tải…"
  },
  "nav": {
    "chat": "Trò chuyện",
    "skills": "Kỹ năng",
    "settings": "Cài đặt"
  },
  "userMenu": {
    "open": "Menu tài khoản",
    "logout": "Đăng xuất"
  },
  "shell": {
    "toggleSidebar": "Ẩn hoặc hiện thanh bên",
    "sidebarTitle": "Thanh bên",
    "sidebarDescription": "Chuyển giữa trò chuyện, kỹ năng và cài đặt."
  },
  "errorPage": {
    "title": "Đã xảy ra lỗi",
    "description": "Không tải được trang này. Vui lòng thử lại.",
    "notFoundTitle": "Không tìm thấy trang",
    "notFoundDescription": "Trang bạn tìm không tồn tại hoặc đã bị xoá.",
    "backHome": "Về trang chủ"
  }
}
```

`frontend/app/i18n/resources/en/common.json`:

```json
{
  "appName": "Pharma Agent",
  "actions": {
    "save": "Save",
    "cancel": "Cancel",
    "retry": "Try again",
    "close": "Close",
    "loading": "Loading…"
  },
  "nav": {
    "chat": "Chat",
    "skills": "Skills",
    "settings": "Settings"
  },
  "userMenu": {
    "open": "Account menu",
    "logout": "Sign out"
  },
  "shell": {
    "toggleSidebar": "Toggle sidebar",
    "sidebarTitle": "Sidebar",
    "sidebarDescription": "Move between chat, skills and settings."
  },
  "errorPage": {
    "title": "Something went wrong",
    "description": "This page could not be loaded. Please try again.",
    "notFoundTitle": "Page not found",
    "notFoundDescription": "The page you are looking for does not exist or was removed.",
    "backHome": "Back to home"
  }
}
```

- [ ] **Step 4: Implement the guard, logout and shell components**

`frontend/app/features/auth/lib/require-user.ts`:

```ts
import { redirect } from "react-router"

import type { UserRead } from "~/api/gen/schemas"
import { isApiError } from "~/api/problem"
import { queryClient } from "~/api/query-client"
import { currentUserQueryOptions } from "~/features/auth/lib/current-user"

export async function requireUser(request: Request): Promise<UserRead> {
  try {
    return await queryClient.ensureQueryData(currentUserQueryOptions())
  } catch (error) {
    if (isApiError(error) && error.status === 401) {
      const url = new URL(request.url)
      throw redirect(
        `/login?next=${encodeURIComponent(url.pathname + url.search)}`
      )
    }
    throw error
  }
}
```

`frontend/app/features/auth/hooks/use-logout.ts`:

```ts
import { useQueryClient } from "@tanstack/react-query"
import { useNavigate } from "react-router"

import { useAuthCookieLogout } from "~/api/gen/endpoints"

export function useLogout(): () => Promise<void> {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const logout = useAuthCookieLogout({
    mutation: { meta: { skipAuthRedirect: true } },
  })

  return async () => {
    try {
      await logout.mutateAsync()
    } finally {
      queryClient.clear()
      await navigate("/login", { replace: true })
    }
  }
}
```

`frontend/app/features/shell/UserMenu.tsx`:

```tsx
import { LogOutIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

import type { UserRead } from "~/api/gen/schemas"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu"
import {
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "~/components/ui/sidebar"
import { useLogout } from "~/features/auth/hooks/use-logout"

export function UserMenu({ user }: { user: UserRead }) {
  const { t } = useTranslation("common")
  const logout = useLogout()
  const displayName =
    user.display_name !== undefined && user.display_name !== ""
      ? user.display_name
      : user.email

  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger
            render={
              <SidebarMenuButton size="lg" aria-label={t("userMenu.open")} />
            }
          >
            <span className="truncate">{displayName}</span>
          </DropdownMenuTrigger>
          <DropdownMenuContent side="top" align="start">
            <DropdownMenuGroup>
              <DropdownMenuLabel>{user.email}</DropdownMenuLabel>
            </DropdownMenuGroup>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={() => {
                void logout()
              }}
            >
              <LogOutIcon aria-hidden="true" />
              {t("userMenu.logout")}
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
```

`frontend/app/features/shell/AppSidebar.tsx`:

```tsx
import { MessageSquareIcon, SettingsIcon, SparklesIcon } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Link, useLocation } from "react-router"

import type { UserRead } from "~/api/gen/schemas"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "~/components/ui/sidebar"
import { UserMenu } from "~/features/shell/UserMenu"

const NAV_ITEMS = [
  { to: "/chat", label: "nav.chat", Icon: MessageSquareIcon },
  { to: "/skills", label: "nav.skills", Icon: SparklesIcon },
  { to: "/settings", label: "nav.settings", Icon: SettingsIcon },
] as const

export function AppSidebar({ user }: { user: UserRead }) {
  const { t } = useTranslation("common")
  const { pathname } = useLocation()

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <span className="truncate px-2 font-semibold">{t("appName")}</span>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV_ITEMS.map(({ to, label, Icon }) => (
                <SidebarMenuItem key={to}>
                  <SidebarMenuButton
                    render={<Link to={to} />}
                    isActive={pathname === to || pathname.startsWith(`${to}/`)}
                    tooltip={t(label)}
                  >
                    <Icon aria-hidden="true" />
                    <span>{t(label)}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <UserMenu user={user} />
      </SidebarFooter>
    </Sidebar>
  )
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cd /home/andv/personal/thesis/frontend && npx vitest run --project browser app/features/auth/lib/require-user.browser.test.tsx app/features/shell`
Expected: PASS, `Test Files 2 passed (2)`, `Tests 4 passed (4)`.

- [ ] **Step 6: Translate the shell's hard-coded labels in the shadcn components**

`frontend/app/components/ui/sidebar.tsx`:
- add `import { useTranslation } from "react-i18next"` below `import * as React from "react"`;
- in `function Sidebar`, add `const { t } = useTranslation("common")` as the first line of the body, directly above `const { isMobile, state, openMobile, setOpenMobile } = useSidebar()`;
- replace

  ```tsx
              <SheetTitle>Sidebar</SheetTitle>
              <SheetDescription>Displays the mobile sidebar.</SheetDescription>
  ```

  with

  ```tsx
              <SheetTitle>{t("shell.sidebarTitle")}</SheetTitle>
              <SheetDescription>{t("shell.sidebarDescription")}</SheetDescription>
  ```

`frontend/app/components/ui/toast.tsx`:
- add `import { useTranslation } from "react-i18next"` below `import * as React from "react"`;
- in `function ToastClose`, add `const { t } = useTranslation("common")` as the first line of the body, and replace `aria-label="Close toast"` with `aria-label={t("actions.close")}`.

`frontend/app/components/ui/toast.browser.test.tsx`: the Toaster now needs an i18next instance (react-i18next warns without one, and warnings fail tests). Add `import { TestProviders } from "../../../tests/utils/providers"` and change the render to

```tsx
  await render(
    <TestProviders>
      <Toaster>
        <SaveButton />
      </Toaster>
    </TestProviders>
  )
```

- [ ] **Step 7: Add the private layout, the placeholder pages and the route table**

`frontend/app/routes/app/layout.tsx`:

```tsx
import { useTranslation } from "react-i18next"
import { Outlet } from "react-router"

import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "~/components/ui/sidebar"
import { Spinner } from "~/components/ui/spinner"
import { requireUser } from "~/features/auth/lib/require-user"
import { AppSidebar } from "~/features/shell/AppSidebar"

import type { Route } from "./+types/layout"

/** Runs before child loaders on every client navigation into the private area. */
export const clientMiddleware: Route.ClientMiddlewareFunction[] = [
  async ({ request }) => {
    await requireUser(request)
  },
]

export async function clientLoader({ request }: Route.ClientLoaderArgs) {
  return { user: await requireUser(request) }
}

// Also run on hydration, so the first document load of /chat is guarded.
clientLoader.hydrate = true as const

export function HydrateFallback() {
  const { t } = useTranslation("common")
  return (
    <main className="flex min-h-svh items-center justify-center gap-2 text-sm text-muted-foreground">
      <Spinner />
      <span>{t("actions.loading")}</span>
    </main>
  )
}

export default function AppLayout({ loaderData }: Route.ComponentProps) {
  const { t } = useTranslation("common")

  return (
    <SidebarProvider>
      <AppSidebar user={loaderData.user} />
      <SidebarInset>
        <header className="flex h-12 items-center gap-2 border-b px-3">
          <SidebarTrigger aria-label={t("shell.toggleSidebar")} />
        </header>
        <Outlet />
      </SidebarInset>
    </SidebarProvider>
  )
}
```

`frontend/app/routes/app/chat.tsx` (Plan 9 replaces this module):

```tsx
import { useTranslation } from "react-i18next"

export default function ChatPage() {
  const { t } = useTranslation("common")
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold">{t("nav.chat")}</h1>
    </div>
  )
}
```

`frontend/app/routes/app/skills.tsx` (Plan 10 replaces this module):

```tsx
import { useTranslation } from "react-i18next"

export default function SkillsPage() {
  const { t } = useTranslation("common")
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold">{t("nav.skills")}</h1>
    </div>
  )
}
```

`frontend/app/routes/app/settings.tsx` (Plan 10 replaces this module):

```tsx
import { useTranslation } from "react-i18next"

export default function SettingsPage() {
  const { t } = useTranslation("common")
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold">{t("nav.settings")}</h1>
    </div>
  )
}
```

`frontend/app/routes.ts`:

```ts
import {
  type RouteConfig,
  index,
  layout,
  route,
} from "@react-router/dev/routes"

export default [
  index("routes/public/landing.tsx", { id: "landing-vi" }),
  route("en", "routes/public/landing.tsx", { id: "landing-en" }),
  route("login", "routes/auth/login.tsx"),
  route("register", "routes/auth/register.tsx"),
  route("auth/google/callback", "routes/auth/google-callback.tsx"),
  layout("routes/app/layout.tsx", [
    route("chat/:conversationId?", "routes/app/chat.tsx"),
    route("skills", "routes/app/skills.tsx"),
    route("settings", "routes/app/settings.tsx"),
  ]),
  route("actions/locale", "routes/actions/locale.ts"),
  route("actions/theme", "routes/actions/theme.ts"),
  route("*", "routes/not-found.tsx"),
] satisfies RouteConfig
```

- [ ] **Step 8: Full check and SSR smoke test**

```bash
cd /home/andv/personal/thesis/frontend
npm run format:write
npm run lint && npm run format && npm run typecheck && npm test
npm run build
PORT=3100 npm start > /tmp/frontend-start.log 2>&1 &
SERVER_PID=$!
timeout 30 bash -c 'until curl -sf http://localhost:3100/ > /dev/null; do sleep 0.5; done'
for path in /chat /chat/7f0c /skills /settings; do
  printf '%s ' "$path"; curl -s -o /tmp/private.html -w '%{http_code} ' "http://localhost:3100$path"; grep -c 'Đang tải…' /tmp/private.html
done
kill "$SERVER_PID"
```

Expected: all checks succeed; each private path answers `200` with the server-rendered `HydrateFallback` (`1`), because the guard runs in the browser. Redirecting a signed-out user to `/login?next=` and signing out are covered by the tests above and by Plan 10's Playwright scenarios against the real backend.

- [ ] **Step 9: Commit**

```bash
cd /home/andv/personal/thesis
git add frontend/app/features/auth/lib/require-user.ts frontend/app/features/auth/lib/require-user.browser.test.tsx \
  frontend/app/features/auth/hooks/use-logout.ts frontend/app/features/shell frontend/app/routes/app \
  frontend/app/routes.ts frontend/app/i18n/resources/vi/common.json frontend/app/i18n/resources/en/common.json \
  frontend/app/components/ui/sidebar.tsx frontend/app/components/ui/toast.tsx frontend/app/components/ui/toast.browser.test.tsx
git commit -m "feat(frontend): guard private routes and add the app shell with logout" \
  -m "$COMMIT_TRAILER"
```

---

## Self-Review

### Spec coverage

| Requirement | Task |
| --- | --- |
| B §2 stack, exact versions | Task 1 (`package.json`), **Tech Stack** |
| B §3 one origin through nginx; `/api` dev proxy; loaders call `API_INTERNAL_URL` and forward `Cookie` | Task 8; Task 1 (`vite.config.ts`); Task 6 (`fetcher`) |
| B §4 folder layout, revealed `entry.server.tsx`, `features/`, `components/ui`, `components/elements`, `api/`, `i18n/`, `openapi.json`, `orval.config.ts`, `tests/e2e`, `Dockerfile` | Tasks 1–3, 5–11 (see **File Structure**) |
| B §5 `/` and `/en`: SSR, `prerender`, `meta`, hreflang vi/en/x-default | Task 7 |
| B §5 `/login`, `/register`: SSR shell, client form | Task 10 |
| B §5 `/auth/google/callback`: client only, `clientLoader` then `next` or `/chat` | Task 10 |
| B §5 `/chat`, `/chat/:conversationId`, `/skills`, `/settings`: client only in `app/layout.tsx` | Task 11 (placeholders; content in Plans 9 and 10) |
| B §5 `*`: SSR 404 with i18n | Task 7 |
| B §6 guard: `clientMiddleware` + `ensureQueryData(getUsersCurrentUserQueryOptions())` + `redirect("/login?next=")` | Task 11 (`requireUser`, via `currentUserQueryOptions` from Task 10) |
| B §6 cookie login (form-urlencoded), invalidate user, go to `next`; signed-in users skip `/login` | Task 10 |
| B §6 Google: authorize → `authorization_url`; same-origin callback | Task 10 |
| B §6 logout: `POST /auth/cookie/logout` → `queryClient.clear()` → `/login` | Task 11 |
| B §6 session expiry: `QueryCache.onError` and `MutationCache.onError` → `/login?next=` | Task 6 |
| B §6 CSRF: `csrftoken` cookie → `x-csrftoken` on POST/PATCH/PUT/DELETE | Tasks 5, 6 (the chat transport is Plan 9) |
| B §7.1 orval: react-query + fetch mutator + v5 + `useInfinite` on `cursor` + MSW; zod v4 output; `api:generate` | Task 9 |
| B §7.2 fetcher: same origin in the browser, internal URL on the server, problem+json → `ApiError`, message by `code` else `title` | Tasks 5, 6 |
| B §7.3 one browser QueryClient, no retry on 4xx | Task 6. The per-request server client is not built: v1 has no public page that calls the API (spec §7.2 says so too). |
| B §7.4 react-hook-form + `zodResolver(orval schema)` + shadcn `Field`; 422 `errors[]` → `setError(field)`, others → `root.server` | Task 10 (`applyApiError`, reusable by Plan 10) |
| B §8 remix-i18next middleware; URL `/en` → cookie `lng` → `Accept-Language` → `vi`; bundled JSON per namespace; `CustomTypeOptions`; `<html lang dir>` | Tasks 3, 7 |
| B §8 hard-coded English labels replaced by keys | Task 11 for the shell's shadcn labels; Plan 9 for registry elements |
| B §9 `shadcn init -t react-router`, `base-nova`, neutral, lucide, `~/components` | Task 1 |
| B §9 base components `Sidebar`, `Toast`, `Sheet`, `Drawer`, `Dialog`, `AlertDialog`, `Card`, `Switch`, `Badge`, `InputGroup`, `Textarea`, `Field` | Task 2 |
| B §9 remix-themes with `createCookieSessionStorage`, `ThemeProvider`, `PreventFlashOnWrongTheme`, `actions/theme.ts`; prerendered pages follow the system theme | Task 4 (Task 7 build check) |
| B §12 scripts `dev`, `build`, `start`, `typecheck`, `lint`, `format`, `test`, `e2e`, `api:generate` | Tasks 1, 9 |
| B §12 Oxlint plugins and categories, hook rules, no disable comments | Task 1 |
| B §12 Prettier with `prettier-plugin-tailwindcss`, `tailwindStylesheet`, `tailwindFunctions` | Task 1 |
| B §12 TypeScript `strict`, `noUncheckedIndexedAccess`, typegen include and `rootDirs` | Task 1 |
| B §12 Vitest `unit` and `browser` projects, MSW `onUnhandledRequest: "error"`, console warnings and errors fail | Task 1 |
| B §12 root pre-commit hooks: oxlint, `prettier --check`, typecheck, orval drift | Tasks 1, 9 |
| B §13 unit rows: problem+json → `ApiError`, CSRF cookie, fetcher | Tasks 5, 6 |
| B §13 component row: login form | Task 10 (other rows belong to Plans 9 and 10) |
| B §14 Dockerfile on `node:24-alpine`, `react-router-serve` on `PORT=3000`; nginx SSE settings; compose `frontend` and `nginx` with `network_mode: host`, `FRONTEND_PORT`, `WEB_PORT` | Task 8 |
| B §15 SSR theme test on React Router 8 | Task 4 |
| A §6 cookie `pharma_session`, CSRF names, OAuth redirect to `/auth/google/callback` | Tasks 5, 6, 8 (`PHARMA_AUTH__FRONTEND_URL`), 10 |
| A §7 problem+json shape and fastapi-users codes | Task 5 |
| A §8 `pharma-agent export-openapi`, committed `frontend/openapi.json` | Task 9 |

### Names against the overview §4

- Used exactly: `frontend/openapi.json` and the `api:openapi` command; `api:generate` runs orval; `app/api/gen/endpoints.ts`; `app/api/gen/zod.ts`; `fetcher<T>(url: string, init?: RequestInit): Promise<T>`; `ApiError { status, code, detail, errors }` and `isApiError`; `readCsrfToken(): string | undefined` and `CSRF_HEADER = "x-csrftoken"`; `queryClient`; the eight namespaces; `app/components/elements/`; `frontend/playwright.config.ts`; `frontend/tests/e2e/`.
- orval layout matches the updated overview: `app/api/gen/schemas/` with `index.ts` (orval 8.32 writes single-file schemas only for Zod) and `app/api/gen/endpoints.msw.ts` (orval names the mock file after the target), imported as `~/api/gen/schemas` and `~/api/gen/endpoints.msw`.
- Test support paths, the toast manager, the action routes, the landing paths `/` and `/en/`, and browser test naming `app/**/*.browser.test.tsx` match overview §4 line by line.
- Passwords: registration uses `passwordSchema` (at least 8 characters, message `auth:passwordTooShort`), the rule Plan 10's change-password form reuses; login keeps `min(1)`.
- Commit steps end with `-m "$COMMIT_TRAILER"`; no session URL is written in the plan.
- `ApiError` fields are `readonly`; code that reads the pinned fields is unaffected.

### Decisions later plans rely on

- The only disabled Oxlint rule is `react/react-in-jsx-scope` (a false positive under the automatic JSX runtime). `node_modules/**` must stay in `ignorePatterns`.
- The English landing page's canonical URL is `/en/`, and `/` is always Vietnamese (`publicPageLocale`).
- `InputGroupAddon` no longer focuses the input when its padding is clicked, and `Spinner` is decorative (callers render translated text). Plan 9's composer should know both.
- Queries and mutations that handle 401 themselves set `meta: { skipAuthRedirect: true }`.
- Private child routes read the user with `useRouteLoaderData<typeof clientLoader>("routes/app/layout")`. Browser tests that render the sidebar set a desktop viewport first.
- Tasks 9–11 need Plans 5 and 7; Tasks 1–8 do not.

### Placeholder and consistency scan

- No step says TBD, "similar to", or "add error handling"; every code step shows the file or an exact before/after snippet.
- Names defined before use: `createTestI18n` (Task 3) → Tasks 4, 5, 7, 10; `apiErrorMessage`, `isErrorCode` (Task 5) → Task 10; `queryClient`, the `skipAuthRedirect` meta (Task 6) → Tasks 10, 11; `publicPageLocale`, `LANDING_PATHS` (Task 7); `currentUserQueryOptions`, `TestProviders` (Task 10) → Task 11; `applyZodLocale` (Task 3) → Task 10 tests.
- Checked in the planning spike: TypeScript 7 plus type-aware Oxlint (this plan's config) on the shadcn base-nova components after the Task 2 fixes, the cast-free `fetcher`/`problem`/`csrf`, and every Task 10 and 11 component, helper and test. Also: orval generating the Task 9 layout (strict TS 7 and Prettier clean), `nginx -t` on the Task 8 template, and `docker compose config` on the Task 8 compose changes. Not executed: the Vitest browser run, the React Router build and the Docker image build (the steps above run them).
