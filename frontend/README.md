# pharma-agent-web

The web frontend of the pharma AI agent: a React Router application with server rendering for the public pages and client-only chat, skills and settings pages behind sign-in.

## End-to-end tests

The Playwright suite in `tests/e2e/` runs the production build of this app against the backend E2E server: real FastAPI, Postgres and Qdrant, with a fake LLM and a fixture corpus. It covers:

- registration, login and logout
- session expiry
- asking a question and seeing the phase, the `[1]` citation, its preview and the full text
- 👎 feedback with a note
- loading older messages
- uploading, enabling, disabling and deleting a skill
- changing language and theme
- the `/en/` hreflang links

### Prerequisites

1. From the repository root: `docker compose up -d postgres qdrant`.
2. Backend dependencies: `uv sync --directory ..` (the uv workspace at the repository root). The E2E server's settings are described in the "Server E2E" section of `backend/README.md`. On every start it recreates the `pharma_e2e` database (override with `E2E_POSTGRES_DSN`) and the Qdrant alias `e2e_chunks_current` (Qdrant URL from `E2E_QDRANT_URL`); your dev data is not touched.
3. Chromium for Playwright: `npx playwright install chromium` (already done if you followed the setup above).

### Run

```bash
npm run e2e                                   # starts every server, runs all specs
npx playwright test tests/e2e/skills.e2e.ts   # one file
npx playwright test --headed --debug          # step through in a browser
npx playwright show-report                    # HTML report of the last run
```

### How the servers are wired

| Server             | Command                                                       | Port             |
| ------------------ | ------------------------------------------------------------- | ---------------- |
| Backend E2E server | `uv run python -m tests.e2e.server --port 8001` in `backend/` | 8001             |
| Frontend           | `npm run build && npm start`                                  | 3100             |
| One-origin proxy   | `node tests/e2e/proxy.ts`                                     | 3200 (`baseURL`) |

In production nginx gives the browser one origin (`/api/*` to FastAPI, everything else to React Router). `tests/e2e/proxy.ts` does the same with http-proxy-3, so cookies, CSRF and streaming behave as deployed and no test-only code enters the app. Locally, servers already running on these ports are reused; with `CI` set, Playwright always starts fresh ones.
