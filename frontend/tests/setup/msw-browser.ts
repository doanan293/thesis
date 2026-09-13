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
