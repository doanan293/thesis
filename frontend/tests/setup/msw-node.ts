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
