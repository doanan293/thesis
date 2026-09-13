import { expect, test } from "vitest"

import { readCsrfToken } from "~/lib/csrf"

test("readCsrfToken returns undefined on the server", () => {
  expect(readCsrfToken()).toBeUndefined()
})
