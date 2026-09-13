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
