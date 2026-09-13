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
