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
