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
