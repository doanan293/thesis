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
