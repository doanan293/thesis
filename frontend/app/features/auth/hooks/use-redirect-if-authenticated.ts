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
