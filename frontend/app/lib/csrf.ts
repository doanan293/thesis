import { parseCookie } from "cookie"

/** Cookie set by the backend CSRF middleware; readable by JavaScript on purpose. */
export const CSRF_COOKIE = "csrftoken"
export const CSRF_HEADER = "x-csrftoken"

export function readCsrfToken(): string | undefined {
  if (typeof document === "undefined") {
    return undefined
  }
  const token = parseCookie(document.cookie)[CSRF_COOKIE]
  return token === "" ? undefined : token
}
