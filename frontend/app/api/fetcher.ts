import { toApiError } from "~/api/problem"
import { CSRF_HEADER, readCsrfToken } from "~/lib/csrf"

const UNSAFE_METHODS = new Set(["POST", "PUT", "PATCH", "DELETE"])
const DEFAULT_API_INTERNAL_URL = "http://127.0.0.1:8000"

/** Same origin in the browser; the backend's internal URL when a loader calls the API. */
export function apiBaseUrl(): string {
  if (typeof document !== "undefined") {
    return ""
  }
  return process.env.API_INTERNAL_URL ?? DEFAULT_API_INTERNAL_URL
}

/** orval mutator for every generated request. */
export async function fetcher<T>(url: string, init?: RequestInit): Promise<T> {
  const method = (init?.method ?? "GET").toUpperCase()
  const headers = new Headers(init?.headers)
  if (!headers.has("Accept")) {
    headers.set("Accept", "application/json, application/problem+json")
  }
  if (UNSAFE_METHODS.has(method)) {
    const token = readCsrfToken()
    if (token !== undefined) {
      headers.set(CSRF_HEADER, token)
    }
  }

  const response = await fetch(`${apiBaseUrl()}${url}`, {
    ...init,
    method,
    headers,
    credentials: "same-origin",
  })
  if (!response.ok) {
    throw await toApiError(response)
  }

  // The generated request type decides T; JSON.parse returns `any`, so no type assertion is needed.
  const text = await response.text()
  return JSON.parse(text === "" ? "null" : text)
}
