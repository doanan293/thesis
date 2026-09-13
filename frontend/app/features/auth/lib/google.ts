import {
  oauthGoogleCookieAuthorize,
  oauthGoogleCookieCallback,
} from "~/api/gen/endpoints"
import { safeNextPath } from "~/features/auth/lib/next-path"

/** fastapi-users' OAuth state does not carry our `next`, so it waits in sessionStorage. */
export const NEXT_PATH_STORAGE_KEY = "pharma-agent:auth-next"

function assignLocation(url: string): void {
  window.location.assign(url)
}

export async function startGoogleLogin(
  nextPath: string,
  assign: (url: string) => void = assignLocation
): Promise<void> {
  const { authorization_url: authorizationUrl } =
    await oauthGoogleCookieAuthorize()
  window.sessionStorage.setItem(NEXT_PATH_STORAGE_KEY, nextPath)
  assign(authorizationUrl)
}

export async function completeGoogleLogin(callbackUrl: URL): Promise<string> {
  const { searchParams } = callbackUrl
  // Same-origin call; the backend reads its OAuth state cookie and sets pharma_session.
  await oauthGoogleCookieCallback({
    code: searchParams.get("code") ?? undefined,
    state: searchParams.get("state") ?? undefined,
  })
  const next = window.sessionStorage.getItem(NEXT_PATH_STORAGE_KEY)
  window.sessionStorage.removeItem(NEXT_PATH_STORAGE_KEY)
  return safeNextPath(next)
}
