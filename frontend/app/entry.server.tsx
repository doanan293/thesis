import { PassThrough } from "node:stream"

import { createReadableStreamFromReadable } from "@react-router/node"
import { isbot } from "isbot"
import type { RenderToPipeableStreamOptions } from "react-dom/server"
import { renderToPipeableStream } from "react-dom/server"
import { I18nextProvider } from "react-i18next"
import type { EntryContext, RouterContextProvider } from "react-router"
import { ServerRouter } from "react-router"

import { getInstance } from "~/i18n/middleware.server"

export const streamTimeout = 5_000

export default function handleRequest(
  request: Request,
  responseStatusCode: number,
  responseHeaders: Headers,
  routerContext: EntryContext,
  loadContext: RouterContextProvider
) {
  // https://httpwg.org/specs/rfc9110.html#HEAD
  if (request.method.toUpperCase() === "HEAD") {
    return new Response(null, {
      status: responseStatusCode,
      headers: responseHeaders,
    })
  }

  return new Promise<Response>((resolve, reject) => {
    let shellRendered = false
    let statusCode = responseStatusCode
    const userAgent = request.headers.get("user-agent")

    // Bots and SPA mode wait for all content so crawlers get the full page.
    const readyOption: keyof RenderToPipeableStreamOptions =
      (userAgent !== null && isbot(userAgent)) || routerContext.isSpaMode
        ? "onAllReady"
        : "onShellReady"

    let timeoutId: ReturnType<typeof setTimeout> | undefined = setTimeout(
      () => {
        abort()
      },
      streamTimeout + 1000
    )

    const { pipe, abort } = renderToPipeableStream(
      <I18nextProvider i18n={getInstance(loadContext)}>
        <ServerRouter context={routerContext} url={request.url} />
      </I18nextProvider>,
      {
        [readyOption]() {
          shellRendered = true
          const body = new PassThrough({
            final(callback) {
              clearTimeout(timeoutId)
              timeoutId = undefined
              callback()
            },
          })
          const stream = createReadableStreamFromReadable(body)

          responseHeaders.set("Content-Type", "text/html")

          pipe(body)

          resolve(
            new Response(stream, {
              headers: responseHeaders,
              status: statusCode,
            })
          )
        },
        onShellError(error: unknown) {
          reject(error instanceof Error ? error : new Error(String(error)))
        },
        onError(error: unknown) {
          statusCode = 500
          // Errors before the shell reject above and are logged by React Router.
          if (shellRendered) {
            console.error(error)
          }
        },
      }
    )
  })
}
