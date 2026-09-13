import { useEffect, type ReactNode } from "react"
import { useTranslation } from "react-i18next"
import {
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  isRouteErrorResponse,
} from "react-router"

import { Document } from "~/components/document"
import { Toaster } from "~/components/ui/toast"
import { TooltipProvider } from "~/components/ui/tooltip"
import { getLocale, i18nextMiddleware } from "~/i18n/middleware.server"

import type { Route } from "./+types/root"
import stylesheet from "./app.css?url"

export const links: Route.LinksFunction = () => [
  { rel: "stylesheet", href: stylesheet },
]

export const middleware = [i18nextMiddleware]

export function loader({ context }: Route.LoaderArgs) {
  return { locale: getLocale(context) }
}

export function Layout({ children }: { children: ReactNode }) {
  return (
    <Document>
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        {children}
        <ScrollRestoration />
        <Scripts />
      </body>
    </Document>
  )
}

export default function App({ loaderData }: Route.ComponentProps) {
  const { i18n } = useTranslation()

  useEffect(() => {
    if (i18n.language !== loaderData.locale) {
      void i18n.changeLanguage(loaderData.locale)
    }
  }, [i18n, loaderData.locale])

  return (
    <Toaster>
      <TooltipProvider>
        <Outlet />
      </TooltipProvider>
    </Toaster>
  )
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  const { t } = useTranslation("common")
  const notFound = isRouteErrorResponse(error) && error.status === 404

  return (
    <main className="mx-auto flex min-h-svh max-w-md flex-col justify-center gap-2 p-6">
      <h1 className="text-2xl font-semibold">
        {notFound ? t("errorPage.notFoundTitle") : t("errorPage.title")}
      </h1>
      <p className="text-muted-foreground">
        {notFound
          ? t("errorPage.notFoundDescription")
          : t("errorPage.description")}
      </p>
      {import.meta.env.DEV && error instanceof Error ? (
        <pre className="overflow-x-auto text-xs">{error.stack}</pre>
      ) : null}
    </main>
  )
}
