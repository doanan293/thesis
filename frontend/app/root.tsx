import { useEffect, type ReactNode } from "react"
import { useTranslation } from "react-i18next"
import {
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  isRouteErrorResponse,
  useRouteLoaderData,
} from "react-router"
import { PreventFlashOnWrongTheme, ThemeProvider } from "remix-themes"

import { Document } from "~/components/document"
import { Toaster } from "~/components/ui/toast"
import { TooltipProvider } from "~/components/ui/tooltip"
import { getLocale, i18nextMiddleware } from "~/i18n/middleware.server"
import { themeSessionResolver } from "~/lib/theme.server"

import type { Route } from "./+types/root"
import stylesheet from "./app.css?url"

export const links: Route.LinksFunction = () => [
  { rel: "stylesheet", href: stylesheet },
]

export const middleware = [i18nextMiddleware]

export async function loader({ request, context }: Route.LoaderArgs) {
  const { getTheme } = await themeSessionResolver(request)
  return { locale: getLocale(context), theme: getTheme() }
}

export function Layout({ children }: { children: ReactNode }) {
  const data = useRouteLoaderData<typeof loader>("root")
  const theme = data?.theme ?? null

  return (
    <ThemeProvider
      specifiedTheme={theme}
      themeAction="/actions/theme"
      disableTransitionOnThemeChange
    >
      <Document>
        <head>
          <meta charSet="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          <Meta />
          <PreventFlashOnWrongTheme ssrTheme={theme !== null} />
          <Links />
        </head>
        <body>
          {children}
          <ScrollRestoration />
          <Scripts />
        </body>
      </Document>
    </ThemeProvider>
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
