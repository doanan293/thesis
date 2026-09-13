import type { ReactNode } from "react"
import { Links, Meta, Outlet, Scripts, ScrollRestoration } from "react-router"

import { Toaster } from "~/components/ui/toast"
import { TooltipProvider } from "~/components/ui/tooltip"

import type { Route } from "./+types/root"
import stylesheet from "./app.css?url"

export const links: Route.LinksFunction = () => [
  { rel: "stylesheet", href: stylesheet },
]

export function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="vi">
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
    </html>
  )
}

export default function App() {
  return (
    <Toaster>
      <TooltipProvider>
        <Outlet />
      </TooltipProvider>
    </Toaster>
  )
}
