import { useTranslation } from "react-i18next"
import { Outlet } from "react-router"

import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "~/components/ui/sidebar"
import { Spinner } from "~/components/ui/spinner"
import { requireUser } from "~/features/auth/lib/require-user"
import { AppSidebar } from "~/features/shell/AppSidebar"

import type { Route } from "./+types/layout"

/** Runs before child loaders on every client navigation into the private area. */
export const clientMiddleware: Route.ClientMiddlewareFunction[] = [
  async ({ request }) => {
    await requireUser(request)
  },
]

export async function clientLoader({ request }: Route.ClientLoaderArgs) {
  return { user: await requireUser(request) }
}

// Also run on hydration, so the first document load of /chat is guarded.
clientLoader.hydrate = true as const

export function HydrateFallback() {
  const { t } = useTranslation("common")
  return (
    <main className="flex min-h-svh items-center justify-center gap-2 text-sm text-muted-foreground">
      <Spinner />
      <span>{t("actions.loading")}</span>
    </main>
  )
}

export default function AppLayout({ loaderData }: Route.ComponentProps) {
  const { t } = useTranslation("common")

  return (
    <SidebarProvider>
      <AppSidebar user={loaderData.user} />
      <SidebarInset>
        <header className="flex h-12 items-center gap-2 border-b px-3">
          <SidebarTrigger aria-label={t("shell.toggleSidebar")} />
        </header>
        <Outlet />
      </SidebarInset>
    </SidebarProvider>
  )
}
