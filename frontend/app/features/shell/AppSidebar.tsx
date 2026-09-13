import { MessageSquareIcon, SettingsIcon, SparklesIcon } from "lucide-react"
import { useTranslation } from "react-i18next"
import { Link, useLocation } from "react-router"

import type { UserRead } from "~/api/gen/schemas"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from "~/components/ui/sidebar"
import { UserMenu } from "~/features/shell/UserMenu"

const NAV_ITEMS = [
  { to: "/chat", label: "nav.chat", Icon: MessageSquareIcon },
  { to: "/skills", label: "nav.skills", Icon: SparklesIcon },
  { to: "/settings", label: "nav.settings", Icon: SettingsIcon },
] as const

export function AppSidebar({ user }: { user: UserRead }) {
  const { t } = useTranslation("common")
  const { pathname } = useLocation()

  return (
    <Sidebar collapsible="icon">
      <SidebarHeader>
        <span className="truncate px-2 font-semibold">{t("appName")}</span>
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupContent>
            <SidebarMenu>
              {NAV_ITEMS.map(({ to, label, Icon }) => (
                <SidebarMenuItem key={to}>
                  <SidebarMenuButton
                    render={<Link to={to} />}
                    isActive={pathname === to || pathname.startsWith(`${to}/`)}
                    tooltip={t(label)}
                  >
                    <Icon aria-hidden="true" />
                    <span>{t(label)}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter>
        <UserMenu user={user} />
      </SidebarFooter>
    </Sidebar>
  )
}
