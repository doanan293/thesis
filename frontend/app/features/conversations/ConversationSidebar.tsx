import {
  MoreHorizontalIcon,
  PencilIcon,
  PlusIcon,
  Trash2Icon,
} from "lucide-react"
import { useEffect, useState } from "react"
import { useTranslation } from "react-i18next"
import { useInView } from "react-intersection-observer"
import { Link, NavLink, useParams } from "react-router"

import { useListConversationsInfinite } from "~/api/gen/endpoints"
import type { ConversationView } from "~/api/gen/schemas"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "~/components/ui/dropdown-menu"
import {
  SidebarGroup,
  SidebarGroupAction,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSkeleton,
} from "~/components/ui/sidebar"

import { DeleteConversationDialog } from "./components/DeleteConversationDialog"
import { RenameConversationDialog } from "./components/RenameConversationDialog"

export const CONVERSATION_PAGE_SIZE = 20

type PendingAction = {
  kind: "rename" | "delete"
  conversation: ConversationView
} | null

const SKELETON_KEYS = ["first", "second", "third"] as const

export function ConversationSidebar() {
  const { t } = useTranslation("chat")
  const { conversationId } = useParams()
  const [pending, setPending] = useState<PendingAction>(null)
  const {
    data,
    isPending,
    isError,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  } = useListConversationsInfinite(
    { limit: CONVERSATION_PAGE_SIZE },
    {
      query: {
        initialPageParam: undefined,
        getNextPageParam: (page) => page.next_cursor ?? undefined,
      },
    }
  )
  const { ref: sentinelRef, inView } = useInView({ rootMargin: "120px" })

  useEffect(() => {
    if (inView && hasNextPage && !isFetchingNextPage) {
      void fetchNextPage()
    }
  }, [inView, hasNextPage, isFetchingNextPage, fetchNextPage])

  const conversations = data?.pages.flatMap((page) => page.items) ?? []

  return (
    <SidebarGroup className="group-data-[collapsible=icon]:hidden">
      <SidebarGroupLabel>{t("conversations.title")}</SidebarGroupLabel>
      <SidebarGroupAction
        render={<Link to="/chat" />}
        aria-label={t("conversations.new")}
      >
        <PlusIcon aria-hidden />
      </SidebarGroupAction>
      <SidebarGroupContent>
        <SidebarMenu>
          {isPending
            ? SKELETON_KEYS.map((key) => (
                <SidebarMenuItem key={key}>
                  <SidebarMenuSkeleton />
                </SidebarMenuItem>
              ))
            : null}
          {isError && data === undefined ? (
            <SidebarMenuItem>
              <p role="alert" className="px-2 text-xs text-destructive">
                {t("conversations.loadFailed")}
              </p>
            </SidebarMenuItem>
          ) : null}
          {data !== undefined && conversations.length === 0 ? (
            <SidebarMenuItem>
              <p className="px-2 text-xs text-muted-foreground">
                {t("conversations.empty")}
              </p>
            </SidebarMenuItem>
          ) : null}
          {conversations.map((conversation) => (
            <SidebarMenuItem key={conversation.id}>
              <SidebarMenuButton
                render={<NavLink to={`/chat/${conversation.id}`} />}
                isActive={conversation.id === conversationId}
              >
                <span className="truncate">{conversation.title}</span>
              </SidebarMenuButton>
              <DropdownMenu>
                <DropdownMenuTrigger
                  render={
                    <SidebarMenuAction
                      showOnHover
                      aria-label={t("conversations.more", {
                        title: conversation.title,
                      })}
                    />
                  }
                >
                  <MoreHorizontalIcon aria-hidden />
                </DropdownMenuTrigger>
                <DropdownMenuContent side="right" align="start">
                  <DropdownMenuItem
                    onClick={() => setPending({ kind: "rename", conversation })}
                  >
                    <PencilIcon aria-hidden />
                    {t("conversations.rename")}
                  </DropdownMenuItem>
                  <DropdownMenuItem
                    variant="destructive"
                    onClick={() => setPending({ kind: "delete", conversation })}
                  >
                    <Trash2Icon aria-hidden />
                    {t("conversations.delete")}
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </SidebarMenuItem>
          ))}
          {hasNextPage ? (
            <SidebarMenuItem ref={sentinelRef}>
              <SidebarMenuSkeleton />
            </SidebarMenuItem>
          ) : null}
        </SidebarMenu>
      </SidebarGroupContent>
      <RenameConversationDialog
        conversation={pending?.kind === "rename" ? pending.conversation : null}
        onClose={() => setPending(null)}
      />
      <DeleteConversationDialog
        conversation={pending?.kind === "delete" ? pending.conversation : null}
        activeConversationId={conversationId}
        onClose={() => setPending(null)}
      />
    </SidebarGroup>
  )
}
