import { useQueryClient } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router"

import { useDeleteConversation } from "~/api/gen/endpoints"
import type { ConversationView } from "~/api/gen/schemas"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "~/components/ui/alert-dialog"
import { toast } from "~/components/ui/toast"

import { removeConversationFromCache } from "../lib/conversation-cache"

export type DeleteConversationDialogProps = {
  conversation: ConversationView | null
  activeConversationId: string | undefined
  onClose: () => void
}

export function DeleteConversationDialog({
  conversation,
  activeConversationId,
  onClose,
}: DeleteConversationDialogProps) {
  const { t } = useTranslation("chat")
  const queryClient = useQueryClient()
  const navigate = useNavigate()
  const deleteConversation = useDeleteConversation()

  function confirm() {
    if (conversation === null) {
      return
    }
    const deletedId = conversation.id
    deleteConversation.mutate(
      { conversationId: deletedId },
      {
        onSuccess: () => {
          removeConversationFromCache(queryClient, deletedId)
          onClose()
          if (deletedId === activeConversationId) {
            void navigate("/chat", { replace: true })
          }
        },
        onError: () => {
          toast.add({ title: t("conversations.deleteFailed"), type: "error" })
        },
      }
    )
  }

  return (
    <AlertDialog
      open={conversation !== null}
      onOpenChange={(open) => {
        if (!open) {
          onClose()
        }
      }}
    >
      <AlertDialogContent>
        <AlertDialogHeader>
          <AlertDialogTitle>{t("conversations.deleteTitle")}</AlertDialogTitle>
          <AlertDialogDescription>
            {t("conversations.deleteDescription", {
              title: conversation?.title ?? "",
            })}
          </AlertDialogDescription>
        </AlertDialogHeader>
        <AlertDialogFooter>
          <AlertDialogCancel>{t("conversations.cancel")}</AlertDialogCancel>
          <AlertDialogAction
            variant="destructive"
            disabled={deleteConversation.isPending}
            onClick={confirm}
          >
            {t("conversations.deleteConfirm")}
          </AlertDialogAction>
        </AlertDialogFooter>
      </AlertDialogContent>
    </AlertDialog>
  )
}
