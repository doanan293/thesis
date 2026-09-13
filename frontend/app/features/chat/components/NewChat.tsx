import { MessageSquareTextIcon } from "lucide-react"
import { useState } from "react"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router"

import { useCreateConversation } from "~/api/gen/endpoints"
import {
  Empty,
  EmptyDescription,
  EmptyHeader,
  EmptyMedia,
  EmptyTitle,
} from "~/components/ui/empty"
import { toast } from "~/components/ui/toast"

import type { ChatLocationState } from "../lib/location-state"

import { Composer } from "./Composer"

export function NewChat() {
  const { t } = useTranslation("chat")
  const navigate = useNavigate()
  const [draft, setDraft] = useState("")
  const createConversation = useCreateConversation()

  function start(question: string) {
    createConversation.mutate(undefined, {
      onSuccess: (conversation) => {
        const state: ChatLocationState = { initialQuestion: question }
        void navigate(`/chat/${conversation.id}`, { replace: true, state })
      },
      onError: () => {
        toast.add({ title: t("newChat.createFailed"), type: "error" })
      },
    })
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <Empty className="flex-1">
        <EmptyHeader>
          <EmptyMedia variant="icon">
            <MessageSquareTextIcon aria-hidden />
          </EmptyMedia>
          <EmptyTitle>{t("newChat.title")}</EmptyTitle>
          <EmptyDescription>{t("newChat.description")}</EmptyDescription>
        </EmptyHeader>
      </Empty>
      <Composer
        value={draft}
        onValueChange={setDraft}
        busy={false}
        disabled={createConversation.isPending}
        error={null}
        onSubmit={start}
        onStop={() => undefined}
      />
    </div>
  )
}
