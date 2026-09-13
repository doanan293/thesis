import { zodResolver } from "@hookform/resolvers/zod"
import { useQueryClient } from "@tanstack/react-query"
import { useEffect, useId } from "react"
import { Controller, useForm } from "react-hook-form"
import { useTranslation } from "react-i18next"
import type { z } from "zod"

import { useRenameConversation } from "~/api/gen/endpoints"
import type { ConversationView } from "~/api/gen/schemas"
import { RenameConversationBody } from "~/api/gen/zod"
import { Button } from "~/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "~/components/ui/dialog"
import { Field, FieldError, FieldLabel } from "~/components/ui/field"
import { Input } from "~/components/ui/input"
import { toast } from "~/components/ui/toast"

import { replaceConversationInCache } from "../lib/conversation-cache"

type RenameValues = z.infer<typeof RenameConversationBody>

export type RenameConversationDialogProps = {
  conversation: ConversationView | null
  onClose: () => void
}

export function RenameConversationDialog({
  conversation,
  onClose,
}: RenameConversationDialogProps) {
  const { t } = useTranslation("chat")
  const inputId = useId()
  const queryClient = useQueryClient()
  const renameConversation = useRenameConversation()
  const form = useForm<RenameValues>({
    resolver: zodResolver(RenameConversationBody),
    defaultValues: { title: "" },
  })

  useEffect(() => {
    if (conversation !== null) {
      form.reset({ title: conversation.title })
    }
  }, [conversation, form])

  function submit(values: RenameValues) {
    if (conversation === null) {
      return
    }
    renameConversation.mutate(
      { conversationId: conversation.id, data: { title: values.title.trim() } },
      {
        onSuccess: (updated) => {
          replaceConversationInCache(queryClient, updated)
          onClose()
        },
        onError: () => {
          toast.add({ title: t("conversations.renameFailed"), type: "error" })
        },
      }
    )
  }

  return (
    <Dialog
      open={conversation !== null}
      onOpenChange={(open) => {
        if (!open) {
          onClose()
        }
      }}
    >
      <DialogContent showCloseButton={false}>
        <form
          noValidate
          className="grid gap-4"
          onSubmit={(event) => void form.handleSubmit(submit)(event)}
        >
          <DialogHeader>
            <DialogTitle>{t("conversations.renameTitle")}</DialogTitle>
          </DialogHeader>
          <Controller
            control={form.control}
            name="title"
            render={({ field, fieldState }) => (
              <Field data-invalid={fieldState.invalid}>
                <FieldLabel htmlFor={inputId}>
                  {t("conversations.titleLabel")}
                </FieldLabel>
                <Input
                  {...field}
                  id={inputId}
                  autoComplete="off"
                  aria-invalid={fieldState.invalid}
                />
                {fieldState.invalid ? (
                  <FieldError errors={[fieldState.error]} />
                ) : null}
              </Field>
            )}
          />
          <DialogFooter>
            <Button type="button" variant="outline" onClick={onClose}>
              {t("conversations.cancel")}
            </Button>
            <Button type="submit" disabled={renameConversation.isPending}>
              {t("conversations.save")}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}
