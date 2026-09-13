import { ArrowUpIcon, SquareIcon } from "lucide-react"
import { type FormEvent, type KeyboardEvent, useId } from "react"
import { useTranslation } from "react-i18next"

import { chatStreamBodyMessageMax } from "~/api/gen/zod"
import {
  InputGroup,
  InputGroupAddon,
  InputGroupButton,
  InputGroupTextarea,
} from "~/components/ui/input-group"
import { cn } from "~/lib/utils"

export type ComposerProps = {
  value: string
  onValueChange: (value: string) => void
  busy: boolean
  error: string | null
  onSubmit: (text: string) => void
  onStop: () => void
  disabled?: boolean
}

export function Composer({
  value,
  onValueChange,
  busy,
  error,
  onSubmit,
  onStop,
  disabled = false,
}: ComposerProps) {
  const { t } = useTranslation("chat")
  const feedbackId = useId()
  const question = value.trim()
  const tooLong = value.length > chatStreamBodyMessageMax
  const canSend = !busy && !disabled && question.length > 0 && !tooLong
  const message = tooLong
    ? t("composer.tooLong", {
        count: value.length,
        max: chatStreamBodyMessageMax,
      })
    : error

  function submit() {
    if (canSend) {
      onSubmit(question)
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    submit()
  }

  function handleKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault()
      submit()
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      className="mx-auto w-full max-w-3xl px-4 pb-4"
    >
      <InputGroup>
        <InputGroupTextarea
          aria-label={t("composer.label")}
          aria-invalid={message !== null}
          aria-describedby={feedbackId}
          placeholder={t("composer.placeholder")}
          rows={2}
          value={value}
          onChange={(event) => onValueChange(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        <InputGroupAddon align="block-end">
          <span
            id={feedbackId}
            className={cn("text-xs", message !== null && "text-destructive")}
          >
            {message}
          </span>
          {busy ? (
            <InputGroupButton
              type="button"
              size="icon-sm"
              variant="default"
              className="ms-auto"
              aria-label={t("composer.stop")}
              onClick={onStop}
            >
              <SquareIcon aria-hidden />
            </InputGroupButton>
          ) : (
            <InputGroupButton
              type="submit"
              size="icon-sm"
              variant="default"
              className="ms-auto"
              aria-label={t("composer.send")}
              disabled={!canSend}
            >
              <ArrowUpIcon aria-hidden />
            </InputGroupButton>
          )}
        </InputGroupAddon>
      </InputGroup>
    </form>
  )
}
