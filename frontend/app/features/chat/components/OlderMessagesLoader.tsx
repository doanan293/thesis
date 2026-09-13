import { useEffect } from "react"
import { useTranslation } from "react-i18next"

import { useMessageScrollerScrollable } from "~/components/ui/message-scroller"
import { Spinner } from "~/components/ui/spinner"

export type OlderMessagesLoaderProps = {
  hasOlder: boolean
  loadingOlder: boolean
  userScrolled: boolean
  onLoadOlder: () => void
}

export function OlderMessagesLoader({
  hasOlder,
  loadingOlder,
  userScrolled,
  onLoadOlder,
}: OlderMessagesLoaderProps) {
  const { t } = useTranslation("chat")
  const { start, end } = useMessageScrollerScrollable()
  const shouldLoad =
    hasOlder && !loadingOlder && !start && (userScrolled || !end)

  useEffect(() => {
    if (shouldLoad) {
      onLoadOlder()
    }
  }, [shouldLoad, onLoadOlder])

  if (!loadingOlder) {
    return null
  }
  return (
    <output className="pointer-events-none absolute inset-x-0 top-2 z-10 flex justify-center">
      <span className="flex items-center gap-2 rounded-full bg-background/90 px-3 py-1 text-xs text-muted-foreground shadow-sm">
        <Spinner />
        {t("thread.loadingOlder")}
      </span>
    </output>
  )
}
