import { useEffect } from "react"
import { useTranslation } from "react-i18next"

import { useMessageScrollerScrollable } from "~/components/ui/message-scroller"
import { Spinner } from "~/components/ui/spinner"

export type OlderMessagesLoaderProps = {
  hasOlder: boolean
  loadingOlder: boolean
  userScrolled: boolean
  contentFits: boolean
  onLoadOlder: () => void
}

export function OlderMessagesLoader({
  hasOlder,
  loadingOlder,
  userScrolled,
  contentFits,
  onLoadOlder,
}: OlderMessagesLoaderProps) {
  const { t } = useTranslation("chat")
  const { start } = useMessageScrollerScrollable()
  // The scrollable state is {start: false, end: false} until the scroller measures anything,
  // which reads as "at the top of a thread that fits". Load only after the user scrolled to the
  // top, or once ChatSession has measured that the thread fits in the viewport.
  const shouldLoad =
    hasOlder && !loadingOlder && ((userScrolled && !start) || contentFits)

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
