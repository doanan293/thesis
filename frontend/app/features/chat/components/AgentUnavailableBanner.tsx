import { CircleAlertIcon } from "lucide-react"
import { useTranslation } from "react-i18next"

import { Alert, AlertDescription, AlertTitle } from "~/components/ui/alert"

export function AgentUnavailableBanner() {
  const { t } = useTranslation("chat")
  return (
    <Alert className="mx-auto mt-3 w-[calc(100%-2rem)] max-w-3xl">
      <CircleAlertIcon aria-hidden />
      <AlertTitle>{t("agentUnavailable.title")}</AlertTitle>
      <AlertDescription>{t("agentUnavailable.description")}</AlertDescription>
    </Alert>
  )
}
