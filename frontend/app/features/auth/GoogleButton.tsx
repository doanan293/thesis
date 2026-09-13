import { useState } from "react"
import { useTranslation } from "react-i18next"

import { Button } from "~/components/ui/button"
import { startGoogleLogin } from "~/features/auth/lib/google"
import { apiErrorMessage } from "~/i18n/error-message"

export function GoogleButton({ nextPath }: { nextPath: string }) {
  const { t } = useTranslation("auth")
  const { t: tErrors } = useTranslation("errors")
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function signIn() {
    setPending(true)
    setError(null)
    try {
      await startGoogleLogin(nextPath)
    } catch (caught) {
      setError(apiErrorMessage(caught, tErrors))
      setPending(false)
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <Button
        type="button"
        variant="outline"
        disabled={pending}
        onClick={() => {
          void signIn()
        }}
      >
        {t("google")}
      </Button>
      {error === null ? null : (
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      )}
    </div>
  )
}
