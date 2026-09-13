import { useTranslation } from "react-i18next"
import { Link, useSearchParams } from "react-router"

import { FieldSeparator } from "~/components/ui/field"
import { AuthCard } from "~/features/auth/AuthCard"
import { GoogleButton } from "~/features/auth/GoogleButton"
import { RegisterForm } from "~/features/auth/RegisterForm"
import { useRedirectIfAuthenticated } from "~/features/auth/hooks/use-redirect-if-authenticated"
import { safeNextPath } from "~/features/auth/lib/next-path"

export default function RegisterPage() {
  const { t } = useTranslation("auth")
  const [searchParams] = useSearchParams()
  const nextPath = safeNextPath(searchParams.get("next"))
  useRedirectIfAuthenticated(nextPath)

  return (
    <AuthCard
      title={t("register.title")}
      description={t("register.description")}
    >
      <title>{t("register.title")}</title>
      <RegisterForm nextPath={nextPath} />
      <FieldSeparator>{t("or")}</FieldSeparator>
      <GoogleButton nextPath={nextPath} />
      <p className="text-center text-sm text-muted-foreground">
        {t("register.hasAccount")}{" "}
        <Link
          to={`/login?next=${encodeURIComponent(nextPath)}`}
          className="underline underline-offset-4"
        >
          {t("register.loginLink")}
        </Link>
      </p>
    </AuthCard>
  )
}
