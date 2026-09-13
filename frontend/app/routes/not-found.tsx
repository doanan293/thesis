import { useTranslation } from "react-i18next"
import { Link, data } from "react-router"

import { buttonVariants } from "~/components/ui/button"

export function loader() {
  return data(null, { status: 404 })
}

export default function NotFound() {
  const { t } = useTranslation("common")

  return (
    <main className="mx-auto flex min-h-svh max-w-md flex-col items-start justify-center gap-3 p-6">
      <h1 className="text-2xl font-semibold">{t("errorPage.notFoundTitle")}</h1>
      <p className="text-muted-foreground">
        {t("errorPage.notFoundDescription")}
      </p>
      <Link to="/" className={buttonVariants({ variant: "outline" })}>
        {t("errorPage.backHome")}
      </Link>
    </main>
  )
}
