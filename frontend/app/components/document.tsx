import type { ReactNode } from "react"
import { useTranslation } from "react-i18next"
import { useTheme } from "remix-themes"

export function Document({ children }: { children: ReactNode }) {
  const { i18n } = useTranslation()
  const [theme] = useTheme()

  return (
    <html
      lang={i18n.language}
      dir={i18n.dir(i18n.language)}
      className={theme ?? undefined}
    >
      {children}
    </html>
  )
}
