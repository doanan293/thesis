import type { ReactNode } from "react"
import { useTranslation } from "react-i18next"

export function Document({ children }: { children: ReactNode }) {
  const { i18n } = useTranslation()
  return (
    <html lang={i18n.language} dir={i18n.dir(i18n.language)}>
      {children}
    </html>
  )
}
