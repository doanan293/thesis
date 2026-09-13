import { useTranslation } from "react-i18next"

export default function SettingsPage() {
  const { t } = useTranslation("common")
  return (
    <div className="p-6">
      <h1 className="text-xl font-semibold">{t("nav.settings")}</h1>
    </div>
  )
}
