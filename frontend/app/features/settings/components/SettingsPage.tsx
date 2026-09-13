import { useTranslation } from "react-i18next"

import { useUsersCurrentUser } from "~/api/gen/endpoints"
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card"
import { Spinner } from "~/components/ui/spinner"

import { LanguageSelect } from "./LanguageSelect"
import { PasswordForm } from "./PasswordForm"
import { ProfileForm } from "./ProfileForm"
import { ThemeRadioGroup } from "./ThemeRadioGroup"

export function SettingsPage() {
  const { t } = useTranslation("settings")
  const user = useUsersCurrentUser()

  return (
    <div className="mx-auto flex w-full max-w-2xl flex-col gap-6 p-4 md:p-8">
      <h1 className="text-2xl font-semibold">{t("title")}</h1>
      <Card>
        <CardHeader>
          <CardTitle>{t("profile.heading")}</CardTitle>
        </CardHeader>
        <CardContent>
          {user.data === undefined ? (
            <Spinner />
          ) : (
            <ProfileForm key={user.data.id} user={user.data} />
          )}
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>{t("password.heading")}</CardTitle>
        </CardHeader>
        <CardContent>
          <PasswordForm />
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>{t("appearance.heading")}</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-6">
          <LanguageSelect />
          <ThemeRadioGroup />
        </CardContent>
      </Card>
    </div>
  )
}
