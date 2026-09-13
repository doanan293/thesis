import { zodResolver } from "@hookform/resolvers/zod"
import { useMemo } from "react"
import { Controller, useForm } from "react-hook-form"
import { useTranslation } from "react-i18next"

import { useUsersPatchCurrentUser } from "~/api/gen/endpoints"
import { isApiError } from "~/api/problem"
import { Button } from "~/components/ui/button"
import { Field, FieldError, FieldLabel } from "~/components/ui/field"
import { Input } from "~/components/ui/input"
import { toast } from "~/components/ui/toast"
import { apiErrorMessage } from "~/i18n/error-message"
import { applyApiError } from "~/lib/form-errors"

import { createPasswordSchema } from "../lib/account-schemas"

const PASSWORD_FIELDS = ["password"] as const

export function PasswordForm() {
  const { t } = useTranslation("settings")
  const { t: tErrors } = useTranslation("errors")
  const patch = useUsersPatchCurrentUser()
  const schema = useMemo(
    () =>
      createPasswordSchema({
        tooShort: t("password.tooShort"),
        tooLong: t("password.tooLong"),
        mismatch: t("password.mismatch"),
      }),
    [t]
  )
  const form = useForm({
    resolver: zodResolver(schema),
    defaultValues: { password: "", confirm: "" },
  })
  const rootError = form.formState.errors.root?.server?.message

  const onSubmit = form.handleSubmit(async (values) => {
    try {
      await patch.mutateAsync({ data: { password: values.password } })
      form.reset({ password: "", confirm: "" })
      toast.add({ type: "success", title: t("password.saved") })
    } catch (error: unknown) {
      if (isApiError(error) && error.code === "UPDATE_USER_INVALID_PASSWORD") {
        form.setError("password", {
          type: "server",
          message: apiErrorMessage(error, tErrors),
        })
        return
      }
      applyApiError(
        error,
        form.setError,
        PASSWORD_FIELDS,
        apiErrorMessage(error, tErrors)
      )
    }
  })

  return (
    <form
      noValidate
      className="flex flex-col gap-4"
      onSubmit={(event) => {
        void onSubmit(event)
      }}
    >
      <Controller
        control={form.control}
        name="password"
        render={({ field, fieldState }) => (
          <Field data-invalid={fieldState.invalid}>
            <FieldLabel htmlFor="settings-new-password">
              {t("password.new")}
            </FieldLabel>
            <Input
              {...field}
              id="settings-new-password"
              type="password"
              autoComplete="new-password"
              aria-invalid={fieldState.invalid}
            />
            <FieldError errors={[fieldState.error]} />
          </Field>
        )}
      />
      <Controller
        control={form.control}
        name="confirm"
        render={({ field, fieldState }) => (
          <Field data-invalid={fieldState.invalid}>
            <FieldLabel htmlFor="settings-confirm-password">
              {t("password.confirm")}
            </FieldLabel>
            <Input
              {...field}
              id="settings-confirm-password"
              type="password"
              autoComplete="new-password"
              aria-invalid={fieldState.invalid}
            />
            <FieldError errors={[fieldState.error]} />
          </Field>
        )}
      />
      {rootError === undefined ? null : (
        <p role="alert" className="text-sm text-destructive">
          {rootError}
        </p>
      )}
      <Button type="submit" className="self-start" disabled={patch.isPending}>
        {t("password.submit")}
      </Button>
    </form>
  )
}
