import { zodResolver } from "@hookform/resolvers/zod"
import { useQueryClient } from "@tanstack/react-query"
import { Controller, useForm } from "react-hook-form"
import { useTranslation } from "react-i18next"
import { useNavigate } from "react-router"
import { z } from "zod"

import {
  getUsersCurrentUserQueryKey,
  useAuthCookieLogin,
} from "~/api/gen/endpoints"
import { AuthCookieLoginBody } from "~/api/gen/zod"
import { Button } from "~/components/ui/button"
import {
  Field,
  FieldError,
  FieldGroup,
  FieldLabel,
} from "~/components/ui/field"
import { Input } from "~/components/ui/input"
import { Spinner } from "~/components/ui/spinner"
import { apiErrorMessage } from "~/i18n/error-message"
import { applyApiError } from "~/lib/form-errors"

const loginSchema = AuthCookieLoginBody.pick({
  username: true,
  password: true,
}).extend({
  username: z.email(),
  password: z.string().min(1),
})

type LoginValues = z.infer<typeof loginSchema>

const LOGIN_FIELDS = ["username", "password"] as const

export function LoginForm({ nextPath }: { nextPath: string }) {
  const { t } = useTranslation("auth")
  const { t: tErrors } = useTranslation("errors")
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const login = useAuthCookieLogin({
    mutation: { meta: { skipAuthRedirect: true } },
  })
  const form = useForm<LoginValues>({
    resolver: zodResolver(loginSchema),
    defaultValues: { username: "", password: "" },
  })

  async function onSubmit(values: LoginValues) {
    try {
      await login.mutateAsync({ data: values })
    } catch (error) {
      applyApiError(
        error,
        form.setError,
        LOGIN_FIELDS,
        apiErrorMessage(error, tErrors)
      )
      return
    }
    await queryClient.invalidateQueries({
      queryKey: getUsersCurrentUserQueryKey(),
    })
    await navigate(nextPath, { replace: true })
  }

  const serverError = form.formState.errors.root?.server?.message

  return (
    <form
      noValidate
      onSubmit={form.handleSubmit(onSubmit)}
      className="flex flex-col gap-6"
    >
      <FieldGroup>
        <Controller
          name="username"
          control={form.control}
          render={({ field, fieldState }) => (
            <Field data-invalid={fieldState.invalid}>
              <FieldLabel htmlFor="login-email">{t("email")}</FieldLabel>
              <Input
                {...field}
                id="login-email"
                type="email"
                autoComplete="email"
                aria-invalid={fieldState.invalid}
              />
              <FieldError errors={[fieldState.error]} />
            </Field>
          )}
        />
        <Controller
          name="password"
          control={form.control}
          render={({ field, fieldState }) => (
            <Field data-invalid={fieldState.invalid}>
              <FieldLabel htmlFor="login-password">{t("password")}</FieldLabel>
              <Input
                {...field}
                id="login-password"
                type="password"
                autoComplete="current-password"
                aria-invalid={fieldState.invalid}
              />
              <FieldError errors={[fieldState.error]} />
            </Field>
          )}
        />
      </FieldGroup>
      {serverError === undefined ? null : (
        <p role="alert" className="text-sm text-destructive">
          {serverError}
        </p>
      )}
      <Button type="submit" disabled={form.formState.isSubmitting}>
        {form.formState.isSubmitting ? <Spinner /> : null}
        {t("login.submit")}
      </Button>
    </form>
  )
}
