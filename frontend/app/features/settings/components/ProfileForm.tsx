import { zodResolver } from "@hookform/resolvers/zod"
import { useQueryClient } from "@tanstack/react-query"
import { useMemo } from "react"
import { Controller, useForm } from "react-hook-form"
import { useTranslation } from "react-i18next"

import {
  getUsersCurrentUserQueryKey,
  useUsersPatchCurrentUser,
} from "~/api/gen/endpoints"
import type { UserRead } from "~/api/gen/schemas"
import { Button } from "~/components/ui/button"
import { Field, FieldError, FieldLabel } from "~/components/ui/field"
import { Input } from "~/components/ui/input"
import { toast } from "~/components/ui/toast"
import { apiErrorMessage } from "~/i18n/error-message"
import { applyApiError } from "~/lib/form-errors"

import { createProfileSchema } from "../lib/account-schemas"

const PROFILE_FIELDS = ["display_name"] as const

export function ProfileForm({ user }: { user: UserRead }) {
  const { t } = useTranslation("settings")
  const { t: tErrors } = useTranslation("errors")
  const queryClient = useQueryClient()
  const patch = useUsersPatchCurrentUser()
  const schema = useMemo(
    () =>
      createProfileSchema({
        required: t("profile.required"),
        tooLong: t("profile.tooLong"),
      }),
    [t]
  )
  const form = useForm({
    resolver: zodResolver(schema),
    defaultValues: { display_name: user.display_name ?? "" },
  })
  const rootError = form.formState.errors.root?.server?.message

  const onSubmit = form.handleSubmit(async (values) => {
    try {
      const updated = await patch.mutateAsync({
        data: { display_name: values.display_name },
      })
      queryClient.setQueryData<UserRead>(getUsersCurrentUserQueryKey(), updated)
      form.reset({ display_name: updated.display_name ?? "" })
      toast.add({ type: "success", title: t("profile.saved") })
    } catch (error: unknown) {
      applyApiError(
        error,
        form.setError,
        PROFILE_FIELDS,
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
        name="display_name"
        render={({ field, fieldState }) => (
          <Field data-invalid={fieldState.invalid}>
            <FieldLabel htmlFor="settings-display-name">
              {t("profile.displayName")}
            </FieldLabel>
            <Input
              {...field}
              id="settings-display-name"
              autoComplete="nickname"
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
      <Button
        type="submit"
        className="self-start"
        disabled={patch.isPending || !form.formState.isDirty}
      >
        {t("profile.submit")}
      </Button>
    </form>
  )
}
