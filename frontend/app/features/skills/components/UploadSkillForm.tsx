import { zodResolver } from "@hookform/resolvers/zod"
import { useQueryClient } from "@tanstack/react-query"
import { FileTextIcon } from "lucide-react"
import { useMemo, useRef } from "react"
import { Controller, useForm, useWatch } from "react-hook-form"
import { useTranslation } from "react-i18next"

import { getListSkillsQueryKey, useUploadSkill } from "~/api/gen/endpoints"
import type { SkillView } from "~/api/gen/schemas"
import {
  Attachment,
  AttachmentContent,
  AttachmentDescription,
  AttachmentMedia,
  AttachmentTitle,
  AttachmentTrigger,
} from "~/components/ui/attachment"
import { Button } from "~/components/ui/button"
import {
  Field,
  FieldDescription,
  FieldError,
  FieldLabel,
} from "~/components/ui/field"
import { Spinner } from "~/components/ui/spinner"
import { toast } from "~/components/ui/toast"

import { toUploadError } from "../lib/upload-errors"
import {
  createUploadSkillSchema,
  type UploadSkillValues,
} from "../lib/upload-schema"

type AttachmentState = "idle" | "uploading" | "error" | "done"

function attachmentState(
  pending: boolean,
  invalid: boolean,
  hasFile: boolean
): AttachmentState {
  if (pending) {
    return "uploading"
  }
  if (invalid) {
    return "error"
  }
  return hasFile ? "done" : "idle"
}

export function UploadSkillForm() {
  const { t } = useTranslation("skills")
  const { t: tErrors } = useTranslation("errors")
  const queryClient = useQueryClient()
  const upload = useUploadSkill()
  const inputRef = useRef<HTMLInputElement | null>(null)
  const schema = useMemo(
    () =>
      createUploadSkillSchema({
        required: t("upload.errors.required"),
        wrongName: t("upload.errors.wrongName"),
        tooLarge: t("upload.errors.tooLarge"),
      }),
    [t]
  )
  const form = useForm({ resolver: zodResolver(schema) })
  const file = useWatch({ control: form.control, name: "file" })
  const rootError = form.formState.errors.root?.server?.message

  async function submit(values: UploadSkillValues) {
    try {
      const created = await upload.mutateAsync({ data: { file: values.file } })
      const queryKey = getListSkillsQueryKey()
      queryClient.setQueryData<SkillView[]>(queryKey, (current) =>
        current === undefined ? [created] : [...current, created]
      )
      await queryClient.invalidateQueries({ queryKey })
      form.reset()
      if (inputRef.current !== null) {
        inputRef.current.value = ""
      }
      toast.add({
        type: "success",
        title: t("upload.success", { title: created.title }),
      })
    } catch (error: unknown) {
      const problem = toUploadError(error, t, tErrors)
      form.setError(problem.target, {
        type: "server",
        message: problem.message,
      })
    }
  }

  return (
    <form
      noValidate
      className="flex flex-col gap-3 rounded-xl border p-4"
      onSubmit={(event) => {
        void form.handleSubmit(submit)(event)
      }}
    >
      <Controller
        control={form.control}
        name="file"
        render={({ field, fieldState }) => (
          <Field data-invalid={fieldState.invalid}>
            <FieldLabel htmlFor="skill-file">
              {t("upload.fileLabel")}
            </FieldLabel>
            <input
              id="skill-file"
              ref={(element) => {
                field.ref(element)
                inputRef.current = element
              }}
              name={field.name}
              type="file"
              accept=".md,text/markdown"
              className="sr-only"
              aria-invalid={fieldState.invalid}
              onBlur={field.onBlur}
              onChange={(event) => {
                field.onChange(event.currentTarget.files?.[0])
              }}
            />
            <Attachment
              state={attachmentState(
                upload.isPending,
                fieldState.invalid,
                file !== undefined
              )}
            >
              <AttachmentMedia>
                <FileTextIcon aria-hidden="true" />
              </AttachmentMedia>
              <AttachmentContent>
                <AttachmentTitle>
                  {file?.name ?? t("upload.choose")}
                </AttachmentTitle>
                <AttachmentDescription>
                  {file === undefined
                    ? t("upload.noFile")
                    : t("upload.size", { size: Math.ceil(file.size / 1024) })}
                </AttachmentDescription>
              </AttachmentContent>
              <AttachmentTrigger
                aria-label={t("upload.choose")}
                onClick={() => {
                  inputRef.current?.click()
                }}
              />
            </Attachment>
            <FieldDescription>{t("upload.description")}</FieldDescription>
            <FieldError errors={[fieldState.error]} />
          </Field>
        )}
      />
      {rootError === undefined ? null : (
        <p role="alert" className="text-sm text-destructive">
          {rootError}
        </p>
      )}
      <Button type="submit" className="self-start" disabled={upload.isPending}>
        {upload.isPending ? <Spinner /> : null}
        {t("upload.submit")}
      </Button>
    </form>
  )
}
