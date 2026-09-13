import { useQueryClient } from "@tanstack/react-query"
import { useTranslation } from "react-i18next"

import {
  getListSkillsQueryKey,
  useDeleteSkill,
  useSetSkillEnabled,
} from "~/api/gen/endpoints"
import type { SkillView } from "~/api/gen/schemas"
import { toast } from "~/components/ui/toast"

type ToggleContext = { previous: SkillView[] | undefined }

/** `PATCH /skills/{name}` with the switch updated in the list cache first. */
export function useToggleSkill() {
  const queryClient = useQueryClient()
  const { t } = useTranslation("skills")
  const queryKey = getListSkillsQueryKey()

  return useSetSkillEnabled<unknown, ToggleContext>({
    mutation: {
      onMutate: async ({ name, data }) => {
        await queryClient.cancelQueries({ queryKey })
        const previous = queryClient.getQueryData<SkillView[]>(queryKey)
        queryClient.setQueryData<SkillView[]>(queryKey, (current) =>
          current?.map((item) =>
            item.name === name ? { ...item, enabled: data.enabled } : item
          )
        )
        return { previous }
      },
      onError: (_error, { name }, context) => {
        if (context?.previous !== undefined) {
          queryClient.setQueryData<SkillView[]>(queryKey, context.previous)
        }
        toast.add({ type: "error", title: t("card.toggleFailed", { name }) })
      },
      onSettled: () => queryClient.invalidateQueries({ queryKey }),
    },
  })
}

/** `DELETE /skills/{name}`; the card disappears once the server confirms. */
export function useRemoveSkill() {
  const queryClient = useQueryClient()
  const { t } = useTranslation("skills")
  const queryKey = getListSkillsQueryKey()

  return useDeleteSkill({
    mutation: {
      onSuccess: (_data, { name }) => {
        queryClient.setQueryData<SkillView[]>(queryKey, (current) =>
          current?.filter((item) => item.name !== name)
        )
        toast.add({ type: "success", title: t("delete.done", { name }) })
      },
      onError: (_error, { name }) => {
        toast.add({ type: "error", title: t("delete.failed", { name }) })
      },
      onSettled: () => queryClient.invalidateQueries({ queryKey }),
    },
  })
}
