import { useTranslation } from "react-i18next"

import { useListSkills } from "~/api/gen/endpoints"
import type { SkillView } from "~/api/gen/schemas"
import { Spinner } from "~/components/ui/spinner"

import { SkillCard } from "./SkillCard"

function SkillList({ items }: { items: SkillView[] }) {
  return (
    <ul className="flex flex-col gap-3">
      {items.map((item) => (
        <li key={item.name}>
          <SkillCard skill={item} />
        </li>
      ))}
    </ul>
  )
}

export function SkillsPage() {
  const { t } = useTranslation("skills")
  const skills = useListSkills()
  const own = skills.data?.filter((item) => !item.is_system) ?? []
  const system = skills.data?.filter((item) => item.is_system) ?? []

  return (
    <div className="mx-auto flex w-full max-w-3xl flex-col gap-8 p-4 md:p-8">
      <header className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold">{t("title")}</h1>
        <p className="text-sm text-muted-foreground">{t("subtitle")}</p>
      </header>
      {skills.isPending ? <Spinner /> : null}
      {skills.isError ? (
        <p role="alert" className="text-sm text-destructive">
          {t("loadFailed")}
        </p>
      ) : null}
      {skills.isSuccess ? (
        <>
          <section
            aria-labelledby="own-skills-heading"
            className="flex flex-col gap-4"
          >
            <h2 id="own-skills-heading" className="text-lg font-medium">
              {t("own.heading")}
            </h2>
            {own.length === 0 ? (
              <p className="text-sm text-muted-foreground">{t("own.empty")}</p>
            ) : (
              <SkillList items={own} />
            )}
          </section>
          <section
            aria-labelledby="system-skills-heading"
            className="flex flex-col gap-4"
          >
            <h2 id="system-skills-heading" className="text-lg font-medium">
              {t("system.heading")}
            </h2>
            <SkillList items={system} />
          </section>
        </>
      ) : null}
    </div>
  )
}
