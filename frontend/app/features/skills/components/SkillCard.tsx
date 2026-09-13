import { useTranslation } from "react-i18next"

import type { SkillView } from "~/api/gen/schemas"
import { Badge } from "~/components/ui/badge"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "~/components/ui/card"
import { Switch } from "~/components/ui/switch"

import { useToggleSkill } from "../hooks/use-skill-mutations"
import { DeleteSkillButton } from "./DeleteSkillButton"

export function SkillCard({ skill }: { skill: SkillView }) {
  const { t } = useTranslation("skills")
  const toggle = useToggleSkill()

  return (
    <Card>
      <CardHeader>
        <CardTitle>{skill.title}</CardTitle>
        <CardDescription>{skill.description}</CardDescription>
        <CardAction>
          {skill.is_system ? (
            <Badge variant="secondary">{t("card.system")}</Badge>
          ) : (
            <div className="flex items-center gap-2">
              <Switch
                checked={skill.enabled}
                aria-label={t("card.enable", { name: skill.name })}
                onCheckedChange={(enabled) => {
                  toggle.mutate({ name: skill.name, data: { enabled } })
                }}
              />
              <DeleteSkillButton name={skill.name} />
            </div>
          )}
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-wrap gap-3 text-xs text-muted-foreground">
        <code>{skill.name}</code>
        <span>{t("card.version", { version: skill.version })}</span>
      </CardContent>
    </Card>
  )
}
