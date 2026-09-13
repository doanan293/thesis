import { useTranslation } from "react-i18next"
import { Theme, type ThemeMetadata, isTheme, useTheme } from "remix-themes"

import { Field, FieldLabel, FieldLegend, FieldSet } from "~/components/ui/field"
import { RadioGroup, RadioGroupItem } from "~/components/ui/radio-group"

const THEME_CHOICES = ["light", "dark", "system"] as const
type ThemeChoice = (typeof THEME_CHOICES)[number]

function choiceOf(
  theme: Theme | null,
  definedBy: ThemeMetadata["definedBy"]
): ThemeChoice {
  if (definedBy === "SYSTEM") {
    return "system"
  }
  return theme === Theme.DARK ? "dark" : "light"
}

export function ThemeRadioGroup() {
  const { t } = useTranslation("settings")
  const [theme, setTheme, { definedBy }] = useTheme()

  return (
    <FieldSet>
      <FieldLegend variant="label">{t("theme.label")}</FieldLegend>
      <RadioGroup
        value={choiceOf(theme, definedBy)}
        onValueChange={(value) => {
          // remix-themes posts {theme} to the provider's themeAction; null follows the system.
          setTheme(isTheme(value) ? value : null)
        }}
      >
        {THEME_CHOICES.map((choice) => (
          <Field key={choice} orientation="horizontal">
            <RadioGroupItem id={`settings-theme-${choice}`} value={choice} />
            <FieldLabel
              htmlFor={`settings-theme-${choice}`}
              className="font-normal"
            >
              {t(`theme.${choice}`)}
            </FieldLabel>
          </Field>
        ))}
      </RadioGroup>
    </FieldSet>
  )
}
