import { useCallback } from "react"
import { useTranslation } from "react-i18next"

export type PageLabel =
  | { key: "pages.single"; page: number }
  | { key: "pages.range"; start: number; end: number }
  | null

export function pageLabel(
  startPage: number | null,
  endPage: number | null
): PageLabel {
  if (startPage === null && endPage === null) {
    return null
  }
  if (startPage === null || endPage === null || startPage === endPage) {
    return { key: "pages.single", page: startPage ?? endPage ?? 0 }
  }
  return { key: "pages.range", start: startPage, end: endPage }
}

export function usePageText(): (
  startPage: number | null,
  endPage: number | null
) => string | null {
  const { t } = useTranslation("citations")
  return useCallback(
    (startPage: number | null, endPage: number | null) => {
      const label = pageLabel(startPage, endPage)
      if (label === null) {
        return null
      }
      return label.key === "pages.single"
        ? t("pages.single", { page: label.page })
        : t("pages.range", { start: label.start, end: label.end })
    },
    [t]
  )
}
