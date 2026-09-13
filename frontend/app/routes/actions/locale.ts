import { setLocale } from "~/i18n/locale.server"

import type { Route } from "./+types/locale"

export function action({ request }: Route.ActionArgs) {
  return setLocale(request)
}
