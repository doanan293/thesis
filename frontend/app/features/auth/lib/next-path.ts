export const DEFAULT_NEXT_PATH = "/chat"

/** Accept only same-origin paths so `next` cannot send the user to another site. */
export function safeNextPath(value: string | null | undefined): string {
  if (
    value === null ||
    value === undefined ||
    !value.startsWith("/") ||
    value.startsWith("//") ||
    value.startsWith("/\\")
  ) {
    return DEFAULT_NEXT_PATH
  }
  return value
}
