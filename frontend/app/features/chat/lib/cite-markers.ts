const CITE_MARKER = /\[(\d{1,3})\](?!\()/g
const CITE_INDEX = /^[1-9]\d{0,2}$/

export function toCiteRefMarkup(text: string): string {
  return text.replaceAll(
    CITE_MARKER,
    (_marker, index: string) => `<cite-ref index="${index}"></cite-ref>`
  )
}

export function parseCiteIndex(value: unknown): number | null {
  return typeof value === "string" && CITE_INDEX.test(value)
    ? Number(value)
    : null
}
