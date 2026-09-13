import { useMemo } from "react"
import { type Components, Streamdown } from "streamdown"

import { CitationMarker } from "~/features/citations/CitationMarker"

import { toCiteRefMarkup } from "../lib/cite-markers"
import { streamdownPlugins } from "../lib/streamdown-config"

const allowedTags = { "cite-ref": ["index"] }
const literalTagContent = ["cite-ref"]
const components: Components = { "cite-ref": CitationMarker }
// While streaming, remend would complete a dangling "[1" into a placeholder link (rendered as a
// button by link safety); citations use that syntax, so "[n" stays text until "]" arrives.
// remend keeps its link handler while either `links` or `images` is enabled, so both are off.
const remendOptions = { links: false, images: false }

export type AnswerMarkdownProps = {
  text: string
  streaming: boolean
}

export function AnswerMarkdown({ text, streaming }: AnswerMarkdownProps) {
  const markup = useMemo(() => toCiteRefMarkup(text), [text])
  return (
    <Streamdown
      className="text-sm leading-relaxed"
      mode={streaming ? "streaming" : "static"}
      isAnimating={streaming}
      plugins={streamdownPlugins}
      remend={remendOptions}
      allowedTags={allowedTags}
      literalTagContent={literalTagContent}
      components={components}
    >
      {markup}
    </Streamdown>
  )
}
