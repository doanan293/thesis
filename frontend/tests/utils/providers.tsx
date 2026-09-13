import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { useState, type ReactNode } from "react"
import { I18nextProvider } from "react-i18next"

import type { Language } from "~/i18n/config"

import { createTestI18n } from "./i18n"

export function createTestQueryClient(): QueryClient {
  return new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  })
}

export function TestProviders({
  children,
  client,
  language = "vi",
}: {
  children: ReactNode
  client?: QueryClient
  language?: Language
}) {
  const [queryClient] = useState(() => client ?? createTestQueryClient())
  const [i18n] = useState(() => createTestI18n(language))

  return (
    <QueryClientProvider client={queryClient}>
      <I18nextProvider i18n={i18n}>{children}</I18nextProvider>
    </QueryClientProvider>
  )
}
