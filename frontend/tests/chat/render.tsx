import type { QueryClient } from "@tanstack/react-query"
import type { ComponentType, ReactNode } from "react"
import { createRoutesStub, useLocation } from "react-router"
import { render, type RenderResult } from "vitest-browser-react"

import { Toaster } from "~/components/ui/toast"

import { createTestQueryClient, TestProviders } from "../utils/providers"

export type RenderRoutesOptions = {
  initialEntry: string | { pathname: string; state: unknown }
  path?: string
  wrap?: ComponentType<{ children: ReactNode }>
}

export type RenderRoutesResult = {
  screen: RenderResult
  queryClient: QueryClient
}

export function LocationProbe() {
  const location = useLocation()
  return (
    <output data-testid="location">{`${location.pathname}${location.search}`}</output>
  )
}

function PassThrough({ children }: { children: ReactNode }) {
  return <>{children}</>
}

export function renderWithProviders(ui: ReactNode): Promise<RenderResult> {
  return render(
    <TestProviders>
      <Toaster>{ui}</Toaster>
    </TestProviders>
  )
}

export async function renderRoutes(
  Page: ComponentType,
  {
    initialEntry,
    path = "/chat/:conversationId?",
    wrap: Wrap = PassThrough,
  }: RenderRoutesOptions
): Promise<RenderRoutesResult> {
  const queryClient = createTestQueryClient()

  function RoutePage() {
    return (
      <Wrap>
        <Page />
        <LocationProbe />
      </Wrap>
    )
  }

  const Stub = createRoutesStub([
    { path, Component: RoutePage },
    { path: "/login", Component: LocationProbe },
  ])

  const screen = await render(
    <TestProviders client={queryClient}>
      <Toaster>
        <Stub initialEntries={[initialEntry]} />
      </Toaster>
    </TestProviders>
  )
  return { screen, queryClient }
}
