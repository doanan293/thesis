import type { ReactNode } from "react"

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "~/components/ui/card"

export function AuthCard({
  title,
  description,
  children,
}: {
  title: string
  description: string
  children: ReactNode
}) {
  return (
    <main className="flex min-h-svh items-center justify-center bg-muted p-6">
      <Card className="w-full max-w-sm">
        <CardHeader>
          <CardTitle>
            <h1 className="text-xl font-semibold">{title}</h1>
          </CardTitle>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-6">{children}</CardContent>
      </Card>
    </main>
  )
}
