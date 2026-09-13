import { type RouteConfig, index, route } from "@react-router/dev/routes"

export default [
  index("routes/public/landing.tsx", { id: "landing-vi" }),
  route("en", "routes/public/landing.tsx", { id: "landing-en" }),
  route("actions/locale", "routes/actions/locale.ts"),
  route("actions/theme", "routes/actions/theme.ts"),
  route("*", "routes/not-found.tsx"),
] satisfies RouteConfig
