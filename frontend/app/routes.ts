import { type RouteConfig, index, route } from "@react-router/dev/routes"

export default [
  index("routes/public/landing.tsx", { id: "landing-vi" }),
  route("actions/locale", "routes/actions/locale.ts"),
] satisfies RouteConfig
