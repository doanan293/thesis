import {
  type RouteConfig,
  index,
  layout,
  route,
} from "@react-router/dev/routes"

export default [
  index("routes/public/landing.tsx", { id: "landing-vi" }),
  route("en", "routes/public/landing.tsx", { id: "landing-en" }),
  route("login", "routes/auth/login.tsx"),
  route("register", "routes/auth/register.tsx"),
  route("auth/google/callback", "routes/auth/google-callback.tsx"),
  layout("routes/app/layout.tsx", [
    route("chat/:conversationId?", "routes/app/chat.tsx"),
    route("settings", "routes/app/settings.tsx"),
  ]),
  route("actions/locale", "routes/actions/locale.ts"),
  route("actions/theme", "routes/actions/theme.ts"),
  route("*", "routes/not-found.tsx"),
] satisfies RouteConfig
