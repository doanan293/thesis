import { createCookieSessionStorage } from "react-router"
import { createThemeSessionResolver } from "remix-themes"

const ONE_YEAR_SECONDS = 60 * 60 * 24 * 365

export const themeSessionResolver = createThemeSessionResolver(
  createCookieSessionStorage({
    cookie: {
      name: "theme",
      path: "/",
      httpOnly: true,
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
      maxAge: ONE_YEAR_SECONDS,
      // The cookie only holds "light" or "dark". React Router warns about unsigned
      // session cookies, so it is signed; the value is not sensitive.
      secrets: [process.env.THEME_COOKIE_SECRET ?? "pharma-agent-theme"],
    },
  })
)
