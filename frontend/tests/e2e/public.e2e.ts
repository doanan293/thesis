import { expect, test } from "@playwright/test"

const ROOT_HREF = /^(https?:\/\/[^/]+)?\/$/
const EN_HREF = /^(https?:\/\/[^/]+)?\/en\/$/

test("/en/ declares its language and hreflang alternates", async ({ page }) => {
  await page.goto("/en/")

  await expect(page.locator("html")).toHaveAttribute("lang", "en")
  await expect(
    page.locator('link[rel="alternate"][hreflang="vi"]')
  ).toHaveAttribute("href", ROOT_HREF)
  await expect(
    page.locator('link[rel="alternate"][hreflang="en"]')
  ).toHaveAttribute("href", EN_HREF)
  await expect(
    page.locator('link[rel="alternate"][hreflang="x-default"]')
  ).toHaveAttribute("href", ROOT_HREF)
})

test("the prerendered /en/ HTML already contains the hreflang links", async ({
  request,
}) => {
  const response = await request.get("/en/")
  expect(response.status()).toBe(200)
  const html = await response.text()

  expect(html).toMatch(/<html[^>]*lang="en"/)
  // React 19 serializes the prop as `hrefLang`; HTML attribute names are case-insensitive.
  expect(html).toMatch(/<link[^>]*hreflang="vi"/i)
  expect(html).toMatch(/<link[^>]*hreflang="en"/i)
  expect(html).toMatch(/<link[^>]*hreflang="x-default"/i)
})
