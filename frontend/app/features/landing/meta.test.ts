import { describe, expect, test } from "vitest"

import { landingMeta } from "~/features/landing/meta"

const SITE = "https://pharma.example"

describe("landingMeta", () => {
  test("describes the Vietnamese page with Open Graph tags and a canonical URL", () => {
    const tags = landingMeta({
      locale: "vi",
      title: "Tiêu đề",
      description: "Mô tả",
      siteUrl: SITE,
    })

    expect(tags).toEqual(
      expect.arrayContaining([
        { title: "Tiêu đề" },
        { name: "description", content: "Mô tả" },
        { property: "og:type", content: "website" },
        { property: "og:title", content: "Tiêu đề" },
        { property: "og:description", content: "Mô tả" },
        { property: "og:url", content: "https://pharma.example/" },
        { property: "og:locale", content: "vi_VN" },
        { tagName: "link", rel: "canonical", href: "https://pharma.example/" },
      ])
    )
  })

  test("points the English page at /en/", () => {
    const tags = landingMeta({
      locale: "en",
      title: "Title",
      description: "Description",
      siteUrl: SITE,
    })

    expect(tags).toEqual(
      expect.arrayContaining([
        { property: "og:locale", content: "en_US" },
        { property: "og:url", content: "https://pharma.example/en/" },
        {
          tagName: "link",
          rel: "canonical",
          href: "https://pharma.example/en/",
        },
      ])
    )
  })

  test.each(["vi", "en"] as const)(
    "links both languages and x-default from the %s page",
    (locale) => {
      const tags = landingMeta({
        locale,
        title: "T",
        description: "D",
        siteUrl: SITE,
      })

      expect(tags).toEqual(
        expect.arrayContaining([
          {
            tagName: "link",
            rel: "alternate",
            hrefLang: "vi",
            href: "https://pharma.example/",
          },
          {
            tagName: "link",
            rel: "alternate",
            hrefLang: "en",
            href: "https://pharma.example/en/",
          },
          {
            tagName: "link",
            rel: "alternate",
            hrefLang: "x-default",
            href: "https://pharma.example/",
          },
        ])
      )
    }
  )
})
