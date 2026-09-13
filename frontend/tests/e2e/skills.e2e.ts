import type { Page, Response } from "@playwright/test"

import { expect, test } from "./support/fixtures"

function skillMarkdown(name: string, title: string): string {
  return [
    "---",
    `name: ${name}`,
    "description: Ưu tiên mục Liều lượng khi hỏi về trẻ em.",
    "---",
    "",
    `# ${title}`,
    "",
    "Ưu tiên mục Liều lượng cho trẻ em.",
    "",
  ].join("\n")
}

async function uploadSkillFile(page: Page, content: string): Promise<void> {
  await page.getByLabel("Tệp SKILL.md", { exact: true }).setInputFiles({
    name: "SKILL.md",
    mimeType: "text/markdown",
    buffer: Buffer.from(content, "utf8"),
  })
  await page.getByRole("button", { name: "Tải lên" }).click()
}

function skillPatched(page: Page, name: string): Promise<Response> {
  return page.waitForResponse(
    (response) =>
      response.request().method() === "PATCH" &&
      new URL(response.url()).pathname === `/api/v1/skills/${name}`
  )
}

test("upload, disable, enable and delete an own skill", async ({
  signedInPage: page,
}) => {
  await page.goto("/skills")
  await expect(
    page.getByRole("heading", { name: "Kỹ năng hệ thống" })
  ).toBeVisible()
  await expect(page.getByText("Bạn chưa tải lên kỹ năng nào.")).toBeVisible()

  await uploadSkillFile(page, skillMarkdown("e2e-notes", "Ghi chú E2E"))
  await expect(page.getByText("Đã thêm kỹ năng Ghi chú E2E.")).toBeVisible()
  const toggle = page.getByRole("switch", { name: "Bật e2e-notes" })
  await expect(toggle).toBeChecked()

  const disabled = skillPatched(page, "e2e-notes")
  await toggle.click()
  expect((await disabled).status()).toBe(200)
  await expect(toggle).not.toBeChecked()
  await page.reload()
  await expect(
    page.getByRole("switch", { name: "Bật e2e-notes" })
  ).not.toBeChecked()

  const enabled = skillPatched(page, "e2e-notes")
  await page.getByRole("switch", { name: "Bật e2e-notes" }).click()
  expect((await enabled).status()).toBe(200)
  await page.reload()
  await expect(
    page.getByRole("switch", { name: "Bật e2e-notes" })
  ).toBeChecked()

  await page.getByRole("button", { name: "Xoá e2e-notes" }).click()
  await page
    .getByRole("alertdialog")
    .getByRole("button", { name: "Xoá", exact: true })
    .click()
  await expect(page.getByText("Đã xoá kỹ năng e2e-notes.")).toBeVisible()
  await expect(page.getByRole("switch", { name: "Bật e2e-notes" })).toHaveCount(
    0
  )
  await page.reload()
  await expect(page.getByText("Bạn chưa tải lên kỹ năng nào.")).toBeVisible()
})

test("the backend validator message is shown for an invalid SKILL.md", async ({
  signedInPage: page,
}) => {
  await page.goto("/skills")

  await uploadSkillFile(page, "---\nname: e2e-broken\n---\n\nThiếu mô tả.\n")

  await expect(
    page.getByRole("alert").filter({ hasText: "SKILL.md không hợp lệ:" })
  ).toBeVisible()
  await expect(
    page.getByRole("switch", { name: "Bật e2e-broken" })
  ).toHaveCount(0)
})

test("uploading the same skill name twice reports the conflict", async ({
  signedInPage: page,
}) => {
  await page.goto("/skills")
  await uploadSkillFile(page, skillMarkdown("e2e-twice", "Hai lần"))
  await expect(
    page.getByRole("switch", { name: "Bật e2e-twice" })
  ).toBeChecked()

  await uploadSkillFile(page, skillMarkdown("e2e-twice", "Hai lần"))

  await expect(
    page.getByRole("alert").filter({ hasText: "Tên kỹ năng đã được dùng." })
  ).toBeVisible()
  await expect(page.getByRole("switch", { name: "Bật e2e-twice" })).toHaveCount(
    1
  )
})
