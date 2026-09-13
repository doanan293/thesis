import { http, HttpResponse } from "msw"
import { describe, expect, test } from "vitest"
import { page } from "vitest/browser"
import { render } from "vitest-browser-react"

import {
  getListSkillsMockHandler,
  getUploadSkillMockHandler,
} from "~/api/gen/endpoints.msw"
import type { SkillView } from "~/api/gen/schemas"
import { Toaster } from "~/components/ui/toast"

import { worker } from "../../../../tests/msw/browser"
import { TestProviders } from "../../../../tests/utils/providers"
import { SkillsPage } from "./SkillsPage"

const SYSTEM: SkillView = {
  name: "drug-monograph",
  title: "Chuyên luận thuốc",
  description: "Tra cứu theo chuyên luận Dược thư.",
  enabled: true,
  is_system: true,
  version: "0123456789ab",
}

const UPLOADED: SkillView = {
  name: "my-notes",
  title: "Ghi chú của tôi",
  description: "Ưu tiên liều cho trẻ em.",
  enabled: true,
  is_system: false,
  version: "ba9876543210",
}

const SKILL_MD = [
  "---",
  "name: my-notes",
  "description: Ưu tiên liều cho trẻ em.",
  "---",
  "",
  "# Ghi chú của tôi",
  "",
  "Ưu tiên mục Liều lượng cho trẻ em.",
  "",
].join("\n")

function skillFile(name = "SKILL.md"): File {
  return new File([SKILL_MD], name, { type: "text/markdown" })
}

function problemResponse(status: number, code: string, detail: string) {
  return HttpResponse.json(
    {
      type: `urn:pharma-agent:problem:${code.toLowerCase().replaceAll("_", "-")}`,
      title: code,
      status,
      code,
      detail,
    },
    { status, headers: { "Content-Type": "application/problem+json" } }
  )
}

async function renderPage(): Promise<void> {
  await render(
    <TestProviders>
      <Toaster>
        <SkillsPage />
      </Toaster>
    </TestProviders>
  )
}

async function chooseAndSubmit(file: File): Promise<void> {
  await page.getByLabelText("Tệp SKILL.md", { exact: true }).upload(file)
  await page.getByRole("button", { name: "Tải lên" }).click()
}

describe("UploadSkillForm", () => {
  test("uploads SKILL.md and shows the new skill", async () => {
    let stored: SkillView[] = [SYSTEM]
    const received: string[] = []
    worker.use(
      getListSkillsMockHandler(() => stored),
      getUploadSkillMockHandler(async ({ request }) => {
        const file = (await request.formData()).get("file")
        received.push(file instanceof File ? file.name : "not-a-file")
        stored = [...stored, UPLOADED]
        return UPLOADED
      })
    )
    await renderPage()
    await expect
      .element(page.getByText("Bạn chưa tải lên kỹ năng nào."))
      .toBeVisible()

    await chooseAndSubmit(skillFile())

    await expect
      .element(page.getByText("Đã thêm kỹ năng Ghi chú của tôi."))
      .toBeVisible()
    await expect
      .element(page.getByRole("switch", { name: "Bật my-notes" }))
      .toBeChecked()
    await expect.element(page.getByText("Chưa chọn tệp")).toBeVisible()
    expect(received).toEqual(["SKILL.md"])
  })

  test("shows 409 SKILL_NAME_TAKEN on the file field", async () => {
    worker.use(
      getListSkillsMockHandler([SYSTEM]),
      http.post("*/api/v1/skills", () =>
        problemResponse(
          409,
          "SKILL_NAME_TAKEN",
          "you already have a skill named 'my-notes'"
        )
      )
    )
    await renderPage()

    await chooseAndSubmit(skillFile())

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Tên kỹ năng đã được dùng.")
  })

  test("shows 413 PAYLOAD_TOO_LARGE on the file field", async () => {
    worker.use(
      getListSkillsMockHandler([SYSTEM]),
      http.post("*/api/v1/skills", () =>
        problemResponse(
          413,
          "PAYLOAD_TOO_LARGE",
          "SKILL.md must be at most 64 KB"
        )
      )
    )
    await renderPage()

    await chooseAndSubmit(skillFile())

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Tệp quá lớn.")
  })

  test("shows the validator message of 422 INVALID_INPUT", async () => {
    worker.use(
      getListSkillsMockHandler([SYSTEM]),
      http.post("*/api/v1/skills", () =>
        problemResponse(
          422,
          "INVALID_INPUT",
          "Missing required field in frontmatter: description"
        )
      )
    )
    await renderPage()

    await chooseAndSubmit(skillFile())

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent(
        "SKILL.md không hợp lệ: Missing required field in frontmatter: description"
      )
  })

  test("rejects a wrongly named file without calling the API", async () => {
    worker.use(getListSkillsMockHandler([SYSTEM]))
    await renderPage()

    await chooseAndSubmit(skillFile("notes.md"))

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Tệp phải có tên SKILL.md.")
  })

  test("asks for a file when none is chosen", async () => {
    worker.use(getListSkillsMockHandler([SYSTEM]))
    await renderPage()

    await page.getByRole("button", { name: "Tải lên" }).click()

    await expect
      .element(page.getByRole("alert"))
      .toHaveTextContent("Hãy chọn tệp SKILL.md.")
  })
})
