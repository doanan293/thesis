import { describe, expect, test } from "vitest"

import { ApiError } from "~/api/problem"

import { createTestI18n } from "../../../../tests/utils/i18n"
import { toUploadError } from "./upload-errors"

const i18n = createTestI18n("vi")
const t = i18n.getFixedT("vi", "skills")
const tErrors = i18n.getFixedT("vi", "errors")

function apiError(
  status: number,
  code: string,
  detail?: string,
  errors: ApiError["errors"] = []
): ApiError {
  return new ApiError({ status, code, title: code, detail, errors })
}

describe("toUploadError", () => {
  test("puts 409 SKILL_NAME_TAKEN on the file field", () => {
    expect(
      toUploadError(apiError(409, "SKILL_NAME_TAKEN"), t, tErrors)
    ).toEqual({ target: "file", message: "Tên kỹ năng đã được dùng." })
  })

  test("puts 413 PAYLOAD_TOO_LARGE on the file field", () => {
    expect(
      toUploadError(apiError(413, "PAYLOAD_TOO_LARGE"), t, tErrors)
    ).toEqual({ target: "file", message: "Tệp quá lớn." })
  })

  test("shows the validator message of 422 INVALID_INPUT", () => {
    expect(
      toUploadError(
        apiError(
          422,
          "INVALID_INPUT",
          "Missing required field in frontmatter: description"
        ),
        t,
        tErrors
      )
    ).toEqual({
      target: "file",
      message:
        "SKILL.md không hợp lệ: Missing required field in frontmatter: description",
    })
  })

  test("joins the items of 422 VALIDATION_ERROR", () => {
    expect(
      toUploadError(
        apiError(422, "VALIDATION_ERROR", undefined, [
          { loc: ["body", "file"], message: "Field required", type: "missing" },
          {
            loc: ["body"],
            message: "Expected UploadFile",
            type: "value_error",
          },
        ]),
        t,
        tErrors
      )
    ).toEqual({
      target: "file",
      message: "SKILL.md không hợp lệ: Field required; Expected UploadFile",
    })
  })

  test("sends other failures to the form root", () => {
    expect(
      toUploadError(apiError(503, "AGENT_UNAVAILABLE"), t, tErrors)
    ).toEqual({
      target: "root.server",
      message: "Agent tạm thời không sẵn sàng.",
    })
    expect(toUploadError(new TypeError("Failed to fetch"), t, tErrors)).toEqual(
      { target: "root.server", message: "Đã xảy ra lỗi. Vui lòng thử lại." }
    )
  })
})
