import { describe, expect, test } from "vitest"

import { MAX_SKILL_BYTES, createUploadSkillSchema } from "./upload-schema"

const schema = createUploadSkillSchema({
  required: "required",
  wrongName: "wrong-name",
  tooLarge: "too-large",
})

function messages(value: unknown): string[] {
  const result = schema.safeParse(value)
  return result.success ? [] : result.error.issues.map((issue) => issue.message)
}

describe("createUploadSkillSchema", () => {
  test("accepts a SKILL.md within the limit", () => {
    expect(
      messages({ file: new File(["---\nname: a\n---\n"], "SKILL.md") })
    ).toEqual([])
  })

  test("requires a file", () => {
    expect(messages({ file: undefined })).toEqual(["required"])
  })

  test("rejects other file names", () => {
    expect(messages({ file: new File(["x"], "notes.md") })).toEqual([
      "wrong-name",
    ])
  })

  test("rejects files over 64 KB", () => {
    expect(
      messages({
        file: new File(["x".repeat(MAX_SKILL_BYTES + 1)], "SKILL.md"),
      })
    ).toEqual(["too-large"])
  })
})
