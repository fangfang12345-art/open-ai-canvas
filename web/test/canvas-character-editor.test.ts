import { expect, test } from "bun:test";
import { resolve } from "node:path";

test("角色卡统一编辑与部分保存重试在独立 DOM 中执行", () => {
    const result = Bun.spawnSync([process.execPath, "test", "./test/fixtures/canvas-character-editor.harness.tsx"], { cwd: resolve(import.meta.dir, ".."), timeout: 30000 });
    const output = new TextDecoder().decode(result.stdout) + new TextDecoder().decode(result.stderr);
    expect(result.exitCode, output).toBe(0);
    expect(output).toContain("5 pass");
    expect(output).toContain("0 fail");
}, 35000);
