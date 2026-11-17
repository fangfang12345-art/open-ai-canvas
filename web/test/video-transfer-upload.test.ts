import { afterEach, expect, test } from "bun:test";
import { apiClient } from "../src/services/api/request";
import { uploadResourceFile } from "../src/services/api/resources";

const originalAdapter = apiClient.defaults.adapter;
afterEach(() => {
    apiClient.defaults.adapter = originalAdapter;
});

test("video transfer always uses the authorized chunk upload purpose and forwards cancellation", async () => {
    const controller = new AbortController();
    const urls: string[] = [];
    const bytes: number[] = [];
    const file = new Blob([new Uint8Array(18)], { type: "video/mp4" });
    apiClient.defaults.adapter = async (config) => {
        urls.push(config.url!);
        expect(config.signal).toBe(controller.signal);
        let data: unknown;
        if (config.url === "/resources/uploads") {
            expect(JSON.parse(config.data)).toMatchObject({ purpose: "video-transfer", size: 18, kind: "video", idempotencyKey: "caller-upload-key" });
            data = { uploadId: "video-session", chunkSize: 8, chunkCount: 3 };
        } else if (config.url?.endsWith("/complete")) {
            expect(config.timeout).toBe(0);
            data = { resource: { id: "video-resource" } };
        } else {
            bytes.push((config.data as Blob).size);
            data = { index: bytes.length - 1 };
        }
        return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, data } };
    };
    expect((await uploadResourceFile(file, "video", { purpose: "video-transfer", fileName: "clip.mp4", idempotencyKey: "caller-upload-key" }, undefined, controller.signal)).id).toBe("video-resource");
    expect(bytes).toEqual([8, 8, 2]);
    expect(urls).toEqual(["/resources/uploads", "/resources/uploads/video-session/chunks/0", "/resources/uploads/video-session/chunks/1", "/resources/uploads/video-session/chunks/2", "/resources/uploads/video-session/complete"]);
});

test("video upload retries reuse one generated idempotency key in the session body", async () => {
    const keys: string[] = [];
    let completions = 0;
    apiClient.defaults.adapter = async (config) => {
        let data: unknown;
        if (config.url === "/resources/uploads") {
            const key = JSON.parse(config.data).idempotencyKey;
            expect(config.headers["X-Idempotency-Key"]).toBe(key);
            keys.push(key);
            data = { uploadId: `retry-session-${keys.length}`, chunkSize: 8, chunkCount: 1 };
        } else if (config.url?.endsWith("/complete")) {
            completions += 1;
            if (completions === 1) throw new Error("connection reset after resource was saved");
            data = { resource: { id: "idempotent-video-resource" } };
        } else {
            data = { index: 0 };
        }
        return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, data } };
    };
    expect((await uploadResourceFile(new Blob(["clip"]), "video", { purpose: "video-transfer" })).id).toBe("idempotent-video-resource");
    expect(completions).toBe(2);
    expect(keys).toHaveLength(2);
    expect(keys[0]).toMatch(/^video-transfer:.+/);
    expect(keys[1]).toBe(keys[0]);
});

test("an aborted video upload releases its session without the aborted signal and never retries", async () => {
    const controller = new AbortController();
    const urls: string[] = [];
    apiClient.defaults.adapter = async (config) => {
        urls.push(config.url!);
        if (config.method === "delete") expect(config.signal).toBeUndefined();
        if (config.url?.includes("/chunks/")) controller.abort();
        return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, data: { uploadId: "cancelled-session", chunkSize: 8, chunkCount: 3 } } };
    };
    await expect(uploadResourceFile(new Blob([new Uint8Array(18)]), "video", { purpose: "video-transfer" }, undefined, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
    expect(urls).toEqual(["/resources/uploads", "/resources/uploads/cancelled-session/chunks/0", "/resources/uploads/cancelled-session"]);
});

test("session cleanup failure preserves the original permanent upload error", async () => {
    const methods: string[] = [];
    apiClient.defaults.adapter = async (config) => {
        methods.push(config.method!);
        if (config.method === "delete") throw new Error("cleanup connection reset");
        if (config.url?.endsWith("/complete")) return { config, status: 403, statusText: "Forbidden", headers: {}, data: { code: 403, msg: "存储配额不足", reason: "quota_exceeded", data: null } };
        return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, data: { uploadId: "failed-session", chunkSize: 8, chunkCount: 1 } } };
    };
    await expect(uploadResourceFile(new Blob(["clip"]), "video", { purpose: "video-transfer" })).rejects.toThrow("存储配额不足");
    expect(methods).toEqual(["post", "put", "post", "delete"]);
});

test("storage quota rejection is reported once without restarting a video upload", async () => {
    let requests = 0;
    apiClient.defaults.adapter = async (config) => {
        requests += 1;
        return { config, status: 403, statusText: "Forbidden", headers: {}, data: { code: 403, msg: "存储配额不足", reason: "quota_exceeded", data: null } };
    };
    await expect(uploadResourceFile(new Blob(["clip"]), "video", { purpose: "video-transfer" })).rejects.toThrow("存储配额不足");
    expect(requests).toBe(1);
});
