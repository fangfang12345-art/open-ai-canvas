import { afterEach, describe, expect, test } from "bun:test";
import { apiClient } from "../src/services/api/request";
import { cancelVideoTransfer, listVideoTransfers, runVideoTransfer, updateVideoTransfer } from "../src/services/api/video-transfer";

const originalAdapter = apiClient.defaults.adapter;
afterEach(() => {
    apiClient.defaults.adapter = originalAdapter;
});

describe("video transfer API contracts", () => {
    test("run includes only the selected branch and optimistic revision, never another output as input", async () => {
        apiClient.defaults.adapter = async (config) => {
            expect(config.url).toBe("/video-transfers/record%2F1/run");
            expect(JSON.parse(config.data)).toEqual({ branch: "moving-v6", expectedRevision: 7 });
            return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, msg: "ok", data: { record: { id: "record/1", revision: 8 } } } };
        };
        expect(await runVideoTransfer("record/1", "moving-v6", 7)).toEqual({ id: "record/1", revision: 8 });
    });

    test("settings patches preserve the two distinct branches and forward request cancellation", async () => {
        const controller = new AbortController();
        const input = {
            expectedRevision: 3,
            branches: {
                "fixed-r16": {
                    settings: {
                        target: "top-right" as const,
                        startSeconds: 0,
                        durationSeconds: 0,
                        fixedRegions: [
                            { roi: { x: 10, y: 20, width: 100, height: 40 }, referenceSeconds: 2, startSeconds: 0, durationSeconds: 0 },
                            { roi: { x: 1400, y: 20, width: 300, height: 80 }, referenceSeconds: 3, startSeconds: 0, durationSeconds: 180 },
                        ],
                    },
                },
                "moving-v6": { settings: { startSeconds: 10, durationSeconds: 5, referenceSeconds: 0, roi: { x: 10, y: 20, width: 30, height: 40 } } },
            },
        };
        apiClient.defaults.adapter = async (config) => {
            expect(config.signal).toBe(controller.signal);
            expect(JSON.parse(config.data)).toEqual(input);
            return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, msg: "ok", data: { record: { id: "record", revision: 4 } } } };
        };
        expect((await updateVideoTransfer("record", input, controller.signal)).revision).toBe(4);
    });

    test("an empty fixed-region list is sent explicitly to restore corner mode", async () => {
        const input = { expectedRevision: 4, branches: { "fixed-r16": { settings: { target: "top-left" as const, startSeconds: 0, durationSeconds: 0, fixedRegions: [] } } } };
        apiClient.defaults.adapter = async (config) => {
            expect(JSON.parse(config.data)).toEqual(input);
            return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, msg: "ok", data: { record: { id: "record", revision: 5 } } } };
        };
        expect((await updateVideoTransfer("record", input)).revision).toBe(5);
    });

    test("a rejected server write is surfaced instead of treated as saved", async () => {
        apiClient.defaults.adapter = async (config) => ({ config, status: 409, statusText: "Conflict", headers: {}, data: { code: 409, msg: "revision changed", reason: "video_transfer_conflict", data: {} } });
        await expect(updateVideoTransfer("record", { expectedRevision: 1, title: "new title" })).rejects.toThrow("revision changed");
    });

    test("aborted reads keep AbortError semantics and never reach the transport", async () => {
        let sent = false;
        apiClient.defaults.adapter = async () => {
            sent = true;
            throw new Error("should not send");
        };
        const controller = new AbortController();
        controller.abort();
        await expect(listVideoTransfers({ page: 1, pageSize: 20, q: "" }, controller.signal)).rejects.toMatchObject({ name: "AbortError" });
        expect(sent).toBe(false);
    });

    test("job cancellation sends a server command separate from aborting a browser request", async () => {
        apiClient.defaults.adapter = async (config) => {
            expect(config.method).toBe("post");
            expect(config.url).toBe("/video-transfers/record/cancel");
            return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, msg: "ok", data: { record: { id: "record" } } } };
        };
        expect((await cancelVideoTransfer("record")).id).toBe("record");
    });
});
