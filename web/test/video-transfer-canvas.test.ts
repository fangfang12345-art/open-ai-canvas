import { afterAll, afterEach, beforeEach, expect, mock, test } from "bun:test";
import { resolve } from "node:path";
import type { InternalAxiosRequestConfig } from "axios";
import type { RemoteResource, ResourceUploadMeta } from "../src/services/api/resources";
import type { VideoTransferRecord } from "../src/services/api/video-transfer";

if (process.env.VIDEO_TRANSFER_DOM_TEST !== "1") {
    test("video transfer canvas executes its save, run, conflict and account lifecycles in an isolated DOM", () => {
        const result = Bun.spawnSync([process.execPath, "test", "./test/video-transfer-canvas.test.ts"], { cwd: resolve(import.meta.dir, ".."), env: { ...process.env, VIDEO_TRANSFER_DOM_TEST: "1" }, timeout: 30000 });
        const output = new TextDecoder().decode(result.stdout) + new TextDecoder().decode(result.stderr);
        expect(result.exitCode, output).toBe(0);
        expect(output).toContain("12 pass");
        expect(output).toContain("0 fail");
    }, 35000);
} else {
    const { Window } = await import("happy-dom");
    const dom = new Window({ url: "http://localhost/video-transfer/record-one" });
    for (const [key, value] of Object.entries({
        window: dom,
        document: dom.document,
        navigator: dom.navigator,
        localStorage: dom.localStorage,
        HTMLElement: dom.HTMLElement,
        HTMLInputElement: dom.HTMLInputElement,
        SVGElement: dom.SVGElement,
        Element: dom.Element,
        Node: dom.Node,
        ShadowRoot: dom.ShadowRoot,
        MutationObserver: dom.MutationObserver,
        ResizeObserver: dom.ResizeObserver,
        getComputedStyle: dom.getComputedStyle.bind(dom),
        requestAnimationFrame: dom.requestAnimationFrame.bind(dom),
        cancelAnimationFrame: dom.cancelAnimationFrame.bind(dom),
        IS_REACT_ACT_ENVIRONMENT: true,
    })) {
        Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
    }
    const React = await import("react");
    const { act, createElement: el } = React;
    const { createRoot } = await import("react-dom/client");
    const { MemoryRouter, Route, Routes } = await import("react-router");
    const { QueryClient, QueryClientProvider } = await import("@tanstack/react-query");
    const antd = await import("antd");
    const resources = await import("../src/services/api/resources");
    let resolveUpload: ((value: RemoteResource) => void) | undefined;
    let uploadMeta: ResourceUploadMeta | undefined;
    let uploadSignal: AbortSignal | undefined;
    mock.module("antd", () => ({ ...antd, App: { ...antd.App, useApp: () => ({ message: { success: () => {}, error: () => {} } }) } }));
    mock.module("../src/lib/video-transfer/media", () => ({ readVideoTransferMetadata: async () => ({ width: 1920, height: 1080, duration: 30 }) }));
    mock.module("../src/services/api/resources", () => ({
        ...resources,
        uploadResourceFile: (_file: Blob, _kind: string, meta: ResourceUploadMeta, _progress: unknown, signal: AbortSignal) => {
            uploadMeta = meta;
            uploadSignal = signal;
            return new Promise<RemoteResource>((finish) => {
                resolveUpload = finish;
            });
        },
    }));
    const { default: VideoTransferCanvasPage } = await import("../src/pages/video-transfer/canvas");
    const { apiClient } = await import("../src/services/api/request");
    const { useUserStore } = await import("../src/stores/use-user-store");
    const { videoTransferKeys } = await import("../src/lib/video-transfer/workflow");
    let host: HTMLDivElement;
    let root: ReturnType<typeof createRoot> | undefined;
    let client: InstanceType<typeof QueryClient>;
    let record: VideoTransferRecord;
    const writes: InternalAxiosRequestConfig[] = [];
    let patchGate: Promise<void> | undefined;
    let rejectPatch = false;
    const capabilities = {
        branches: [
            { id: "fixed-r16", available: true },
            { id: "moving-v6", available: true },
        ],
        limits: { maxSourceBytes: 2 ** 31, maxSourceDurationSeconds: 7200, maxDurationSeconds: 600, maxPixels: 8294400, maxWorkers: 16, maxMovingRoiPixels: 196608, maxMovingRoiDimension: 1024 },
    };

    async function flush() {
        await act(async () => {
            await new Promise((finish) => setTimeout(finish, 0));
        });
    }
    async function click(label: string, selector = "button") {
        const button = [...host.querySelectorAll<HTMLButtonElement>(selector)].find((button) => button.textContent?.trim() === label);
        expect(button, `Missing ${label}`).toBeDefined();
        await act(async () => button!.click());
        await flush();
    }
    async function number(label: string, value: number) {
        const input = host.querySelector<HTMLInputElement>(`input[aria-label="${label}"]`);
        expect(input, `Missing input ${label}`).toBeDefined();
        await act(async () => {
            Object.getOwnPropertyDescriptor(dom.HTMLInputElement.prototype, "value")!.set!.call(input, String(value));
            input!.dispatchEvent(new dom.Event("input", { bubbles: true }));
        });
    }
    async function upload(size?: number) {
        const input = host.querySelector<HTMLInputElement>('input[type="file"]')!;
        const file = new dom.File(["video"], "replacement.mp4", { type: "video/mp4" });
        if (size) Object.defineProperty(file, "size", { value: size });
        Object.defineProperty(input, "files", { configurable: true, value: [file] });
        await act(async () => input.dispatchEvent(new dom.Event("change", { bubbles: true })));
        await flush();
        expect(resolveUpload).toBeDefined();
    }

    beforeEach(async () => {
        record = {
            id: "record-one",
            revision: 1,
            title: "测试原片",
            sourceResourceId: "original-source",
            sourceName: "original.mp4",
            sourceWidth: 1920,
            sourceHeight: 1080,
            sourceDuration: 30,
            selectedBranch: "fixed-r16",
            createdAt: "2026-01-01",
            updatedAt: "2026-01-01",
            branches: {
                "fixed-r16": { settings: { target: "top-right", startSeconds: 0, durationSeconds: 0 }, status: "succeeded", progress: 100, outputResourceId: "fixed-result" },
                "moving-v6": { settings: { referenceSeconds: 0, startSeconds: 10, durationSeconds: 10 }, status: "succeeded", progress: 100, outputResourceId: "moving-result" },
            },
        };
        writes.length = 0;
        rejectPatch = false;
        patchGate = undefined;
        resolveUpload = undefined;
        uploadMeta = undefined;
        uploadSignal = undefined;
        useUserStore.setState({ user: { id: "account-a" } as NonNullable<ReturnType<typeof useUserStore.getState>["user"]> });
        apiClient.defaults.adapter = async (config) => {
            if (config.method === "get") return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, msg: "ok", data: config.url?.endsWith("/capabilities") ? capabilities : { record: structuredClone(record) } } };
            writes.push(config);
            const input = config.data ? JSON.parse(config.data) : {};
            if (config.method === "patch") {
                await patchGate;
                if (rejectPatch) return { config, status: 409, statusText: "Conflict", headers: {}, data: { code: 409, msg: "revision changed", reason: "video_transfer_conflict", data: {} } };
                if (input.sourceResourceId) {
                    record = {
                        ...record,
                        sourceResourceId: input.sourceResourceId,
                        sourceName: input.sourceName,
                        branches: {
                            "fixed-r16": { settings: { target: "top-right", startSeconds: 0, durationSeconds: 0, workers: 10 }, status: "idle", progress: 0 },
                            "moving-v6": { settings: { referenceSeconds: 0, startSeconds: 0, durationSeconds: 0, workers: 10 }, status: "idle", progress: 0 },
                        },
                    };
                }
                record = {
                    ...record,
                    title: input.title || record.title,
                    selectedBranch: input.selectedBranch || record.selectedBranch,
                    revision: record.revision + 1,
                    branches: {
                        "fixed-r16": { ...record.branches["fixed-r16"], settings: input.branches?.["fixed-r16"]?.settings || record.branches["fixed-r16"].settings },
                        "moving-v6": { ...record.branches["moving-v6"], settings: input.branches?.["moving-v6"]?.settings || record.branches["moving-v6"].settings },
                    },
                };
            } else if (config.url?.endsWith("/run"))
                record = { ...record, revision: record.revision + 1, branches: { ...record.branches, [input.branch]: { ...record.branches[input.branch as keyof typeof record.branches], status: "queued", progress: 0 } } };
            return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, msg: "ok", data: { record: structuredClone(record) } } };
        };
        client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 } } });
        client.setQueryData(videoTransferKeys.record("account-a", record.id), structuredClone(record));
        client.setQueryData(videoTransferKeys.capabilities("account-a"), capabilities);
        host = document.createElement("div");
        document.body.append(host);
        root = createRoot(host);
        await act(async () =>
            root!.render(
                el(
                    QueryClientProvider,
                    { client },
                    el(
                        MemoryRouter,
                        { initialEntries: ["/video-transfer/record-one"] },
                        el(Routes, null, el(Route, { path: "/video-transfer/:id", element: el(VideoTransferCanvasPage) }), el(Route, { path: "/video-transfer", element: el("p", null, "搬运列表路由") })),
                    ),
                ),
            ),
        );
        await flush();
    });
    afterEach(async () => {
        if (root) await act(async () => root!.unmount());
        root = undefined;
        client.clear();
        host.remove();
    });
    afterAll(() => {
        mock.restore();
        void dom.happyDOM.abort();
    });

    test("the header back icon navigates directly to the list from a loaded record", async () => {
        const icon = host.querySelector<SVGElement>('button[aria-label="返回搬运列表"] svg path');
        expect(icon).not.toBeNull();
        await act(async () => icon!.dispatchEvent(new dom.MouseEvent("click", { bubbles: true })));
        await flush();
        expect(host.textContent).toBe("搬运列表路由");
        expect(writes).toHaveLength(0);
    });

    test("select moving, edit ROI and reference, then await save before run with the returned revision", async () => {
        await click("移动水印 V6", ".video-transfer-inspector button");
        for (const [label, value] of [
            ["参考时刻", 3],
            ["左侧 X像素", 100],
            ["顶部 Y像素", 80],
            ["宽度像素", 120],
            ["高度像素", 40],
        ] as const)
            await number(label, value);
        let finish!: () => void;
        patchGate = new Promise<void>((resolve) => {
            finish = resolve;
        });
        await click("保存并开始处理");
        expect(writes).toHaveLength(1);
        expect(writes[0].method).toBe("patch");
        const patch = JSON.parse(writes[0].data);
        expect(patch).toMatchObject({
            selectedBranch: "moving-v6",
            expectedRevision: 1,
            branches: { "moving-v6": { settings: { referenceSeconds: 3, roi: { x: 100, y: 80, width: 120, height: 40 } } }, "fixed-r16": { settings: { target: "top-right" } } },
        });
        await act(async () => finish());
        await flush();
        expect(writes).toHaveLength(2);
        expect(writes[1].url).toBe("/video-transfers/record-one/run");
        expect(JSON.parse(writes[1].data)).toEqual({ branch: "moving-v6", expectedRevision: 2 });
        expect(host.textContent).toContain("取消处理任务");
    });

    test("a version conflict retains the edited draft and does not submit a run", async () => {
        await number("开始时间", 2);
        rejectPatch = true;
        await click("保存并开始处理");
        expect(writes).toHaveLength(1);
        expect(host.textContent).toContain("当前修改已保留");
        expect(host.querySelector<HTMLInputElement>('input[aria-label="开始时间"]')!.value).toBe("2.000");
        expect(host.textContent).toContain("载入服务器版本");
    });

    test("fixed regions share one frame editor and save distinct absolute windows before running", async () => {
        expect(host.querySelector('[aria-label="固定水印位置"]')).not.toBeNull();
        await click("精确区域");
        for (const [label, value] of [
            ["区域左侧 X像素", 10],
            ["区域顶部 Y像素", 20],
            ["区域宽度像素", 120],
            ["区域高度像素", 40],
            ["区域参考时刻", 1.117],
        ] as const)
            await number(label, value);
        const referenceVideo = host.querySelector<HTMLVideoElement>(".video-transfer-roi video")!;
        Object.defineProperty(referenceVideo, "readyState", { configurable: true, value: 2 });
        referenceVideo.currentTime = 1.117;
        await number("区域参考时刻", 1.122);
        expect(referenceVideo.currentTime).toBe(1.122);
        await click("添加区域");
        for (const [label, value] of [
            ["区域左侧 X像素", 1500],
            ["区域顶部 Y像素", 20],
            ["区域宽度像素", 300],
            ["区域高度像素", 80],
            ["区域参考时刻", 2.117],
            ["区域开始时间", 1.017],
            ["区域有效时长", 8.133],
        ] as const)
            await number(label, value);
        expect(host.querySelectorAll(".video-transfer-roi video")).toHaveLength(1);
        expect(host.querySelectorAll(".video-transfer-algorithm")).toHaveLength(1);
        await click("区域 1");
        expect(host.querySelector<HTMLInputElement>('input[aria-label="区域参考时刻"]')!.value).toBe("1.122");
        expect(host.querySelector<HTMLInputElement>('input[aria-label="区域左侧 X像素"]')!.value).toBe("10");
        await click("保存并开始处理");
        expect(writes).toHaveLength(2);
        const patch = JSON.parse(writes[0].data);
        expect(patch.branches["fixed-r16"].settings.fixedRegions).toEqual([
            { roi: { x: 10, y: 20, width: 120, height: 40 }, referenceSeconds: 1.122, startSeconds: 0, durationSeconds: 0 },
            { roi: { x: 1500, y: 20, width: 300, height: 80 }, referenceSeconds: 2.117, startSeconds: 1.017, durationSeconds: 8.133 },
        ]);
        expect(patch.branches["moving-v6"].settings).not.toHaveProperty("fixedRegions");
        expect(JSON.parse(writes[1].data)).toEqual({ branch: "fixed-r16", expectedRevision: 2 });
        expect(host.querySelector<HTMLInputElement>('input[aria-label="区域宽度像素"]')!.disabled).toBe(true);
    });

    test("fixed region editing limits the region count and clears explicitly back to corner mode", async () => {
        await click("精确区域");
        await number("区域宽度像素", 120);
        await number("区域高度像素", 40);
        await click("添加区域");
        await click("添加区域");
        await click("添加区域");
        const add = [...host.querySelectorAll<HTMLButtonElement>("button")].find((button) => button.textContent?.trim() === "添加区域")!;
        expect(add.disabled).toBe(true);
        expect(host.querySelectorAll(".video-transfer-region-list button[aria-pressed]")).toHaveLength(4);
        await click("删除区域");
        expect(host.querySelectorAll(".video-transfer-region-list button[aria-pressed]")).toHaveLength(3);
        await click("区域 1");
        expect(host.querySelector<HTMLInputElement>('input[aria-label="区域宽度像素"]')!.value).toBe("120");
        await click("自动角落");
        expect(host.querySelector('[aria-label="固定水印位置"]')).not.toBeNull();
        expect(host.querySelectorAll(".video-transfer-roi video")).toHaveLength(0);
        await click("保存设置");
        expect(JSON.parse(writes[0].data).branches["fixed-r16"].settings.fixedRegions).toEqual([]);
        await click("移动水印 V6", ".video-transfer-inspector button");
        expect(host.querySelector('[aria-label="参考时刻"]')).not.toBeNull();
        expect(host.querySelector('[aria-label="固定水印定位方式"]')).toBeNull();
    });

    test("branch changes retain the original source and expose each branch's own result", async () => {
        await act(async () => {
            client.setQueryData(videoTransferKeys.record("account-a", record.id), {
                ...record,
                branches: {
                    ...record.branches,
                    "moving-v6": {
                        ...record.branches["moving-v6"],
                        summary: { frames: 30, trackedFrames: 24, untrackedFrames: 6, trackingCoverage: 0.8, limitations: ["Inversion is disabled per pixel when held-out calibration cannot establish its gain.", "请检查输出中的文字细节"] },
                    },
                },
            });
        });
        await flush();
        const videos = [...host.querySelectorAll<HTMLVideoElement>(".video-transfer-graph video")].map((video) => video.getAttribute("src"));
        expect(videos).toEqual(["/api/resources/original-source/file", "/api/resources/fixed-result/file"]);
        await click("移动水印 V6", ".video-transfer-inspector button");
        expect(host.querySelectorAll(".video-transfer-algorithm")).toHaveLength(1);
        expect(host.querySelector(".video-transfer-algorithm")?.textContent).toContain("移动水印 V6");
        expect(host.querySelector(".video-transfer-algorithm")?.textContent).not.toContain("固定水印 R16");
        expect([...host.querySelectorAll<HTMLVideoElement>(".video-transfer-graph video")].map((video) => video.getAttribute("src"))).toEqual(["/api/resources/original-source/file", "/api/resources/moving-result/file"]);
        await click("预览与下载结果");
        await flush();
        expect(document.querySelector<HTMLVideoElement>(".video-transfer-preview-video")?.getAttribute("src")).toBe("/api/resources/moving-result/file");
        expect(document.body.textContent).toContain("跟踪覆盖 24/30 帧（80.0%）");
        expect(document.body.textContent).toContain("有 6 帧未跟踪到水印，已保留原画面");
        expect(document.body.textContent).toContain("无法可靠估计透明度的区域不会进行反算");
        expect(document.body.textContent).toContain("请检查输出中的文字细节");
        expect(document.body.textContent).not.toContain("held-out");
        expect(writes).toHaveLength(0);
    });

    test("replacing the source omits old branch settings and accepts the server reset", async () => {
        await click("移动水印 V6", ".video-transfer-inspector button");
        for (const [label, value] of [
            ["参考时刻", 3],
            ["左侧 X像素", 100],
            ["顶部 Y像素", 80],
            ["宽度像素", 120],
            ["高度像素", 40],
            ["开始时间", 8],
        ] as const)
            await number(label, value);
        await upload();
        await act(async () => resolveUpload!({ id: "replacement-resource" } as RemoteResource));
        await flush();
        expect(writes).toHaveLength(1);
        expect(JSON.parse(writes[0].data)).toEqual({ expectedRevision: 1, title: "测试原片", selectedBranch: "moving-v6", sourceResourceId: "replacement-resource", sourceName: "replacement.mp4" });
        expect(host.querySelector<HTMLInputElement>('input[aria-label="开始时间"]')!.value).toBe("0.000");
        expect(host.querySelector<HTMLInputElement>('input[aria-label="宽度像素"]')!.value).toBe("");
        expect([...host.querySelectorAll<HTMLVideoElement>(".video-transfer-graph video")].map((video) => video.getAttribute("src"))).toEqual(["/api/resources/replacement-resource/file"]);
    });

    test("a late upload completion after unmount cannot attach its resource", async () => {
        await upload();
        await act(async () => root!.unmount());
        root = undefined;
        expect(uploadSignal?.aborted).toBe(true);
        await act(async () => resolveUpload!({ id: "late-resource" } as RemoteResource));
        await flush();
        expect(writes).toHaveLength(0);
    });

    test("a late upload completion after changing accounts cannot patch either account", async () => {
        await upload();
        await act(async () => useUserStore.setState({ user: { id: "account-b" } as NonNullable<ReturnType<typeof useUserStore.getState>["user"]> }));
        await act(async () => resolveUpload!({ id: "late-resource" } as RemoteResource));
        await flush();
        expect(writes).toHaveLength(0);
    });

    test("large uploads use the video transfer upload purpose without a browser size limit", async () => {
        await upload(3 * 1024 ** 3);
        expect(uploadMeta?.purpose).toBe("video-transfer");
        expect(uploadSignal?.aborted).toBe(false);
        expect(host.textContent).not.toContain("视频文件超过");
        await act(async () => resolveUpload!({ id: "large-resource" } as RemoteResource));
        await flush();
        expect(JSON.parse(writes[0].data).sourceResourceId).toBe("large-resource");
    });

    test("one task shows aggregate progress and waits for the complete video before preview or download", async () => {
        await act(async () =>
            client.setQueryData(videoTransferKeys.record("account-a", record.id), {
                ...record,
                sourceDuration: 21.3,
                branches: {
                    ...record.branches,
                    "fixed-r16": {
                        ...record.branches["fixed-r16"],
                        outputResourceId: undefined,
                        status: "running",
                        progress: 40,
                        stage: "processing_segments",
                        segments: [
                            { id: "segment-1", index: 0, startSeconds: 0, durationSeconds: 10, status: "succeeded", progress: 100, outputResourceId: "first-output" },
                            { id: "segment-2", index: 1, startSeconds: 10, durationSeconds: 10, status: "running", progress: 25, stage: "repairing" },
                            { id: "segment-3", index: 2, startSeconds: 20, durationSeconds: 1.3, status: "queued", progress: 0 },
                        ],
                    },
                },
            }),
        );
        await flush();
        expect(host.querySelectorAll(".video-transfer-node")).toHaveLength(2);
        expect(host.querySelectorAll(".video-transfer-algorithm")).toHaveLength(1);
        expect(host.querySelector(".video-transfer-algorithm")?.textContent).toContain("40%");
        expect(host.querySelector(".video-transfer-algorithm")?.textContent).toContain("正在处理视频");
        expect(host.querySelector(".video-transfer-algorithm")?.textContent).not.toContain("25%");
        expect(host.querySelectorAll(".video-transfer-segment")).toHaveLength(0);
        expect(host.textContent).not.toMatch(/片段|分段|每段|下一页|上一页/);
        expect(host.querySelector(".video-transfer-result-video")).toBeNull();
        expect(host.textContent).not.toContain("预览与下载");
        const current = client.getQueryData<VideoTransferRecord>(videoTransferKeys.record("account-a", record.id))!;
        await act(async () =>
            client.setQueryData(videoTransferKeys.record("account-a", record.id), {
                ...current,
                branches: {
                    ...current.branches,
                    "fixed-r16": {
                        ...current.branches["fixed-r16"],
                        progress: 95,
                        stage: "正在生成完整视频",
                        outputResourceId: "stale-output",
                        segments: current.branches["fixed-r16"].segments!.map((segment) => ({ ...segment, status: "succeeded", progress: 100 })),
                    },
                },
            }),
        );
        await flush();
        expect(host.querySelector(".video-transfer-algorithm")?.textContent).toContain("95%");
        expect(host.querySelector(".video-transfer-algorithm")?.textContent).toContain("正在生成完整视频");
        expect(host.querySelector(".video-transfer-result-video")).toBeNull();
        expect(host.textContent).not.toContain("预览与下载");
        await act(async () =>
            client.setQueryData(videoTransferKeys.record("account-a", record.id), {
                ...current,
                branches: { ...current.branches, "fixed-r16": { ...current.branches["fixed-r16"], status: "succeeded", progress: 100, stage: "处理完成", outputResourceId: "complete-video" } },
            }),
        );
        await flush();
        expect(host.querySelector<HTMLVideoElement>(".video-transfer-result-video")?.getAttribute("src")).toBe("/api/resources/complete-video/file");
        await click("预览与下载完整视频");
        expect(document.querySelector<HTMLVideoElement>(".video-transfer-preview-video")?.getAttribute("src")).toBe("/api/resources/complete-video/file");
        expect(document.querySelector<HTMLAnchorElement>("a[download]")?.getAttribute("href")).toBe("/api/resources/complete-video/file");
        expect(document.body.innerHTML).not.toContain("first-output/file");
    });

    test("failed, cancelled and missing-output tasks never expose internal outputs", async () => {
        const failureReason = "固定水印标定失败：所选角落未找到可靠水印，请检查水印位置。";
        const segments = Array.from({ length: 8 }, (_, index) => ({
            id: `segment-${index + 1}`,
            index,
            startSeconds: index * 10,
            durationSeconds: 10,
            status: "succeeded" as const,
            progress: 100,
            outputResourceId: `internal-output-${index + 1}`,
        }));
        for (const status of ["failed", "cancelled"] as const) {
            await act(async () =>
                client.setQueryData(videoTransferKeys.record("account-a", record.id), {
                    ...record,
                    branches: { ...record.branches, "fixed-r16": { ...record.branches["fixed-r16"], status, progress: 95, error: status === "failed" ? failureReason : undefined, summary: { frames: 9600 }, segments } },
                }),
            );
            await flush();
            const node = host.querySelector(".video-transfer-algorithm")!;
            expect(host.querySelectorAll(".video-transfer-node")).toHaveLength(2);
            expect(node.textContent).toContain(status === "failed" ? failureReason : "处理已取消");
            expect(host.textContent).toContain("当前已处理 9600 帧，完整视频尚未生成。");
            if (status === "failed") {
                expect(host.textContent!.split(failureReason)).toHaveLength(2);
                expect(node.querySelector(".video-transfer-result-empty")?.textContent).toContain("请根据失败原因调整配置或排查服务，再重新处理。");
                expect(node.querySelector(".video-transfer-result-empty")?.textContent).not.toContain("在配置中选择算法");
            }
            expect(node.querySelector("video")).toBeNull();
            expect(host.textContent).not.toContain("预览与下载");
            expect(host.textContent).not.toMatch(/片段|分段|每段|下一页|上一页/);
            expect(host.innerHTML).not.toContain("internal-output-");
        }
        await act(async () => client.setQueryData(videoTransferKeys.record("account-a", record.id), { ...record, branches: { ...record.branches, "fixed-r16": { ...record.branches["fixed-r16"], outputResourceId: undefined, segments } } }));
        await flush();
        expect(host.querySelector(".video-transfer-algorithm")?.textContent).toContain("完整视频尚未就绪");
        expect(host.querySelector(".video-transfer-algorithm")?.textContent).not.toContain("等待开始处理");
        expect(host.textContent).not.toContain("预览与下载");
        await click("移动水印 V6", ".video-transfer-inspector button");
        await act(async () =>
            client.setQueryData(videoTransferKeys.record("account-a", record.id), {
                ...record,
                branches: { ...record.branches, "moving-v6": { ...record.branches["moving-v6"], status: "failed", outputResourceId: undefined, summary: { frames: 30, trackedFrames: 24, untrackedFrames: 6, trackingCoverage: 0.8 } } },
            }),
        );
        await flush();
        expect(host.querySelector(".video-transfer-result-empty")?.textContent).toContain("本次处理未完成，未生成完整视频。");
        expect(host.textContent).toContain("已处理范围内的跟踪覆盖 24/30 帧（80.0%）");
        expect(host.textContent).toContain("上述处理范围内有 6 帧未跟踪到水印");
        expect(host.textContent).not.toContain("请预览检查效果");
    });
}
