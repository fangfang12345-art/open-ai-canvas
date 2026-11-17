import { afterAll, afterEach, beforeEach, expect, mock, test } from "bun:test";
import { Window } from "happy-dom";
import type { InternalAxiosRequestConfig } from "axios";
import type { CanvasNodeData } from "@/types/canvas";
import type { YouTubeVideoImport } from "@/services/api/resources";

const dom = new Window({ url: "http://localhost/" });
for (const [key, value] of Object.entries({
    window: dom,
    document: dom.document,
    navigator: dom.navigator,
    localStorage: dom.localStorage,
    HTMLElement: dom.HTMLElement,
    SVGElement: dom.SVGElement,
    Element: dom.Element,
    Node: dom.Node,
    ShadowRoot: dom.ShadowRoot,
    MutationObserver: dom.MutationObserver,
    ResizeObserver: dom.ResizeObserver,
    getComputedStyle: dom.getComputedStyle.bind(dom),
    IS_REACT_ACT_ENVIRONMENT: true,
})) {
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
}
const React = await import("react");
const { act, StrictMode } = React;
const { createRoot } = await import("react-dom/client");
const antd = await import("antd");
const { QueryClient, QueryClientProvider } = await import("@tanstack/react-query");
const { setActiveUserScope } = await import("@/lib/user-scope");
const { useUserStore } = await import("@/stores/use-user-store");
const { apiClient } = await import("@/services/api/request");
const { clearResourceAccessCache, getResource, importYouTubeVideo } = await import("@/services/api/resources");
const silent = () => {};
const persisted: Array<{ canvasId: string; node: CanvasNodeData }> = [];
let persistHook: (() => Promise<void>) | undefined;
mock.module("antd", () => ({ ...antd, App: { ...antd.App, useApp: () => ({ message: { info: silent, error: silent, warning: silent, success: silent } }) } }));
mock.module("@/components/canvas/canvas-asset-tray", () => ({ CANVAS_IMAGE_ASSET_DND_TYPE: "canvas-asset" }));
mock.module("@/services/project-asset-sync", () => ({
    ensureCanvasNodeAsset: async ({ canvasId, node }: { canvasId: string; node: CanvasNodeData }) => {
        persisted.push({ canvasId, node });
        await persistHook?.();
        return { assetId: `asset:${node.id}`, syncPending: false };
    },
}));
const { useCanvasUpload } = await import("@/pages/canvas/use-canvas-upload");
const { useRemixVideoImport } = await import("@/pages/remix/use-remix-video-import");

type PendingRequest = { config: InternalAxiosRequestConfig; finish: (data: unknown, code?: number, msg?: string) => void };
const requests: PendingRequest[] = [];
const nodesRef = { current: [] as CanvasNodeData[] };
const insertedSnapshots: CanvasNodeData[][] = [];
let state: ReturnType<typeof useRemixVideoImport>;
let host: HTMLDivElement;
let root: ReturnType<typeof createRoot> | undefined;
let queryClient: InstanceType<typeof QueryClient>;

function video(id = "resource-one"): YouTubeVideoImport {
    return {
        title: "参考原片",
        videoId: "BaW_jenozKc",
        sourceUrl: "https://www.youtube.com/watch?v=BaW_jenozKc",
        resource: {
            id,
            userId: "account-a",
            kind: "video",
            status: "ready",
            provider: "local",
            endpoint: "",
            bucket: "",
            objectKey: `${id}.mp4`,
            publicUrl: "",
            mimeType: "video/mp4",
            size: 1234,
            width: 1080,
            height: 1920,
            durationMs: 10000,
            createdAt: "",
            updatedAt: "",
        },
    };
}

function Harness({ canvasId }: { canvasId: string }) {
    const upload = useCanvasUpload({
        canvasId,
        nodesRef,
        selectedNodeIdsRef: { current: new Set() },
        getCanvasCenter: () => ({ x: 500, y: 400 }),
        screenToCanvas: () => ({ x: 500, y: 400 }),
        setNodes: (action: React.SetStateAction<CanvasNodeData[]>) => {
            nodesRef.current = typeof action === "function" ? action(nodesRef.current) : action;
            insertedSnapshots.push(structuredClone(nodesRef.current));
        },
        setSelectedNodeIds: silent,
        setSelectedConnectionId: silent,
        setContextMenu: silent,
        setDialogNodeId: silent,
    });
    state = useRemixVideoImport(canvasId, upload.insertYouTubeVideo);
    return null;
}

async function render(canvasId = "remix_first") {
    await act(async () =>
        root!.render(
            <StrictMode>
                <QueryClientProvider client={queryClient}>
                    <Harness canvasId={canvasId} />
                </QueryClientProvider>
            </StrictMode>,
        ),
    );
}

async function startImport(url = "https://youtu.be/BaW_jenozKc") {
    await act(async () => state.setUrl(url));
    let task!: Promise<void>;
    await act(async () => {
        task = state.onImport();
    });
    return { task };
}

beforeEach(async () => {
    setActiveUserScope("account-a");
    useUserStore.setState({ user: { id: "account-a" } as NonNullable<ReturnType<typeof useUserStore.getState>["user"]> });
    clearResourceAccessCache();
    requests.length = 0;
    persisted.length = 0;
    persistHook = undefined;
    insertedSnapshots.length = 0;
    nodesRef.current = [];
    apiClient.defaults.adapter = (config) =>
        new Promise((resolve) => {
            requests.push({ config, finish: (data, code = 0, msg = "ok") => resolve({ config, status: 200, statusText: "OK", headers: {}, data: { code, data, msg } }) });
        });
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
    queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    await render();
});

afterEach(async () => {
    if (root) await act(async () => root!.unmount());
    root = undefined;
    queryClient.clear();
    host.remove();
    clearResourceAccessCache();
});
afterAll(() => {
    mock.restore();
    dom.happyDOM.abort();
});

test("double submit downloads once and inserts a persistent resource video with provenance", async () => {
    const { task } = await startImport("  https://youtu.be/BaW_jenozKc  ");
    await act(async () => state.onImport());
    expect(requests).toHaveLength(1);
    expect(requests[0].config.url).toBe("/resources/import-youtube");
    expect(JSON.parse(requests[0].config.data)).toEqual({ url: "https://youtu.be/BaW_jenozKc" });
    expect(requests[0].config.timeout).toBe(360000);
    expect(state.busy).toBe(true);
    const result = video();
    await act(async () => {
        requests[0].finish(result);
        await task;
    });
    expect(nodesRef.current).toHaveLength(1);
    expect(nodesRef.current[0]).toMatchObject({
        title: result.title,
        metadata: {
            content: "/api/resources/resource-one/file",
            storageKey: "resource:resource-one",
            naturalWidth: 1080,
            naturalHeight: 1920,
            durationMs: 10000,
            importSource: { provider: "youtube", videoId: result.videoId, url: result.sourceUrl },
            assetId: expect.any(String),
        },
    });
    expect(persisted).toHaveLength(1);
    expect(insertedSnapshots[0]?.[0]?.metadata?.assetId).toBe(`asset:${nodesRef.current[0].id}`);
    expect(persisted[0].canvasId).toBe("remix_first");
    expect(state.busy).toBe(false);
    expect(state.url).toBe("");
    expect(await getResource(result.resource.id)).toEqual(result.resource);
    expect(requests).toHaveLength(1);
});

test("asset registration must finish before the video enters the canvas", async () => {
    let release!: () => void;
    persistHook = () =>
        new Promise<void>((resolve) => {
            release = resolve;
        });
    const { task } = await startImport();
    await act(async () => {
        requests[0].finish(video());
    });
    expect(persisted).toHaveLength(1);
    expect(insertedSnapshots).toEqual([]);
    expect(nodesRef.current).toEqual([]);
    expect(state.busy).toBe(true);
    await act(async () => {
        release();
        await task;
    });
    expect(insertedSnapshots).toHaveLength(1);
    expect(insertedSnapshots[0]?.[0]?.metadata?.assetId).toBe(`asset:${nodesRef.current[0].id}`);
});

test("asset registration failure leaves the canvas empty and preserves an actionable error", async () => {
    persistHook = async () => {
        throw new Error("素材登记失败，请重试");
    };
    const { task } = await startImport();
    await act(async () => {
        requests[0].finish(video());
        await task;
    });
    expect(insertedSnapshots).toEqual([]);
    expect(nodesRef.current).toEqual([]);
    expect(state.busy).toBe(false);
    expect(state.error).toBe("素材登记失败，请重试");
});

test("cancelling during asset registration prevents a late canvas insertion", async () => {
    let release!: () => void;
    persistHook = () =>
        new Promise<void>((resolve) => {
            release = resolve;
        });
    const { task } = await startImport();
    await act(async () => {
        requests[0].finish(video());
    });
    expect(persisted).toHaveLength(1);
    await act(async () => state.onCancel());
    await act(async () => {
        release();
        await task;
    });
    expect(insertedSnapshots).toEqual([]);
    expect(nodesRef.current).toEqual([]);
    expect(state).toMatchObject({ busy: false, error: "" });
});

test("cancelled late response cannot insert or clear a newer import", async () => {
    const first = await startImport();
    await act(async () => state.onCancel());
    expect(requests[0].config.signal?.aborted).toBe(true);
    const second = await startImport("https://www.youtube.com/shorts/BaW_jenozKc");
    await act(async () => {
        requests[0].finish(video("cancelled"));
        await first.task;
    });
    expect(nodesRef.current).toEqual([]);
    expect(state.busy).toBe(true);
    expect(state.error).toBe("");
    await act(async () => {
        requests[1].finish(video("newer"));
        await second.task;
    });
    expect(nodesRef.current).toHaveLength(1);
    expect(nodesRef.current[0].metadata?.storageKey).toBe("resource:newer");
});

test("changing canvas aborts a late download and keeps the next canvas empty", async () => {
    const { task } = await startImport();
    await render("remix_second");
    expect(requests[0].config.signal?.aborted).toBe(true);
    await act(async () => {
        requests[0].finish(video());
        await task;
    });
    expect(nodesRef.current).toEqual([]);
    expect(persisted).toEqual([]);
    expect(state).toMatchObject({ url: "", busy: false, error: "" });
});

test("changing accounts blocks a late response before any asset is added", async () => {
    const { task } = await startImport();
    await act(async () => {
        setActiveUserScope("account-b");
        useUserStore.setState({ user: { id: "account-b" } as NonNullable<ReturnType<typeof useUserStore.getState>["user"]> });
    });
    expect(requests[0].config.signal?.aborted).toBe(true);
    await act(async () => {
        requests[0].finish(video());
        await task;
    });
    expect(nodesRef.current).toEqual([]);
    expect(persisted).toEqual([]);
    expect(state.error).toBe("");
});

test("unmount aborts the transport and late completion creates no node", async () => {
    const { task } = await startImport();
    await act(async () => root!.unmount());
    root = undefined;
    expect(requests[0].config.signal?.aborted).toBe(true);
    requests[0].finish(video());
    await task;
    expect(nodesRef.current).toEqual([]);
    expect(persisted).toEqual([]);
});

test("backend download errors remain actionable and a retry can succeed", async () => {
    const { task } = await startImport();
    await act(async () => {
        requests[0].finish(null, 400, "当前仅支持 YouTube 视频链接");
        await task;
    });
    expect(state.error).toBe("当前仅支持 YouTube 视频链接");
    expect(state.busy).toBe(false);
    expect(nodesRef.current).toEqual([]);
    const retry = await startImport();
    await act(async () => {
        requests[1].finish(video());
        await retry.task;
    });
    expect(state.error).toBe("");
    expect(nodesRef.current).toHaveLength(1);
});

test("API rejects a response from a previous account even without a React render", async () => {
    const task = importYouTubeVideo("https://youtu.be/BaW_jenozKc", new AbortController().signal);
    setActiveUserScope("account-b");
    requests[0].finish(video());
    await expect(task).rejects.toMatchObject({ name: "AbortError" });
    const fresh = getResource("resource-one");
    expect(requests).toHaveLength(2);
    requests[1].finish({ resource: { ...video().resource, userId: "account-b" } });
    expect((await fresh).userId).toBe("account-b");
});

test("clearing a same-account session prevents old import results from repopulating the cache", async () => {
    const task = importYouTubeVideo("https://youtu.be/BaW_jenozKc", new AbortController().signal);
    clearResourceAccessCache();
    requests[0].finish(video());
    await expect(task).rejects.toMatchObject({ name: "AbortError" });
    expect(nodesRef.current).toEqual([]);
});
