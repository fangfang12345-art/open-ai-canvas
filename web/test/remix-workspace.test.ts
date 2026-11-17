import { afterEach, beforeEach, expect, test } from "bun:test";
import { createRemixDraft, forgetRemixCanvas, isRemixCanvas, rememberRemixCanvas, resolveRemixCanvas } from "@/lib/canvas/remix-workspace";
import { setActiveUserScope, withUserScopedPersistenceSuppressed } from "@/lib/user-scope";
import { apiClient } from "@/services/api/request";
import { useCanvasStore, withCanvasStorePersistenceSuppressed } from "@/stores/canvas/use-canvas-store";
import { useUserStore } from "@/stores/use-user-store";

const originalWindow = globalThis.window;
const originalAdapter = apiClient.defaults.adapter;
const originalUser = useUserStore.getState().user;
const storage = new Map<string, string>();
let sequence = 0;

function changeUser(id?: string) {
    setActiveUserScope(id);
    withUserScopedPersistenceSuppressed(() => useUserStore.setState({ user: id ? ({ id } as NonNullable<typeof originalUser>) : null }));
    withCanvasStorePersistenceSuppressed(() => useCanvasStore.setState({ projects: [], drafts: {} }));
}

beforeEach(() => {
    storage.clear();
    Object.defineProperty(globalThis, "window", {
        configurable: true,
        value: {
            localStorage: {
                getItem: (key: string) => storage.get(key) ?? null,
                setItem: (key: string, value: string) => storage.set(key, value),
                removeItem: (key: string) => storage.delete(key),
            },
        },
    });
    changeUser(`remix-test-${++sequence}`);
    apiClient.defaults.adapter = async (config) => ({ config, status: 200, statusText: "OK", headers: {}, data: { code: 0, data: { projects: [] }, msg: "ok" } });
});

afterEach(() => {
    withCanvasStorePersistenceSuppressed(() => useCanvasStore.setState({ projects: [], drafts: {} }));
    withUserScopedPersistenceSuppressed(() => useUserStore.setState({ user: originalUser }));
    apiClient.defaults.adapter = originalAdapter;
    Object.defineProperty(globalThis, "window", { configurable: true, value: originalWindow });
});

test("concurrent entrances share one untouched draft and remember only its ID", async () => {
    const [first, second] = await Promise.all([resolveRemixCanvas(), resolveRemixCanvas()]);
    expect(first).toBe(second);
    expect(isRemixCanvas(first)).toBe(true);
    expect(Object.keys(useCanvasStore.getState().drafts)).toEqual([first]);
    expect(useCanvasStore.getState().projects).toEqual([]);
    expect([...storage.entries()].filter(([key]) => key.startsWith("yingce:last-remix-canvas"))).toEqual([[expect.any(String), first]]);
    expect(await resolveRemixCanvas()).toBe(first);
});

test("a missing pointer resumes the latest saved remix without selecting an ordinary canvas", async () => {
    const firstId = createRemixDraft();
    const secondId = createRemixDraft();
    const store = useCanvasStore.getState();
    const first = { ...store.openProject(firstId)!, updatedAt: "2026-01-01" };
    const second = { ...store.openProject(secondId)!, updatedAt: "2026-02-01" };
    const ordinary = { ...second, id: "ordinary-canvas", updatedAt: "2026-03-01" };
    withCanvasStorePersistenceSuppressed(() => useCanvasStore.setState({ projects: [first, ordinary, second], drafts: {} }));
    forgetRemixCanvas();
    expect(await resolveRemixCanvas()).toBe(secondId);
});

test("a stale pointer recovers a saved remote remix and never recreates a deleted ID", async () => {
    rememberRemixCanvas("remix_deleted");
    apiClient.defaults.adapter = async (config) => ({ config, status: 200, statusText: "OK", headers: {}, data: { code: 0, data: { projects: [{ id: "remix_saved", updatedAt: "2026-01-01" }] }, msg: "ok" } });
    expect(await resolveRemixCanvas()).toBe("remix_saved");
    expect(useCanvasStore.getState().openProject("remix_deleted")).toBeNull();

    apiClient.defaults.adapter = async (config) => ({ config, status: 200, statusText: "OK", headers: {}, data: { code: 0, data: { projects: [] }, msg: "ok" } });
    const replacement = await resolveRemixCanvas();
    expect(replacement).not.toBe("remix_saved");
    expect(isRemixCanvas(replacement)).toBe(true);
});

test("last canvas pointers are isolated by account", async () => {
    changeUser("remix-account-a");
    const first = createRemixDraft();
    const firstDrafts = useCanvasStore.getState().drafts;
    changeUser("remix-account-b");
    const second = await resolveRemixCanvas();
    expect(second).not.toBe(first);
    changeUser("remix-account-a");
    withCanvasStorePersistenceSuppressed(() => useCanvasStore.setState({ drafts: firstDrafts }));
    expect(await resolveRemixCanvas()).toBe(first);
});

test("failed lookup does not create a replacement workspace", async () => {
    apiClient.defaults.adapter = async () => {
        throw new Error("offline");
    };
    await expect(resolveRemixCanvas()).rejects.toThrow("offline");
    expect(useCanvasStore.getState().drafts).toEqual({});
});

test("an account switch during remote lookup cannot write into the new account", async () => {
    let finish: (() => void) | undefined;
    const waiting = new Promise<void>((resolve) => {
        finish = resolve;
    });
    apiClient.defaults.adapter = async (config) => {
        await waiting;
        return { config, status: 200, statusText: "OK", headers: {}, data: { code: 0, data: { projects: [{ id: "remix_old-account", updatedAt: "2026-01-01" }] }, msg: "ok" } };
    };
    const opening = resolveRemixCanvas();
    changeUser("remix-new-account");
    finish!();
    await expect(opening).rejects.toThrow("账号已切换");
    expect(useCanvasStore.getState().drafts).toEqual({});
    expect([...storage.keys()].some((key) => key === "yingce:last-remix-canvas:user:remix-new-account")).toBe(false);
});
