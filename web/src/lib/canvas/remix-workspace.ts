import { nanoid } from "nanoid";
import { getActiveUserScope, scopedLocalStorage } from "@/lib/user-scope";
import { listRemoteCanvasProjects } from "@/services/api/user-data";
import { useCanvasStore } from "@/stores/canvas/use-canvas-store";
import { useUserStore } from "@/stores/use-user-store";

const REMIX_CANVAS_PREFIX = "remix_";
const LAST_REMIX_CANVAS_KEY = "yingce:last-remix-canvas";
const pendingEntrances = new Map<string, Promise<string>>();

export function isRemixCanvas(id?: string | null) {
    return Boolean(id?.startsWith(REMIX_CANVAS_PREFIX) && id.length > REMIX_CANVAS_PREFIX.length);
}

export function remixCanvasPath(id: string) {
    return `/remix/${encodeURIComponent(id)}`;
}

function lastRemixCanvas() {
    try {
        const id = scopedLocalStorage.getItem(LAST_REMIX_CANVAS_KEY);
        return isRemixCanvas(id) ? id : null;
    } catch {
        return null;
    }
}

export function rememberRemixCanvas(id: string) {
    if (!isRemixCanvas(id)) return;
    try {
        scopedLocalStorage.setItem(LAST_REMIX_CANVAS_KEY, id);
    } catch {
        // Storage restrictions should not prevent editing the current canvas.
    }
}

export function forgetRemixCanvas(id?: string) {
    if (id && lastRemixCanvas() !== id) return;
    try {
        scopedLocalStorage.removeItem(LAST_REMIX_CANVAS_KEY);
    } catch {
        // A missing canvas is also discarded by the entrance lookup.
    }
}

export function createRemixDraft() {
    const store = useCanvasStore.getState();
    const scope = getActiveUserScope();
    const count = store.projects.filter((project) => isRemixCanvas(project.id)).length + Object.values(store.drafts).filter((draft) => draft.scope === scope && isRemixCanvas(draft.project.id)).length;
    const id = store.createDraftProject(`二创短视频 ${count + 1}`, `${REMIX_CANVAS_PREFIX}${nanoid()}`);
    rememberRemixCanvas(id);
    return id;
}

async function resolveEntrance(scope: string, userId?: string) {
    const ensureCurrentScope = () => {
        if (getActiveUserScope() !== scope || useUserStore.getState().user?.id !== userId) throw new Error("账号已切换，请重新打开二创短视频");
    };
    const lastId = lastRemixCanvas();
    const store = useCanvasStore.getState();
    if (lastId && store.openProject(lastId)) return lastId;

    const localProjects = [
        ...store.projects,
        ...Object.values(store.drafts)
            .filter((draft) => draft.scope === scope)
            .map((draft) => draft.project),
    ].filter((project) => isRemixCanvas(project.id));
    let candidates: { id: string; updatedAt: string }[] = localProjects;
    // The summary endpoint contains IDs and dates only. It can recover the last
    // workspace on another browser without downloading every canvas document.
    if (userId && (lastId || !localProjects.length)) {
        const { projects } = await listRemoteCanvasProjects();
        ensureCurrentScope();
        candidates = [...localProjects, ...projects.filter((project) => isRemixCanvas(project.id))];
    }
    ensureCurrentScope();
    const remembered = candidates.find((project) => project.id === lastId);
    const latest = remembered || candidates.sort((left, right) => right.updatedAt.localeCompare(left.updatedAt))[0];
    if (latest) {
        rememberRemixCanvas(latest.id);
        return latest.id;
    }
    // An absent ID may have been deleted remotely. Never recreate that ID.
    forgetRemixCanvas();
    return createRemixDraft();
}

export function resolveRemixCanvas() {
    const scope = getActiveUserScope();
    const existing = pendingEntrances.get(scope);
    if (existing) return existing;
    const pending = resolveEntrance(scope, useUserStore.getState().user?.id);
    pendingEntrances.set(scope, pending);
    const clear = () => {
        if (pendingEntrances.get(scope) === pending) pendingEntrances.delete(scope);
    };
    void pending.then(clear, clear);
    return pending;
}
