import { afterAll, afterEach, beforeEach, expect, mock, test } from "bun:test";
import { Window } from "happy-dom";
import type { ReactNode } from "react";
import type { ProjectCharacterDetail } from "@/services/api/projects";
import type { CanvasNodeData } from "@/types/canvas";

const dom = new Window({ url: "http://localhost/" });
for (const [key, value] of Object.entries({
    window: dom,
    document: dom.document,
    navigator: dom.navigator,
    localStorage: dom.localStorage,
    HTMLElement: dom.HTMLElement,
    HTMLBodyElement: dom.HTMLBodyElement,
    HTMLHtmlElement: dom.HTMLHtmlElement,
    HTMLInputElement: dom.HTMLInputElement,
    HTMLTextAreaElement: dom.HTMLTextAreaElement,
    Element: dom.Element,
    Node: dom.Node,
    SVGElement: dom.SVGElement,
    ShadowRoot: dom.ShadowRoot,
    MutationObserver: dom.MutationObserver,
    ResizeObserver: dom.ResizeObserver,
    getComputedStyle: dom.getComputedStyle.bind(dom),
    requestAnimationFrame: dom.requestAnimationFrame.bind(dom),
    cancelAnimationFrame: dom.cancelAnimationFrame.bind(dom),
    IS_REACT_ACT_ENVIRONMENT: true,
}))
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });

const { act, StrictMode } = await import("react");
const { createRoot } = await import("react-dom/client");
const { QueryClient, QueryClientProvider } = await import("@tanstack/react-query");
const antd = await import("antd");
await import("@/i18n");

type RepresentationInput = { role: string; resourceId: string };
type ProfileInput = { name: string; definition: Record<string, unknown> };
type Pending<T> = { assetId: string; input: T; resolve: (detail: ProjectCharacterDetail) => void; reject: (error: Error) => void };
const profiles: Pending<ProfileInput>[] = [];
const images: Pending<RepresentationInput[]>[] = [];
const updated: ProjectCharacterDetail[] = [];
const errors: string[] = [];
const successes: string[] = [];
let server: ProjectCharacterDetail;
let host: HTMLDivElement;
let root: ReturnType<typeof createRoot>;
let client: InstanceType<typeof QueryClient>;

mock.module("antd", () => ({ ...antd, App: { ...antd.App, useApp: () => ({ message: { error: (message: string) => errors.push(message), success: (message: string) => successes.push(message) } }) } }));
// The form, React Query updates and input events stay real; unrelated modal/media I/O is isolated.
mock.module("@/components/ui/product/app-modal", () => ({ AppModal: ({ open, children }: { open: boolean; children: ReactNode }) => (open ? <div role="dialog">{children}</div> : null) }));
mock.module("@/components/canvas/canvas-character-reference-node", () => ({ useCharacterCoverSrc: (url?: string) => url || "" }));
mock.module("@/components/canvas/character-media-library", () => ({
    CharacterImageLibrary: ({ onPick, busyKey }: { onPick: (id: string) => void; busyKey: string }) => (
        <div aria-label="素材库图片">
            <input type="search" aria-label="搜索图片" />
            {["portrait-new", "views-new", "expressions-new"].map((id) => (
                <button type="button" key={id} disabled={Boolean(busyKey)} onClick={() => onPick(id)}>
                    {id}
                </button>
            ))}
        </div>
    ),
}));
mock.module("@/services/api/resources", () => ({
    resourceFileUrl: (id: string) => `/api/resources/${id}/file`,
    resourceIdFromStorageKey: (key?: string) => (key?.startsWith("resource:") ? key.slice(9) : ""),
    uploadResourceFile: async () => ({ id: "uploaded-image" }),
}));
mock.module("@/services/api/projects", () => ({
    getCharacter: async () => structuredClone(server),
    updateCharacter: (assetId: string, input: ProfileInput) => new Promise<ProjectCharacterDetail>((resolve, reject) => profiles.push({ assetId, input: structuredClone(input), resolve, reject })),
    replaceCharacterRepresentations: (assetId: string, input: RepresentationInput[]) => new Promise<ProjectCharacterDetail>((resolve, reject) => images.push({ assetId, input: structuredClone(input), resolve, reject })),
}));
const { CanvasCharacterReferenceModal } = await import("@/components/canvas/canvas-character-reference-modal");

const imageFields = ["斜前方全身图", "三视图", "表情图"] as const;
const initialImages: RepresentationInput[] = [
    { role: "primary", resourceId: "portrait-old" },
    { role: "turnaround_sheet", resourceId: "views-old" },
    { role: "expression_sheet", resourceId: "expressions-old" },
];
function fixture(representations = initialImages): ProjectCharacterDetail {
    return {
        asset: { id: "character-a", title: "林溪", category: "character", mediaType: "entity", status: "ready", versionCount: 1, usages: [], position: 0, updatedAt: "" },
        character: {
            versionId: "version-1",
            version: 1,
            definition: { personality: "沉稳", appearance: "旧外观设定", voiceTimbre: "旧声音设定" },
            visualStatus: "ready",
            voiceStatus: "missing",
            representations: representations.map((item, index) => ({ ...item, id: `representation-${index}`, mediaType: "image" })),
        },
    };
}
const node: CanvasNodeData = { id: "node-a", type: "text" as CanvasNodeData["type"], title: "林溪", position: { x: 0, y: 0 }, width: 240, height: 320, metadata: { workflowKind: "character", characterAssetId: "character-a" } };

beforeEach(() => {
    profiles.length = 0;
    images.length = 0;
    updated.length = 0;
    errors.length = 0;
    successes.length = 0;
    server = fixture();
    host = document.createElement("div");
    document.body.append(host);
    root = createRoot(host);
    client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: Infinity } } });
});
afterEach(async () => {
    await act(async () => root.unmount());
    client.clear();
    host.remove();
});
afterAll(() => {
    mock.restore();
    dom.happyDOM.abort();
});

async function settleUntil(predicate: () => boolean) {
    for (let attempt = 0; attempt < 30 && !predicate(); attempt++) await act(async () => new Promise((resolve) => setTimeout(resolve, 0)));
    expect(predicate()).toBe(true);
}
async function render() {
    await act(async () =>
        root.render(
            <StrictMode>
                <QueryClientProvider client={client}>
                    <antd.ConfigProvider theme={{ token: { motion: false } }}>
                        <CanvasCharacterReferenceModal open node={node} onClose={() => {}} onUpdated={(detail) => updated.push(detail)} />
                    </antd.ConfigProvider>
                </QueryClientProvider>
            </StrictMode>,
        ),
    );
    await settleUntil(() => Boolean(host.querySelector("textarea")));
}
function button(label: string, scope: ParentNode = host) {
    const found = [...scope.querySelectorAll<HTMLButtonElement>("button")].find((item) => item.textContent?.trim() === label || item.getAttribute("aria-label") === label);
    expect(found, `button ${label}`).toBeDefined();
    return found!;
}
async function click(label: string, scope: ParentNode = host) {
    await act(async () => button(label, scope).click());
}
function input(label: string) {
    const wrapper = [...host.querySelectorAll("label")].find((item) => item.querySelector("span")?.textContent === label);
    const found = wrapper?.querySelector<HTMLInputElement | HTMLTextAreaElement>("input, textarea");
    expect(found, `field ${label}`).toBeDefined();
    return found!;
}
async function fill(label: string, value: string) {
    const field = input(label);
    const prototype = field.tagName === "TEXTAREA" ? dom.HTMLTextAreaElement.prototype : dom.HTMLInputElement.prototype;
    await act(async () => {
        Object.getOwnPropertyDescriptor(prototype, "value")!.set!.call(field, value);
        field.dispatchEvent(new dom.Event("input", { bubbles: true }));
    });
}
function imageField(label: string) {
    const field = host.querySelector<HTMLElement>(`section[aria-label="${label}"]`);
    expect(field).not.toBeNull();
    return field!;
}
function selectedImages() {
    return imageFields.map(
        (label) =>
            imageField(label)
                .querySelector("img")
                ?.getAttribute("src")
                ?.match(/\/resources\/([^/]+)\/file/)?.[1] || "",
    );
}
async function chooseImage(label: string, resourceId: string) {
    await click("从素材库选择", imageField(label));
    await click(resourceId);
}
async function save() {
    await click("保存角色卡");
}
async function finishProfile(index = 0) {
    const request = profiles[index];
    server = { ...server, asset: { ...server.asset, title: request.input.name }, character: { ...server.character, versionId: `profile-${index}`, definition: request.input.definition } };
    await act(async () => request.resolve(structuredClone(server)));
}
async function finishImages(index = 0) {
    const request = images[index];
    server = { ...server, character: { ...server.character, versionId: `images-${index}`, representations: request.input.map((item, ordinal) => ({ ...item, id: `saved-${ordinal}`, mediaType: "image" })) } };
    await act(async () => request.resolve(structuredClone(server)));
    await settleUntil(() => button("保存角色卡").disabled);
}

test("五项信息同屏，三图独立暂存且统一保存前不提交绑定", async () => {
    server = fixture([]);
    await render();
    expect(host.querySelector("[role=tablist]")).toBeNull();
    expect(host.querySelectorAll("label")).toHaveLength(2);
    await fill("名称", "林溪新名");
    await fill("性格特征", "沉稳且敏锐");
    await click("从素材库选择", imageField("斜前方全身图"));
    const search = host.querySelector<HTMLInputElement>('input[aria-label="搜索图片"]')!;
    const enter = new dom.KeyboardEvent("keydown", { key: "Enter", bubbles: true, cancelable: true });
    await act(async () => search.dispatchEvent(enter));
    expect(enter.defaultPrevented).toBe(true);
    expect(profiles).toHaveLength(0);
    expect(images).toHaveLength(0);
    await click("收起素材库");
    for (const [label, resourceId] of imageFields.map((label, index) => [label, ["portrait-new", "views-new", "expressions-new"][index]])) await chooseImage(label, resourceId);
    expect(selectedImages()).toEqual(["portrait-new", "views-new", "expressions-new"]);
    expect(input("名称").value).toBe("林溪新名");
    expect(input("性格特征").value).toBe("沉稳且敏锐");
    expect(profiles).toHaveLength(0);
    expect(images).toHaveLength(0);
    await save();
    expect(profiles[0]).toMatchObject({ assetId: "character-a", input: { name: "林溪新名", definition: { personality: "沉稳且敏锐" } } });
    await finishProfile();
    await settleUntil(() => images.length === 1);
    expect(images[0].input).toEqual(initialImages.map((item) => ({ ...item, resourceId: item.resourceId.replace("old", "new") })));
    expect(input("名称").value).toBe("林溪新名");
    expect(input("性格特征").value).toBe("沉稳且敏锐");
    await finishImages();
    expect(updated).toHaveLength(2);
    expect(successes).toHaveLength(1);
    expect(errors).toEqual([]);
});

test("替换三视图保留另外两图和其他工作流的单视角资源", async () => {
    server = fixture([...initialImages, { role: "front", resourceId: "legacy-front" }]);
    await render();
    await chooseImage("三视图", "views-new");
    expect(selectedImages()).toEqual(["portrait-old", "views-new", "expressions-old"]);
    await save();
    expect(profiles).toHaveLength(0);
    expect(images[0].input).toEqual([
        { role: "front", resourceId: "legacy-front" },
        { role: "primary", resourceId: "portrait-old" },
        { role: "turnaround_sheet", resourceId: "views-new" },
        { role: "expression_sheet", resourceId: "expressions-old" },
    ]);
    await finishImages();
    expect(selectedImages()).toEqual(["portrait-old", "views-new", "expressions-old"]);
    expect(input("名称").value).toBe("林溪");
    expect(input("性格特征").value).toBe("沉稳");
});

test("后台资料刷新与图片选择不会覆盖未保存的名称和性格", async () => {
    await render();
    await fill("名称", "草稿名称");
    await fill("性格特征", "草稿性格");
    await act(async () => client.setQueryData(["character-studio", "character-a"], { ...server, character: { ...server.character, versionId: "remote-version" } }));
    await chooseImage("表情图", "expressions-new");
    expect(input("名称").value).toBe("草稿名称");
    expect(input("性格特征").value).toBe("草稿性格");
    await save();
    expect(profiles[0].input).toEqual({ name: "草稿名称", definition: { personality: "草稿性格" } });
    await finishProfile();
    await settleUntil(() => images.length === 1);
    expect(input("名称").value).toBe("草稿名称");
    expect(input("性格特征").value).toBe("草稿性格");
    await finishImages();
});

test("文字成功而图片失败时保留草稿，重试只重新保存图片", async () => {
    await render();
    await fill("名称", "  保存后的名字  ");
    await fill("性格特征", "  保存后的性格  ");
    await chooseImage("斜前方全身图", "portrait-new");
    await save();
    await finishProfile();
    await settleUntil(() => images.length === 1);
    await act(async () => images[0].reject(new Error("图片服务暂时不可用")));
    await settleUntil(() => errors.length === 1 && !input("名称").disabled);
    expect(errors[0]).toContain("部分信息已保存");
    expect(successes).toEqual([]);
    expect(updated).toHaveLength(1);
    expect(input("名称").value.trim()).toBe("保存后的名字");
    expect(input("性格特征").value.trim()).toBe("保存后的性格");
    expect(selectedImages()).toEqual(["portrait-new", "views-old", "expressions-old"]);
    expect(button("保存角色卡").disabled).toBe(false);
    await save();
    expect(profiles).toHaveLength(1);
    expect(images).toHaveLength(2);
    expect(images[1].input).toEqual(images[0].input);
    await finishImages(1);
    expect(input("名称").value).toBe("保存后的名字");
    expect(input("性格特征").value).toBe("保存后的性格");
    expect(successes).toHaveLength(1);
});

test("空白名称拒绝保存且不丢失性格与图片草稿", async () => {
    await render();
    await fill("名称", "   ");
    await fill("性格特征", "未提交的性格");
    await chooseImage("表情图", "expressions-new");
    await save();
    expect(errors).toEqual(["角色名称不能为空"]);
    expect(profiles).toEqual([]);
    expect(images).toEqual([]);
    expect(input("性格特征").value).toBe("未提交的性格");
    expect(selectedImages()).toEqual(["portrait-old", "views-old", "expressions-new"]);
});
