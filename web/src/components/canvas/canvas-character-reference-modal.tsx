import { App, Button, Input } from "antd";
import { useEffect, useMemo, useRef, useState, type FormEvent } from "react";
import { useTranslation } from "react-i18next";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { AppModal } from "@/components/ui/product/app-modal";
import { CharacterImageLibrary, type CanvasMediaTitles } from "@/components/canvas/character-media-library";
import { useCharacterCoverSrc } from "@/components/canvas/canvas-character-reference-node";
import { resourceFileUrl, resourceIdFromStorageKey, uploadResourceFile } from "@/services/api/resources";
import { getCharacter, replaceCharacterRepresentations, updateCharacter, type ProjectCharacterDetail } from "@/services/api/projects";
import type { CanvasNodeData } from "@/types/canvas";

const IMAGE_FIELDS = [
    { role: "primary", label: "primaryLabel", hint: "primaryHint" },
    { role: "turnaround_sheet", label: "turnaroundLabel", hint: "turnaroundHint" },
    { role: "expression_sheet", label: "expressionLabel", hint: "expressionHint" },
] as const;
type ImageRole = (typeof IMAGE_FIELDS)[number]["role"];
type Drafts = { name: string; personality: string; images: Record<ImageRole, string> };

function draftsFrom(detail: ProjectCharacterDetail): Drafts {
    const personality = detail.character.definition.personality;
    return {
        name: detail.asset.title,
        personality: typeof personality === "string" ? personality : "",
        images: {
            primary: detail.character.representations.find((item) => item.role === "primary")?.resourceId || "",
            turnaround_sheet: detail.character.representations.find((item) => item.role === "turnaround_sheet")?.resourceId || "",
            expression_sheet: detail.character.representations.find((item) => item.role === "expression_sheet")?.resourceId || "",
        },
    };
}

function sameDrafts(left: Drafts, right: Drafts) {
    return left.name === right.name && left.personality === right.personality && IMAGE_FIELDS.every(({ role }) => left.images[role] === right.images[role]);
}

export function CanvasCharacterReferenceModal({ node, canvasNodes, open, onClose, onUpdated }: { node: CanvasNodeData | null; canvasNodes?: CanvasNodeData[]; open: boolean; onClose: () => void; onUpdated?: (detail: ProjectCharacterDetail) => void }) {
    const { t } = useTranslation();
    const [busy, setBusy] = useState(false);
    const canvasTitles = useMemo<CanvasMediaTitles>(() => {
        const titles = new Map<string, string>();
        for (const item of canvasNodes || []) {
            const resourceId = resourceIdFromStorageKey(item.metadata?.storageKey);
            const title = item.title?.trim();
            if (resourceId && title && !titles.has(resourceId)) titles.set(resourceId, title);
        }
        return titles;
    }, [canvasNodes]);
    return (
        <AppModal
            open={open && Boolean(node)}
            title={null}
            aria-label={t("canvasA.characterStudio.title")}
            footer={null}
            destroyOnHidden
            centered
            width="min(1000px, calc(100vw - 32px))"
            onCancel={onClose}
            closable={!busy}
            keyboard={!busy}
            mask={{ closable: !busy }}
            flush
        >
            {node?.metadata?.characterAssetId ? <CharacterStudio key={node.metadata.characterAssetId} assetId={node.metadata.characterAssetId} canvasTitles={canvasTitles} onUpdated={onUpdated} onClose={onClose} onBusyChange={setBusy} /> : null}
        </AppModal>
    );
}

function CharacterStudio({ assetId, canvasTitles, onUpdated, onClose, onBusyChange }: { assetId: string; canvasTitles: CanvasMediaTitles; onUpdated?: (detail: ProjectCharacterDetail) => void; onClose: () => void; onBusyChange: (busy: boolean) => void }) {
    const { t } = useTranslation();
    const detailQuery = useQuery({ queryKey: ["character-studio", assetId], queryFn: () => getCharacter(assetId), refetchOnWindowFocus: false });
    return (
        <section className="flex max-h-[calc(100dvh-48px)] flex-col bg-background text-foreground" data-canvas-no-zoom data-canvas-wheel-scroll>
            <header className="shrink-0 px-5 pb-4 pr-14 pt-6 md:px-7 md:pr-14">
                <h2 className="text-[var(--fs-heading)] font-semibold">{t("canvasA.characterStudio.title")}</h2>
                <p className="mt-1 text-sm text-foreground/50">{t("canvasA.characterStudio.description")}</p>
            </header>
            {detailQuery.data ? (
                <CharacterForm detail={detailQuery.data} canvasTitles={canvasTitles} onUpdated={onUpdated} onClose={onClose} onBusyChange={onBusyChange} />
            ) : (
                <div className="space-y-4 px-7 py-16 text-center text-sm text-foreground/55" role="status">
                    <p>{t(detailQuery.isError ? "canvasA.characterStudio.loadFailed" : "canvasA.characterStudio.loading")}</p>
                    {detailQuery.isError ? <Button onClick={() => void detailQuery.refetch()}>{t("canvasA.characterStudio.retry")}</Button> : null}
                </div>
            )}
        </section>
    );
}

function CharacterForm({
    detail,
    canvasTitles,
    onUpdated,
    onClose,
    onBusyChange,
}: {
    detail: ProjectCharacterDetail;
    canvasTitles: CanvasMediaTitles;
    onUpdated?: (detail: ProjectCharacterDetail) => void;
    onClose: () => void;
    onBusyChange: (busy: boolean) => void;
}) {
    const { t } = useTranslation();
    const { message } = App.useApp();
    const queryClient = useQueryClient();
    const assetId = detail.asset.id;
    const baseline = useMemo(() => draftsFrom(detail), [detail]);
    const previousBaseline = useRef(baseline);
    const [drafts, setDrafts] = useState(baseline);
    const [pickerRole, setPickerRole] = useState<ImageRole | null>(null);
    const [uploading, setUploading] = useState<ImageRole | null>(null);
    const [saving, setSaving] = useState(false);
    const operationInFlight = useRef(false);
    const mounted = useRef(true);
    const pickerRef = useRef<HTMLDivElement>(null);
    const busy = saving || Boolean(uploading);
    const dirty = !sameDrafts(drafts, baseline);

    // 后台刷新仅同步未编辑的表单，保存部分成功时也保留剩余草稿供重试。
    useEffect(() => {
        const previous = previousBaseline.current;
        previousBaseline.current = baseline;
        setDrafts((current) => (sameDrafts(current, previous) ? baseline : current));
    }, [baseline]);
    useEffect(() => {
        onBusyChange(busy);
    }, [busy, onBusyChange]);
    useEffect(() => {
        mounted.current = true;
        return () => {
            mounted.current = false;
            onBusyChange(false);
        };
    }, [onBusyChange]);
    useEffect(() => {
        if (pickerRole) pickerRef.current?.scrollIntoView({ block: "nearest" });
    }, [pickerRole]);

    const pickImage = (role: ImageRole, resourceId: string) => {
        setDrafts((current) => ({ ...current, images: { ...current.images, [role]: resourceId } }));
        setPickerRole(null);
    };
    const uploadImage = async (role: ImageRole, file: File) => {
        if (operationInFlight.current) return;
        if (!file.type.startsWith("image/")) {
            message.error(t("canvasA.characterStudio.invalidImage"));
            return;
        }
        operationInFlight.current = true;
        setUploading(role);
        try {
            const resource = await uploadResourceFile(file, "image", { fileName: file.name });
            if (mounted.current) pickImage(role, resource.id);
        } catch (error) {
            if (mounted.current) message.error(error instanceof Error ? error.message : t("canvasA.characterStudio.uploadFailed"));
        } finally {
            operationInFlight.current = false;
            if (mounted.current) setUploading(null);
        }
    };
    const publish = (next: ProjectCharacterDetail) => {
        queryClient.setQueryData(["character-studio", assetId], next);
        void queryClient.invalidateQueries({ queryKey: ["characters"] });
        onUpdated?.(next);
    };
    const save = async (event: FormEvent) => {
        event.preventDefault();
        if (operationInFlight.current || !dirty) return;
        const name = drafts.name.trim();
        if (!name) {
            message.error(t("canvasA.characterStudio.nameRequired"));
            return;
        }
        operationInFlight.current = true;
        setSaving(true);
        let profileSaved = false;
        try {
            let next = detail;
            if (name !== baseline.name || drafts.personality.trim() !== baseline.personality) {
                next = await updateCharacter(assetId, { name, definition: { personality: drafts.personality.trim() } });
                profileSaved = true;
                publish(next);
            }
            if (IMAGE_FIELDS.some(({ role }) => drafts.images[role] !== baseline.images[role])) {
                // 图片接口整体替换，提交全部图片槽并保留其他工作流已有的单视角资源。
                const otherImages = next.character.representations.filter((item) => !IMAGE_FIELDS.some(({ role }) => role === item.role)).map(({ role, resourceId }) => ({ role, resourceId }));
                const images = IMAGE_FIELDS.flatMap(({ role }) => (drafts.images[role] ? [{ role, resourceId: drafts.images[role] }] : []));
                next = await replaceCharacterRepresentations(assetId, [...otherImages, ...images]);
                publish(next);
            }
            if (mounted.current) {
                setDrafts(draftsFrom(next));
                setPickerRole(null);
                message.success(t("canvasA.characterStudio.saved"));
            }
        } catch (error) {
            if (mounted.current) message.error(profileSaved ? t("canvasA.characterStudio.savePartial") : error instanceof Error ? error.message : t("canvasA.characterStudio.saveFailed"));
        } finally {
            operationInFlight.current = false;
            if (mounted.current) setSaving(false);
        }
    };

    const selectedField = IMAGE_FIELDS.find(({ role }) => role === pickerRole);
    return (
        <form onSubmit={(event) => void save(event)} className="flex min-h-0 flex-col">
            <div className="thin-scrollbar min-h-0 overflow-y-auto px-5 pb-6 md:px-7">
                <div className="grid grid-cols-1 gap-5 md:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
                    <label className="block">
                        <span className="mb-2 block text-sm font-medium">{t("canvasA.characterStudio.nameLabel")}</span>
                        <Input value={drafts.name} maxLength={80} disabled={saving} placeholder={t("canvasA.characterLibrary.namePlaceholder")} onChange={(event) => setDrafts((current) => ({ ...current, name: event.target.value }))} />
                    </label>
                    <label className="block">
                        <span className="mb-2 block text-sm font-medium">{t("canvasA.characterStudio.personalityLabel")}</span>
                        <Input.TextArea
                            value={drafts.personality}
                            rows={3}
                            disabled={saving}
                            placeholder={t("canvasA.characterStudio.personalityPlaceholder")}
                            onChange={(event) => setDrafts((current) => ({ ...current, personality: event.target.value }))}
                            className="!resize-none"
                        />
                    </label>
                </div>
                <div className="mt-6 grid grid-cols-1 gap-5 md:grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)]">
                    {IMAGE_FIELDS.map((field) => (
                        <CharacterImageField
                            key={field.role}
                            field={field}
                            resourceId={drafts.images[field.role]}
                            disabled={busy}
                            uploading={uploading === field.role}
                            pending={drafts.images[field.role] !== baseline.images[field.role]}
                            selected={pickerRole === field.role}
                            onUpload={(file) => void uploadImage(field.role, file)}
                            onSelect={() => setPickerRole((current) => (current === field.role ? null : field.role))}
                        />
                    ))}
                </div>
                {selectedField ? (
                    <div
                        ref={pickerRef}
                        className="mt-6 rounded-[var(--card-radius)] border border-border p-4"
                        aria-label={t("canvasA.characterStudio.selectImageFor", { label: t(`canvasA.characterStudio.${selectedField.label}`) })}
                        onKeyDown={(event) => {
                            if (event.key === "Enter" && event.target instanceof HTMLInputElement) event.preventDefault();
                        }}
                    >
                        <div className="mb-4 flex items-center justify-between gap-3">
                            <h3 className="text-sm font-medium">{t("canvasA.characterStudio.selectImageFor", { label: t(`canvasA.characterStudio.${selectedField.label}`) })}</h3>
                            <Button type="text" size="small" disabled={busy} onClick={() => setPickerRole(null)}>
                                {t("canvasA.characterStudio.closePicker")}
                            </Button>
                        </div>
                        <CharacterImageLibrary
                            currentResourceId={drafts.images[selectedField.role]}
                            busyKey={uploading ? "upload" : saving ? "save" : ""}
                            canvasTitles={canvasTitles}
                            onPick={(id) => pickImage(selectedField.role, id)}
                            onUpload={(file) => void uploadImage(selectedField.role, file)}
                        />
                    </div>
                ) : null}
            </div>
            <footer className="flex shrink-0 items-center justify-end gap-2 border-t border-border px-5 py-4 md:px-7">
                <Button disabled={busy} onClick={onClose}>
                    {t("canvasA.characterStudio.cancel")}
                </Button>
                <Button type="primary" htmlType="submit" loading={saving} disabled={!dirty || Boolean(uploading)}>
                    {t("canvasA.characterStudio.save")}
                </Button>
            </footer>
        </form>
    );
}

function CharacterImageField({
    field,
    resourceId,
    disabled,
    uploading,
    pending,
    selected,
    onUpload,
    onSelect,
}: {
    field: (typeof IMAGE_FIELDS)[number];
    resourceId: string;
    disabled: boolean;
    uploading: boolean;
    pending: boolean;
    selected: boolean;
    onUpload: (file: File) => void;
    onSelect: () => void;
}) {
    const { t } = useTranslation();
    const inputRef = useRef<HTMLInputElement>(null);
    const src = useCharacterCoverSrc(resourceId ? resourceFileUrl(resourceId) : undefined);
    const label = t(`canvasA.characterStudio.${field.label}`);
    const primary = field.role === "primary";
    return (
        <section className={`flex min-w-0 flex-col ${primary ? "md:row-span-2" : ""}`} aria-label={label}>
            <div className="mb-2 flex min-h-5 items-center justify-between gap-2">
                <h3 className="text-sm font-medium">{label}</h3>
                {pending ? <span className="text-xs text-foreground/45">{t("canvasA.characterStudio.imagePending")}</span> : null}
            </div>
            <button
                type="button"
                disabled={disabled}
                onClick={() => inputRef.current?.click()}
                aria-label={`${t(resourceId ? "canvasA.characterStudio.replaceImage" : "canvasA.characterStudio.uploadImage")}：${label}`}
                className={`relative flex w-full items-center justify-center overflow-hidden rounded-[var(--card-radius)] border border-dashed border-border bg-[var(--media-surface)] p-3 outline-none hover:border-[var(--workspace-accent)] focus-visible:ring-2 focus-visible:ring-[var(--workspace-accent)] disabled:cursor-wait ${primary ? "h-72 md:h-auto md:min-h-64 md:flex-1" : "h-28"}`}
            >
                {src ? <img src={src} alt={label} draggable={false} className="absolute inset-0 h-full w-full object-contain p-2" /> : <span className="text-center text-sm text-foreground/45">{t(`canvasA.characterStudio.${field.hint}`)}</span>}
                {uploading ? (
                    <span role="status" className="absolute inset-0 grid place-items-center bg-background/80 text-sm">
                        {t("canvasA.characterStudio.uploadPending")}
                    </span>
                ) : null}
            </button>
            <div className="mt-2 flex flex-wrap items-center gap-1">
                <Button size="small" type="text" disabled={disabled} loading={uploading} onClick={() => inputRef.current?.click()}>
                    {t(resourceId ? "canvasA.characterStudio.replaceImage" : "canvasA.characterStudio.uploadImage")}
                </Button>
                <Button size="small" type="text" disabled={disabled} aria-pressed={selected} onClick={onSelect}>
                    {t("canvasA.characterStudio.selectImage")}
                </Button>
            </div>
            <input
                ref={inputRef}
                type="file"
                accept="image/*"
                aria-label={`${t("canvasA.characterStudio.uploadImage")}：${label}`}
                className="hidden"
                disabled={disabled}
                onChange={(event) => {
                    const file = event.target.files?.[0];
                    event.target.value = "";
                    if (file) onUpload(file);
                }}
            />
        </section>
    );
}
