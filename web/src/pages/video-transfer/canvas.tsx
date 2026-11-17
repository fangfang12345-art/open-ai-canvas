import { useEffect, useRef, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { App, Button, Input, InputNumber, Progress, Spin } from "antd";
import { ArrowLeftOutlined, EditOutlined } from "@ant-design/icons";
import { useNavigate, useParams } from "react-router";
import { WorkspacePage } from "@/components/layout/workspace-page";
import { WorkspaceErrorState, WorkspaceForbiddenState } from "@/components/layout/workspace-state";
import { AppModal } from "@/components/ui/product/app-modal";
import { readVideoTransferMetadata } from "@/lib/video-transfer/media";
import { VIDEO_TRANSFER_BRANCHES, VIDEO_TRANSFER_BRANCH_LABELS, VIDEO_TRANSFER_STATUS_LABELS, formatVideoSeconds, formatVideoTransferStage, isVideoTransferActive, validateVideoTransferSettings, videoTransferKeys } from "@/lib/video-transfer/workflow";
import { ApiError } from "@/services/api/request";
import { resourceFileUrl, uploadResourceFile } from "@/services/api/resources";
import {
    cancelVideoTransfer,
    getVideoTransfer,
    getVideoTransferCapabilities,
    runVideoTransfer,
    updateVideoTransfer,
    type VideoTransferBranch,
    type VideoTransferRecord,
    type VideoTransferROI,
    type VideoTransferSettings,
} from "@/services/api/video-transfer";
import { useUserStore } from "@/stores/use-user-store";
import { WorkflowCanvas } from "./workflow-canvas";
import { ROISelector } from "./roi-selector";
import { FixedRegionSettings } from "./fixed-region-settings";
import { ResultSummary } from "./result-summary";
import "./video-transfer.css";

export default function VideoTransferCanvasPage() {
    const { id } = useParams<{ id: string }>();
    const userId = useUserStore((state) => state.user?.id);
    const query = useQuery({
        queryKey: videoTransferKeys.record(userId || "", id || ""),
        enabled: Boolean(id && userId),
        queryFn: ({ signal }) => getVideoTransfer(id!, signal),
        refetchInterval: (query) => (query.state.data && isVideoTransferActive(query.state.data) ? 1500 : false),
    });
    if (!userId)
        return (
            <WorkspacePage>
                <WorkspaceForbiddenState title="登录后打开搬运画布" />
            </WorkspacePage>
        );
    if (query.isLoading)
        return (
            <WorkspacePage>
                <div className="video-transfer-loading">
                    <Spin />
                    <span>正在读取搬运记录</span>
                </div>
            </WorkspacePage>
        );
    if (!query.data)
        return (
            <WorkspacePage>
                <WorkspaceErrorState description={query.error instanceof Error ? query.error.message : "没有找到这条搬运记录"} onRetry={() => void query.refetch()} />
            </WorkspacePage>
        );
    return <VideoTransferEditor key={`${userId}:${id}`} userId={userId} record={query.data} pollingError={query.error instanceof Error ? query.error.message : undefined} onRetry={() => void query.refetch()} />;
}

function VideoTransferEditor({ userId, record, pollingError, onRetry }: { userId: string; record: VideoTransferRecord; pollingError?: string; onRetry: () => void }) {
    const navigate = useNavigate();
    const queryClient = useQueryClient();
    const { message } = App.useApp();
    const [title, setTitle] = useState(record.title);
    const [editingTitle, setEditingTitle] = useState(false);
    const [selectedBranch, setSelectedBranch] = useState(record.selectedBranch);
    const [settings, setSettings] = useState<Record<VideoTransferBranch, VideoTransferSettings>>(() => branchSettings(record));
    const [savedDraft, setSavedDraft] = useState(() => ({ title: record.title, selectedBranch: record.selectedBranch, settings: branchSettings(record) }));
    const [baseRevision, setBaseRevision] = useState(record.revision);
    const [busy, setBusy] = useState<"upload" | "save" | "run" | "cancel" | "reload" | null>(null);
    const [uploadProgress, setUploadProgress] = useState<number | null>(null);
    const [error, setError] = useState("");
    const [conflict, setConflict] = useState(false);
    const [inspectorOpen, setInspectorOpen] = useState(true);
    const [preview, setPreview] = useState<{ resourceId: string; label: string } | null>(null);
    const fileRef = useRef<HTMLInputElement>(null);
    const requestRef = useRef<AbortController | null>(null);
    const active = isVideoTransferActive(record);
    const locked = active || busy !== null;
    const branchState = record.branches[selectedBranch];
    const outputResourceId = branchState.status === "succeeded" ? branchState.outputResourceId : undefined;
    const draft = settings[selectedBranch];
    const dirty = JSON.stringify({ title, selectedBranch, settings }) !== JSON.stringify(savedDraft);

    const capabilities = useQuery({
        queryKey: videoTransferKeys.capabilities(userId),
        queryFn: ({ signal }) => getVideoTransferCapabilities(signal),
        staleTime: 30_000,
    });
    const capability = capabilities.data?.branches.find((item) => item.id === selectedBranch);
    const limits = capabilities.data?.limits;
    const runProblem = record.sourceResourceId ? validateVideoTransferSettings(selectedBranch, draft, record, limits) : "请先上传原视频";
    useEffect(
        () => () => {
            requestRef.current?.abort();
        },
        [],
    );
    useEffect(() => {
        if (record.revision === baseRevision || dirty || busy) return;
        const nextSettings = branchSettings(record);
        setTitle(record.title);
        setSelectedBranch(record.selectedBranch);
        setSettings(nextSettings);
        setSavedDraft({ title: record.title, selectedBranch: record.selectedBranch, settings: nextSettings });
        setBaseRevision(record.revision);
    }, [record, baseRevision, dirty, busy]);

    function setDraft(patch: Partial<VideoTransferSettings>) {
        setSettings((old) => ({ ...old, [selectedBranch]: { ...old[selectedBranch], ...patch } }));
    }

    function acceptRecord(next: VideoTransferRecord) {
        queryClient.setQueryData(videoTransferKeys.record(userId, record.id), next);
        void queryClient.invalidateQueries({ queryKey: [...videoTransferKeys.all(userId), "list"] });
        const nextSettings = branchSettings(next);
        setTitle(next.title);
        setSelectedBranch(next.selectedBranch);
        setSettings(nextSettings);
        setSavedDraft({ title: next.title, selectedBranch: next.selectedBranch, settings: nextSettings });
        setBaseRevision(next.revision);
        setConflict(false);
    }

    function stillCurrent(request: AbortController) {
        return !request.signal.aborted && requestRef.current === request && useUserStore.getState().user?.id === userId;
    }

    async function perform(action: NonNullable<typeof busy>, operation: (request: AbortController) => Promise<void>) {
        if (requestRef.current) return;
        const request = new AbortController();
        requestRef.current = request;
        setBusy(action);
        setError("");
        try {
            await queryClient.cancelQueries({ queryKey: videoTransferKeys.record(userId, record.id) });
            if (!stillCurrent(request)) return;
            await operation(request);
        } catch (cause) {
            if (!stillCurrent(request)) return;
            const isConflict = cause instanceof ApiError && cause.status === 409;
            setConflict(isConflict);
            setError(isConflict ? "这条记录已在其他窗口更新。当前修改已保留，请载入服务器版本后重新编辑。" : cause instanceof Error ? cause.message : "操作失败，请重试");
            if (isConflict) void queryClient.invalidateQueries({ queryKey: videoTransferKeys.record(userId, record.id) });
        } finally {
            if (requestRef.current === request) {
                requestRef.current = null;
                setBusy(null);
                setUploadProgress(null);
            }
        }
    }

    async function save(request: AbortController) {
        if (!title.trim()) throw new Error("请填写记录名称");
        const next = await updateVideoTransfer(
            record.id,
            {
                expectedRevision: baseRevision,
                title: title.trim(),
                selectedBranch,
                branches: { "fixed-r16": { settings: settings["fixed-r16"] }, "moving-v6": { settings: settings["moving-v6"] } },
            },
            request.signal,
        );
        if (stillCurrent(request)) acceptRecord(next);
        return next;
    }

    function saveOnly() {
        void perform("save", async (request) => {
            await save(request);
            if (stillCurrent(request)) message.success("设置已保存到服务器");
        });
    }

    function start() {
        if (runProblem) {
            setError(runProblem);
            return;
        }
        if (!capability?.available) return;
        void perform("run", async (request) => {
            const saved = await save(request);
            if (!stillCurrent(request)) return;
            const next = await runVideoTransfer(record.id, selectedBranch, saved.revision, request.signal);
            if (stillCurrent(request)) {
                acceptRecord(next);
                message.success("处理任务已提交，可在画布中查看总进度");
            }
        });
    }

    function cancel() {
        void perform("cancel", async (request) => {
            const next = await cancelVideoTransfer(record.id, request.signal);
            if (stillCurrent(request)) {
                acceptRecord(next);
                message.success("已发送取消请求");
            }
        });
    }

    function upload(file: File) {
        if (locked) return;
        if (!title.trim()) {
            setError("请先填写记录名称");
            return;
        }
        if (!file.type.startsWith("video/") && !/\.(mp4|mov|m4v|webm|mkv)$/i.test(file.name)) {
            setError("请选择视频文件");
            return;
        }
        void perform("upload", async (request) => {
            setUploadProgress(0);
            const metadata = await readVideoTransferMetadata(file, request.signal).catch((cause: unknown) => {
                if (request.signal.aborted) throw cause;
                // 浏览器不支持的封装仍可交给后端 ffprobe 校验，不能用预览能力限制上传格式。
                return undefined;
            });
            if (!stillCurrent(request)) return;
            if (limits && limits.maxSourceDurationSeconds > 0 && metadata && metadata.duration > limits.maxSourceDurationSeconds) throw new Error(`原视频超过当前服务时长上限 ${limits.maxSourceDurationSeconds} 秒`);
            if (limits && metadata && metadata.width * metadata.height > limits.maxPixels) throw new Error(`原视频像素量超过当前服务上限 ${limits.maxPixels}`);
            const resource = await uploadResourceFile(
                file,
                "video",
                { purpose: "video-transfer", fileName: file.name, width: metadata?.width, height: metadata?.height, durationMs: metadata ? Math.round(metadata.duration * 1000) : undefined },
                (loaded, total) => {
                    if (stillCurrent(request) && total > 0) setUploadProgress(Math.round((loaded / total) * 100));
                },
                request.signal,
            );
            if (!stillCurrent(request)) return;
            const next = await updateVideoTransfer(
                record.id,
                {
                    expectedRevision: baseRevision,
                    title: title.trim(),
                    selectedBranch,
                    sourceResourceId: resource.id,
                    sourceName: file.name,
                },
                request.signal,
            );
            if (stillCurrent(request)) {
                acceptRecord(next);
                message.success("原视频已保存，可以配置水印并开始处理");
            }
        });
    }

    return (
        <WorkspacePage fluid scroll={false} className="video-transfer-canvas-page">
            <div className="video-transfer-editor">
                <header className="video-transfer-editor-header">
                    <Button type="text" icon={<ArrowLeftOutlined />} aria-label="返回搬运列表" onClick={() => navigate("/video-transfer")} />
                    <div className="video-transfer-editor-title">
                        {editingTitle ? (
                            <Input
                                autoFocus
                                aria-label="修改搬运记录名称"
                                value={title}
                                maxLength={120}
                                disabled={locked}
                                onChange={(event) => setTitle(event.target.value)}
                                onPressEnter={() => setEditingTitle(false)}
                                onBlur={() => setEditingTitle(false)}
                            />
                        ) : (
                            <button type="button" disabled={locked} onClick={() => setEditingTitle(true)} title="单击修改名称">
                                <strong>{title}</strong>
                                <EditOutlined />
                            </button>
                        )}
                        <span>{dirty ? "有未保存的修改" : "已保存到服务器"}</span>
                    </div>
                    <div className="video-transfer-header-actions">
                        <Button disabled={locked || !dirty} loading={busy === "save"} onClick={saveOnly}>
                            保存设置
                        </Button>
                        <Button aria-pressed={inspectorOpen} onClick={() => setInspectorOpen((open) => !open)}>
                            {inspectorOpen ? "收起配置" : "展开配置"}
                        </Button>
                    </div>
                </header>
                {pollingError ? (
                    <div className="video-transfer-banner" role="alert">
                        <span>无法刷新处理进度：{pollingError}</span>
                        <Button size="small" onClick={onRetry}>
                            重新连接
                        </Button>
                    </div>
                ) : null}
                {error ? (
                    <div className="video-transfer-banner is-error" role="alert">
                        <span>{error}</span>
                        {conflict ? (
                            <Button
                                size="small"
                                disabled={busy !== null}
                                onClick={() =>
                                    void perform("reload", async (request) => {
                                        const next = await getVideoTransfer(record.id, request.signal);
                                        if (stillCurrent(request)) {
                                            acceptRecord(next);
                                            setError("");
                                        }
                                    })
                                }
                            >
                                载入服务器版本
                            </Button>
                        ) : (
                            <Button size="small" type="text" onClick={() => setError("")}>
                                关闭
                            </Button>
                        )}
                    </div>
                ) : null}
                <div className={`video-transfer-editor-body${inspectorOpen ? " has-inspector" : ""}`}>
                    <WorkflowCanvas record={record} selectedBranch={selectedBranch} onUpload={() => fileRef.current?.click()} onPreview={(resourceId, label) => setPreview({ resourceId, label })} uploadProgress={uploadProgress} disabled={locked} />
                    {inspectorOpen ? (
                        <aside className="video-transfer-inspector" data-canvas-no-zoom data-canvas-wheel-scroll aria-label="视频处理设置">
                            <div className="video-transfer-inspector-heading">
                                <h2>处理配置</h2>
                                <span>{VIDEO_TRANSFER_BRANCH_LABELS[selectedBranch]}</span>
                            </div>
                            <div className="video-transfer-branch-switch">
                                {VIDEO_TRANSFER_BRANCHES.map((branch) => (
                                    <button key={branch} type="button" disabled={locked} aria-pressed={selectedBranch === branch} onClick={() => setSelectedBranch(branch)}>
                                        {branch === "fixed-r16" ? "固定水印 R16" : "移动水印 V6"}
                                    </button>
                                ))}
                            </div>
                            <p className="video-transfer-hint">上传不设单文件大小上限，仍占用账号存储空间。选择算法并配置水印，完成后可预览和下载完整视频。</p>
                            <fieldset disabled={locked} className="video-transfer-settings">
                                <legend>{selectedBranch === "fixed-r16" ? "水印位置" : "水印参考区域"}</legend>
                                {selectedBranch === "fixed-r16" ? (
                                    <FixedRegionSettings settings={draft} source={record} disabled={locked} onChange={setDraft} />
                                ) : (
                                    <>
                                        <label>
                                            参考时刻（原视频秒数）
                                            <InputNumber
                                                aria-label="参考时刻"
                                                disabled={locked}
                                                min={0}
                                                max={record.sourceDuration}
                                                step={0.1}
                                                precision={2}
                                                value={draft.referenceSeconds ?? 0}
                                                onChange={(value) => setDraft({ referenceSeconds: value ?? 0 })}
                                            />
                                        </label>
                                        <p className="video-transfer-hint">适合形状稳定、仅平移的标志。请紧贴水印框选，四周背景应接近均匀；暂不支持旋转、缩放或字形变化。</p>
                                        {record.sourceResourceId && record.sourceWidth && record.sourceHeight ? (
                                            <ROISelector
                                                key={record.sourceResourceId}
                                                src={resourceFileUrl(record.sourceResourceId)}
                                                width={record.sourceWidth}
                                                height={record.sourceHeight}
                                                seconds={draft.referenceSeconds ?? 0}
                                                roi={draft.roi}
                                                disabled={locked}
                                                onChange={(roi) => setDraft({ roi })}
                                            />
                                        ) : (
                                            <p className="video-transfer-hint">上传视频后，可以在参考画面中拖框选择水印。</p>
                                        )}
                                        <div className="video-transfer-roi-inputs">
                                            {(
                                                [
                                                    ["x", "左侧 X"],
                                                    ["y", "顶部 Y"],
                                                    ["width", "宽度"],
                                                    ["height", "高度"],
                                                ] as Array<[keyof VideoTransferROI, string]>
                                            ).map(([key, label]) => (
                                                <label key={key}>
                                                    {label}（像素）
                                                    <InputNumber
                                                        aria-label={`${label}像素`}
                                                        disabled={locked}
                                                        min={key === "width" || key === "height" ? 4 : 0}
                                                        precision={0}
                                                        value={draft.roi?.[key]}
                                                        onChange={(value) => setDraft({ roi: { x: 0, y: 0, width: 0, height: 0, ...draft.roi, [key]: value ?? 0 } })}
                                                    />
                                                </label>
                                            ))}
                                        </div>
                                    </>
                                )}
                            </fieldset>
                            <fieldset disabled={locked} className="video-transfer-settings">
                                <legend>处理范围</legend>
                                <div className="video-transfer-time-inputs">
                                    <label>
                                        开始时间（秒）
                                        <InputNumber aria-label="开始时间" disabled={locked} min={0} max={record.sourceDuration} step={0.1} precision={3} value={draft.startSeconds} onChange={(value) => setDraft({ startSeconds: value ?? 0 })} />
                                    </label>
                                    <label>
                                        处理时长（秒）
                                        <InputNumber aria-label="处理时长" disabled={locked} min={0} max={record.sourceDuration} step={0.1} precision={3} value={draft.durationSeconds} onChange={(value) => setDraft({ durationSeconds: value ?? 0 })} />
                                    </label>
                                </div>
                                <p className="video-transfer-hint">
                                    时长为 0 时处理至片尾。{record.sourceDuration ? `原视频时长 ${formatVideoSeconds(record.sourceDuration)}。` : ""}
                                    {limits?.maxDurationSeconds ? `当前服务单次最多处理 ${limits.maxDurationSeconds} 秒。` : ""}
                                </p>
                                <label>
                                    帧处理并发总数
                                    <InputNumber
                                        aria-label="并行处理数"
                                        disabled={locked}
                                        min={1}
                                        max={limits?.maxWorkers}
                                        precision={0}
                                        placeholder="使用已保存值"
                                        value={draft.workers}
                                        onChange={(value) => setDraft({ workers: value ?? branchState.settings.workers })}
                                    />
                                </label>
                                <p className="video-transfer-hint">同时处理的帧数越高，占用的内存越多。服务器会按此总数分配处理资源。</p>
                            </fieldset>
                            <div className="video-transfer-run-state" aria-live="polite">
                                <div>
                                    <span className={`video-transfer-status-dot is-${branchState.status}`} />
                                    <strong>{VIDEO_TRANSFER_STATUS_LABELS[branchState.status]}</strong>
                                    {branchState.elapsedSeconds ? <span>用时 {formatVideoSeconds(branchState.elapsedSeconds)}</span> : null}
                                </div>
                                {active ? <Progress percent={Math.round(branchState.progress || 0)} status="active" size="small" strokeColor="var(--workspace-accent)" /> : null}
                                {branchState.stage && branchState.status !== "failed" ? <p>{formatVideoTransferStage(branchState.stage)}</p> : null}
                                {outputResourceId && JSON.stringify(draft) !== JSON.stringify(savedDraft.settings[selectedBranch]) ? <p>当前结果对应上次保存的设置，新参数将在保存后生效。</p> : null}
                            </div>
                            <ResultSummary branch={selectedBranch} summary={branchState.summary} complete={Boolean(outputResourceId)} />
                            <div className="video-transfer-run-actions">
                                {capabilities.isError ? (
                                    <div className="video-transfer-hint" role="alert">
                                        无法读取处理服务状态。
                                        <Button size="small" type="link" onClick={() => void capabilities.refetch()}>
                                            重试
                                        </Button>
                                    </div>
                                ) : !capability?.available ? (
                                    <p className="video-transfer-hint">{capabilities.isLoading ? "正在检查处理服务…" : capability?.reason || "当前处理服务尚未就绪"}</p>
                                ) : null}
                                {!active && runProblem ? <p className="video-transfer-hint">{runProblem}</p> : null}
                                {active ? (
                                    <Button danger block loading={busy === "cancel"} disabled={busy !== null} onClick={cancel}>
                                        取消处理任务
                                    </Button>
                                ) : (
                                    <Button type="primary" block loading={busy === "run"} disabled={locked || !capability?.available || Boolean(runProblem) || !title.trim()} onClick={start}>
                                        保存并开始处理
                                    </Button>
                                )}
                                {outputResourceId ? (
                                    <Button block onClick={() => setPreview({ resourceId: outputResourceId, label: VIDEO_TRANSFER_BRANCH_LABELS[selectedBranch] })}>
                                        预览与下载结果
                                    </Button>
                                ) : null}
                            </div>
                        </aside>
                    ) : null}
                </div>
                <input
                    ref={fileRef}
                    type="file"
                    accept="video/*,.mkv,.mov"
                    hidden
                    onChange={(event) => {
                        const file = event.target.files?.[0];
                        if (file) upload(file);
                        event.target.value = "";
                    }}
                />
            </div>
            <AppModal
                title={preview ? `${preview.label} - 处理结果` : "处理结果"}
                open={Boolean(preview)}
                width={1000}
                onCancel={() => setPreview(null)}
                footer={
                    preview ? (
                        <Button href={resourceFileUrl(preview.resourceId)} download={`${record.title}-${preview.label}.mp4`}>
                            下载视频
                        </Button>
                    ) : null
                }
            >
                {preview ? <video key={preview.resourceId} src={resourceFileUrl(preview.resourceId)} controls playsInline preload="metadata" className="video-transfer-preview-video" /> : null}
                {preview
                    ? VIDEO_TRANSFER_BRANCHES.map((branch) => {
                          const state = record.branches[branch];
                          const summary = state.outputResourceId === preview.resourceId ? state.summary : undefined;
                          return summary ? <ResultSummary key={branch} branch={branch} summary={summary} /> : null;
                      })
                    : null}
            </AppModal>
        </WorkspacePage>
    );
}

function branchSettings(record: VideoTransferRecord): Record<VideoTransferBranch, VideoTransferSettings> {
    return {
        "fixed-r16": { target: "top-right", ...record.branches["fixed-r16"].settings },
        "moving-v6": { referenceSeconds: 0, ...record.branches["moving-v6"].settings },
    };
}
