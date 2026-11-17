import { useCallback, useEffect, useRef, useState, type PointerEvent } from "react";
import { Button, Progress } from "antd";
import { VIDEO_TRANSFER_BRANCH_LABELS, VIDEO_TRANSFER_STATUS_LABELS, formatVideoSeconds, formatVideoTransferStage } from "@/lib/video-transfer/workflow";
import { resourceFileUrl } from "@/services/api/resources";
import type { VideoTransferBranch, VideoTransferRecord } from "@/services/api/video-transfer";

const GRAPH_WIDTH = 840;
const GRAPH_HEIGHT = 500;
const MIN_ZOOM = 0.15;
const MAX_ZOOM = 1.6;

export function WorkflowCanvas({
    record,
    selectedBranch,
    onUpload,
    onPreview,
    uploadProgress,
    disabled,
}: {
    record: VideoTransferRecord;
    selectedBranch: VideoTransferBranch;
    onUpload: () => void;
    onPreview: (resourceId: string, label: string) => void;
    uploadProgress: number | null;
    disabled: boolean;
}) {
    const hostRef = useRef<HTMLDivElement>(null);
    const state = record.branches[selectedBranch];
    const processing = state.status === "running" || state.status === "queued";
    const outputResourceId = state.status === "succeeded" ? state.outputResourceId : undefined;
    const [view, setView] = useState({ x: 24, y: 24, zoom: 0.8 });
    const [panning, setPanning] = useState(false);
    const uploadLabel = uploadProgress !== null ? (uploadProgress === 100 ? "正在保存视频" : `上传 ${uploadProgress}%`) : record.sourceResourceId ? "更换原视频" : "上传视频";
    const dragRef = useRef<{ id: number; x: number; y: number; originX: number; originY: number } | null>(null);
    const fit = useCallback(() => {
        const host = hostRef.current;
        if (!host || !host.clientWidth || !host.clientHeight) return;
        const zoom = Math.max(MIN_ZOOM, Math.min(1, (host.clientWidth - 64) / GRAPH_WIDTH, (host.clientHeight - 80) / GRAPH_HEIGHT));
        setView({ x: Math.max(16, (host.clientWidth - GRAPH_WIDTH * zoom) / 2), y: Math.max(16, (host.clientHeight - GRAPH_HEIGHT * zoom) / 2), zoom });
    }, []);

    useEffect(() => {
        fit();
        const host = hostRef.current;
        if (!host) return;
        const wheel = (event: WheelEvent) => {
            if (window.matchMedia("(max-width: 767px)").matches) return;
            if ((event.target as HTMLElement).closest("[data-canvas-no-zoom],video,input,button,select,textarea,[role=dialog]")) return;
            event.preventDefault();
            const bounds = host.getBoundingClientRect();
            setView((old) => {
                if (!event.ctrlKey && !event.metaKey) return { ...old, x: old.x - event.deltaX, y: old.y - event.deltaY };
                const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, old.zoom * Math.exp(-event.deltaY * 0.006)));
                const x = event.clientX - bounds.left;
                const y = event.clientY - bounds.top;
                return { zoom, x: x - ((x - old.x) * zoom) / old.zoom, y: y - ((y - old.y) * zoom) / old.zoom };
            });
        };
        host.addEventListener("wheel", wheel, { passive: false });
        const observer = new ResizeObserver(fit);
        observer.observe(host);
        return () => {
            host.removeEventListener("wheel", wheel);
            observer.disconnect();
        };
    }, [fit]);

    function zoomBy(factor: number) {
        const host = hostRef.current;
        if (!host) return;
        setView((old) => {
            const zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, old.zoom * factor));
            return { zoom, x: host.clientWidth / 2 - ((host.clientWidth / 2 - old.x) * zoom) / old.zoom, y: host.clientHeight / 2 - ((host.clientHeight / 2 - old.y) * zoom) / old.zoom };
        });
    }

    function startPan(event: PointerEvent<HTMLDivElement>) {
        if (window.matchMedia("(max-width: 767px)").matches) return;
        if (event.button !== 0 || (event.target as HTMLElement).closest(".video-transfer-node,[data-canvas-no-zoom]")) return;
        event.preventDefault();
        event.currentTarget.focus();
        event.currentTarget.setPointerCapture(event.pointerId);
        dragRef.current = { id: event.pointerId, x: event.clientX, y: event.clientY, originX: view.x, originY: view.y };
        setPanning(true);
    }

    function endPan(event: PointerEvent<HTMLDivElement>) {
        if (dragRef.current?.id !== event.pointerId) return;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
        dragRef.current = null;
        setPanning(false);
    }

    return (
        <div
            ref={hostRef}
            className={`video-transfer-viewport${panning ? " is-panning" : ""}`}
            tabIndex={0}
            aria-label="视频搬运流程画布，拖动空白处平移，按加减键缩放，按 0 适应画布"
            onPointerDown={startPan}
            onPointerMove={(event) => {
                const drag = dragRef.current;
                if (drag?.id === event.pointerId) setView((old) => ({ ...old, x: drag.originX + event.clientX - drag.x, y: drag.originY + event.clientY - drag.y }));
            }}
            onPointerUp={endPan}
            onPointerCancel={endPan}
            onLostPointerCapture={() => {
                dragRef.current = null;
                setPanning(false);
            }}
            onKeyDown={(event) => {
                if (window.matchMedia("(max-width: 767px)").matches) return;
                if (event.target !== event.currentTarget) return;
                if (event.key === "+" || event.key === "=") {
                    event.preventDefault();
                    zoomBy(1.2);
                } else if (event.key === "-") {
                    event.preventDefault();
                    zoomBy(1 / 1.2);
                } else if (event.key === "0") {
                    event.preventDefault();
                    fit();
                } else if (["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"].includes(event.key)) {
                    event.preventDefault();
                    setView((old) => ({ ...old, x: old.x + (event.key === "ArrowLeft" ? 40 : event.key === "ArrowRight" ? -40 : 0), y: old.y + (event.key === "ArrowUp" ? 40 : event.key === "ArrowDown" ? -40 : 0) }));
                }
            }}
        >
            <div className="video-transfer-graph" style={{ width: GRAPH_WIDTH, height: GRAPH_HEIGHT, transform: `translate(${view.x}px, ${view.y}px) scale(${view.zoom})` }}>
                <svg className="video-transfer-edges" width={GRAPH_WIDTH} height={GRAPH_HEIGHT} aria-hidden>
                    <g className="is-selected">
                        <path d="M 340 240 C 380 240, 400 240, 440 240" />
                        <circle cx={340} cy={240} r={5} />
                        <circle cx={440} cy={240} r={5} />
                    </g>
                </svg>
                <article className="video-transfer-node video-transfer-source" style={{ left: 40, top: 60, width: 300 }}>
                    <header>
                        <strong>原视频</strong>
                        <span className="video-transfer-node-kind">输入</span>
                    </header>
                    {record.sourceResourceId ? (
                        <video key={record.sourceResourceId} src={resourceFileUrl(record.sourceResourceId)} controls preload="metadata" playsInline className="video-transfer-node-video" />
                    ) : (
                        <div className="video-transfer-source-empty">
                            <span>上传待处理的视频</span>
                            <p>上传后选择算法并配置水印区域</p>
                        </div>
                    )}
                    <div className="video-transfer-node-body">
                        <p className="video-transfer-source-name" title={record.sourceName}>
                            {record.sourceName || "尚未上传"}
                        </p>
                        {record.sourceResourceId ? (
                            <p className="video-transfer-hint">
                                {record.sourceWidth} × {record.sourceHeight} / {formatVideoSeconds(record.sourceDuration)}
                            </p>
                        ) : null}
                        <Button block disabled={disabled} loading={uploadProgress !== null} onClick={onUpload}>
                            {uploadLabel}
                        </Button>
                        <p className="video-transfer-hint">不设单文件大小上限，按账号可用空间上传。</p>
                    </div>
                </article>
                <article className="video-transfer-node video-transfer-algorithm is-selected" style={{ left: 440, top: 60, width: 360 }} aria-label="视频处理任务">
                    <header>
                        <strong>{VIDEO_TRANSFER_BRANCH_LABELS[selectedBranch]}</strong>
                        <span className={`video-transfer-status is-${state.status}`}>{VIDEO_TRANSFER_STATUS_LABELS[state.status]}</span>
                    </header>
                    {outputResourceId ? (
                        <video key={outputResourceId} src={resourceFileUrl(outputResourceId)} aria-label="完整处理结果" controls preload="metadata" playsInline className="video-transfer-node-video video-transfer-result-video" />
                    ) : (
                        <div className="video-transfer-result-empty" aria-live="polite">
                            <strong>{processing ? "正在处理视频" : state.status === "failed" ? "视频处理失败" : state.status === "cancelled" ? "处理已取消" : state.status === "succeeded" ? "完整视频尚未就绪" : "等待开始处理"}</strong>
                            {state.status === "failed" ? (
                                <>
                                    <p className="video-transfer-error-text">{state.error || "本次处理未完成，未生成完整视频。"}</p>
                                    <p>请根据失败原因调整配置或排查服务，再重新处理。</p>
                                </>
                            ) : (
                                <p>{processing ? "完成后将在这里展示完整视频。" : state.status === "succeeded" ? "请重新开始处理以生成完整视频。" : "在配置中选择算法和处理范围，保存后开始处理。"}</p>
                            )}
                        </div>
                    )}
                    <div className="video-transfer-node-body" aria-live="polite">
                        {processing ? (
                            <div>
                                <p className="video-transfer-hint">总进度</p>
                                <Progress aria-label="总处理进度" percent={Math.round(state.progress || 0)} size="small" status={state.status === "running" ? "active" : "normal"} strokeColor="var(--workspace-accent)" />
                            </div>
                        ) : null}
                        {state.stage && state.status !== "failed" ? <p className="video-transfer-hint">{formatVideoTransferStage(state.stage)}</p> : null}
                        {state.error && state.status !== "failed" ? <p className="video-transfer-error-text">{state.error}</p> : null}
                        {state.elapsedSeconds ? <p className="video-transfer-hint">用时 {formatVideoSeconds(state.elapsedSeconds)}</p> : null}
                        {outputResourceId ? (
                            <Button block onClick={() => onPreview(outputResourceId, VIDEO_TRANSFER_BRANCH_LABELS[selectedBranch])}>
                                预览与下载完整视频
                            </Button>
                        ) : null}
                    </div>
                </article>
            </div>
            <div className="video-transfer-viewport-help">拖动空白处平移 / Ctrl + 滚轮缩放</div>
            <div className="video-transfer-viewport-controls" data-canvas-no-zoom>
                <Button size="small" aria-label="缩小画布" onClick={() => zoomBy(1 / 1.2)}>
                    -
                </Button>
                <span>{Math.round(view.zoom * 100)}%</span>
                <Button size="small" aria-label="放大画布" onClick={() => zoomBy(1.2)}>
                    +
                </Button>
                <Button size="small" onClick={fit}>
                    适应画布
                </Button>
            </div>
        </div>
    );
}
