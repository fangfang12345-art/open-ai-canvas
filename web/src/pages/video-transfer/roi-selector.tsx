import { useEffect, useRef, useState, type PointerEvent } from "react";
import { roiFromPoints } from "@/lib/video-transfer/workflow";
import type { VideoTransferROI } from "@/services/api/video-transfer";

export function ROISelector({
    src,
    width,
    height,
    seconds,
    roi,
    disabled,
    onChange,
    ariaLabel = "移动水印选区，拖动框选，也可在下方输入坐标",
}: {
    src: string;
    width: number;
    height: number;
    seconds: number;
    roi?: VideoTransferROI;
    disabled: boolean;
    onChange: (roi: VideoTransferROI) => void;
    ariaLabel?: string;
}) {
    const videoRef = useRef<HTMLVideoElement>(null);
    const startRef = useRef<{ pointer: number; x: number; y: number } | null>(null);
    const [frameError, setFrameError] = useState(false);
    const [frameReady, setFrameReady] = useState(false);
    useEffect(() => {
        const video = videoRef.current;
        setFrameReady(false);
        setFrameError(false);
        if (video && video.readyState >= 1 && Math.abs(video.currentTime - seconds) > 0.000001) video.currentTime = seconds;
        else if (video && video.readyState >= 2) setFrameReady(true);
    }, [seconds, src]);

    function point(event: PointerEvent<HTMLDivElement>) {
        const rect = event.currentTarget.getBoundingClientRect();
        return { x: ((event.clientX - rect.left) / rect.width) * width, y: ((event.clientY - rect.top) / rect.height) * height };
    }

    function finish(event: PointerEvent<HTMLDivElement>) {
        const start = startRef.current;
        if (!start || start.pointer !== event.pointerId) return;
        const next = roiFromPoints(start, point(event), width, height);
        if (next.width >= 1 && next.height >= 1) onChange(next);
        startRef.current = null;
        if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
    }

    return (
        <div className="video-transfer-roi">
            <div className="video-transfer-roi-frame" style={{ aspectRatio: `${width} / ${height}` }}>
                <video
                    ref={videoRef}
                    src={src}
                    muted
                    preload="auto"
                    playsInline
                    onLoadedMetadata={() => {
                        if (videoRef.current) videoRef.current.currentTime = seconds;
                    }}
                    onLoadedData={() => setFrameReady(true)}
                    onSeeked={() => setFrameReady(true)}
                    onError={() => setFrameError(true)}
                />
                <div
                    className="video-transfer-roi-overlay"
                    role="img"
                    aria-label={ariaLabel}
                    data-disabled={disabled || !frameReady || frameError}
                    onPointerDown={(event) => {
                        if (disabled || !frameReady || frameError || event.button !== 0) return;
                        event.preventDefault();
                        const p = point(event);
                        startRef.current = { pointer: event.pointerId, ...p };
                        event.currentTarget.setPointerCapture(event.pointerId);
                    }}
                    onPointerMove={(event) => {
                        const start = startRef.current;
                        if (start?.pointer === event.pointerId) onChange(roiFromPoints(start, point(event), width, height));
                    }}
                    onPointerUp={finish}
                    onPointerCancel={() => {
                        startRef.current = null;
                    }}
                    onLostPointerCapture={() => {
                        startRef.current = null;
                    }}
                >
                    {roi ? <span className="video-transfer-roi-box" style={{ left: `${(roi.x / width) * 100}%`, top: `${(roi.y / height) * 100}%`, width: `${(roi.width / width) * 100}%`, height: `${(roi.height / height) * 100}%` }} /> : null}
                </div>
            </div>
            <p className="video-transfer-hint">{frameError ? "参考画面加载失败，可在下方直接输入像素坐标。" : !frameReady ? "正在读取参考画面…" : "在画面上拖动框住水印，坐标按原视频像素保存。"}</p>
        </div>
    );
}
