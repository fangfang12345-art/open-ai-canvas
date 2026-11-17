import type { VideoTransferBranch, VideoTransferSummary } from "@/services/api/video-transfer";
import { formatVideoTransferLimitation } from "@/lib/video-transfer/workflow";

export function ResultSummary({ branch, summary, complete = true }: { branch: VideoTransferBranch; summary?: VideoTransferSummary; complete?: boolean }) {
    if (!summary) return null;
    return (
        <div className="video-transfer-result-summary">
            {!complete && summary.frames ? <p>当前已处理 {summary.frames} 帧，完整视频尚未生成。</p> : null}
            {branch === "moving-v6" && typeof summary.trackingCoverage === "number" ? (
                <p>
                    {complete ? "跟踪覆盖" : "已处理范围内的跟踪覆盖"} {summary.trackedFrames ?? 0}/{summary.frames ?? 0} 帧（{(summary.trackingCoverage * 100).toFixed(1)}%）
                </p>
            ) : complete && summary.frames ? (
                <p>已处理 {summary.frames} 帧</p>
            ) : null}
            {branch === "moving-v6" && summary.untrackedFrames ? (
                <p>
                    {complete ? "有" : "上述处理范围内有"} {summary.untrackedFrames} 帧未跟踪到水印，已保留原画面。{complete ? "请预览检查效果。" : ""}
                </p>
            ) : null}
            {summary.limitations?.length ? (
                <ul>
                    {summary.limitations.map((value, index) => (
                        <li key={index}>{formatVideoTransferLimitation(value)}</li>
                    ))}
                </ul>
            ) : null}
        </div>
    );
}
