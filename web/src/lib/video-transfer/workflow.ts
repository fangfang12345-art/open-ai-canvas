import type { VideoTransferBranch, VideoTransferLimits, VideoTransferRecord, VideoTransferROI, VideoTransferSettings, VideoTransferStatus } from "@/services/api/video-transfer";

export const VIDEO_TRANSFER_BRANCHES: VideoTransferBranch[] = ["fixed-r16", "moving-v6"];
export const VIDEO_TRANSFER_BRANCH_LABELS: Record<VideoTransferBranch, string> = { "fixed-r16": "固定水印 R16", "moving-v6": "移动水印 V6" };
export const VIDEO_TRANSFER_STATUS_LABELS: Record<VideoTransferStatus, string> = {
    idle: "未开始",
    queued: "排队中",
    running: "处理中",
    succeeded: "已完成",
    failed: "失败",
    cancelled: "已取消",
};

const VIDEO_TRANSFER_STAGE_LABELS: Record<string, string> = {
    probing: "读取视频信息",
    preparing_native_kernels: "准备视频处理",
    calibrating_fixed_mask: "识别固定水印区域",
    fixed_mask_ready: "固定水印已定位",
    calibrating_fixed_references: "识别各区域的固定水印",
    fixed_references_ready: "固定水印参考区域已就绪",
    tracking_moving_mask: "跟踪水印位置",
    calibrating_moving_inverse: "估计水印颜色与透明度",
    moving_model_ready: "移动水印已定位",
    repairing: "修复水印区域",
    encoding: "导出处理视频",
    completed: "视频处理完成",
    processing_segments: "正在处理视频",
};

export function formatVideoTransferStage(stage?: string) {
    const value = stage?.trim();
    return value ? VIDEO_TRANSFER_STAGE_LABELS[value] || value : undefined;
}

const VIDEO_TRANSFER_LIMITATION_LABELS: Record<string, string> = {
    "Stationary bright neutral overlays only; persistent bright scene detail can be ambiguous.": "仅适用于位置固定的亮色中性水印；持续存在的明亮背景细节可能被误识别。",
    "Frames outside explicitly configured activity intervals are preserved without target verification.": "各区域仅在设定的有效时段内识别和处理，时段外保留原画面。",
    "Active frames must match the shared bright-overlay reference; uncertainty fails the segment.": "有效时段内的水印须与参考画面一致；无法确认时会停止处理，请检查参考画面和有效时段。",
    "Opaque hidden content is estimated spatially, not recovered as ground truth.": "完全遮挡的内容依靠周围像素估计，无法保证还原真实细节。",
    "Requires a stable translation-only logo and a near-uniform reference ROI border.": "水印需形状稳定、仅平移，参考选区四周的背景应接近均匀。",
    "Uncertain tracked frames are preserved, and never used as clean donor evidence.": "跟踪位置不确定的帧保留原画面，也不会作为其他帧的修复依据。",
    "V6 uses the historical spatial fallback when independent clean donor frames are unavailable.": "缺少可靠的其他帧背景时，使用水印周围的像素填补。",
    "V6 estimates missing background using evidence-gated spatial continuity; hidden texture is not ground truth.": "缺少可靠的其他帧背景时，按可见背景和边缘连续性填补，无法保证还原被遮挡的真实纹理。",
    "Large holes with no validated planar background may use spatial exemplar synthesis, not observed background truth.": "较大的遮挡区域缺少经验证的平面背景时，可能采用周围纹理合成填补，无法保证还原真实背景。",
    "Inversion is disabled per pixel when held-out calibration cannot establish its gain.": "无法可靠估计透明度的区域不会进行反算，以免放大残点和噪声。",
    "Occlusion, rotation, scaling, changing glyphs, and indistinguishable scene detail are unsupported.": "暂不支持遮挡、旋转、缩放、字形变化或与背景难以区分的水印。",
};

export function formatVideoTransferLimitation(value: string) {
    return VIDEO_TRANSFER_LIMITATION_LABELS[value] || value;
}

export const videoTransferKeys = {
    all: (userId: string) => ["video-transfers", userId] as const,
    list: (userId: string, params: { page: number; pageSize: number; q: string }) => ["video-transfers", userId, "list", params] as const,
    record: (userId: string, id: string) => ["video-transfers", userId, "record", id] as const,
    capabilities: (userId: string) => ["video-transfers", userId, "capabilities"] as const,
};

export function isVideoTransferActive(record: VideoTransferRecord) {
    return VIDEO_TRANSFER_BRANCHES.some((branch) => {
        const state = record.branches[branch];
        return state.status === "running" || state.status === "queued" || state.segments?.some((segment) => segment.status === "running" || segment.status === "queued");
    });
}

export function validateVideoTransferSettings(branch: VideoTransferBranch, settings: VideoTransferSettings, source: Pick<VideoTransferRecord, "sourceWidth" | "sourceHeight" | "sourceDuration">, limits?: VideoTransferLimits): string | undefined {
    if (!Number.isFinite(settings.startSeconds) || settings.startSeconds < 0) return "开始时间必须是大于或等于 0 的秒数";
    if (!Number.isFinite(settings.durationSeconds) || settings.durationSeconds < 0) return "处理时长必须是大于或等于 0 的秒数";
    if (source.sourceDuration && settings.startSeconds >= source.sourceDuration) return "开始时间必须早于视频结束时间";
    if (source.sourceDuration && settings.durationSeconds > 0 && settings.startSeconds + settings.durationSeconds > source.sourceDuration + 0.01) return "处理范围超出了视频时长";
    const duration = settings.durationSeconds || (source.sourceDuration ? source.sourceDuration - settings.startSeconds : 0);
    if (limits && limits.maxDurationSeconds > 0 && duration > limits.maxDurationSeconds) return `当前服务单次最多处理 ${limits.maxDurationSeconds} 秒，请缩短处理时长`;
    if (limits && limits.maxSourceDurationSeconds > 0 && source.sourceDuration && source.sourceDuration > limits.maxSourceDurationSeconds) return `原视频时长超出服务上限 ${limits.maxSourceDurationSeconds} 秒`;
    if (limits && source.sourceWidth && source.sourceHeight && source.sourceWidth * source.sourceHeight > limits.maxPixels) return `原视频像素量超出服务上限 ${limits.maxPixels}，请更换视频`;
    if (settings.workers !== undefined && (!Number.isInteger(settings.workers) || settings.workers < 1 || (limits && settings.workers > limits.maxWorkers))) return `并行处理数必须是 1 到 ${limits?.maxWorkers || 16} 之间的整数`;
    if (branch === "fixed-r16") {
        const regions = settings.fixedRegions;
        if (!regions?.length) return ["top-left", "top-right", "bottom-left", "bottom-right"].includes(settings.target || "") ? undefined : "请选择固定水印所在的角落";
        if (regions.length > 4) return "固定水印最多支持 4 个精确区域";
        for (const [index, region] of regions.entries()) {
            const label = `区域 ${index + 1}`;
            const roiProblem = validateVideoTransferROI(region.roi, source);
            if (roiProblem) return `${label}：${roiProblem}`;
            if (!Number.isFinite(region.startSeconds) || region.startSeconds < 0 || !Number.isFinite(region.durationSeconds) || region.durationSeconds < 0) return `${label}的有效开始与时长必须是大于或等于 0 的秒数`;
            const end = region.durationSeconds > 0 ? region.startSeconds + region.durationSeconds : source.sourceDuration;
            if (source.sourceDuration !== undefined && (region.startSeconds >= source.sourceDuration || (end !== undefined && end > source.sourceDuration + 0.01))) return `${label}的有效时段超出了原视频时长`;
            if (!Number.isFinite(region.referenceSeconds) || region.referenceSeconds < region.startSeconds || (end !== undefined && region.referenceSeconds >= end)) return `${label}的参考时刻必须在有效时段内，且早于结束时间`;
            if (regions.slice(0, index).some(({ roi }) => roi.x < region.roi.x + region.roi.width && roi.x + roi.width > region.roi.x && roi.y < region.roi.y + region.roi.height && roi.y + roi.height > region.roi.y))
                return `${label}与其他水印区域重叠，请调整选区`;
        }
        const processEnd = settings.durationSeconds > 0 ? settings.startSeconds + settings.durationSeconds : (source.sourceDuration ?? Infinity);
        if (!regions.some((region) => region.startSeconds < processEnd && (region.durationSeconds > 0 ? region.startSeconds + region.durationSeconds : (source.sourceDuration ?? Infinity)) > settings.startSeconds))
            return "处理范围内没有生效的水印区域，请调整处理范围或区域有效时段";
        return undefined;
    }
    const referenceSeconds = settings.referenceSeconds;
    if (referenceSeconds === undefined || !Number.isFinite(referenceSeconds) || referenceSeconds < 0) return "参考时刻必须是大于或等于 0 的秒数";
    if (source.sourceDuration !== undefined && referenceSeconds >= source.sourceDuration) return "参考时刻必须早于原视频结束时间";
    const roi = settings.roi;
    const roiProblem = validateVideoTransferROI(roi, source);
    if (roiProblem || !roi) return roiProblem;
    const maxDimension = limits?.maxMovingRoiDimension ?? 1024;
    const maxArea = limits?.maxMovingRoiPixels ?? 196608;
    if (roi.width > maxDimension || roi.height > maxDimension || roi.width * roi.height > maxArea) return `选区过大，请仅框选水印：单边不超过 ${maxDimension} 像素，面积不超过 ${maxArea} 像素`;
    return undefined;
}

function validateVideoTransferROI(roi: VideoTransferSettings["roi"], source: Pick<VideoTransferRecord, "sourceWidth" | "sourceHeight">): string | undefined {
    if (!roi || ![roi.x, roi.y, roi.width, roi.height].every(Number.isInteger) || roi.x < 0 || roi.y < 0 || roi.width < 4 || roi.height < 4) return "请填写整数像素坐标，选区宽高至少为 4 像素";
    if (!source.sourceWidth || !source.sourceHeight) return "无法读取原视频尺寸，请重新上传可读取的视频";
    if (roi.x + roi.width > source.sourceWidth || roi.y + roi.height > source.sourceHeight) return "水印选区超出了原视频画面";
    return undefined;
}

// 拖框始终映射到原视频像素，缩放预览不会改变算法输入坐标。
export function roiFromPoints(start: { x: number; y: number }, end: { x: number; y: number }, width: number, height: number): VideoTransferROI {
    const clamp = (value: number, max: number) => Math.round(Math.min(max, Math.max(0, value)));
    const x1 = clamp(Math.min(start.x, end.x), width);
    const y1 = clamp(Math.min(start.y, end.y), height);
    const x2 = clamp(Math.max(start.x, end.x), width);
    const y2 = clamp(Math.max(start.y, end.y), height);
    return { x: x1, y: y1, width: x2 - x1, height: y2 - y1 };
}

export function formatVideoSeconds(seconds?: number) {
    if (seconds === undefined || !Number.isFinite(seconds)) return "时长未知";
    const total = Math.floor(Math.max(0, seconds));
    const hours = Math.floor(total / 3600);
    return `${hours ? `${hours}:` : ""}${String(Math.floor(total / 60) % 60).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
}

export function formatVideoSegmentRange(startSeconds: number, durationSeconds: number) {
    const time = (value: number) => {
        const milliseconds = Math.round(value * 1000);
        const remainder = milliseconds % 1000;
        return `${formatVideoSeconds(Math.floor(milliseconds / 1000))}${remainder ? `.${String(remainder).padStart(3, "0").replace(/0+$/, "")}` : ""}`;
    };
    return `${time(startSeconds)} - ${time(startSeconds + durationSeconds)}`;
}
