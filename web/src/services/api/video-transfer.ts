import { compactApiParams, http } from "@/services/api/request";

export type VideoTransferBranch = "fixed-r16" | "moving-v6";
export type VideoTransferTarget = "top-left" | "top-right" | "bottom-left" | "bottom-right";
export type VideoTransferStatus = "idle" | "queued" | "running" | "succeeded" | "failed" | "cancelled";
export type VideoTransferROI = { x: number; y: number; width: number; height: number };
export type VideoTransferFixedRegion = {
    roi: VideoTransferROI;
    referenceSeconds: number;
    startSeconds: number;
    durationSeconds: number;
};
export type VideoTransferSummary = {
    frames?: number;
    trackedFrames?: number;
    untrackedFrames?: number;
    trackingCoverage?: number;
    limitations?: string[];
};

export type VideoTransferSettings = {
    target?: VideoTransferTarget;
    fixedRegions?: VideoTransferFixedRegion[];
    referenceSeconds?: number;
    roi?: VideoTransferROI;
    startSeconds: number;
    durationSeconds: number;
    workers?: number;
};

export type VideoTransferBranchState = {
    settings: VideoTransferSettings;
    status: VideoTransferStatus;
    progress: number;
    stage?: string;
    error?: string;
    outputResourceId?: string;
    elapsedSeconds?: number;
    taskId?: string;
    summary?: VideoTransferSummary;
    segments?: VideoTransferSegment[];
};

export type VideoTransferSegment = {
    id: string;
    index: number;
    startSeconds: number;
    durationSeconds: number;
    status: VideoTransferStatus;
    progress: number;
    stage?: string;
    error?: string;
    outputResourceId?: string;
    elapsedSeconds?: number;
    summary?: VideoTransferSummary;
};

export type VideoTransferRecord = {
    id: string;
    revision: number;
    title: string;
    sourceResourceId?: string;
    sourceName?: string;
    sourceWidth?: number;
    sourceHeight?: number;
    sourceDuration?: number;
    selectedBranch: VideoTransferBranch;
    branches: Record<VideoTransferBranch, VideoTransferBranchState>;
    createdAt: string;
    updatedAt: string;
};

export type VideoTransferPatch = Partial<Pick<VideoTransferRecord, "title" | "sourceResourceId" | "sourceName" | "selectedBranch">> & {
    expectedRevision: number;
    branches?: Partial<Record<VideoTransferBranch, { settings: VideoTransferSettings }>>;
};

export type VideoTransferCapability = { id: VideoTransferBranch; available: boolean; reason?: string };
export type VideoTransferLimits = {
    maxSourceBytes: number;
    maxSourceDurationSeconds: number;
    maxDurationSeconds: number;
    maxPixels: number;
    maxWorkers: number;
    maxMovingRoiPixels: number;
    maxMovingRoiDimension: number;
    segmentDurationSeconds?: number;
    maxConcurrentSegments?: number;
};
export type VideoTransferCapabilities = { branches: VideoTransferCapability[]; limits: VideoTransferLimits };
export type VideoTransferList = { records: VideoTransferRecord[]; total: number; page: number; pageSize: number; hasMore: boolean };

export function listVideoTransfers(params: { page: number; pageSize: number; q?: string }, signal?: AbortSignal) {
    return http.get<VideoTransferList>("/video-transfers", { params: compactApiParams(params), signal });
}

export async function getVideoTransfer(id: string, signal?: AbortSignal) {
    const data = await http.get<{ record: VideoTransferRecord }>(`/video-transfers/${encodeURIComponent(id)}`, { signal });
    return data.record;
}

export async function createVideoTransfer(input: { title: string; selectedBranch: VideoTransferBranch }, signal?: AbortSignal) {
    const data = await http.post<{ record: VideoTransferRecord }>("/video-transfers", input, { signal });
    return data.record;
}

export async function updateVideoTransfer(id: string, input: VideoTransferPatch, signal?: AbortSignal) {
    const data = await http.patch<{ record: VideoTransferRecord }>(`/video-transfers/${encodeURIComponent(id)}`, input, { signal });
    return data.record;
}

export async function runVideoTransfer(id: string, branch: VideoTransferBranch, expectedRevision: number, signal?: AbortSignal) {
    const data = await http.post<{ record: VideoTransferRecord }>(`/video-transfers/${encodeURIComponent(id)}/run`, { branch, expectedRevision }, { signal });
    return data.record;
}

export async function cancelVideoTransfer(id: string, signal?: AbortSignal) {
    const data = await http.post<{ record: VideoTransferRecord }>(`/video-transfers/${encodeURIComponent(id)}/cancel`, undefined, { signal });
    return data.record;
}

export async function getVideoTransferCapabilities(signal?: AbortSignal) {
    return http.get<VideoTransferCapabilities>("/video-transfers/capabilities", { signal });
}
