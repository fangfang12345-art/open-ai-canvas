import { describe, expect, test } from "bun:test";
import { formatVideoSegmentRange, formatVideoTransferLimitation, formatVideoTransferStage, isVideoTransferActive, roiFromPoints, validateVideoTransferSettings, videoTransferKeys } from "../src/lib/video-transfer/workflow";
import type { VideoTransferLimits, VideoTransferRecord, VideoTransferSettings } from "../src/services/api/video-transfer";

const source = { sourceWidth: 3840, sourceHeight: 2160, sourceDuration: 30 };
const moving: VideoTransferSettings = { startSeconds: 10, durationSeconds: 10, referenceSeconds: 0, roi: { x: 3100, y: 100, width: 150, height: 64 } };

describe("video transfer branch input validation", () => {
    test("result limitations explain both algorithms without dropping unknown warnings", () => {
        const limitations = [
            "Stationary bright neutral overlays only; persistent bright scene detail can be ambiguous.",
            "Frames outside explicitly configured activity intervals are preserved without target verification.",
            "Active frames must match the shared bright-overlay reference; uncertainty fails the segment.",
            "Opaque hidden content is estimated spatially, not recovered as ground truth.",
            "Requires a stable translation-only logo and a near-uniform reference ROI border.",
            "Uncertain tracked frames are preserved, and never used as clean donor evidence.",
            "V6 uses the historical spatial fallback when independent clean donor frames are unavailable.",
            "Inversion is disabled per pixel when held-out calibration cannot establish its gain.",
            "Occlusion, rotation, scaling, changing glyphs, and indistinguishable scene detail are unsupported.",
        ];
        for (const value of limitations) {
            expect(formatVideoTransferLimitation(value)).toMatch(/[\u4e00-\u9fff]/);
            expect(formatVideoTransferLimitation(value)).not.toContain("held-out");
        }
        expect(formatVideoTransferLimitation("New unsupported source condition")).toBe("New unsupported source condition");
        expect(formatVideoTransferLimitation("请检查输出中的文字细节")).toBe("请检查输出中的文字细节");
    });

    test("algorithm progress is readable while server messages remain intact", () => {
        expect(formatVideoTransferStage("calibrating_moving_inverse")).toBe("估计水印颜色与透明度");
        expect(formatVideoTransferStage("processing_segments")).toBe("正在处理视频");
        expect(formatVideoTransferStage("正在生成完整视频")).toBe("正在生成完整视频");
        for (const stage of [
            "probing",
            "preparing_native_kernels",
            "calibrating_fixed_mask",
            "fixed_mask_ready",
            "calibrating_fixed_references",
            "fixed_references_ready",
            "tracking_moving_mask",
            "moving_model_ready",
            "repairing",
            "encoding",
            "completed",
        ]) {
            expect(formatVideoTransferStage(stage)).not.toBe(stage);
            expect(formatVideoTransferStage(stage)).toMatch(/[\u4e00-\u9fff]/);
        }
        expect(formatVideoTransferStage("保存处理结果")).toBe("保存处理结果");
        expect(formatVideoTransferStage("future_stage")).toBe("future_stage");
        expect(formatVideoTransferStage(undefined)).toBeUndefined();
        expect(formatVideoTransferStage(" ")).toBeUndefined();
    });

    test("a moving watermark reference is an absolute source time outside the processed clip", () => {
        expect(validateVideoTransferSettings("moving-v6", moving, source)).toBeUndefined();
        expect(validateVideoTransferSettings("moving-v6", { ...moving, referenceSeconds: 29.9 }, source)).toBeUndefined();
        expect(validateVideoTransferSettings("moving-v6", { ...moving, referenceSeconds: 30 }, source)).toContain("原视频结束");
    });

    test("fixed branch accepts a 4K source without requiring the other branch's ROI", () => {
        expect(validateVideoTransferSettings("fixed-r16", { startSeconds: 0, durationSeconds: 0, target: "bottom-right" }, source)).toBeUndefined();
        expect(validateVideoTransferSettings("fixed-r16", { startSeconds: 0, durationSeconds: 0 }, source)).toContain("角落");
    });

    test("fixed regions use source pixels and independent absolute active windows without changing V6", () => {
        const fixedRegions = [
            { roi: { x: 0, y: 0, width: 1100, height: 300 }, referenceSeconds: 2, startSeconds: 0, durationSeconds: 0 },
            { roi: { x: 2700, y: 0, width: 1100, height: 300 }, referenceSeconds: 3, startSeconds: 1, durationSeconds: 9 },
        ];
        const fixed: VideoTransferSettings = { startSeconds: 15, durationSeconds: 10, fixedRegions };
        expect(validateVideoTransferSettings("fixed-r16", fixed, source)).toBeUndefined();
        expect(validateVideoTransferSettings("moving-v6", { ...moving, fixedRegions }, source)).toBeUndefined();
        expect(validateVideoTransferSettings("fixed-r16", { ...fixed, fixedRegions: [], target: "top-left" }, source)).toBeUndefined();
        expect(validateVideoTransferSettings("fixed-r16", { ...fixed, fixedRegions: [fixedRegions[1]] }, source)).toContain("没有生效");
        expect(validateVideoTransferSettings("fixed-r16", { ...fixed, fixedRegions: Array(5).fill(fixedRegions[0]) }, source)).toContain("最多支持 4");
        expect(validateVideoTransferSettings("fixed-r16", { ...fixed, fixedRegions: [fixedRegions[0], { ...fixedRegions[1], roi: { ...fixedRegions[1].roi, x: 1000 } }] }, source)).toContain("重叠");
        expect(validateVideoTransferSettings("fixed-r16", { ...fixed, fixedRegions: [fixedRegions[0], { ...fixedRegions[1], roi: { ...fixedRegions[1].roi, x: 1100 } }] }, source)).toBeUndefined();
        for (const patch of [
            { referenceSeconds: 0 },
            { referenceSeconds: 10 },
            { referenceSeconds: NaN },
            { startSeconds: -1 },
            { durationSeconds: Infinity },
            { durationSeconds: -1 },
            { durationSeconds: 30 },
            { startSeconds: 30, durationSeconds: 0 },
            { roi: { x: 3800, y: 0, width: 100, height: 20 } },
            { roi: { x: 0, y: 0, width: 3, height: 20 } },
        ]) {
            expect(validateVideoTransferSettings("fixed-r16", { ...fixed, fixedRegions: [fixedRegions[0], { ...fixedRegions[1], ...patch }] }, source)).toBeDefined();
        }
        expect(validateVideoTransferSettings("fixed-r16", { ...fixed, fixedRegions: [{ ...fixedRegions[0], referenceSeconds: 30 }] }, source)).toContain("参考时刻");
    });

    test("uses server limits for duration and workers while allowing supported 4K sources", () => {
        const limits: VideoTransferLimits = { maxSourceBytes: 2 ** 31, maxSourceDurationSeconds: 7200, maxDurationSeconds: 12, maxPixels: 3840 * 2160, maxWorkers: 6, maxMovingRoiPixels: 196608, maxMovingRoiDimension: 1024 };
        expect(validateVideoTransferSettings("moving-v6", { ...moving, workers: 6 }, source, limits)).toBeUndefined();
        expect(validateVideoTransferSettings("moving-v6", { ...moving, durationSeconds: 0 }, source, limits)).toContain("12 秒");
        expect(validateVideoTransferSettings("moving-v6", { ...moving, workers: 7 }, source, limits)).toContain("1 到 6");
        expect(validateVideoTransferSettings("moving-v6", moving, source, { ...limits, maxPixels: 1920 * 1080 })).toContain("像素量");
        expect(validateVideoTransferSettings("fixed-r16", { startSeconds: 0, durationSeconds: 0, target: "top-right" }, { ...source, sourceDuration: 86400 }, { ...limits, maxSourceDurationSeconds: 0, maxDurationSeconds: 0 })).toBeUndefined();
    });

    test("rejects out of range, non-finite and negative processing windows", () => {
        for (const patch of [{ startSeconds: -1 }, { startSeconds: 30 }, { startSeconds: NaN }, { durationSeconds: Infinity }, { durationSeconds: -1 }, { durationSeconds: 21 }]) {
            expect(validateVideoTransferSettings("moving-v6", { ...moving, ...patch }, source)).toBeDefined();
        }
        expect(validateVideoTransferSettings("moving-v6", { ...moving, durationSeconds: 0 }, source)).toBeUndefined();
    });

    test("rejects fractional, too small, oversized, and out of bounds regions", () => {
        for (const roi of [
            { x: 0.5, y: 0, width: 50, height: 20 },
            { x: 0, y: 0, width: 3, height: 20 },
            { x: 3800, y: 0, width: 80, height: 20 },
            { x: 0, y: 2150, width: 80, height: 20 },
            { x: 0, y: 0, width: 1025, height: 20 },
            { x: 0, y: 0, width: 600, height: 600 },
        ])
            expect(validateVideoTransferSettings("moving-v6", { ...moving, roi }, source)).toBeDefined();
        expect(validateVideoTransferSettings("moving-v6", { ...moving, roi: { x: 0, y: 0, width: 4, height: 4 } }, source)).toBeUndefined();
    });

    test("dragging backwards or outside the preview produces bounded source pixels", () => {
        expect(roiFromPoints({ x: 140.4, y: 80.4 }, { x: 50.2, y: 20.2 }, 200, 100)).toEqual({ x: 50, y: 20, width: 90, height: 60 });
        expect(roiFromPoints({ x: 170, y: 80 }, { x: 250, y: -50 }, 200, 100)).toEqual({ x: 170, y: 0, width: 30, height: 80 });
    });

    test("query caches isolate account and record identities", () => {
        expect(videoTransferKeys.record("account-a", "record-1")).not.toEqual(videoTransferKeys.record("account-b", "record-1"));
        expect(videoTransferKeys.record("account-a", "record-1")).not.toEqual(videoTransferKeys.record("account-a", "record-2"));
        expect(videoTransferKeys.list("account-a", { page: 1, pageSize: 20, q: "" })).not.toEqual(videoTransferKeys.list("account-b", { page: 1, pageSize: 20, q: "" }));
    });

    test("the non-selected running branch still locks source replacement", () => {
        const record = { selectedBranch: "fixed-r16", branches: { "fixed-r16": { status: "succeeded" }, "moving-v6": { status: "running" } } } as VideoTransferRecord;
        expect(isVideoTransferActive(record)).toBe(true);
        record.branches["moving-v6"].status = "cancelled";
        expect(isVideoTransferActive(record)).toBe(false);
        record.branches["moving-v6"].segments = [{ id: "segment-1", index: 0, startSeconds: 0, durationSeconds: 10, progress: 0, status: "queued" }];
        expect(isVideoTransferActive(record)).toBe(true);
    });

    test("segment ranges preserve a short final fragment and millisecond boundaries", () => {
        expect(formatVideoSegmentRange(0, 10)).toBe("00:00 - 00:10");
        expect(formatVideoSegmentRange(20, 1.3)).toBe("00:20 - 00:21.3");
        expect(formatVideoSegmentRange(3599.999, 0.001)).toBe("59:59.999 - 1:00:00");
    });
});
