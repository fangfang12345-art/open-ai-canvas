"""Bidirectional classical tracking of a user-selected opaque overlay anchor.

The CLI accepts the source, reference frame, overall overlay ROI, and optional
smaller stable anchor ROI. No content, trajectory, or video is built in.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def subpixel_peak(response, xy):
    x, y = xy
    offsets = []
    for axis in (0, 1):
        coord = x if axis == 0 else y
        length = response.shape[1] if axis == 0 else response.shape[0]
        if not 0 < coord < length - 1:
            offsets.append(0.)
            continue
        before = response[y, x - 1] if axis == 0 else response[y - 1, x]
        after = response[y, x + 1] if axis == 0 else response[y + 1, x]
        denominator = float(before - 2 * response[y, x] + after)
        offset = .5 * float(before - after) / denominator if denominator < -1e-8 else 0.
        offsets.append(float(np.clip(offset, -.5, .5)))
    return np.array([x + offsets[0], y + offsets[1]], np.float64)


def reference_support(patch, dilation=1):
    """Extract support when the selected reference has a near-uniform border.

    This is a conservative image-statistics mask, not an alpha matte. Its
    quality report lets callers decline extraction on a textured reference.
    """
    border = np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])
    background = np.median(border.astype(np.float32), axis=0)
    difference = np.max(np.abs(patch.astype(np.float32) - background), axis=2)
    border_noise = np.max(np.abs(border.astype(np.float32) - background), axis=1)
    threshold = max(6., float(np.percentile(border_noise, 95)) + 3.)
    binary = (difference > threshold).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    clean = np.zeros_like(binary)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 2:
            clean[labels == label] = 1
    if dilation:
        size = dilation * 2 + 1
        clean = cv2.dilate(clean, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)))
    return clean, {
        "background_bgr": background.tolist(), "threshold": threshold,
        "border_noise_p95": float(np.percentile(border_noise, 95)),
        "mask_pixels": int(clean.sum()), "dilation": dilation,
        "assumption": "Overlay reference has a near-uniform background at the ROI border",
    }


class TemplateAnchorTracker:
    def __init__(self, reference_gray, anchor_roi, min_confidence=.60, search_margin=16):
        x0, y0, x1, y1 = anchor_roi
        self.template = reference_gray[y0:y1, x0:x1].copy()
        if self.template.size == 0 or self.template.std() < 1:
            raise ValueError("Anchor must contain nonuniform image structure")
        self.height, self.width = reference_gray.shape
        self.anchor_height, self.anchor_width = self.template.shape
        self.position = np.array([x0, y0], np.float64)
        self.velocity = np.zeros(2, np.float64)
        self.min_confidence = min_confidence
        self.search_margin = search_margin
        self.lost = False

    def match(self, gray, global_search=False):
        predicted = self.position + self.velocity
        if global_search:
            x0 = y0 = 0
            x1, y1 = self.width, self.height
        else:
            x0 = max(0, int(np.floor(predicted[0] - self.search_margin)))
            y0 = max(0, int(np.floor(predicted[1] - self.search_margin)))
            x1 = min(self.width, int(np.ceil(predicted[0] + self.anchor_width + self.search_margin)))
            y1 = min(self.height, int(np.ceil(predicted[1] + self.anchor_height + self.search_margin)))
            # Velocity can predict beyond the image while the logo is still
            # fully visible. Let update reacquire globally before matching a
            # crop that is smaller than the template in either dimension.
            if x1 - x0 < self.anchor_width or y1 - y0 < self.anchor_height:
                return -1., self.position.copy(), 0.
        response = cv2.matchTemplate(gray[y0:y1, x0:x1], self.template, cv2.TM_CCOEFF_NORMED)
        _, score, _, location = cv2.minMaxLoc(response)
        position = subpixel_peak(response, location) + [x0, y0]
        sx, sy = location
        response[max(0, sy - 4):sy + 5, max(0, sx - 4):sx + 5] = -1
        competing = float(response.max())
        return float(score), position, float(score - competing)

    def update(self, gray):
        started = time.perf_counter()
        search = "global" if self.lost else "local"
        score, position, peak_gap = self.match(gray, global_search=self.lost)
        if search == "local" and score < self.min_confidence:
            score, position, peak_gap = self.match(gray, global_search=True)
            search = "global"
        threshold = max(.70, self.min_confidence) if search == "global" else self.min_confidence
        accepted = score >= threshold and peak_gap >= .05
        if accepted:
            self.velocity = .7 * self.velocity + .3 * (position - self.position) if search == "local" else np.zeros(2)
            self.position = position
        else:
            self.velocity[:] = 0
        self.lost = not accepted
        return {
            "anchor_subpixel_xy": position.tolist() if accepted else None,
            "candidate_subpixel_xy": position.tolist(),
            "confidence": score, "peak_gap": peak_gap,
            "accepted": bool(accepted), "search": search,
            "tracking_ms": (time.perf_counter() - started) * 1000,
        }


def interpolate_short_gaps(records, box_size, anchor_offset, max_gap=3,
                           anchor_confidence=.65, max_velocity_error=1.0):
    """Fill only bounded short gaps corroborated by motion on both sides.

    The detector's score and accepted flag remain separately recorded. This
    function never interpolates at either end or through a direction change.
    """
    filled = []
    for record in records:
        record["detector_accepted"] = record["accepted"]
        record["acceptance"] = "matched" if record["accepted"] else "rejected"
    index = 0
    while index < len(records):
        if records[index]["detector_accepted"]:
            index += 1
            continue
        start = index
        while index < len(records) and not records[index]["detector_accepted"]:
            index += 1
        end = index
        left, right = start - 1, end
        if end - start > max_gap or left < 2 or right + 2 >= len(records):
            continue
        context = [records[i] for i in [left - 2, left, right, right + 2]]
        if not all(r["detector_accepted"] for r in context):
            continue
        if min(records[left]["confidence"], records[right]["confidence"]) < anchor_confidence:
            continue
        points = [np.array(r["subpixel_xy"], np.float64) for r in context]
        before = (points[1] - points[0]) / 2
        after = (points[3] - points[2]) / 2
        across = (points[2] - points[1]) / (right - left)
        error = max(float(np.linalg.norm(before - after)),
                    float(np.linalg.norm(before - across)),
                    float(np.linalg.norm(after - across)))
        if error > max_velocity_error:
            continue
        for missing in range(start, end):
            xy = points[1] + across * (missing - left)
            ix, iy = np.round(xy).astype(int)
            record = records[missing]
            record.update({
                "subpixel_xy": xy.tolist(),
                "anchor_subpixel_xy": (xy - anchor_offset).tolist(),
                "bbox": [int(ix), int(iy), int(ix + box_size[0]), int(iy + box_size[1])],
                "accepted": True, "acceptance": "interpolated",
                "interpolation": {"endpoints": [left, right], "velocity_error": error},
            })
            filled.append(missing)
    return filled


def track_video(source, reference_frame, reference_roi, anchor_roi=None,
                frames=300, min_confidence=.60, search_margin=16, mask_dilation=1,
                max_gap=3):
    started = time.perf_counter()
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    gray_frames = []
    reference_bgr = None
    for frame_index in range(frames):
        success, value = capture.read()
        if not success:
            break
        gray_frames.append(cv2.cvtColor(value, cv2.COLOR_BGR2GRAY))
        if frame_index == reference_frame:
            reference_bgr = value.copy()
    capture.release()
    if reference_bgr is None:
        raise ValueError("Reference frame is outside the decoded range")
    anchor_roi = tuple(anchor_roi or reference_roi)
    reference_roi = tuple(reference_roi)
    height, width = reference_bgr.shape[:2]
    for x0, y0, x1, y1 in [anchor_roi, reference_roi]:
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError("ROIs must be inside the original frame")
    ox, oy = reference_roi[0] - anchor_roi[0], reference_roi[1] - anchor_roi[1]
    bw, bh = reference_roi[2] - reference_roi[0], reference_roi[3] - reference_roi[1]
    records = [None] * len(gray_frames)
    for direction in (-1, 1):
        tracker = TemplateAnchorTracker(gray_frames[reference_frame], anchor_roi, min_confidence, search_margin)
        indices = range(reference_frame, -1, -1) if direction == -1 else range(reference_frame, len(gray_frames))
        for index in indices:
            record = tracker.update(gray_frames[index])
            record.update({"frame": index, "seconds": index / fps, "direction": direction})
            position = record["anchor_subpixel_xy"]
            if position is not None:
                x, y = position[0] + ox, position[1] + oy
                ix, iy = round(x), round(y)
                record["subpixel_xy"] = [x, y]
                record["bbox"] = [ix, iy, ix + bw, iy + bh]
            else:
                record["subpixel_xy"] = record["bbox"] = None
            records[index] = record
    raw_rejected = [r["frame"] for r in records if not r["accepted"]]
    interpolated = interpolate_short_gaps(records, (bw, bh), np.array([ox, oy]), max_gap=max_gap)
    x0, y0, x1, y1 = reference_roi
    patch = reference_bgr[y0:y1, x0:x1].copy()
    mask, mask_detail = reference_support(patch, mask_dilation)
    report = {
        "source": str(source), "frames": len(records), "fps": fps,
        "width": width, "height": height, "reference_frame": reference_frame,
        "reference_roi": reference_roi, "anchor_roi": anchor_roi,
        "mask_shape": list(mask.shape), "mask_details": mask_detail,
        "coordinates": "Source native pixels; bbox is xyxy with exclusive right/bottom edges",
        "mask_placement": "reference_mask[y,x] belongs at subpixel_xy + [x,y]. Use warpAffine translation of subpixel_xy, preserving fractional phase; bbox rounds that origin only.",
        "tracking_parameters": {"local_min_confidence": min_confidence, "global_min_confidence": max(.70, min_confidence), "search_margin": search_margin, "max_interpolation_gap": max_gap},
        "raw_rejected_frames": raw_rejected, "interpolated_frames": interpolated,
        "confidence_semantics": "Detector correlation, including on interpolated frames; acceptance distinguishes observed matches from motion interpolation",
        "rejected_frames": [r["frame"] for r in records if not r["accepted"]],
        "tracking_ms_mean": float(np.mean([r["tracking_ms"] for r in records])),
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    return report, patch, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--reference-frame", type=int, required=True)
    parser.add_argument("--roi", type=int, nargs=4, required=True, metavar=("X0", "Y0", "X1", "Y1"))
    parser.add_argument("--anchor-roi", type=int, nargs=4)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--min-confidence", type=float, default=.60)
    parser.add_argument("--search-margin", type=int, default=16)
    parser.add_argument("--mask-dilation", type=int, default=1)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report, patch, mask = track_video(args.source, args.reference_frame, args.roi,
                                     args.anchor_roi, args.frames, args.min_confidence,
                                     args.search_margin, args.mask_dilation, args.max_gap)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "moving_logo_track.json").write_text(json.dumps(report, indent=2))
    np.save(args.output_dir / "reference_mask.npy", mask)
    cv2.imwrite(str(args.output_dir / "reference_mask.png"), mask * 255)
    cv2.imwrite(str(args.output_dir / "reference_patch.png"), patch)
    summary = {key: value for key, value in report.items() if key != "records"}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
