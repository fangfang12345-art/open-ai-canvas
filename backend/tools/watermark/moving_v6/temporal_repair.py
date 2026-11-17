"""Conservative CPU temporal repair using clean, corroborated neighboring pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import cv2
import numpy as np


def _dilate(mask, radius):
    return cv2.dilate((mask != 0).astype(np.uint8),
                      np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)) != 0


def _warp(image, translation, *, nearest=False, border=0):
    height, width = image.shape[:2]
    translation = np.asarray(translation)
    matrix = (translation.astype(np.float32) if translation.shape == (2, 3)
              else np.float32([[1, 0, translation[0]], [0, 1, translation[1]]]))
    interpolation = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    if not nearest and image.ndim == 3 and image.dtype == np.uint8:
        interpolation = cv2.INTER_CUBIC
    return cv2.warpAffine(image, matrix, (width, height),
                          flags=interpolation,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def _alignment_metrics(target, donor, valid):
    if np.count_nonzero(valid) < 300:
        return None
    difference = np.mean(np.abs(target.astype(np.float32) - donor.astype(np.float32)), axis=2)
    values = difference[valid]
    return dict(median=float(np.median(values)), p90=float(np.percentile(values, 90)),
                mean=float(np.mean(values)), bad_fraction=float(np.mean(values > 18)))


def _cubic_source_exclusion(mask):
    """Cover extra cubic taps, including samples beyond the cropped image."""
    excluded = _dilate(mask, 1)
    excluded[:2, :] = excluded[-2:, :] = True
    excluded[:, :2] = excluded[:, -2:] = True
    return excluded


def _background_transforms(target, donor, target_mask, donor_mask):
    """Find up to two coherent content layers without using watermark features."""
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    donor_gray = cv2.cvtColor(donor, cv2.COLOR_BGR2GRAY)
    target_excluded = _dilate(target_mask, 4)
    donor_excluded = _dilate(donor_mask, 4)
    points = cv2.goodFeaturesToTrack(target_gray, 200, 0.01, 4,
                                    mask=(~target_excluded).astype(np.uint8) * 255, blockSize=5)
    if points is None or len(points) < 10:
        return []
    moved, status, _ = cv2.calcOpticalFlowPyrLK(
        target_gray, donor_gray, points, None, winSize=(19, 19), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, .02))
    if moved is None:
        return []
    reverse, back_status, _ = cv2.calcOpticalFlowPyrLK(
        donor_gray, target_gray, moved, None, winSize=(19, 19), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, .02))
    if reverse is None:
        return []
    original = points[:, 0]
    moved = moved[:, 0]
    height, width = target_mask.shape
    rounded = np.rint(moved).astype(int)
    valid = status[:, 0].astype(bool) & back_status[:, 0].astype(bool)
    valid &= np.linalg.norm(reverse[:, 0] - original, axis=1) < .8
    valid &= (rounded[:, 0] >= 0) & (rounded[:, 0] < width)
    valid &= (rounded[:, 1] >= 0) & (rounded[:, 1] < height)
    selected = np.flatnonzero(valid)
    valid[selected] &= ~donor_excluded[rounded[selected, 1], rounded[selected, 0]]
    src, dst = moved[valid], original[valid]
    proposals = []
    if len(src) >= 10:
        delta = dst - src
        median = np.median(delta, axis=0)
        inliers = np.linalg.norm(delta - median, axis=1) < .8
        if inliers.sum() >= 10 and inliers.mean() >= .65 and np.linalg.norm(median) <= 24:
            proposals.append(np.float32([[1, 0, median[0]], [0, 1, median[1]]]))
    remaining = np.ones(len(src), bool)
    for _ in range(2):
        if remaining.sum() < 10:
            break
        matrix, inlier_flags = cv2.estimateAffinePartial2D(
            src[remaining], dst[remaining], method=cv2.RANSAC,
            ransacReprojThreshold=1.0, maxIters=600, confidence=.99, refineIters=8)
        if matrix is None or inlier_flags is None:
            break
        inliers = inlier_flags[:, 0] != 0
        if inliers.sum() < 10:
            break
        scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
        angle = abs(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
        source_points = src[remaining][inliers]
        spread = np.ptp(source_points, axis=0)
        center = np.array([width/2, height/2, 1.])
        center_displacement = np.linalg.norm(matrix @ center - center[:2])
        if (.75 <= scale <= 1.33 and angle <= 12 and center_displacement <= 55
                and spread.min() >= 10 and spread.max() >= 35):
            proposals.append(matrix.astype(np.float32))
        positions = np.flatnonzero(remaining)
        remaining[positions[inliers]] = False
    return proposals


def _candidate(target, target_mask, donor, donor_mask, *, max_context_mae=6.0):
    excluded = _dilate(target_mask | donor_mask, 4)
    valid = ~excluded
    identity = _alignment_metrics(target, donor, valid)
    if identity is None:
        return None, {"reason": "insufficient_visible_context"}
    identity_matrix = np.float32([[1, 0, 0], [0, 1, 0]])
    proposals = [identity_matrix]
    # Identity wins ties, while coherent local scales can explain animated text.
    if identity["mean"] > 2.5:
        proposals += _background_transforms(target, donor, target_mask, donor_mask)
    best_image = donor.copy()
    best_score = np.full(target_mask.shape, np.inf, np.float32)
    best_clean = np.zeros(target_mask.shape, bool)
    model_reports = []
    identity_local = None
    for number, matrix in enumerate(proposals):
        aligned = donor if number == 0 else _warp(donor, matrix)
        # Cubic RGB reads 4x4 source samples. One source-pixel expansion before
        # linear mask warping covers those taps; target dilation adds a margin.
        aligned_mask = donor_mask if number == 0 else (
            _warp(_cubic_source_exclusion(donor_mask).astype(np.float32), matrix, border=1) > .01)
        valid_pixels = _warp(np.ones(target_mask.shape, np.uint8), matrix, nearest=True) != 0
        valid = ~_dilate(target_mask | aligned_mask, 4) & valid_pixels
        metrics = _alignment_metrics(target, aligned, valid)
        if metrics is None:
            continue
        difference = np.mean(abs(target.astype(np.float32) - aligned.astype(np.float32)), axis=2)
        count = cv2.boxFilter(valid.astype(np.float32), -1, (65, 65), normalize=False)
        total = cv2.boxFilter(difference * valid, -1, (65, 65), normalize=False)
        mean = total / np.maximum(1, count)
        local_bad = cv2.boxFilter(((difference > 18) & valid).astype(np.float32),
                                 -1, (65, 65), normalize=False) / np.maximum(1, count)
        if number == 0:
            identity_local = mean.copy()
        clean = (~_dilate(aligned_mask, 2) & valid_pixels & (count >= 180)
                 & (mean <= max_context_mae) & (local_bad <= .10))
        if number:
            clean &= mean <= identity_local * .75
        if metrics["median"] > 20 and metrics["bad_fraction"] > .65:
            clean[:] = False
        score = mean + number * .05
        take = clean & (score < best_score)
        best_image[take] = aligned[take]
        best_score[take] = score[take]
        best_clean |= take
        model_reports.append(dict(matrix=matrix.tolist(), context=metrics,
                                  eligible_pixels=int(np.count_nonzero(clean & target_mask))))
    count = int(np.count_nonzero(best_clean & target_mask))
    diagnostic = dict(identity_context=identity, models=model_reports,
                      reason="accepted_context" if count else "visible_content_changed", eligible_pixels=count)
    return ((best_image, best_clean) if count else None), diagnostic


def _donor_agreement(first, second, threshold, radius):
    """Require clean local shape agreement, not an isolated equal-valued pixel."""
    difference = np.max(np.abs(first["image"].astype(np.int16)
                               - second["image"].astype(np.int16)), axis=2).astype(np.float32)
    joint = first["clean"] & second["clean"]
    agreement = joint & (difference <= threshold)
    if radius:
        size = 2 * radius + 1
        valid = joint.astype(np.float32)
        count = cv2.boxFilter(valid, -1, (size, size), normalize=False)
        total = cv2.boxFilter(difference * valid, -1, (size, size), normalize=False)
        bad = cv2.boxFilter(((difference > threshold * 1.5) & joint).astype(np.float32),
                            -1, (size, size), normalize=False)
        supported = ((count >= max(9, size * size // 3))
                     & (total <= threshold * count) & (bad <= .10 * count))
        if radius >= 2:
            # A square mixes both sides of an anti-aliased stroke. A narrow
            # tangent strip is admissible only when both edge vectors agree.
            gradients = []
            for candidate in (first, second):
                if "gradient" not in candidate:
                    gray = cv2.cvtColor(candidate["image"], cv2.COLOR_BGR2GRAY).astype(np.float32)
                    gray = cv2.GaussianBlur(gray, (3, 3), .6)
                    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
                    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
                    candidate["gradient"] = gx, gy, np.hypot(gx, gy)
                gradients.append(candidate["gradient"])
            (ax, ay, an), (bx, by, bn) = gradients
            edge_consistent = ((np.minimum(an, bn) >= 8)
                               & (ax * bx + ay * by >= .98 * an * bn)
                               & (np.maximum(an, bn) <= 1.5 * np.minimum(an, bn)))
            normal = (np.rint(np.arctan2(ay + by, ax + bx) / (np.pi / 4)).astype(int) % 4)
            yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
            for direction in range(4):
                angle = direction * np.pi / 4
                kernel = (np.abs(xx * np.cos(angle) + yy * np.sin(angle)) <= 1.01).astype(np.float32)
                local_count = cv2.filter2D(valid, -1, kernel)
                local_total = cv2.filter2D(difference * valid, -1, kernel)
                local_bad = cv2.filter2D(((difference > threshold * 1.5) & joint).astype(np.float32),
                                        -1, kernel)
                supported |= ((normal == direction) & edge_consistent
                              & (local_count >= max(9, int(kernel.sum()) // 2))
                              & (local_total <= threshold * local_count)
                              & (local_bad <= .10 * local_count))
        agreement &= supported
    return agreement


def repair_patch(target_patch, target_mask, donors, *, min_donors=2,
                 max_context_mae=6.0, agreement_threshold=8.0, agreement_radius=3):
    """Return a repaired BGR patch and diagnostics containing recovered_mask.

    Each donor is {image: same-size BGR array, mask: own overlay bool array,
    index: frame identifier, distance: signed distance from target, cut_ok: bool}.
    Rejected pixels retain the target unchanged. No inpainting fallback occurs.
    """
    if min_donors < 2:
        raise ValueError("At least two independent donors are required")
    target_mask = np.asarray(target_mask) != 0
    if target_patch.shape[:2] != target_mask.shape:
        raise ValueError("Target mask/image dimensions disagree")
    started = time.perf_counter()
    candidates, diagnostics = [], []
    for donor in sorted(donors, key=lambda item: abs(item.get("distance", 0))):
        info = {"index": int(donor["index"]), "distance": int(donor.get("distance", 0))}
        if not donor.get("cut_ok", True):
            diagnostics.append(dict(info, reason="scene_cut"))
            continue
        if donor["image"].shape != target_patch.shape or donor["mask"].shape != target_mask.shape:
            raise ValueError("Donor arrays must match target patch dimensions")
        candidate, assessment = _candidate(target_patch, target_mask, donor["image"],
                                            np.asarray(donor["mask"]) != 0,
                                            max_context_mae=max_context_mae)
        diagnostics.append(dict(info, **assessment))
        if candidate is not None:
            image, clean = candidate
            candidates.append(dict(image=image, clean=clean, **info))
    result = target_patch.copy()
    recovered = np.zeros(target_mask.shape, bool)
    yy, xx = np.nonzero(target_mask)
    target_count = len(yy)
    votes_used = []
    if len(candidates) >= min_donors and target_count:
        pair_agreement = {}
        for first_id, first in enumerate(candidates):
            votes = first["clean"][yy, xx].astype(np.int16)
            independent = np.zeros(target_count, bool)
            for second_id, second in enumerate(candidates):
                if first is second or first["index"] == second["index"]:
                    continue
                # Adjacent codec frames alone are weak corroboration.
                if abs(first["index"] - second["index"]) < 3:
                    continue
                key = tuple(sorted((first_id, second_id)))
                if key not in pair_agreement:
                    pair_agreement[key] = _donor_agreement(
                        first, second, agreement_threshold, agreement_radius)[yy, xx]
                match = pair_agreement[key]
                votes += match
                independent |= match
            accepted = ((votes >= min_donors) & independent & first["clean"][yy, xx]
                        & ~recovered[yy, xx])
            if accepted.any():
                result[yy[accepted], xx[accepted]] = first["image"][yy[accepted], xx[accepted]]
                recovered[yy[accepted], xx[accepted]] = True
                votes_used.append({"donor": first["index"], "pixels": int(accepted.sum())})
    return result, dict(recovered_mask=recovered, coverage=float(recovered.sum() / max(1, target_count)),
                        recovered_pixels=int(recovered.sum()), target_pixels=target_count,
                        candidates=diagnostics, donor_usage=votes_used,
                        repair_seconds=time.perf_counter() - started,
                        limitation="Unseen changes wholly hidden under the target overlay cannot be ruled out; matching donors are corroboration, not ground truth.")


