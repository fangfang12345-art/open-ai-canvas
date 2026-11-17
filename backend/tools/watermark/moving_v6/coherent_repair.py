"""Corroborated temporal repair with coherent, unblended source labels.

The caller supplies independently tracked donor masks. No pixel under an
unverified mask is evidence; source-frame agreement is not hidden-text truth.
"""
from __future__ import annotations

import sys
import time

sys.dont_write_bytecode = True

import cv2
import numpy as np

from .background_registration import descriptor_transforms, translation_transforms

from .temporal_repair import (
    _alignment_metrics, _background_transforms, _cubic_source_exclusion,
    _dilate, _donor_agreement, _warp,
)


def candidate_models(target, target_mask, donor, donor_mask, *, max_context_mae=6.0):
    """Keep transforms separate so one donor cannot synthesize mixed strokes."""
    valid = ~_dilate(target_mask | donor_mask, 4)
    identity = _alignment_metrics(target, donor, valid)
    if identity is None:
        return [], {"reason": "insufficient_visible_context", "models": []}
    proposals = [np.float32([[1, 0, 0], [0, 1, 0]])]
    if identity["mean"] > 2.5:
        proposals.extend(_background_transforms(target, donor, target_mask, donor_mask))
        proposals.extend(descriptor_transforms(target, donor, target_mask, donor_mask))
        proposals.extend(translation_transforms(target, donor, target_mask, donor_mask))
    models, reports = [], []
    identity_local = None
    for model_id, matrix in enumerate(proposals):
        aligned = donor if model_id == 0 else _warp(donor, matrix)
        aligned_mask = donor_mask if model_id == 0 else (
            _warp(_cubic_source_exclusion(donor_mask).astype(np.float32), matrix, border=1) > .01)
        inside = _warp(np.ones(target_mask.shape, np.uint8), matrix, nearest=True) != 0
        valid = ~_dilate(target_mask | aligned_mask, 4) & inside
        metrics = _alignment_metrics(target, aligned, valid)
        if metrics is None:
            continue
        difference = np.mean(abs(target.astype(np.float32) - aligned.astype(np.float32)), axis=2)
        count = cv2.boxFilter(valid.astype(np.float32), -1, (65, 65), normalize=False)
        mean = cv2.boxFilter(difference * valid, -1, (65, 65), normalize=False) / np.maximum(1, count)
        bad = cv2.boxFilter(((difference > 18) & valid).astype(np.float32),
                            -1, (65, 65), normalize=False) / np.maximum(1, count)
        if model_id == 0:
            identity_local = mean.copy()
        clean = (~_dilate(aligned_mask, 2) & inside & (count >= 180)
                 & (mean <= max_context_mae) & (bad <= .10))
        if model_id:
            if identity_local is None:
                clean[:] = False
            else:
                clean &= mean <= identity_local * .75
        if metrics["median"] > 20 and metrics["bad_fraction"] > .65:
            clean[:] = False
        eligible = int(np.count_nonzero(clean & target_mask))
        reports.append(dict(model_id=model_id, matrix=matrix.tolist(), context=metrics,
                            eligible_pixels=eligible))
        if eligible:
            models.append(dict(image=aligned, clean=clean, score=mean + model_id * .05,
                               model_id=model_id, matrix=matrix))
    return models, dict(identity_context=identity, models=reports,
                        eligible_pixels=max((r["eligible_pixels"] for r in reports), default=0),
                        reason="accepted_context" if models else "visible_content_changed")


def coherent_labels(costs, eligible, *, coherence_weight=.8, iterations=3):
    """Choose trusted sources with local Potts continuity, never blend RGB.

    A source cannot gain eligibility through neighboring labels. Checkerboard
    updates decrease the four-neighbor label energy rather than oscillating
    two competing checkerboard patterns under simultaneous updates.
    """
    costs = np.asarray(costs, np.float32)
    eligible = np.asarray(eligible, bool)
    if costs.ndim != 3 or eligible.shape != costs.shape:
        raise ValueError("Costs and eligibility must both be K x H x W")
    if not np.isfinite(costs[eligible]).all():
        raise ValueError("Eligible source costs must be finite")
    if not len(costs):
        return np.full(costs.shape[1:], -1, np.int32)
    if coherence_weight < 0 or iterations < 0:
        raise ValueError("Continuity weight and iterations must be nonnegative")
    bounded = np.where(eligible, costs, np.inf)
    labels = np.argmin(bounded, axis=0).astype(np.int32)
    active = eligible.any(axis=0)
    labels[~active] = -1
    yy, xx = np.indices(active.shape)
    kernel = np.float32([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    degree = cv2.filter2D(active.astype(np.float32), -1, kernel, borderType=cv2.BORDER_CONSTANT)
    for _ in range(iterations):
        changed = False
        for parity in (0, 1):
            scores = bounded.copy()
            for index in range(len(costs)):
                same = cv2.filter2D((labels == index).astype(np.float32), -1, kernel,
                                    borderType=cv2.BORDER_CONSTANT)
                scores[index] += coherence_weight * (degree - same)
            new = np.argmin(scores, axis=0).astype(np.int32)
            selected = active & ((xx + yy) % 2 == parity)
            changed |= bool(np.any(labels[selected] != new[selected]))
            labels[selected] = new[selected]
        if not changed:
            break
    return labels


def repair_patch(target_patch, target_mask, donors, *, min_donors=2,
                 max_context_mae=6.0, agreement_threshold=8.0, agreement_radius=3,
                 coherence_weight=.8, coherence_iterations=3):
    """Copy corroborated candidate pixels inside the supplied target mask only.

    Donors follow the V4 contract: image, mask, index, distance, cut_ok. A
    missing mask is unknown, and is rejected. Multiple transformations of one
    source frame count as one witness, not multiple independent observations.
    """
    if min_donors < 2:
        raise ValueError("At least two distinct source frames are required")
    target_mask = np.asarray(target_mask) != 0
    if target_patch.ndim != 3 or target_patch.shape[2] != 3 or target_patch.shape[:2] != target_mask.shape:
        raise ValueError("Target must be a BGR image with an H x W mask")
    started = time.perf_counter()
    candidates, diagnostics = [], []
    for donor in sorted(donors, key=lambda item: (abs(item.get("distance", 0)), int(item["index"]))):
        info = {"index": int(donor["index"]), "distance": int(donor.get("distance", 0))}
        if not donor.get("cut_ok", True):
            diagnostics.append(dict(info, reason="scene_cut"))
            continue
        if donor.get("mask") is None:
            diagnostics.append(dict(info, reason="unknown_donor_mask"))
            continue
        if donor["image"].shape != target_patch.shape or np.asarray(donor["mask"]).shape != target_mask.shape:
            raise ValueError("Donor arrays must match target patch dimensions")
        models, assessment = candidate_models(target_patch, target_mask, donor["image"],
                                               np.asarray(donor["mask"]) != 0,
                                               max_context_mae=max_context_mae)
        diagnostics.append(dict(info, **assessment))
        candidates.extend(dict(model, **info) for model in models)
    result = target_patch.copy()
    source_map = np.full(target_mask.shape, -1, np.int32)
    transform_map = np.full(target_mask.shape, -1, np.int16)
    label_map = np.full(target_mask.shape, -1, np.int32)
    recovered = np.zeros(target_mask.shape, bool)
    usage, comparisons = [], 0
    yy, xx = np.nonzero(target_mask)
    if len(candidates) >= min_donors and len(yy):
        # Work on the actual hole's bounding rectangle, not the context crop.
        y0, y1, x0, x1 = int(yy.min()), int(yy.max())+1, int(xx.min()), int(xx.max())+1
        region = np.s_[y0:y1, x0:x1]
        local_target = target_mask[region]
        witnesses = [{} for _ in candidates]
        for first_id, first in enumerate(candidates):
            for second_id in range(first_id + 1, len(candidates)):
                second = candidates[second_id]
                if abs(first["index"] - second["index"]) < 3:
                    continue
                # Different transforms with no common trusted hole pixels
                # cannot corroborate each other and need no image comparison.
                if not np.any(first["clean"][region] & second["clean"][region] & local_target):
                    continue
                match = _donor_agreement(first, second, agreement_threshold, agreement_radius)[region]
                comparisons += 1
                for candidate_id, witness_index in ((first_id, second["index"]), (second_id, first["index"])):
                    old = witnesses[candidate_id].get(witness_index)
                    witnesses[candidate_id][witness_index] = match.copy() if old is None else old | match
        eligible, costs = [], []
        for candidate_id, candidate in enumerate(candidates):
            votes = np.zeros(local_target.shape, np.int16)
            for match in witnesses[candidate_id].values():
                votes += match
            trusted = local_target & candidate["clean"][region] & (votes >= min_donors-1)
            eligible.append(trusted)
            # Visible-context fit dominates a modest preference for nearby
            # frames. A far reference can win when it explains content better.
            costs.append(candidate["score"][region] + .002 * min(abs(candidate["distance"]), 300))
        labels = coherent_labels(np.stack(costs), np.stack(eligible),
                                 coherence_weight=coherence_weight, iterations=coherence_iterations)
        label_map[region] = labels
        for candidate_id, candidate in enumerate(candidates):
            selected = labels == candidate_id
            if not selected.any():
                continue
            result[region][selected] = candidate["image"][region][selected]
            source_map[region][selected] = candidate["index"]
            transform_map[region][selected] = candidate["model_id"]
            recovered[region][selected] = True
            usage.append(dict(donor=candidate["index"], model_id=candidate["model_id"],
                              pixels=int(selected.sum()), distance=candidate["distance"]))
    if not np.array_equal(result[~target_mask], target_patch[~target_mask]):
        raise RuntimeError("Temporal repair escaped the target mask")
    boundaries = int(np.count_nonzero((label_map[:, 1:] != label_map[:, :-1])
                                      & recovered[:, 1:] & recovered[:, :-1])
                     + np.count_nonzero((label_map[1:] != label_map[:-1])
                                        & recovered[1:] & recovered[:-1]))
    return result, dict(recovered_mask=recovered, source_map=source_map, transform_map=transform_map,
                        coverage=float(recovered.sum()/max(1, target_mask.sum())),
                        recovered_pixels=int(recovered.sum()), target_pixels=int(target_mask.sum()),
                        candidates=diagnostics, donor_usage=usage, model_candidates=len(candidates),
                        pair_comparisons=comparisons, source_boundaries=boundaries,
                        repair_seconds=time.perf_counter()-started,
                        limitation="Corroborated frames cannot prove unseen changes beneath an opaque overlay.")
