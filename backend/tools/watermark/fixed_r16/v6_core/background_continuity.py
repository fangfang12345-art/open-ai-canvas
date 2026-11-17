"""Local smooth-background models fitted only to observed clean pixels.

The partition follows agreement of overlapping color planes, not watermark
components. Every model must explain held-out observations and even isolated
visible outliers, so a thin true stroke cannot disappear in a percentile test.
"""
from functools import lru_cache
import time

import cv2
import numpy as np
import native_background


@lru_cache(maxsize=6)
def _geometry(shape, packed_mask):
    mask = np.frombuffer(packed_mask, np.uint8).reshape(shape) > 0
    h, w = shape
    if not mask.any():
        return (), np.zeros(shape, np.int32), np.array([h*w])
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    radius = int(np.clip(np.ceil(np.percentile(distance[mask], 85)*3.), 16, 64))
    step = max(8, radius//2)
    windows = []
    yy_target, xx_target = np.nonzero(mask)
    for cy in range(int(yy_target.min())//step*step, int(yy_target.max())+step, step):
        for cx in range(int(xx_target.min())//step*step, int(xx_target.max())+step, step):
            x0, x1 = max(0, cx-radius), min(w, cx+radius+1)
            y0, y1 = max(0, cy-radius), min(h, cy+radius+1)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            local = mask[y0:y1, x0:x1]
            target = local & (abs(xx-cx) <= step) & (abs(yy-cy) <= step)
            if not target.any():
                continue
            sy, sx = yy[~local], xx[~local]
            if len(sx) < 64:
                continue
            u, v = (sx-cx)/radius, (sy-cy)/radius
            if min(np.ptp(u), np.ptp(v)) < 1.2:
                continue
            design = np.column_stack((np.ones(len(sx)), u, v)).astype(np.float32)
            heldout = ((sx + sy*3) % 5 == 0)
            training = ~heldout
            if heldout.sum() < 12 or training.sum() < 36:
                continue
            matrix = design[training]
            gram = matrix.T @ matrix
            if np.linalg.cond(gram) > 40:
                continue
            projector = np.linalg.solve(gram, matrix.T).astype(np.float32)
            ty, tx = yy[target], xx[target]
            supported = ((tx >= sx.min()) & (tx <= sx.max()) &
                         (ty >= sy.min()) & (ty <= sy.max()))
            ty, tx = ty[supported], tx[supported]
            if len(tx) == 0:
                continue
            target_design = np.column_stack((np.ones(len(tx)), (tx-cx)/radius,
                                            (ty-cy)/radius)).astype(np.float32)
            weight = ((1.-abs(tx-cx)/(step+1.)) *
                      (1.-abs(ty-cy)/(step+1.))).astype(np.float32)
            windows.append((sy, sx, design, training, heldout, projector,
                            ty, tx, target_design, weight))
    # NS needs nearby observed pixels as well as its radius-three color stencil.
    # Only fully fitted, separated groups may bypass it. Partially fitted groups
    # retain the original solve, never using a new plane as an uncertain boundary.
    grouped = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8))
    _, groups = cv2.connectedComponents(grouped, 8)
    counts = np.bincount(groups[mask])
    return tuple(windows), groups, counts


def fit_background(patch, evidence_mask, write_mask=None):
    started = time.perf_counter()
    evidence = np.asarray(evidence_mask, bool)
    write = evidence if write_mask is None else np.asarray(write_mask, bool)
    windows, groups, group_counts = _geometry(write.shape, write.astype(np.uint8).tobytes())
    native = native_background.fit(patch, evidence, write, windows)
    if native is not None:
        result, accepted, uncertainty, statistics = native
        statistics['proposal_seconds'] = time.perf_counter()-started
        return result, accepted, uncertainty, groups, group_counts, statistics
    h, w = write.shape
    total = np.zeros((h, w, 3), np.float32)
    weights = np.zeros((h, w), np.float32)
    uncertainty = np.zeros((h, w), np.float32)
    low = np.full((h, w, 3), np.inf, np.float32)
    high = np.full((h, w, 3), -np.inf, np.float32)
    accepted_windows = 0
    for sy, sx, design, training, heldout, projector, ty, tx, target_design, weight in windows:
        usable = ~evidence[sy, sx]
        train = training & usable
        test = heldout & usable
        if train.sum() < 36 or test.sum() < 12:
            continue
        colors = patch[sy, sx].astype(np.float32)
        if usable.all():
            coefficients = projector @ colors[training]
        else:
            matrix = design[train]
            gram = matrix.T @ matrix
            if np.linalg.cond(gram) > 40:
                continue
            coefficients = np.linalg.solve(gram, matrix.T @ colors[train])
        residual = np.max(np.abs(design @ coefficients - colors), axis=1)
        # Check all visible samples as well as independent held-out samples.
        # This is deliberately stricter than accepting an average smooth patch.
        if (np.max(residual[usable]) > 7. or
                np.mean(residual[test]**2) > 2.25 or
                np.percentile(residual[usable], 95) > 3.):
            continue
        prediction = target_design @ coefficients
        if prediction.min() < -.5 or prediction.max() > 255.5:
            continue
        total[ty, tx] += prediction * weight[:, None]
        weights[ty, tx] += weight
        uncertainty[ty, tx] += float(np.sqrt(np.mean(residual[test]**2))) * weight
        low[ty, tx] = np.minimum(low[ty, tx], prediction)
        high[ty, tx] = np.maximum(high[ty, tx], prediction)
        accepted_windows += 1
    accepted = write & (weights > .05)
    accepted &= np.max(high-low, axis=2) <= 3.
    result = patch.copy()
    result[accepted] = np.clip(np.rint(total[accepted]/weights[accepted, None]), 0, 255).astype(np.uint8)
    uncertainty[accepted] /= weights[accepted]
    statistics = dict(filled_pixels=int(accepted.sum()), accepted_windows=accepted_windows,
                      candidate_windows=len(windows), proposal_seconds=time.perf_counter()-started,
                      backend='numpy')
    return result, accepted, uncertainty, groups, group_counts, statistics


def independent_ns_mask(mask, accepted, groups, group_counts):
    """Skip only entire isolated groups; keep partial-group NS evidence intact."""
    unknown = np.asarray(mask, bool)
    trusted = np.asarray(accepted, bool) & unknown
    # Evidence exclusion can add pixels not present in the cached write mask.
    # A group touching such pixels must still be solved in full.
    unavailable = unknown & ~trusted
    blocked = np.bincount(groups[unavailable], minlength=len(group_counts)) > 0
    if len(blocked):
        blocked[0] = True
    skipped = trusted & ~blocked[groups]
    return unknown & ~skipped, skipped
