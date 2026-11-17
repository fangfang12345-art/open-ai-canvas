"""Spatial texture synthesis from immutable clean patches, never ground truth.

Only caller-authorized pixels are written. Synthesized pixels can constrain a
later seam with reduced weight, but can never enter the clean source library.
"""
import cv2
import numpy as np


def _validated_plane(image, component, exclusion, radius):
    """Protect smooth interpolation only after spatially held-out validation."""
    kernel = np.ones((2 * radius + 1,) * 2, np.uint8)
    ring = (cv2.dilate(component.astype(np.uint8), kernel) > 0) & ~exclusion
    yy, xx = np.nonzero(ring)
    if len(xx) < 48:
        return False
    cy, cx = np.mean(np.argwhere(component), axis=0)
    # All four surrounding quadrants must provide independent observations;
    # a one-sided image border cannot validate a plane across the whole hole.
    partitions = (xx >= cx).astype(int) + 2 * (yy >= cy)
    design = np.column_stack([np.ones(len(xx)), (xx-cx)/max(1, np.ptp(xx)),
                              (yy-cy)/max(1, np.ptp(yy))])
    colors = image[yy, xx].astype(np.float64)
    for quadrant in range(4):
        held = partitions == quadrant
        if held.sum() < 8 or (~held).sum() < 24:
            return False
        coefficients, _, rank, _ = np.linalg.lstsq(design[~held], colors[~held], rcond=None)
        if rank != 3:
            return False
        errors = np.abs(design[held] @ coefficients - colors[held])
        # Two levels allow integer quantization, not textured-model residuals.
        if np.percentile(errors, 95) > 2 or errors.max() > 4:
            return False
    return True


def synthesize(image, write_mask, evidence_mask, *, radius=5):
    image = np.asarray(image)
    write = np.asarray(write_mask, bool)
    exclusion = np.asarray(evidence_mask, bool) | write
    if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3
            or write.shape != image.shape[:2] or exclusion.shape != write.shape
            or radius < 1):
        raise ValueError('Expected a uint8 BGR patch, matching masks and positive radius')
    height, width = write.shape
    size = 2 * radius + 1
    output = image.copy()
    synthesized = np.zeros(write.shape, bool)
    report = dict(pixels=0, patches=0, source_patches=0, mean_match_rms=None,
                  max_match_rms=None, narrow_components=0, smooth_components=0,
                  limitation='Spatial exemplar texture is synthesized, not observed hidden background.')
    if min(height, width) < size or not write.any():
        return output, synthesized, report
    # Geometry follows missing observations, not the caller's remaining writes.
    # Temporal estimates remain excluded evidence: they cannot turn a large
    # unobserved hole into a narrow gap with known pixels on both sides.
    distance = cv2.distanceTransform(exclusion.astype(np.uint8), cv2.DIST_L2, 5)
    count, labels = cv2.connectedComponents(exclusion.astype(np.uint8), 8)
    sizes = np.bincount(labels[exclusion], minlength=count)
    interiors = np.bincount(labels[exclusion & (distance >= radius + 1)], minlength=count)
    requested = np.bincount(labels[write], minlength=count) > 0
    eligible = requested & (sizes > 0) & (interiors * 4 >= sizes)
    eligible[0] = False
    report['narrow_components'] = int((requested & ~eligible)[1:].sum())
    for component in np.flatnonzero(eligible):
        if _validated_plane(image, labels == component, exclusion, radius):
            eligible[component] = False
            report['smooth_components'] += 1
    fill = exclusion & eligible[labels]
    if not fill.any():
        return output, synthesized, report
    observed = ~exclusion
    safe = cv2.erode(observed.astype(np.uint8), np.ones((size, size), np.uint8),
                     borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    source_ok = safe[radius:height-radius, radius:width-radius]
    report['source_patches'] = int(source_ok.sum())
    if not source_ok.any():
        return output, synthesized, report
    source = np.where(observed[..., None], image, 0).astype(np.float32)
    # Advance through the whole unseen component in a private working image.
    # Internal synthesis cannot enter the immutable source library, and only
    # the caller's write mask is copied back at the end.
    unknown, costs = fill.copy(), []
    while unknown.any():
        # Generated RGB only receives a seam weight. The immutable source and
        # clean-evidence mask above never change as the front advances.
        confidence = observed.astype(np.float32) + synthesized.astype(np.float32) * .25
        boundary = unknown & (cv2.dilate((confidence > 0).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
        boundary[:radius] = boundary[-radius:] = False
        boundary[:, :radius] = boundary[:, -radius:] = False
        if not boundary.any():
            break
        coverage = cv2.boxFilter(confidence, -1, (size, size), normalize=False)
        coverage[~boundary] = -1
        y, x = np.unravel_index(np.argmax(coverage), coverage.shape)
        rows, columns = slice(y-radius, y+radius+1), slice(x-radius, x+radius+1)
        weights = confidence[rows, columns]
        if weights.sum() < 4:
            break
        template = output[rows, columns].astype(np.float32)
        # OpenCV SQDIFF squares mask values; sqrt supplies linear weights.
        match_mask = np.sqrt(weights)[..., None] * np.ones((1, 1, 3), np.float32)
        cost = cv2.matchTemplate(source, template, cv2.TM_SQDIFF, mask=match_mask)
        cost = np.maximum(cost, 0) / (weights.sum() * 3)
        cost[~source_ok] = np.inf
        sy, sx = np.unravel_index(np.argmin(cost), cost.shape)
        if not np.isfinite(cost[sy, sx]):
            break
        selected = unknown[rows, columns].copy()
        patch = source[sy:sy+size, sx:sx+size]
        output[rows, columns][selected] = patch[selected].astype(np.uint8)
        synthesized[rows, columns] |= selected
        unknown[rows, columns][selected] = False
        costs.append(float(np.sqrt(cost[sy, sx])))
    synthesized &= write
    result = image.copy()
    result[synthesized] = output[synthesized]
    report.update(pixels=int(synthesized.sum()), patches=len(costs),
                  mean_match_rms=float(np.mean(costs)) if costs else None,
                  max_match_rms=max(costs) if costs else None)
    return result, synthesized, report
