"""Bounded reference exclusion and agreement-checked generated-image seams."""
from functools import lru_cache

import cv2
import numpy as np


@lru_cache(maxsize=8)
def evidence_geometry(shape, packed_mask):
    write_mask = np.frombuffer(packed_mask, np.uint8).reshape(shape) > 0
    if not write_mask.any():
        return write_mask, write_mask, 0
    distance = cv2.distanceTransform(write_mask.astype(np.uint8), cv2.DIST_L2, 5)
    # The guard scales with stroke width, stays narrow, and never authorizes
    # writing to extra pixels. It separates uncertain samples from target pixels.
    radius = int(np.clip(np.rint(np.percentile(distance[write_mask], 85) * .15), 1, 2))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    guard = cv2.dilate(write_mask.astype(np.uint8), kernel) > 0
    return write_mask, guard, radius


def clean_evidence_mask(patch, mask):
    """Exclude bright fringe outliers without discarding an entire clean ring.

    The existing detector targets light overlays. An external pixel is suspect
    only when it is near that target, bright relative to its observed core, and
    brighter than every clean nearby sample. A genuine continued light edge has
    equally bright outer support and is therefore kept as evidence.
    """
    write_mask, guard, radius = evidence_geometry(mask.shape, mask.tobytes())
    if not write_mask.any():
        return write_mask, radius
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    reference = np.where(~guard, gray, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * (radius + 3) + 1,) * 2)
    outer_max = cv2.dilate(reference, kernel).astype(np.int16)
    outer_count = cv2.dilate((~guard).astype(np.uint8), kernel) > 0
    brightness = max(128., float(np.percentile(gray[write_mask], 75)) * .70)
    suspect = guard & ~write_mask & outer_count & (gray >= brightness)
    suspect &= gray.astype(np.int16) > outer_max + 6
    return write_mask | suspect, radius


def merge_generated(base, candidate, accepted, write_mask, width=2, *, evidence_mask=None):
    """Soften only compatible model joins, retaining sharp or uncertain contours.

    Both images must already be generated repairs. Original masked pixels must
    never be used as a feathering endpoint. A disagreement about edge position
    rejects blending instead of averaging two differently located contours.
    """
    accepted = np.asarray(accepted, bool) & np.asarray(write_mask, bool)
    result = base.copy()
    result[accepted] = candidate[accepted]
    if not accepted.any():
        return result, 0
    other_model = np.asarray(write_mask, bool) & ~accepted
    if not other_model.any():
        return result, 0
    # A model joining the observed image is already independently validated.
    # Reconcile only internal joins between two generated repairs, not every
    # watermark boundary, where blending would unnecessarily weaken a good fit.
    distance = cv2.distanceTransform((~other_model).astype(np.uint8), cv2.DIST_L2, 5)
    seam = accepted & (distance <= width)
    if not seam.any():
        return result, 0
    a, b = base.astype(np.float32), candidate.astype(np.float32)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    ax = cv2.Sobel(ga, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    ay = cv2.Sobel(ga, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    bx = cv2.Sobel(gb, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    by = cv2.Sobel(gb, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    # The candidate may contain a hard switch at its support boundary. Compare
    # only broad gradient disagreements and reject high contrast joins entirely.
    magnitude = np.maximum(np.hypot(ax, ay), np.hypot(bx, by))
    protected_edge = cv2.dilate((magnitude > 12).astype(np.uint8),
                               np.ones((2 * width + 3,) * 2, np.uint8)) > 0
    compatible = (np.max(np.abs(a-b), axis=2) <= 18)
    compatible &= (np.hypot(ax-bx, ay-by) <= 5) & ~protected_edge
    blend = seam & compatible
    weight = np.clip(distance / (width + 1), 0., 1.)
    if not blend.any():
        return result, 0
    proposal = a * (1 - weight[..., None]) + b * weight[..., None]
    # Agreement between two generated images alone does not establish quality:
    # both can be wrong. Require an independently observed locally flat ring and
    # an improvement towards that ring before softening a model boundary.
    unknown = np.asarray(evidence_mask if evidence_mask is not None else write_mask, bool)
    known = (~unknown).astype(np.float32)
    span = 2 * (width + 4) + 1
    kernel = (span, span)
    support = cv2.boxFilter(known, cv2.CV_32F, kernel, normalize=True)
    count = np.maximum(support, 1e-6)[..., None]
    mean = cv2.boxFilter(a * known[..., None], cv2.CV_32F, kernel, normalize=True) / count
    square = cv2.boxFilter(a * a * known[..., None], cv2.CV_32F, kernel, normalize=True) / count
    variance = np.maximum(square - mean * mean, 0.)
    supported = (support >= .2) & (np.max(variance, axis=2) <= 9.)
    improvement = (np.max(np.abs(b - mean), axis=2)
                   - np.max(np.abs(proposal - mean), axis=2)) >= .5
    blend &= supported & improvement
    result[blend] = np.clip(np.rint(proposal[blend]), 0, 255).astype(np.uint8)
    return result, int(blend.sum())
