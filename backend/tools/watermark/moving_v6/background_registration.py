"""Large-displacement background proposals, excluding both overlay supports.

These are registration hypotheses only. The caller still checks visible local
context and requires independent clean donors to agree before copying pixels.
Only OpenCV 4.8 APIs are used; no learned descriptors or external weights.
"""
import cv2
import numpy as np


def _visible(mask):
    return cv2.dilate(np.asarray(mask, np.uint8), np.ones((9, 9), np.uint8)) == 0


def descriptor_transforms(target, donor, target_mask, donor_mask):
    """Propose similarity transforms from mutually distinct SIFT observations."""
    sift = cv2.SIFT_create(nfeatures=800, contrastThreshold=.015, edgeThreshold=15)
    points, descriptors = [], []
    for image, mask in ((donor, donor_mask), (target, target_mask)):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        visible = _visible(mask)
        # A keypoint center alone is insufficient: its descriptor neighborhood
        # must not include the watermark, even when the center is outside it.
        distance = cv2.distanceTransform(visible.astype(np.uint8), cv2.DIST_L2, 5)
        gray = np.where(visible, gray, 0).astype(np.uint8)
        keys, features = sift.detectAndCompute(gray, visible.astype(np.uint8) * 255)
        selected = [index for index, key in enumerate(keys)
                    if distance[min(gray.shape[0] - 1, round(key.pt[1])),
                                min(gray.shape[1] - 1, round(key.pt[0]))] > key.size * 7 + 2]
        if len(selected) < 8:
            return []
        points.append(np.float32([keys[index].pt for index in selected]))
        descriptors.append(features[selected])
    matcher = cv2.BFMatcher()

    def unique(first, second):
        return {a.queryIdx: a.trainIdx for pair in matcher.knnMatch(first, second, k=2)
                if len(pair) == 2 for a, b in [pair] if a.distance < .7 * b.distance}

    forward = unique(descriptors[0], descriptors[1])
    backward = unique(descriptors[1], descriptors[0])
    matches = [(a, b) for a, b in forward.items() if backward.get(b) == a]
    if len(matches) < 8:
        return []
    source = np.float32([points[0][a] for a, _ in matches])
    destination = np.float32([points[1][b] for _, b in matches])
    matrix, flags = cv2.estimateAffinePartial2D(
        source, destination, method=cv2.RANSAC, ransacReprojThreshold=1.,
        maxIters=2000, confidence=.995, refineIters=10)
    if matrix is None or flags is None or int(flags.sum()) < 8:
        return []
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    angle = abs(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))))
    spread = np.ptp(source[flags[:, 0] != 0], axis=0)
    if not (.8 <= scale <= 1.25 and angle <= 12 and spread.min() >= 10 and spread.max() >= 35):
        return []
    return [matrix.astype(np.float32)]


def translation_transforms(target, donor, target_mask, donor_mask):
    """Masked photometric search for motion beyond the local flow capture range.

    Both masks move with their own image in each overlap score. The bounded
    search returns separated minima; repeated texture cannot itself authorize a
    repair, because these proposals go through the same corroboration checks.
    """
    height, width = target_mask.shape
    rx, ry = min(96, width // 3), min(64, height // 3)
    target_valid, donor_valid = [_visible(mask).astype(np.float32)
                               for mask in (target_mask, donor_mask)]
    target_gray, donor_gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
                              for image in (target, donor)]

    def correlate(first, second):
        padded = cv2.copyMakeBorder(second, ry, ry, rx, rx, cv2.BORDER_CONSTANT)
        return cv2.matchTemplate(padded, first, cv2.TM_CCORR)

    overlap = correlate(target_valid, donor_valid)
    first = correlate(target_valid * target_gray ** 2, donor_valid)
    second = correlate(target_valid, donor_valid * donor_gray ** 2)
    cross = correlate(target_valid * target_gray, donor_valid * donor_gray)
    cost = (first + second - 2 * cross) / np.maximum(overlap, 1)
    cost[overlap < max(500, min(target_valid.sum(), donor_valid.sum()) * .45)] = np.inf
    # A minimum on the search boundary only says the optimum may lie outside
    # the measured range. Sliding disagreeing structure out of the overlap can
    # otherwise look like a confident camera translation on a flat background.
    original_cost = cost.copy()
    best_y, best_x = np.unravel_index(np.argmin(cost), cost.shape)
    if best_y in (0, cost.shape[0]-1) or best_x in (0, cost.shape[1]-1):
        return []
    # Correlation subtraction in float32 has cancellation error on flat fields.
    # A numerically insignificant ripple is not a measured two-axis minimum.
    precision = (8 * np.finfo(np.float32).eps *
                 (np.abs(first) + np.abs(second) + 2 * np.abs(cross)) / np.maximum(overlap, 1))
    local_minimum = original_cost + precision < cv2.erode(original_cost, np.array(
        [[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8))
    local_minimum[[0, -1], :] = False
    local_minimum[:, [0, -1]] = False
    cost[~local_minimum] = np.inf
    proposals = []
    for _ in range(2):
        y, x = np.unravel_index(np.argmin(cost), cost.shape)
        if not np.isfinite(cost[y, x]):
            break
        offsets = []
        for axis in (0, 1):
            p, length = (x, cost.shape[1]) if axis == 0 else (y, cost.shape[0])
            offset = 0.
            if 0 < p < length - 1:
                before = original_cost[y, x - 1] if axis == 0 else original_cost[y - 1, x]
                after = original_cost[y, x + 1] if axis == 0 else original_cost[y + 1, x]
                curvature = before - 2 * original_cost[y, x] + after
                if np.isfinite(curvature) and curvature > 1e-8:
                    offset = float(np.clip(.5 * (before - after) / curvature, -.5, .5))
            offsets.append(offset)
        proposals.append(np.float32([[1, 0, rx - x - offsets[0]], [0, 1, ry - y - offsets[1]]]))
        cost[max(0, y - 6):y + 7, max(0, x - 6):x + 7] = np.inf
    return proposals
