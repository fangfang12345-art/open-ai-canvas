"""Independent small-image NumPy oracle for the frozen R07 local affine rule.

This fallback deliberately computes explicit local designs and residuals instead
of sharing the native implementation's integral moments or analytic inverse.
It keeps no image state and has no real-time performance guarantee.
"""
import cv2
import numpy as np


def reference_predict(image, exclusion, write, radius):
    image = np.asarray(image, np.uint8)
    exclusion = np.asarray(exclusion, bool)
    write = np.asarray(write, bool)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('image must have shape H x W x 3')
    if exclusion.shape != image.shape[:2] or write.shape != exclusion.shape:
        raise ValueError('mask shape must match the image')
    if radius <= 0:
        raise ValueError('radius must be positive')
    height, width = exclusion.shape
    prediction = np.zeros(image.shape, np.float64)
    alpha = np.zeros(exclusion.shape, np.float64)
    stats = dict(write_pixels=int(write.sum()), protected_pixels=0,
                 unsupported_pixels=0, disagreement_pixels=0,
                 accepted_pixels=0, full_weight_pixels=0)
    if not write.any():
        return prediction, alpha, stats
    clean = ~exclusion
    reliable = cv2.erode(clean.astype(np.uint8), np.ones((3, 3), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    colors = image.astype(np.float64)
    gx, gy = np.zeros_like(colors), np.zeros_like(colors)
    gx[:, 1:-1] = (colors[:, 2:] - colors[:, :-2]) * .5
    gy[1:-1] = (colors[2:] - colors[:-2]) * .5
    strong = reliable & np.any(np.hypot(gx, gy) > 6., axis=2)

    def fit(x, y, span):
        x0, x1 = max(0, x-span), min(width, x+span+1)
        y0, y1 = max(0, y-span), min(height, y+span+1)
        selected = clean[y0:y1, x0:x1]
        sy, sx = np.nonzero(selected)
        count = len(sx)
        if count < .30 * selected.size:
            return None
        dx, dy = sx+x0-x, sy+y0-y
        for horizontal, vertical in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            if np.count_nonzero((dx*horizontal > 0) & (dy*vertical > 0)) < 8:
                return None
        coordinates = np.column_stack((dx/radius, dy/radius))
        centered = coordinates - coordinates.mean(axis=0)
        covariance = centered.T @ centered/count
        if np.linalg.eigvalsh(covariance)[0] < .08:
            return None
        design = np.column_stack((np.ones(count), coordinates))
        observed = colors[sy+y0, sx+x0]
        try:
            coefficient, _, rank, _ = np.linalg.lstsq(design, observed, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if rank != 3 or not np.isfinite(coefficient).all():
            return None
        rms = np.sqrt(np.mean((design @ coefficient-observed)**2, axis=0))
        point = coefficient[0]
        mean, deviation = observed.mean(axis=0), observed.std(axis=0)
        if not np.isfinite(rms).all() or np.max(rms) > 2.:
            return None
        if np.any(np.abs(point-mean) > 3*deviation+2):
            return None
        return point, float(np.max(rms))

    for y, x in zip(*np.nonzero(write)):
        x0, x1 = max(0, x-2*radius), min(width, x+2*radius+1)
        y0, y1 = max(0, y-2*radius), min(height, y+2*radius+1)
        if strong[y0:y1, x0:x1].any():
            stats['protected_pixels'] += 1
            continue
        small, large = fit(x, y, radius), fit(x, y, 2*radius)
        if small is None or large is None:
            stats['unsupported_pixels'] += 1
            continue
        difference = float(np.max(np.abs(small[0]-large[0])))
        if difference > 2.:
            stats['disagreement_pixels'] += 1
            continue
        weight = float(np.clip(2-max(small[1], large[1]), 0, 1)
                       * np.clip(2-difference, 0, 1))
        prediction[y, x] = small[0]
        alpha[y, x] = weight
    stats['accepted_pixels'] = int(np.count_nonzero(alpha > 0))
    stats['full_weight_pixels'] = int(np.count_nonzero(alpha == 1))
    return prediction, alpha, stats
