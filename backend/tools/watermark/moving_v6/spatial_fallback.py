"""Conservative spatial fallback for a supplied watermark mask.

Horizontal interpolation is allowed only when the entire component boundary
agrees with a smooth color plane. It cannot reconstruct hidden text strokes.
"""
import cv2
import numpy as np


def spatial_fallback(patch, mask, radius=3):
    mask = (mask > 0).astype(np.uint8)
    result = cv2.inpaint(patch, mask * 255, radius, cv2.INPAINT_NS)
    flat_mask = np.zeros(mask.shape, np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    for component in range(1, count):
        x, y, cw, ch, area = stats[component]
        # A clipped boundary cannot establish that the surrounding content is flat.
        if x < 5 or y < 5 or x + cw + 5 > w or y + ch + 5 > h:
            continue
        region = (labels == component).astype(np.uint8)
        ring = (cv2.dilate(region, kernel) > 0) & (mask == 0)
        yy, xx = np.nonzero(ring)
        if len(xx) < 24:
            continue
        design = np.column_stack((np.ones(len(xx)), (xx-x)/max(cw, 1),
                                  (yy-y)/max(ch, 1))).astype(np.float32)
        colors = patch[yy, xx].astype(np.float32)
        coefficients = np.linalg.lstsq(design, colors, rcond=None)[0]
        residual = np.max(np.abs(design @ coefficients - colors), axis=1)
        if np.percentile(residual, 95) > 3 or np.percentile(residual, 99) > 7:
            continue
        # Do not turn a large edge or a steep brightness change into a smooth fill.
        if np.max(np.abs(coefficients[1:])) > 15:
            continue
        for row in range(y, y + ch):
            positions = np.flatnonzero(region[row])
            if positions.size == 0:
                continue
            for run in np.split(positions, np.flatnonzero(np.diff(positions) > 1) + 1):
                left, right = int(run[0])-1, int(run[-1])+1
                if left < 2 or right + 2 >= w:
                    continue
                if mask[row, left-2:left+1].any() or mask[row, right:right+3].any():
                    continue
                # Percentile checks can miss a short text stroke. Do not spread
                # any endpoint sample that disagrees with the surrounding plane.
                endpoint_x = np.array([left-2, left-1, left, right, right+1, right+2])
                endpoint_design = np.column_stack((np.ones(6), (endpoint_x-x)/max(cw, 1),
                                                   np.full(6, (row-y)/max(ch, 1))))
                endpoint_colors = patch[row, endpoint_x].astype(np.float32)
                if np.max(np.abs(endpoint_design @ coefficients - endpoint_colors)) > 7:
                    continue
                a = np.median(patch[row, left-2:left+1], axis=0)
                b = np.median(patch[row, right:right+3], axis=0)
                t = ((run - left) / (right-left))[:, None]
                values = (1-t)*a + t*b
                result[row, run] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
                flat_mask[row, run] = 1
    if not np.array_equal(result[mask == 0], patch[mask == 0]):
        raise RuntimeError('Spatial correction escaped the supplied mask')
    return result, flat_mask
