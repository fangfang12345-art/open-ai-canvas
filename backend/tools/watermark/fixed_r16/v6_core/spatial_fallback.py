"""Current-frame NS/flat base plus clean tensor screened harmonic continuation."""
from functools import lru_cache
import time
import cv2
import numpy as np
from robust_continuation import repair


@lru_cache(maxsize=4)
def _geometry(shape, mask_bytes):
    """Cache only immutable geometry; any mask or shape change is a new key."""
    mask = np.frombuffer(mask_bytes, np.uint8).reshape(shape)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    components = []
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
        rows, endpoints, endpoint_designs = [], [], []
        fill_rows, fill_columns, fill_run, blend = [], [], [], []
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
                fill_rows.extend([row]*len(run))
                fill_columns.extend(run)
                fill_run.extend([len(rows)]*len(run))
                blend.extend((run-left)/(right-left))
                rows.append(row)
                endpoints.append(endpoint_x)
                endpoint_designs.append(endpoint_design)
        if rows:
            components.append((yy, xx, design, np.asarray(rows), np.asarray(endpoints),
                               np.asarray(endpoint_designs), np.asarray(fill_rows),
                               np.asarray(fill_columns), np.asarray(fill_run), np.asarray(blend)[:, None]))
    return tuple(components)


def spatial_fallback(patch, mask, radius=3, *, write_mask=None, diagnostics=False):
    mask = (mask > 0).astype(np.uint8)
    result = patch.copy()
    flat_mask = np.zeros(mask.shape, np.uint8)
    geometry_mask = mask if write_mask is None else np.asarray(write_mask, np.uint8)
    for yy, xx, design, rows, endpoints, endpoint_design, fy, fx, runs, t in _geometry(mask.shape, geometry_mask.tobytes()):
        clean = mask[yy, xx] == 0
        if clean.sum() < 24:
            continue
        colors = patch[yy, xx].astype(np.float32)
        coefficients = np.linalg.lstsq(design[clean], colors[clean], rcond=None)[0]
        residual = np.max(np.abs(design[clean] @ coefficients - colors[clean]), axis=1)
        if np.percentile(residual, 95) > 3 or np.percentile(residual, 99) > 7:
            continue
        if np.max(np.abs(coefficients[1:])) > 15:
            continue
        endpoint_colors = patch[rows[:, None], endpoints].astype(np.float32)
        accepted = np.max(np.abs(endpoint_design @ coefficients - endpoint_colors), axis=(1, 2)) <= 7
        accepted &= ~np.any(mask[rows[:, None], endpoints], axis=1)
        a = np.median(endpoint_colors[:, :3], axis=1).astype(np.float64)
        b = np.median(endpoint_colors[:, 3:], axis=1).astype(np.float64)
        keep = accepted[runs]
        values = (1-t[keep])*a[runs[keep]] + t[keep]*b[runs[keep]]
        result[fy[keep], fx[keep]] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
        flat_mask[fy[keep], fx[keep]] = 1
    # NS sees the original patch and excludes all uncertain evidence. Existing
    # validated flat pixels become fixed vertices of the subsequent graph.
    solve_mask = mask
    started = time.perf_counter()
    if solve_mask.any():
        fallback = cv2.inpaint(patch, solve_mask.astype(np.uint8)*255, radius, cv2.INPAINT_NS)
        fallback[flat_mask > 0] = result[flat_mask > 0]
        result = fallback
    ns_seconds = time.perf_counter()-started
    result, statistics = repair(patch, mask, geometry_mask, result, flat_mask)
    statistics.update(skipped_ns_pixels=0,
                      ns_pixels=int(solve_mask.sum()), ns_seconds=ns_seconds,
                      blend_seconds=0.,
                      total_flat_pixels=int(flat_mask.sum()))
    if not np.array_equal(result[mask == 0], patch[mask == 0]):
        raise RuntimeError('Spatial correction escaped the supplied mask')
    if diagnostics:
        return result, flat_mask, statistics
    return result, flat_mask
