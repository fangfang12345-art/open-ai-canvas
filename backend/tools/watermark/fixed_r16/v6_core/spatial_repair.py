"""Apply spatial repair only inside a supplied local watermark mask."""
import numpy as np

from spatial_fallback import spatial_fallback
from edge_continuity import edge_continuity
from repair_evidence import clean_evidence_mask, merge_generated
from affine_postprocess import repair as affine_repair


def local_spatial(base, roi, mask, margin=12, *, edge_enabled=True, diagnostics=False):
    x0, y0, x1, y1 = map(int, roi)
    h, w = base.shape[:2]
    xa, ya, xb, yb = max(0, x0-margin), max(0, y0-margin), min(w, x1+margin), min(h, y1+margin)
    context_mask = np.zeros((yb-ya, xb-xa), np.uint8)
    # Clip only the image border; the tracking coordinate contract remains intact.
    xx0, yy0, xx1, yy1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    context_mask[yy0-ya:yy1-ya, xx0-xa:xx1-xa] = mask[yy0-y0:yy1-y0, xx0-x0:xx1-x0]
    original_patch = base[ya:yb, xa:xb]
    exclusion, guard_radius = clean_evidence_mask(original_patch, context_mask)
    patch, flat, background_stats = spatial_fallback(
        original_patch, exclusion, write_mask=context_mask, diagnostics=True)
    edge_stats = dict(repaired_pixels=0, seam_pixels=0)
    if edge_enabled and np.any((context_mask > 0) & (flat == 0)):
        corrected, edge_mask, edge_stats = edge_continuity(original_patch, exclusion, patch)
        # The older flat branch has stronger evidence on uniform backgrounds.
        accepted = (edge_mask > 0) & (flat == 0) & (context_mask > 0)
        patch, seam_pixels = merge_generated(patch, corrected, accepted, context_mask,
                                             evidence_mask=exclusion)
        edge_stats = dict(edge_stats, repaired_pixels=int(accepted.sum()), seam_pixels=seam_pixels)
    # Independent clean evidence may correct old flat pixels too. Keep this
    # after every R02 generated-image layer so no older model overwrites it.
    patch, affine_stats = affine_repair(original_patch, exclusion, context_mask, patch)
    # Sampling uncertainty must not silently enlarge the authorized write mask.
    patch[context_mask == 0] = original_patch[context_mask == 0]
    edge_stats.update(evidence_guard_radius=guard_radius,
                      excluded_reference_pixels=int((exclusion & (context_mask == 0)).sum()),
                      background_continuity=background_stats,
                      affine_postprocess=affine_stats)
    result = base.copy()
    result[ya:yb, xa:xb] = patch
    if diagnostics:
        return result, int((flat & context_mask).sum()), edge_stats
    return result, int((flat & context_mask).sum())
