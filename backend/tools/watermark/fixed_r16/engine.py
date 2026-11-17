"""Generic input preparation around the original R16 core."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import cv2
import numpy as np

from media import Reader
from .calibrate_fixed import compile_presence_geometry, detect_persistent_light, frame_mask

sys.path.insert(0, str(Path(__file__).resolve().parent / "v6_core"))
from spatial_repair import local_spatial


def repair_task(frame, box, mask):
    cv2.setNumThreads(1)
    statistics = {}
    if mask.any():
        result, _, statistics = local_spatial(frame, box, mask.astype(np.uint8), edge_enabled=True, diagnostics=True)
    else:
        result = frame.copy()
    x0, y0, x1, y1 = box
    changed = np.any(frame != result, axis=2)
    if changed.sum() != (changed[y0:y1, x0:x1] & mask).sum():
        raise RuntimeError("R16 repair escaped its verified mask")
    solver = statistics.get("background_continuity", {})
    return result, int(changed.sum()), int(mask.sum()), dict(
        robustNativeFrames=int(solver.get("native", False)),
        robustNumpyFrames=int(solver.get("robust_pcg_rounds", 0) == 4 and not solver.get("native", False)),
        robustFallbackFrames=int(solver.get("failure_fallbacks", 0)),
        affineFallbackFrames=int(statistics.get("affine_postprocess", {}).get("failure_fallbacks", 0)))


class FixedRepair:
    def __init__(self, source, info, first, count, roi, progress):
        reader = Reader(source)
        try:
            x0, y0, x1, y1 = roi
            sample_limit = min(48, max(3, 24000000 // ((x1 - x0) * (y1 - y0))))
            indices = np.unique(np.linspace(first, first + count - 1, min(count, sample_limit)).round().astype(int))
            if len(indices) < 3:
                raise ValueError("R16 requires at least three calibration frames")
            samples = []
            for offset, index in enumerate(indices):
                samples.append(reader.frame(int(index))[y0:y1, x0:x1].copy())
                progress(5 + 25 * (offset + 1) / len(indices), "calibrating_fixed_mask")
            support, core, _, self.statistics = detect_persistent_light(np.stack(samples))
            yy, xx = np.nonzero(support)
            left, top = max(0, xx.min() - 10), max(0, yy.min() - 10)
            right, bottom = min(x1 - x0, xx.max() + 11), min(y1 - y0, yy.max() + 11)
            local = np.s_[top:bottom, left:right]
            self.model = dict(source_width=info["width"], source_height=info["height"],
                              roi=[x0 + int(left), y0 + int(top), x0 + int(right), y0 + int(bottom)],
                              mask=support[local], core=core[local],
                              presence_gate=dict(brightness=150., max_chroma=35., min_fraction=.85,
                                                 min_contrast=20., ring_radius=8))
            geometry = compile_presence_geometry(self.model["mask"], self.model["core"])
            self.model.update(mask=geometry.support, core=geometry.core, _presence_geometry=geometry)
        finally:
            reader.close()
        self.changed = self.targeted = 0
        self.solver_totals = dict(robustNativeFrames=0, robustNumpyFrames=0, robustFallbackFrames=0, affineFallbackFrames=0)
        progress(40, "fixed_mask_ready")

    def frames(self, reader, first, count, workers):
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = deque()

            def finish():
                future, original, bounds = pending.popleft()
                patch, changed, targeted, solvers = future.result()
                xa, ya, xb, yb = bounds
                original[ya:yb, xa:xb] = patch
                self.changed += changed
                self.targeted += targeted
                for key, value in solvers.items():
                    self.solver_totals[key] += value
                return original

            for index in range(first, first + count):
                original = reader.frame(index)
                box, mask, _ = frame_mask(original, self.model)
                x0, y0, x1, y1 = box
                xa, ya = max(0, x0 - 12), max(0, y0 - 12)
                xb, yb = min(original.shape[1], x1 + 12), min(original.shape[0], y1 + 12)
                local_roi = (x0 - xa, y0 - ya, x1 - xa, y1 - ya)
                future = pool.submit(repair_task, np.ascontiguousarray(original[ya:yb, xa:xb]), local_roi, mask)
                pending.append((future, original, (xa, ya, xb, yb)))
                if len(pending) >= workers * 2:
                    yield finish()
            while pending:
                yield finish()

    def report(self):
        return dict(algorithm="R16 four-round robust continuation + edge continuity + affine postprocess",
                    changedPixels=self.changed, targetPixels=self.targeted, calibration=self.statistics,
                    solverUsage=self.solver_totals,
                    limitations=["Stationary bright neutral overlays only; persistent bright scene detail can be ambiguous.",
                                 "Opaque hidden content is estimated spatially, not recovered as ground truth."])

    def close(self):
        pass
