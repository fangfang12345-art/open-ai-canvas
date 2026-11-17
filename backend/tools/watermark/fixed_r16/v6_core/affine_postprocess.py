"""R07 clean-evidence dual-scale affine correction after the full R02 path.

Build with: python v6_core/affine_postprocess.py
Set R07_DISABLE_NATIVE=1 to use the independent small-example NumPy fallback.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
from _native import library_path
LIBRARY = library_path('native_affine')
try:
    NATIVE = ctypes.CDLL(str(LIBRARY))
    U8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    NATIVE.affine_predict.argtypes = [U8, U8, U8, ctypes.c_int, ctypes.c_int,
                                    ctypes.c_int, F64, F64, F64]
    NATIVE.affine_predict.restype = ctypes.c_int
except (OSError, AttributeError):
    NATIVE = None


def predict(image, exclusion, write, radius=None, native=True):
    started = time.perf_counter()
    image = np.ascontiguousarray(image, np.uint8)
    exclusion = np.ascontiguousarray(exclusion, np.uint8)
    write = np.ascontiguousarray(write, np.uint8)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != write.shape or exclusion.shape != write.shape:
        raise ValueError('Expected matching HWC color image and HW masks')
    if radius is None:
        distance = cv2.distanceTransform((write > 0).astype(np.uint8), cv2.DIST_L2, 5)
        radius = int(np.clip(np.ceil(3*np.percentile(distance[write > 0], 85)), 6, 18)) if write.any() else 6
    native = bool(native and NATIVE is not None and os.environ.get('R07_DISABLE_NATIVE') != '1')
    stats = dict(radius=int(radius), native=native, failure_fallbacks=0)
    prediction = np.zeros(image.shape, np.float64)
    alpha = np.zeros(write.shape, np.float64)
    if native:
        raw = np.zeros(12, np.float64)
        status = NATIVE.affine_predict(image, exclusion, write, *write.shape, int(radius), prediction, alpha, raw)
        if status:
            stats.update(failure_fallbacks=1, fallback_reason='Native status '+str(status))
            prediction.fill(0); alpha.fill(0)
        names = ('write_pixels', 'strong_gradient_pixels', 'structure_protected_pixels',
                 'small_scale_rejected_pixels', 'large_scale_rejected_pixels',
                 'scale_disagreement_pixels', 'accepted_pixels', 'full_weight_pixels',
                 'gradient_seconds', 'moment_seconds', 'fit_seconds')
        stats.update({name: float(value) if name.endswith('seconds') else int(value)
                      for name, value in zip(names, raw)})
    else:
        from affine_reference import reference_predict
        prediction, alpha, reference_stats = reference_predict(image, exclusion > 0, write > 0, int(radius))
        stats.update(reference_stats)
    stats['prediction_seconds'] = time.perf_counter()-started
    return prediction, alpha, stats


def repair(image, exclusion, write, base, acceleration=False):
    started = time.perf_counter()
    prediction, alpha, stats = predict(image, exclusion, write)
    output = base.copy()
    accepted = np.asarray(write, bool) & (alpha > 0)
    mixing = time.perf_counter()
    if accepted.any():
        weight = alpha[accepted, None]
        values = (1-weight)*base[accepted]+weight*prediction[accepted]
        output[accepted] = np.clip(np.rint(values), 0, 255).astype(np.uint8)
    stats.update(changed_pixels=int(np.any(output != base, axis=2)[np.asarray(write, bool)].sum()),
                 blend_seconds=time.perf_counter()-mixing, total_seconds=time.perf_counter()-started,
                 acceleration_requested=bool(acceleration), acceleration_effective=False,
                 skipped_ns_pixels=0, skipped_tensor_pixels=0, saved_seconds=0.,
                 acceleration_disabled_reason='No proof of exact NS and finite-budget global tensor equivalence')
    return output, stats


if __name__ == '__main__':
    flags = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
    subprocess.run(['c++', '-std=c++17', '-O3', '-ffp-contract=off', '-fno-fast-math', *flags,
                    str(HERE/'native_affine.cpp'), '-o', str(LIBRARY)], check=True)
    print(LIBRARY)
