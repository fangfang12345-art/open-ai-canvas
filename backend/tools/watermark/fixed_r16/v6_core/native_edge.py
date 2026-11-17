"""Optional native loops; CDLL releases the GIL for every kernel invocation.

Run this file to build with the local C++ compiler. An absent or incompatible
library uses the unchanged NumPy implementation in edge_continuity instead.
"""
import ctypes
import os
import operator
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
from _native import library_path
LIBRARY = library_path('native_edge')
lib = None
if os.environ.get('V6_DISABLE_NATIVE') != '1':
    try:
        lib = ctypes.CDLL(str(LIBRARY))
    except OSError:
        pass

F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
I64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
BOOL = np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS')
U8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
bounded_kernel = None
if lib is not None:
    lib.edge_profile.argtypes = [F64, F64, F64, ctypes.c_int64, ctypes.c_int, F64, F64]
    lib.edge_profile.restype = None
    lib.edge_direction_variance.argtypes = [F64, F64, F64, F64, ctypes.c_int64,
                                            ctypes.c_double, ctypes.c_double, F64, ctypes.c_int]
    lib.edge_direction_variance.restype = ctypes.c_int
    lib.edge_sample.argtypes = [F64, ctypes.c_int, F64, ctypes.c_int64, F64]
    lib.edge_sample.restype = None
    lib.edge_bracket.argtypes = [I64, F64, BOOL, ctypes.c_int64, ctypes.c_int, F64, F64]
    lib.edge_bracket.restype = None
    # Older optional libraries still provide all existing edge kernels.
    if hasattr(lib, 'edge_bounded'):
        bounded_kernel = lib.edge_bounded
        bounded_kernel.argtypes = [U8, U8] + [ctypes.c_int64]*6 + [U8]
        bounded_kernel.restype = ctypes.c_int


def profile(u, colors, weights, bins):
    result, support = np.empty((bins, 3)), np.empty(bins)
    lib.edge_profile(np.ascontiguousarray(u, dtype=np.float64),
                     np.ascontiguousarray(colors, dtype=np.float64),
                     np.ascontiguousarray(weights, dtype=np.float64),
                     u.size, bins, result, support)
    return result, support


def direction_scorer(x, y, values, weights):
    x, y, values, weights = [np.ascontiguousarray(a, dtype=np.float64)
                             for a in (x, y, values, weights)]
    capacity = int(np.ceil((np.ptp(x)+np.ptp(y))*2))+4
    variance = np.empty(capacity)

    def score(angle):
        bins = lib.edge_direction_variance(x, y, values, weights, x.size,
                                           np.cos(angle), np.sin(angle), variance, capacity)
        if bins < 1:
            raise RuntimeError('Native direction score exceeded its geometric bound')
        # Keep NumPy's original pairwise reduction and exact occupied bin count.
        return float(np.sum(variance[:bins]))
    return score


def sample(profile, u):
    result = np.empty((*u.shape, 3))
    lib.edge_sample(np.ascontiguousarray(profile), len(profile),
                    np.ascontiguousarray(u), u.size, result)
    return result


def bracket(indices, tangent, observed, bins):
    lower, upper = np.empty(bins), np.empty(bins)
    lib.edge_bracket(np.ascontiguousarray(indices, dtype=np.int64),
                     np.ascontiguousarray(tangent), np.ascontiguousarray(observed),
                     indices.size, bins, lower, upper)
    return lower, upper


def bounded(original, result, roi, mask):
    """Compare every pixel outside the supplied ROI mask, without frame copies.

    Return None when the optional native kernel is unavailable. Invalid shapes,
    types, strides or coordinates raise ValueError before any native access.
    """
    if (not isinstance(original, np.ndarray) or not isinstance(result, np.ndarray)
            or original.dtype != np.uint8 or result.dtype != np.uint8
            or original.ndim != 3 or original.shape[-1] != 3
            or original.shape != result.shape
            or not original.flags.c_contiguous or not result.flags.c_contiguous):
        raise ValueError('Bounded check requires matching contiguous uint8 BGR frames')
    height, width = original.shape[:2]
    try:
        x0, y0, x1, y1 = (operator.index(value) for value in roi)
    except (TypeError, ValueError):
        raise ValueError('ROI must contain four integer coordinates') from None
    if not (height > 0 and width > 0 and 0 <= x0 <= x1 <= width
            and 0 <= y0 <= y1 <= height):
        raise ValueError('ROI must lie inside the image')
    if (not isinstance(mask, np.ndarray) or mask.shape != (y1-y0, x1-x0)
            or mask.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8))):
        raise ValueError('Mask must be bool or uint8 and match the ROI')
    if bounded_kernel is None:
        return None
    mask_bytes = np.ascontiguousarray(mask).view(np.uint8)
    status = bounded_kernel(original, result, height, width, x0, y0, x1, y1, mask_bytes)
    if status < 0:
        raise ValueError('Native bounded check rejected invalid image dimensions')
    return bool(status)


if __name__ == '__main__':
    subprocess.run(['xcrun', 'clang++', '-std=c++17', '-O3', '-ffp-contract=off',
                    '-fno-fast-math', '-dynamiclib', str(HERE/'native_edge.cpp'),
                    '-o', str(LIBRARY)], check=True)
    print(LIBRARY)
