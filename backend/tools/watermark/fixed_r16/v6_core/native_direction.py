"""R12 optional two-batch direction variance, with call-local workspace.

Rebuild only this library with ``python -B native_direction.py``. The unchanged
edge and verified tensor arena libraries must not be rebuilt by this command.
Missing libraries and all batch failures cause edge_continuity to replay the
complete old coarse/fine direction search, including its NumPy fallback.
"""
import ctypes
import os
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
from _native import library_path
LIBRARY = library_path('native_direction')
lib = None
if os.environ.get('V6_DISABLE_NATIVE') != '1':
    try:
        lib = ctypes.CDLL(str(LIBRARY))
    except OSError:
        pass

F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
if lib is not None:
    try:
        lib.direction_variance_batch.argtypes = [F64, F64, F64, F64, ctypes.c_int64,
            F64, F64, ctypes.c_int, F64, ctypes.c_int, I32]
        lib.direction_variance_batch.restype = ctypes.c_int
    except AttributeError:
        lib = None

# Test-only fault injection, never enabled in the processing path.
FAULT = None


def direction_scorer(x, y, values, weights, stats=None):
    x, y, values, weights = [np.ascontiguousarray(a, dtype=np.float64)
                             for a in (x, y, values, weights)]
    if x.ndim != 1 or not x.size or any(a.shape != x.shape for a in (y, values, weights)):
        raise ValueError('Direction batches require equal nonempty sample vectors')
    capacity = int(np.ceil((np.ptp(x)+np.ptp(y))*2))+4

    def batch(angles):
        # Scalar ufuncs preserve the exact old trigonometric rounding.
        cosine, sine = np.empty(len(angles)), np.empty(len(angles))
        for index, angle in enumerate(angles):
            cosine[index], sine[index] = np.cos(angle), np.sin(angle)
        variance = np.empty((len(angles), capacity), np.float64)
        bins = np.empty(len(angles), np.int32)
        if FAULT == 'exception':
            raise RuntimeError('Injected direction batch exception')
        if FAULT == 'allocation':
            raise MemoryError('Injected direction allocation failure')
        if stats is not None:
            stats['direction_native_calls'] += 1
        status = lib.direction_variance_batch(x, y, values, weights, x.size,
            cosine, sine, len(angles), variance, capacity, bins)
        if FAULT == 'status':
            status = -4
        if FAULT == 'bins':
            bins[0] = capacity+1
        if status != 0:
            raise RuntimeError('Native direction batch failed: %s' % status)
        if np.any(bins < 1) or np.any(bins > capacity):
            raise RuntimeError('Native direction batch returned invalid occupied bins')
        return bins, variance
    return batch


def scores(bins, variance):
    # Never reduce padded bins or combine angle rows into one reduction.
    return [float(np.sum(row[:count])) for count, row in zip(bins, variance)]


if __name__ == '__main__':
    subprocess.run(['xcrun', 'clang++', '-std=c++17', '-O3', '-ffp-contract=off',
                    '-fno-fast-math', '-dynamiclib', str(HERE/'native_direction.cpp'),
                    '-o', str(LIBRARY)], check=True)
    print(LIBRARY)
