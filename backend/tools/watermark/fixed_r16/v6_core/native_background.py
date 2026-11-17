"""Optional Accelerate implementation of the existing local-plane fit.

Geometry and projectors alone are cached. Frame colors and evidence are supplied
on every invocation. Run this file to build; a missing library leaves the NumPy
oracle available. V8_DISABLE_NATIVE_BACKGROUND=1 explicitly selects that oracle.
"""
from collections import OrderedDict
import ctypes
import os
from pathlib import Path
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
from _native import library_path
LIBRARY = library_path('native_background')
lib = None
try:
    lib = ctypes.CDLL(str(LIBRARY))
    lib.background_fit
except (OSError, AttributeError):
    lib = None

I64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
U8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
if lib is not None:
    lib.background_fit.argtypes = [U8, U8, U8, ctypes.c_int64, ctypes.c_int64,
                                  I64, I64, F32, U8, F32, I64, F32, F32,
                                  U8, U8, F32]
    lib.background_fit.restype = ctypes.c_int64

_packed_cache = OrderedDict()


def _pack(windows, shape):
    key = (id(windows), shape)
    if key in _packed_cache:
        _packed_cache.move_to_end(key)
        return _packed_cache[key][1]
    width = shape[1]
    metadata, samples, designs, training_flags, projectors = [], [], [], [], []
    targets, target_designs, weights = [], [], []
    sample_offset = target_offset = projector_offset = 0
    for sy, sx, design, training, heldout, projector, ty, tx, target_design, weight in windows:
        # Training and held-out samples must partition the original observations.
        if not np.array_equal(~training, heldout):
            raise ValueError('Background geometry has inconsistent holdout samples')
        metadata.append((sample_offset, len(sx), target_offset, len(tx),
                         projector_offset, int(training.sum())))
        samples.append(sy * width + sx)
        designs.append(design)
        training_flags.append(training.astype(np.uint8))
        projectors.append(projector.ravel())
        targets.append(ty * width + tx)
        target_designs.append(target_design)
        weights.append(weight)
        sample_offset += len(sx)
        target_offset += len(tx)
        projector_offset += projector.size

    def concat(parts, dtype):
        return np.ascontiguousarray(np.concatenate(parts) if parts else [], dtype=dtype)

    packed = (np.ascontiguousarray(metadata, dtype=np.int64).reshape(-1, 6),
              concat(samples, np.int64), concat(designs, np.float32),
              concat(training_flags, np.uint8), concat(projectors, np.float32),
              concat(targets, np.int64), concat(target_designs, np.float32),
              concat(weights, np.float32))
    _packed_cache[key] = (windows, packed)
    while len(_packed_cache) > 6:
        _packed_cache.popitem(last=False)
    return packed


def fit(patch, evidence, write, windows):
    """Return the native proposal, or None to use the unchanged NumPy oracle."""
    if (lib is None or os.environ.get('V8_DISABLE_NATIVE_BACKGROUND') == '1'
            or os.environ.get('V6_DISABLE_NATIVE') == '1'):
        return None
    started = time.perf_counter()
    if (not isinstance(patch, np.ndarray) or patch.dtype != np.uint8
            or patch.ndim != 3 or patch.shape[2] != 3
            or np.shape(evidence) != patch.shape[:2]
            or np.shape(write) != patch.shape[:2]):
        raise ValueError('Background fit requires uint8 BGR and matching evidence/write masks')
    shape = patch.shape[:2]
    if not all(shape):
        raise ValueError('Background fit requires a nonempty patch')
    packed = _pack(windows, shape)
    patch = np.ascontiguousarray(patch)
    evidence = np.ascontiguousarray(evidence, dtype=np.bool_).view(np.uint8)
    write = np.ascontiguousarray(write, dtype=np.bool_).view(np.uint8)
    result = np.empty_like(patch)
    accepted = np.empty(shape, np.uint8)
    uncertainty = np.empty(shape, np.float32)
    accepted_windows = lib.background_fit(patch, evidence, write, write.size,
                                          len(windows), *packed, result,
                                          accepted, uncertainty)
    if accepted_windows < 0:
        raise RuntimeError('Native background fit rejected its packed geometry')
    accepted = accepted.view(np.bool_)
    statistics = dict(filled_pixels=int(accepted.sum()),
                      accepted_windows=int(accepted_windows),
                      candidate_windows=len(windows),
                      proposal_seconds=time.perf_counter()-started)
    return result, accepted, uncertainty, statistics


if __name__ == '__main__':
    subprocess.run(['xcrun', 'clang++', '-std=c++17', '-O3', '-ffp-contract=off',
                    '-fno-fast-math', '-Wno-deprecated-declarations', '-dynamiclib',
                    str(HERE/'native_background.cpp'), '-framework', 'Accelerate',
                    '-o', str(LIBRARY)], check=True)
    print(LIBRARY)
