"""Optional R10 transport kernels with a complete NumPy fallback.

Rebuild this independent library with ``python -B native_transport.py``. The
legacy edge and verified tensor libraries are neither rebuilt nor modified.
"""
import ctypes
import os
from pathlib import Path
import subprocess

import numpy as np
import transport_reference

HERE = Path(__file__).resolve().parent
from _native import library_path
LIBRARY = library_path('native_transport')
lib = None
if os.environ.get('V6_DISABLE_NATIVE') != '1':
    try:
        lib = ctypes.CDLL(str(LIBRARY))
    except OSError:
        pass

F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
if lib is not None:
    lib.transport_profile.argtypes = [F64, F64, F64, F64,
                                     ctypes.c_int64, ctypes.c_int, F64]
    lib.transport_profile.restype = None
    lib.transport_paired.argtypes = [F64, F64, F64, F64,
                                    ctypes.c_int64, ctypes.c_int,
                                    ctypes.c_double, ctypes.c_double, F64]
    lib.transport_paired.restype = None


def _inputs(u, colors, weights, coordinate, bins):
    u, colors, weights, coordinate = [np.ascontiguousarray(x, dtype=np.float64)
                                      for x in (u, colors, weights, coordinate)]
    if (u.ndim != 1 or colors.shape != (u.size, 3)
            or weights.shape != u.shape or coordinate.shape != u.shape
            or bins < 2):
        raise ValueError('Transport profile requires aligned samples and at least two bins')
    return u, colors, weights, coordinate


def _unpack(output, bins):
    p0 = output[:bins*3].reshape(bins, 3)
    support = output[bins*3:bins*4]
    value = output[bins*4:bins*7].reshape(bins, 3)
    beta = output[bins*7:bins*7+3]
    return dict(p0=p0, support=support, profile=value, beta=beta,
                ratio=float(output[bins*7+3]), valid=bool(output[bins*7+4]))


def profile(u, colors, weights, tau, bins):
    u, colors, weights, tau = _inputs(u, colors, weights, tau, bins)
    if lib is None:
        return transport_reference.profile(u, colors, weights, tau, bins)
    output = np.empty(bins*7+5, np.float64)
    lib.transport_profile(u, colors, weights, tau, u.size, bins, output)
    return _unpack(output, bins)


def paired(u, colors, weights, tangent, cutoff, radius, bins):
    u, colors, weights, tangent = _inputs(u, colors, weights, tangent, bins)
    if radius <= 0 or cutoff < 0:
        raise ValueError('Transport paired profile requires a positive radius and nonnegative cutoff')
    if lib is None:
        return transport_reference.paired(u, colors, weights, tangent, cutoff, radius, bins)
    output = np.empty((2, bins*7+5), np.float64)
    lib.transport_paired(u, colors, weights, tangent, u.size, bins,
                         cutoff, radius, output)
    return _unpack(output[0], bins), _unpack(output[1], bins)


if __name__ == '__main__':
    subprocess.run(['xcrun', 'clang++', '-std=c++17', '-O3', '-ffp-contract=off',
                    '-fno-fast-math', '-dynamiclib', str(HERE/'native_transport.cpp'),
                    '-o', str(LIBRARY)], check=True)
    print(LIBRARY)
