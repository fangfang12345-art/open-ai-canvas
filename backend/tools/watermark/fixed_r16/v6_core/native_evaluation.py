"""R11 independent, optional residual and target evaluation kernels.

Rebuild with ``python -B native_evaluation.py``. This never rebuilds the R08
edge/tensor or R10 transport libraries. Missing libraries use the complete
NumPy implementations below. Native failures are raised to the caller, which
must replay the full R10 window before committing any accumulation.
"""
import ctypes
import os
from pathlib import Path
import subprocess

import numpy as np

HERE = Path(__file__).resolve().parent
from _native import library_path
LIBRARY = library_path('native_evaluation')
lib = None
if os.environ.get('V6_DISABLE_NATIVE') != '1':
    try:
        lib = ctypes.CDLL(str(LIBRARY))
    except OSError:
        pass

F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
F32 = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
I64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
BOOL = np.ctypeslib.ndpointer(dtype=np.bool_, flags='C_CONTIGUOUS')
if lib is not None:
    lib.evaluation_observed.argtypes = [F64, ctypes.c_int, F64, F64, F64,
        BOOL, ctypes.c_int64, F64, ctypes.c_int, F64, ctypes.c_int64]
    lib.evaluation_observed.restype = ctypes.c_int64
    lib.evaluation_targets.argtypes = [F64, F64, F64, I64, BOOL, BOOL, BOOL,
        F64, ctypes.c_int, ctypes.c_int, F64, F64, F64, F64, F64, F64, F64,
        ctypes.c_int64, ctypes.c_int, ctypes.c_int64, I64, F32, F32, I64]
    lib.evaluation_targets.restype = ctypes.c_int64

# Test-only injection; the normal path is always None.
FAULT = None
_ZERO_BETA = np.zeros(3, np.float64)


def _profile(value, bins=None):
    value = np.ascontiguousarray(value, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or len(value) < 2:
        raise ValueError('Evaluation requires a Bx3 profile with B >= 2')
    if bins is not None and len(value) != bins:
        raise ValueError('Evaluation profiles must share bin count')
    return value


def _beta(value):
    value = np.ascontiguousarray(value, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError('Evaluation requires three transport coefficients')
    return value


def _aligned(value, shape, dtype):
    value = np.asarray(value)
    if value.shape != shape:
        raise ValueError('Evaluation inputs must share sample shape')
    return np.ascontiguousarray(value, dtype=dtype).reshape(-1)


def _sample(profile, u):
    low = np.clip(np.floor(u).astype(np.int32), 0, len(profile)-2)
    f = np.clip(u-low, 0., 1.)
    return profile[low]*(1-f[..., None]) + profile[low+1]*f[..., None]


def _inject(status, output):
    if FAULT == 'exception':
        raise RuntimeError('Injected evaluation exception')
    if FAULT == 'allocation':
        raise MemoryError('Injected evaluation allocation failure')
    if FAULT == 'status':
        status = -4
    if FAULT == 'nonfinite':
        if output.size:
            output.reshape(-1)[0] = np.nan
        raise FloatingPointError('Injected nonfinite evaluation output')
    if status < 0:
        raise FloatingPointError('Native evaluation rejected window: %s' % status)


def observed(profile, u, tau, colors, observed, beta=None, *, native=True):
    """Return every observed residual, preserving original row-major order."""
    profile = _profile(profile)
    u = np.asarray(u)
    shape = u.shape
    u = np.ascontiguousarray(u, dtype=np.float64).reshape(-1)
    tau = _aligned(tau, shape, np.float64)
    observed = _aligned(observed, shape, np.bool_)
    colors = np.ascontiguousarray(colors, dtype=np.float64)
    if colors.shape != (u.size, 3):
        raise ValueError('Observed evaluation requires Nx3 colors')
    use_beta = beta is not None
    beta = _beta(beta) if use_beta else _ZERO_BETA
    if not native or lib is None:
        prediction = _sample(profile, u)
        if use_beta:
            prediction = prediction + tau[:, None]*beta
        return np.max(np.abs(prediction-colors), axis=1)[observed]
    count = int(np.count_nonzero(observed))
    residual = np.empty(count, np.float64)
    status = lib.evaluation_observed(profile, len(profile), u, tau, colors,
        observed, u.size, beta, int(use_beta), residual, count)
    _inject(status, residual)
    if status != count:
        raise RuntimeError('Native observed evaluation omitted samples')
    return residual


def targets(u, tau, spatial, ui, central, common, trusted, disagreement,
            old_valid, new_valid, p0, new_profile, new_beta, left_profile,
            left_beta, right_profile, right_beta, *, native=True):
    """Return private unique target contributions; never touch accumulators."""
    p0 = _profile(p0)
    bins = len(p0)
    new_profile = _profile(new_profile, bins)
    left_profile = _profile(left_profile, bins)
    right_profile = _profile(right_profile, bins)
    new_beta, left_beta, right_beta = map(_beta, (new_beta, left_beta, right_beta))
    u = np.asarray(u)
    shape = u.shape
    u = np.ascontiguousarray(u, dtype=np.float64).reshape(-1)
    tau, spatial = [_aligned(x, shape, np.float64) for x in (tau, spatial)]
    ui = _aligned(ui, shape, np.int64)
    central, trusted = [_aligned(x, shape, np.bool_) for x in (central, trusted)]
    common = _aligned(common, (bins,), np.bool_)
    disagreement = _aligned(disagreement, (bins,), np.float64)
    if not native or lib is None:
        return _targets_numpy(u, tau, spatial, ui, central, common, trusted,
            disagreement, old_valid, new_valid, p0, new_profile, new_beta,
            left_profile, left_beta, right_profile, right_beta)
    capacity = int(np.count_nonzero(central))
    indices = np.empty(capacity, np.int64)
    weights = np.empty(capacity, np.float32)
    contributions = np.empty((capacity, 3), np.float32)
    counts = np.empty(3, np.int64)
    count = lib.evaluation_targets(u, tau, spatial, ui, central, common, trusted,
        disagreement, int(old_valid), int(new_valid), p0, new_profile, new_beta,
        left_profile, left_beta, right_profile, right_beta, u.size, bins,
        capacity, indices, weights, contributions, counts)
    _inject(count, contributions)
    if count > capacity or int(counts[1]+counts[2]) != count:
        raise RuntimeError('Native target evaluation returned invalid counts')
    return dict(indices=indices[:count], weights=weights[:count],
        contributions=contributions[:count], old_supported=bool(counts[0]),
        new_supported=bool(counts[2]), old_count=int(counts[1]),
        new_count=int(counts[2]), backend='native')


def _targets_numpy(u, tau, spatial, ui, central, common, trusted, disagreement,
                   old_valid, new_valid, p0, profile, beta, left, left_beta,
                   right, right_beta):
    if np.any(ui < 0) or np.any(ui >= len(p0)-1):
        raise ValueError('Evaluation bin indices exceed profile')
    active = central & trusted & old_valid
    new_active = np.zeros(u.shape, bool)
    if new_valid:
        target_disagreement = np.max(np.abs(
            (_sample(left, u)+tau[:, None]*left_beta)
            - (_sample(right, u)+tau[:, None]*right_beta)), axis=1)
        new_active = (central & common[ui] & common[ui+1]
                      & (target_disagreement < 9))
    old_active = active & ~new_active
    agreement_weight = np.clip(1-disagreement[ui]/12, 0, 1)
    weights = (spatial*agreement_weight*old_active).astype(np.float32)
    prediction = _sample(p0, u) if old_valid else np.zeros((u.size, 3))
    if new_active.any():
        new_weights = (spatial*np.clip(1-target_disagreement/12, 0, 1)
                       *new_active).astype(np.float32)
        weights[new_active] = new_weights[new_active]
        prediction = prediction.copy()
        new_prediction = _sample(profile, u)+tau[:, None]*beta
        prediction[new_active] = new_prediction[new_active]
    indices = np.flatnonzero(old_active | new_active)
    contributions = prediction.astype(np.float32)*weights[:, None]
    return dict(indices=indices, weights=weights[indices],
        contributions=contributions[indices], old_supported=bool(active.any()),
        new_supported=bool(new_active.any()), old_count=int(old_active.sum()),
        new_count=int(new_active.sum()), backend='numpy')


if __name__ == '__main__':
    subprocess.run(['xcrun', 'clang++', '-std=c++17', '-O3', '-ffp-contract=off',
                    '-fno-fast-math', '-dynamiclib', str(HERE/'native_evaluation.cpp'),
                    '-o', str(LIBRARY)], check=True)
    print(LIBRARY)
