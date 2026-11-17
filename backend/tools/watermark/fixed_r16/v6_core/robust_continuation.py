"""R16 complete four-round vector Cauchy continuation with atomic B fallback.

Build: PYTHONDONTWRITEBYTECODE=1 python -B v6_core/robust_continuation.py
R16_DISABLE_NATIVE=1 uses the complete independent NumPy mathematical oracle.
"""
import ctypes
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from tensor_harmonic import tensor_field

HERE = Path(__file__).resolve().parent
from _native import library_path
LIBRARY = library_path('native_robust_continuation')
try:
    NATIVE = ctypes.CDLL(str(LIBRARY))
    U8 = np.ctypeslib.ndpointer(dtype=np.uint8, flags='C_CONTIGUOUS')
    F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    NATIVE.robust_solve.argtypes = [U8, U8, U8, F64, F64, F64, ctypes.c_int, ctypes.c_int,
                                  F64, F64, F64, ctypes.c_int]
    NATIVE.robust_solve.restype = ctypes.c_int
except (OSError, AttributeError):
    NATIVE = None


def solve_with_fields(image, base, unknown, flat, c, q, tx, ty, *, native=True, fault=0, debug=False):
    image, base = np.ascontiguousarray(image), np.ascontiguousarray(base)
    unknown, flat = (np.ascontiguousarray(v, bool) for v in (unknown, flat))
    shape = unknown.shape
    if (len(shape) != 2 or min(shape) <= 0 or image.shape != (*shape, 3) or base.shape != image.shape
            or image.dtype != np.uint8 or base.dtype != np.uint8 or flat.shape != shape
            or np.any(unknown & flat)):
        raise ValueError('Invalid input shapes, types or overlapping U/F')
    fields = tuple(np.ascontiguousarray(v, np.float64) for v in (c, q, tx, ty))
    if any(v.shape != shape or not np.isfinite(v).all() for v in fields):
        raise ValueError('Invalid tensor shape or nonfinite field')
    c, q, tx, ty = fields
    if np.any((c < 0)|(c > 1)) or np.any(np.abs(np.sqrt(tx*tx+ty*ty)-1) > 1e-5):
        raise ValueError('Invalid coherence or nonunit tangent')
    fixed = image.copy(); fixed[flat] = base[flat]
    native = bool(native and NATIVE is not None and os.environ.get('R16_DISABLE_NATIVE') != '1')
    if not unknown.any():
        return base.copy(), dict(native=native, rounds=[], unknown_pixels=0), None
    if native:
        values = np.empty(base.shape, np.float64); summary = np.zeros(24); rounds = np.zeros((4, 32))
        status = NATIVE.robust_solve(fixed, base, unknown.astype(np.uint8), c, tx, ty,
                                    *shape, values, summary, rounds, fault)
        if status:
            raise ArithmeticError('Native robust status '+str(status))
        details = None
    else:
        from robust_reference import reference_solve
        values, summary, rounds, details = reference_solve(fixed, base, unknown, c, tx, ty, fault, debug)
    if (not np.isfinite(values).all() or not np.isfinite(summary).all() or not np.isfinite(rounds).all()
            or summary[0] != np.count_nonzero(unknown) or not np.all(rounds[:, 27] == 1)
            or not np.array_equal(values[~unknown], fixed[~unknown].astype(np.float64))):
        raise ArithmeticError('Incomplete or nonfinite robust result')
    if (np.any(rounds[:, 7] > rounds[:, 6]+1e-9*np.maximum(np.abs(rounds[:, 6]), 1))
            or np.any(rounds[:, 5] > rounds[:, 4]+1e-9*np.maximum(np.abs(rounds[:, 4]), 1))):
        raise ArithmeticError('Increasing audited energy')
    begin = time.perf_counter()
    output = fixed.copy(); output[unknown] = np.clip(np.rint(values[unknown]), 0, 255).astype(np.uint8)
    changed = np.any(output != base, axis=2) & unknown
    stats = dict(native=native, unknown_pixels=int(summary[0]), graph_edges=int(summary[1]),
                 unknown_edges=int(summary[2]), components=int(summary[3]),
                 component_seconds=float(summary[4]), base_edges_seconds=float(summary[5]),
                 initial_clip_count=int(summary[6]), initial_clip_max=float(summary[7]),
                 initial_energy=float(summary[8]), final_energy=float(summary[9]),
                 quantized_energy=float(summary[10]), commit_seconds=time.perf_counter()-begin+float(summary[11]),
                 native_total_seconds=float(summary[12]), boundary_edges=int(summary[16]),
                 q_zero_pixels=int(np.count_nonzero(unknown & (q == 0))),
                 c_zero_pixels=int(np.count_nonzero(unknown & (c == 0))),
                 q_zero_changed=int(np.count_nonzero(changed & (q == 0))),
                 c_zero_changed=int(np.count_nonzero(changed & (c == 0))),
                 q_mean=float(q[unknown].mean()), changed_pixels=int(changed.sum()),
                 robust_pcg_rounds=4, old_graph_calls=0, old_pcg_calls=0,
                 robust_pcg_steps=int(rounds[:, 12:15].sum()), rounds=[])
    names = ('weight_seconds', 'pcg_seconds', 'clip_seconds', 'energy_seconds', 'Q_before', 'Q_after',
             'J_before', 'J_after', 'weight_min', 'weight_max', 'clip_count', 'clip_max')
    for row in rounds:
        item = {name: float(row[i]) for i, name in enumerate(names)}
        item.update(steps=row[12:15].astype(int).tolist(), initial_residual=row[15:18].tolist(),
                    final_true_residual=row[18:21].tolist(), unconverged=row[21:24].astype(int).tolist(),
                    recurrence_terminal_residual=row[24:27].tolist(), complete=bool(row[27]))
        item['post_projection_residual'] = row[28:31].tolist()
        stats['rounds'].append(item)
    if debug:
        details = dict(details or {}, values=values, raw_summary=summary, raw_rounds=rounds)
    return output, stats, details


def repair(image, exclusion, write, base, flat):
    started = time.perf_counter()
    stats = dict(tensor_seconds=0., failure_fallbacks=0, filled_pixels=0,
                 old_graph_calls=0, old_pcg_calls=0, robust_pcg_rounds=0, native=False)
    try:
        exclusion, write, flat = (np.asarray(v, bool) for v in (exclusion, write, flat))
        if exclusion.shape != write.shape or flat.shape != exclusion.shape:
            raise ValueError('Invalid mask shape')
        unknown = exclusion & ~flat
        stats.update(unknown_pixels=int(unknown.sum()), fixed_flat_pixels=int((exclusion & flat).sum()))
        if not unknown.any() or not write.any():
            return base.copy(), stats
        begin = time.perf_counter(); fields = tensor_field(image, exclusion, write)
        stats['tensor_seconds'] = time.perf_counter()-begin
        output, extra, _ = solve_with_fields(image, base, unknown, exclusion & flat, *fields)
        stats.update(extra)
        stats['filled_pixels'] = int(np.any(output != base, axis=2)[write].sum())
        stats['total_seconds'] = time.perf_counter()-started
        return output, stats
    except Exception as exc:
        stats.update(failure_fallbacks=1, fallback_reason=type(exc).__name__+': '+str(exc),
                     total_seconds=time.perf_counter()-started)
        return base.copy(), stats


if __name__ == '__main__':
    flags = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
    subprocess.run(['c++', '-std=c++17', '-O3', '-ffp-contract=off', '-fno-fast-math', *flags,
                    str(HERE/'native_robust_continuation.cpp'), '-o', str(LIBRARY)], check=True)
    print(LIBRARY)
