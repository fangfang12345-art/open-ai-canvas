"""NumPy reference for the frozen Schur-complement tangential transport model.

The p0 assembly and solve retain the original NumPy profile expressions. Ordered
NumPy cumulative sums make the new scalar reductions comparable to C++ without
introducing a dependency on a BLAS reduction implementation.
"""
import numpy as np


def _solve(diagonal, off, rhs):
    d, value = diagonal.copy(), rhs.copy()
    for i in range(1, len(d)):
        factor = off[i-1]/d[i-1]
        d[i] -= factor*off[i-1]
        value[i] -= factor*value[i-1]
    value[-1] /= d[-1]
    for i in range(len(d)-2, -1, -1):
        value[i] = (value[i]-off[i]*value[i+1])/d[i]
    return value


def _ordered_sum(values):
    if len(values) == 0:
        return np.zeros(values.shape[1:], np.float64)
    return np.cumsum(values, axis=0)[-1]


def profile(u, colors, weights, tau, bins):
    u, colors, weights, tau = [np.asarray(x, dtype=np.float64)
                               for x in (u, colors, weights, tau)]
    lower = np.floor(u).astype(np.int32)
    f = u - lower
    keep = (lower >= 0) & (lower + 1 < bins) & (weights > 0)
    lower, f, c, wt = lower[keep], f[keep], colors[keep], weights[keep]
    t = tau[keep]
    a, b = 1 - f, f
    diagonal = (np.bincount(lower, wt*a*a, minlength=bins)
                + np.bincount(lower+1, wt*b*b, minlength=bins))
    cross = np.bincount(lower, wt*a*b, minlength=bins)
    support = (np.bincount(lower, wt*a, minlength=bins)
               + np.bincount(lower+1, wt*b, minlength=bins))
    diagonal = diagonal + 2*.06 + 1e-5
    off = cross[:-1] - .06
    rhs = np.column_stack([
        np.bincount(lower, wt*a*c[:, k], minlength=bins)
        + np.bincount(lower+1, wt*b*c[:, k], minlength=bins)
        for k in range(3)
    ]).astype(np.float64, copy=False)
    p0 = _solve(diagonal, off, rhs)
    q = (np.bincount(lower, wt*a*t, minlength=bins)
         + np.bincount(lower+1, wt*b*t, minlength=bins)).astype(np.float64, copy=False)
    v = _solve(diagonal, off, q)
    h = _ordered_sum((wt*t)*t) + 1e-5
    z = _ordered_sum((wt*t)[:, None]*c)
    schur = h - _ordered_sum(q*v)
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = (z - _ordered_sum(q[:, None]*p0))/schur
        value = p0 - v[:, None]*beta
        ratio = float(schur/h)
    valid = (np.isfinite(ratio) and ratio >= .02
             and np.isfinite(p0).all() and np.isfinite(support).all()
             and np.isfinite(value).all() and np.isfinite(beta).all()
             and np.max(np.abs(beta)) <= 12)
    return dict(p0=p0, support=support, profile=value, beta=beta,
                ratio=ratio, valid=bool(valid))


def paired(u, colors, weights, tangent, cutoff, radius, bins):
    tangent = np.asarray(tangent, dtype=np.float64)
    tau = tangent/radius
    return (profile(u, colors, weights*(tangent < -cutoff), tau, bins),
            profile(u, colors, weights*(tangent > cutoff), tau, bins))
