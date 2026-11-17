"""Independent float64 NumPy oracle for the frozen R16 vector Cauchy energy."""
import time
import cv2
import numpy as np

DIRECTIONS = ((1, 0), (0, 1), (1, 1), (-1, 1))


def graph(fixed, base, unknown, c, tx, ty):
    started = time.perf_counter()
    h, width = unknown.shape
    size = h * width
    pixels = np.flatnonzero(unknown)
    index = np.full(size, -1, np.int32)
    index[pixels] = np.arange(len(pixels))
    ids = np.arange(size).reshape(h, width)
    ps, zs, coefficients = [], [], []
    cf, xf, yf = (v.ravel() for v in (c, tx, ty))
    for dx, dy in DIRECTIONS:
        x0, x1 = max(0, -dx), min(width, width-dx)
        y0, y1 = 0, h-dy
        p = ids[y0:y1, x0:x1].ravel()
        z = ids[y0+dy:y1+dy, x0+dx:x1+dx].ravel()
        keep = (index[p] >= 0) | (index[z] >= 0)
        p, z = p[keep], z[keep]
        length = dx*dx + dy*dy
        def a(v):
            dot = -dx*yf[v] + dy*xf[v]
            return (1-cf[v]) + cf[v]/(1+64*dot*dot/length)
        ps.append(p); zs.append(z)
        coefficients.append((a(p)+a(z))/(2*length))
    p, z, ae = np.concatenate(ps), np.concatenate(zs), np.concatenate(coefficients)
    a, b = index[p], index[z]
    base_edges_seconds = time.perf_counter()-started
    started = time.perf_counter()
    count, labels = cv2.connectedComponents(unknown.astype(np.uint8), connectivity=8)
    groups = labels.ravel()[pixels]-1
    low = np.full((count-1, 3), np.inf)
    high = np.full((count-1, 3), -np.inf)
    colors = fixed.reshape(-1, 3).astype(np.float64)
    boundary_count = 0
    for source, destination in ((a[b < 0], z[b < 0]), (b[a < 0], p[a < 0])):
        np.minimum.at(low, groups[source], colors[destination])
        np.maximum.at(high, groups[source], colors[destination])
        boundary_count += len(source)
    if not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ArithmeticError('No fixed boundary for unknown component')
    return dict(p=p, z=z, ae=ae, a=a, b=b, pixels=pixels, groups=groups,
                low=low[groups], high=high[groups], components=count-1,
                fixed=colors, initial=base.reshape(-1, 3)[pixels].astype(np.float64),
                boundary_count=boundary_count, component_seconds=time.perf_counter()-started,
                base_edges_seconds=base_edges_seconds)


def edge_differences(g, u):
    colors = g['fixed'].copy()
    colors[g['pixels']] = u
    difference = colors[g['p']] - colors[g['z']]
    return np.sum(difference*difference, axis=1)/3


def energy(g, u):
    return float(np.sum(g['ae']*64*np.log1p(edge_differences(g, u)/64)))


def linear_system(g, weights):
    n = len(g['pixels']); a, b = g['a'], g['b']
    diag = np.zeros(n); rhs = np.zeros((n, 3))
    for src, dst, select in ((a, g['z'], a >= 0), (b, g['p'], b >= 0)):
        np.add.at(diag, src[select], weights[select])
    for src, dst, select in ((a, g['z'], b < 0), (b, g['p'], a < 0)):
        np.add.at(rhs, src[select], weights[select, None]*g['fixed'][dst[select]])
    both = (a >= 0) & (b >= 0)
    aa, bb, ww = a[both], b[both], weights[both]
    def mat(v):
        out = diag[:, None]*v
        np.add.at(out, aa, -ww[:, None]*v[bb])
        np.add.at(out, bb, -ww[:, None]*v[aa])
        return out
    if not np.isfinite(diag).all() or np.any(diag <= 0):
        raise ArithmeticError('Invalid positive diagonal')
    return diag, rhs, mat


def pcg(diag, rhs, mat, current, fault=0):
    u = current.copy()
    residual = rhs-mat(u)
    norm = np.maximum(np.sqrt(np.sum(rhs*rhs, axis=0)), 1)
    relative = lambda r: np.sqrt(np.sum(r*r, axis=0))/norm
    initial = relative(residual)
    z = residual/diag[:, None]; direction = z.copy()
    rz = np.sum(residual*z, axis=0); steps = np.zeros(3, int)
    if fault == 5:
        raise ArithmeticError('Injected nonpositive PCG denominator')
    for _ in range(16):
        live = relative(residual) > 1e-4
        if not live.any():
            break
        direction[:, ~live] = 0
        product = mat(direction)
        denominator = np.sum(direction*product, axis=0)
        if np.any(denominator[live] <= 0) or not np.isfinite(denominator).all():
            raise ArithmeticError('Nonpositive or nonfinite PCG denominator')
        alpha = np.divide(rz, denominator, out=np.zeros(3), where=live)
        steps += live
        u += direction*alpha
        residual -= product*alpha
        z = residual/diag[:, None]
        new = np.sum(residual*z, axis=0)
        beta = np.divide(new, rz, out=np.zeros(3), where=live)
        direction = z + direction*beta
        rz = new
        if not np.isfinite(u).all() or not np.isfinite(residual).all():
            raise ArithmeticError('Nonfinite PCG state')
    return u, steps, initial, relative(rhs-mat(u)), relative(residual)


def reference_solve(fixed, base, unknown, c, tx, ty, fault=0, debug=False):
    started = time.perf_counter()
    if fault == 1:
        raise MemoryError('Injected allocation failure')
    if fault in (2, 3):
        raise ArithmeticError('Injected native/partial state')
    summary = np.zeros(24); rounds = np.zeros((4, 32))
    begin = time.perf_counter(); g = graph(fixed, base, unknown, c, tx, ty)
    summary[4], summary[5] = g['component_seconds'], g['base_edges_seconds']
    summary[:4] = len(g['pixels']), len(g['p']), np.count_nonzero((g['a'] >= 0)&(g['b'] >= 0)), g['components']
    summary[16] = g['boundary_count']
    u = np.clip(g['initial'], g['low'], g['high'])
    summary[6] = np.count_nonzero(u != g['initial'])
    summary[7] = float(np.max(np.abs(u-g['initial']), initial=0))
    summary[8] = energy(g, u)
    trace = []
    for k in range(4):
        row = rounds[k]; before = u.copy()
        begin = time.perf_counter()
        difference = edge_differences(g, u)
        weights = g['ae']/(1+difference/64)
        diag, rhs, mat = linear_system(g, weights)
        row[0] = time.perf_counter()-begin
        row[4] = np.sum(weights*difference); row[6] = energy(g, u)
        row[8:10] = weights.min(), weights.max()
        begin = time.perf_counter()
        solved, steps, initial, true_residual, recurrence_residual = pcg(diag, rhs, mat, u, fault)
        row[1] = time.perf_counter()-begin
        row[12:15], row[15:18], row[18:21] = steps, initial, true_residual
        row[21:24], row[24:27] = true_residual > 1e-4, recurrence_residual
        begin = time.perf_counter(); u = np.clip(solved, g['low'], g['high'])
        row[10] = np.count_nonzero(u != solved)
        row[11] = np.max(np.abs(u-solved), initial=0)
        row[2] = time.perf_counter()-begin
        row[28:31] = np.sqrt(np.sum((rhs-mat(u))**2, axis=0))/np.maximum(np.sqrt(np.sum(rhs*rhs, axis=0)), 1)
        begin = time.perf_counter()
        row[5], row[7] = np.sum(weights*edge_differences(g, u)), energy(g, u)
        if fault == 4:
            row[7] = row[6]+max(abs(row[6]), 1)
        if fault == 6:
            row[5] = row[4]+max(abs(row[4]), 1)
        if not np.isfinite(row).all() or row[7] > row[6]+1e-9*max(abs(row[6]), 1):
            raise ArithmeticError('Invalid increasing Cauchy energy')
        if row[5] > row[4]+1e-9*max(abs(row[4]), 1):
            raise ArithmeticError('Invalid increasing quadratic energy')
        row[3] = time.perf_counter()-begin; row[27] = 1
        if debug:
            trace.append(dict(before=before, solved=solved, projected=u.copy(), weights=weights,
                              diag=diag, rhs=rhs))
    summary[9], summary[10] = energy(g, u), energy(g, np.clip(np.rint(u), 0, 255))
    output = g['fixed'].copy(); output[g['pixels']] = u
    summary[12] = time.perf_counter()-started
    return output.reshape(base.shape), summary, rounds, dict(graph=g, trace=trace) if debug else None
