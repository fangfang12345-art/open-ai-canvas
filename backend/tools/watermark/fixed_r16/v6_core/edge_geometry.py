"""Optional quadratic geometry inferred solely from visible image derivatives."""
import numpy as np


def curved_normal(xx, yy, nx, ny, gx, gy, valid, spatial, radius):
    normal = xx*nx + yy*ny
    tangent = -xx*ny + yy*nx
    projection = gx*nx + gy*ny
    magnitude = np.hypot(gx, gy)
    use = valid & (magnitude > 4) & (abs(projection) > .85*magnitude)
    if np.count_nonzero(use) < 24:
        return None
    polarity = 1 if np.sum((projection.clip(0)**2)[use]) >= np.sum((projection.clip(max=0)**2)[use]) else -1
    use &= projection*polarity > max(4, float(np.percentile(abs(projection[use]), 70))*.65)
    tt = tangent[use]/radius
    nn = normal[use]
    weights = np.sqrt(abs(projection[use]))*spatial[use]
    if len(tt) < 24 or np.ptp(tt) < .7:
        return None
    design = np.column_stack((np.ones_like(tt), tt, tt*tt))
    robust = np.ones_like(tt)
    for _ in range(5):
        ww = np.sqrt(weights*robust)
        coef = np.linalg.lstsq(design*ww[:,None], nn*ww, rcond=None)[0]
        residual = nn-design@coef
        scale = max(.6, float(np.median(abs(residual-np.median(residual))))*1.4826)
        robust = np.minimum(1., 1.5*scale/np.maximum(abs(residual),1e-6))
    if np.percentile(abs(residual),80) > 4 or abs(coef[1])/radius > .8:
        return None
    if abs(coef[2])/radius > .55:
        return None
    # A curve must be observed on both sides of the local hole, not extrapolated.
    if np.count_nonzero(tt < -.2) < 8 or np.count_nonzero(tt > .2) < 8:
        return None
    field = normal-(coef[0] + coef[1]*tangent/radius + coef[2]*(tangent/radius)**2)
    return field, tangent
