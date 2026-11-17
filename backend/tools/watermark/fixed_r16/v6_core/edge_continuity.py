"""Conservative local continuation of a supported one-dimensional edge profile.

Only unmasked pixels estimate orientation and color. A local model must explain
visible samples on both sides of the gap. Curves or texture that cannot be
explained by this model retain the caller's spatial fallback.
"""
import time
import os
from functools import lru_cache

import cv2
import numpy as np
from edge_geometry import curved_normal
import native_edge
import native_transport
import native_evaluation
import native_direction


NATIVE_EVALUATION = True


FORCE_TRANSPORT_REJECT = os.environ.get('V6_FORCE_TRANSPORT_REJECT') == '1'


def _reject(stats, reason):
    reasons = stats['transport_rejections']
    reasons[reason] = reasons.get(reason, 0) + 1


def _model_failure(model):
    if not (np.isfinite(model['p0']).all() and np.isfinite(model['support']).all()
            and np.isfinite(model['profile']).all()
            and np.isfinite(model['beta']).all() and np.isfinite(model['ratio'])):
        return 'nonfinite'
    if model['ratio'] < .02:
        return 'schur_ratio'
    if np.max(np.abs(model['beta'])) > 12:
        return 'beta_limit'
    return None


def _cross_evidence(left, right, u, tau, colors, left_side, right_side,
                    common, clean_strong):
    """Predict held-out observed colors, including every supported strong sample."""
    ui = np.clip(np.floor(u).astype(int), 0, len(common)-2)
    supported = common[ui] & common[ui+1]
    for model, held_out in ((left, right_side), (right, left_side)):
        checked = supported & held_out
        if not checked.any():
            return 'cross_empty'
        predicted = (_sample(model['profile'], u[checked])
                     + tau[checked, None]*model['beta'])
        error = np.max(np.abs(predicted-colors[checked]), axis=1)
        if np.percentile(error, 90) > 6:
            return 'cross_p90'
        if np.percentile(error, 98) > 14:
            return 'cross_p98'
        if np.any(error[clean_strong[checked]] > 14):
            return 'cross_strong_sample'
    return None


def _solve_tridiagonal(diagonal, off, rhs):
    d, value = diagonal.copy(), rhs.copy()
    for i in range(1, len(d)):
        factor = off[i-1]/d[i-1]
        d[i] -= factor*off[i-1]
        value[i] -= factor*value[i-1]
    value[-1] /= d[-1]
    for i in range(len(d)-2, -1, -1):
        value[i] = (value[i]-off[i]*value[i+1])/d[i]
    return value


def _profile(u, colors, weights, bins):
    if native_edge.lib is not None:
        return native_edge.profile(u, colors, weights, bins)
    lower = np.floor(u).astype(np.int32)
    f = u - lower
    keep = (lower >= 0) & (lower + 1 < bins) & (weights > 0)
    lower, f, c, wt = lower[keep], f[keep], colors[keep], weights[keep]
    a, b = 1 - f, f
    diagonal = (np.bincount(lower, wt*a*a, minlength=bins)
                + np.bincount(lower+1, wt*b*b, minlength=bins))
    cross = np.bincount(lower, wt*a*b, minlength=bins)
    support = (np.bincount(lower, wt*a, minlength=bins)
               + np.bincount(lower+1, wt*b, minlength=bins))
    # A weak smoothness prior stabilizes empty boundary bins. Those bins are
    # never trusted for reconstruction without independent observed support.
    regularization = .06
    ab = np.zeros((3, bins), np.float64)
    ab[1] = diagonal + 2*regularization + 1e-5
    ab[0, 1:] = cross[:-1] - regularization
    ab[2, :-1] = cross[:-1] - regularization
    rhs = np.column_stack([
        np.bincount(lower, wt*a*c[:, k], minlength=bins)
        + np.bincount(lower+1, wt*b*c[:, k], minlength=bins)
        for k in range(3)
    ])
    return _solve_tridiagonal(ab[1], ab[0, 1:], rhs), support


def _sample(profile, u):
    if native_edge.lib is not None:
        return native_edge.sample(profile, u)
    low = np.clip(np.floor(u).astype(np.int32), 0, len(profile)-2)
    f = np.clip(u-low, 0., 1.)
    return profile[low]*(1-f[..., None]) + profile[low+1]*f[..., None]


def _refine_direction_legacy(xx, yy, nx, ny, gray, observed, spatial):
    """Calibrate a continuous angle against visible profile variance.

    A tensor direction is biased by weak neighboring contours. Evaluate the
    actual intensity model locally instead of snapping to a horizontal axis.
    Final left/right agreement is checked separately after this search.
    """
    x, y = xx[observed], yy[observed]
    values = gray[observed].astype(np.float64)
    weights = spatial[observed]
    if len(x) > 4096:
        selected = np.linspace(0, len(x)-1, 4096).astype(np.int32)
        x, y, values, weights = x[selected], y[selected], values[selected], weights[selected]
    origin = float(np.arctan2(ny, nx))

    def score(angle):
        coordinate = (x*np.cos(angle)+y*np.sin(angle))*2
        indices = np.rint(coordinate-coordinate.min()).astype(np.int32)
        count = np.bincount(indices, weights)
        total = np.bincount(indices, weights*values)
        square = np.bincount(indices, weights*values*values)
        return float(np.sum(square-total*total/np.maximum(count, 1e-10)))

    if native_edge.lib is not None:
        score = native_edge.direction_scorer(x, y, values, weights)

    angles = origin + np.deg2rad(np.arange(-3., 3.01, .5))
    best = min(angles, key=score)
    fine = best + np.deg2rad(np.arange(-.4, .401, .1))
    best = min(fine, key=score)
    return np.cos(best), np.sin(best)


def _refine_direction(xx, yy, nx, ny, gray, observed, spatial, stats=None):
    """Commit a direction only after both unchanged candidate batches succeed."""
    if native_edge.lib is not None and native_direction.lib is not None:
        try:
            x, y = xx[observed], yy[observed]
            values = gray[observed].astype(np.float64)
            weights = spatial[observed]
            if len(x) > 4096:
                selected = np.linspace(0, len(x)-1, 4096).astype(np.int32)
                x, y, values, weights = x[selected], y[selected], values[selected], weights[selected]
            origin = float(np.arctan2(ny, nx))
            batch = native_direction.direction_scorer(x, y, values, weights, stats)
            angles = origin + np.deg2rad(np.arange(-3., 3.01, .5))
            coarse_scores = iter(native_direction.scores(*batch(angles)))
            best = min(angles, key=lambda angle: next(coarse_scores))
            fine = best + np.deg2rad(np.arange(-.4, .401, .1))
            fine_scores = iter(native_direction.scores(*batch(fine)))
            best = min(fine, key=lambda angle: next(fine_scores))
            result = np.cos(best), np.sin(best)
            if stats is not None:
                stats['direction_native_windows'] += 1
            return result
        except Exception as error:
            if stats is not None:
                reasons = stats['direction_fallback_reasons']
                name = type(error).__name__
                reasons[name] = reasons.get(name, 0)+1
    elif stats is not None:
        reasons = stats['direction_fallback_reasons']
        reasons['unavailable'] = reasons.get('unavailable', 0)+1
    if stats is not None:
        stats['direction_fallback_windows'] += 1
    # Re-run from original samples and origin, even if the fine batch failed.
    # Exceptions from this original path intentionally propagate unchanged.
    return _refine_direction_legacy(xx, yy, nx, ny, gray, observed, spatial)


@lru_cache(maxsize=8)
def _mask_geometry(shape, packed_mask):
    mask = np.frombuffer(packed_mask, dtype=np.bool_).reshape(shape)
    known = ~mask
    valid = cv2.erode(known.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    thickness = float(np.percentile(distance[mask], 85)) if mask.any() else 1.
    return known, valid, thickness


@lru_cache(maxsize=64)
def _coordinates(x0, x1, y0, y1, radius):
    xx, yy = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
    spatial = np.exp(-(xx*xx+yy*yy)/(radius*radius*.8))
    return xx, yy, spatial


def _prepare_evidence(patch, mask):
    known, valid, thickness = _mask_geometry(mask.shape, mask.tobytes())
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    energy = (gx*gx + gy*gy) * valid
    return known, gray, gx, gy, valid, energy, thickness


def _evaluation_stats():
    return dict(old_rejections={}, transport_rejections={}, old_valid_models=0,
                transport_valid_models=0, beta_count=0, beta_abs_sum=[0., 0., 0.],
                beta_abs_max=[0., 0., 0.], supported_models=0, old_supported_models=0,
                transport_supported_models=0, old_retained_pixel_contributions=0,
                transport_pixel_contributions=0, observed_values=0,
                observation_seconds=0., target_seconds=0., evaluation_native_windows=0,
                evaluation_numpy_windows=0, evaluation_fallback_windows=0)


def _evaluate_window(patch, colors, u, tau, spatial, tangent, observed, target_tile,
                     step, bins, common, disagreement, full_model, left_model,
                     right_model, old_valid, new_valid, reason, *, fast=True):
    """Prepare a complete window without touching caller accumulators or stats."""
    stats = _evaluation_stats()
    if not fast:
        stats['evaluation_numpy_windows'] = 1
    profile, support = full_model['p0'], full_model['support']
    if fast and old_valid:
        # Dense R10 evaluates p0 even at inactive targets before multiplying by
        # zero. Keep that behavior for pathological nonfinite/overflow profiles.
        if (not np.isfinite(profile).all()
                or np.max(np.abs(profile)) > np.finfo(np.float32).max/4):
            raise FloatingPointError('Dense legacy prediction requires full fallback')
    prediction = None
    if old_valid:
        residual_started = time.perf_counter()
        if fast:
            error = native_evaluation.observed(profile, u, tau, colors, observed)
        else:
            prediction = _sample(profile, u)
            error = np.max(np.abs(prediction-patch), axis=2)[observed]
        stats['observation_seconds'] += time.perf_counter()-residual_started
        stats['observed_values'] += int(error.size)
        old_reason = ('full_p90' if np.percentile(error, 90) > 6 else
                      'full_p98' if np.percentile(error, 98) > 14 else
                      'full_contrast' if np.max(np.ptp(profile[common], axis=0)) < 18 else None)
        old_valid = old_reason is None
        if old_reason is not None:
            stats['old_rejections'][old_reason] = stats['old_rejections'].get(old_reason, 0)+1
    new_prediction = None
    if new_valid:
        reason = _model_failure(full_model)
        if reason is None:
            residual_started = time.perf_counter()
            if fast:
                new_error = native_evaluation.observed(
                    full_model['profile'], u, tau, colors, observed, full_model['beta'])
            else:
                new_prediction = (_sample(full_model['profile'], u)
                                  + tau[..., None]*full_model['beta'])
                new_error = np.max(np.abs(new_prediction-patch), axis=2)[observed]
            stats['observation_seconds'] += time.perf_counter()-residual_started
            stats['observed_values'] += int(new_error.size)
            if np.percentile(new_error, 90) > 6:
                reason = 'full_p90'
            elif np.percentile(new_error, 98) > 14:
                reason = 'full_p98'
            elif np.max(np.ptp(full_model['profile'][common], axis=0)) < 18:
                reason = 'full_contrast'
        new_valid = reason is None
    if reason is not None:
        _reject(stats, reason)
    stats['old_valid_models'] += int(old_valid)
    stats['transport_valid_models'] += int(new_valid)
    if new_valid:
        beta_abs = np.abs(full_model['beta'])
        stats['beta_count'] += 1
        stats['beta_abs_sum'] = (np.asarray(stats['beta_abs_sum'])+beta_abs).tolist()
        stats['beta_abs_max'] = np.maximum(stats['beta_abs_max'], beta_abs).tolist()
    if not (old_valid or new_valid):
        return None, stats
    ui = np.clip(np.floor(u).astype(int), 0, bins-2)
    trusted = common[ui] & common[ui+1]
    trusted &= (disagreement[ui] < 9) & (disagreement[ui+1] < 9)
    # Apply only to the central tile: each prediction is bracketed by
    # independently fitted left and right context, not extrapolated.
    if native_edge.lib is not None:
        lower_t, upper_t = native_edge.bracket(ui, tangent, observed, bins)
    else:
        lower_t = np.full(bins, np.inf)
        upper_t = np.full(bins, -np.inf)
        for indices in (ui[observed], ui[observed]+1):
            np.minimum.at(lower_t, indices, tangent[observed])
            np.maximum.at(upper_t, indices, tangent[observed])
    bracketed = ((tangent > np.maximum(lower_t[ui], lower_t[ui+1])+2)
                 & (tangent < np.minimum(upper_t[ui], upper_t[ui+1])-2))
    central = target_tile & bracketed & (np.abs(tangent) <= step*.8)
    target_started = time.perf_counter()
    if fast:
        selected = native_evaluation.targets(
            u, tau, spatial, ui, central, common, trusted, disagreement,
            old_valid, new_valid, profile, full_model['profile'], full_model['beta'],
            left_model['profile'], left_model['beta'],
            right_model['profile'], right_model['beta'])
        stats['target_seconds'] += time.perf_counter()-target_started
        stats['evaluation_native_windows' if selected['backend'] == 'native'
              else 'evaluation_numpy_windows'] += 1
        if not selected['indices'].size:
            return None, stats
        stats['supported_models'] += 1
        stats['old_supported_models'] += int(selected['old_supported'])
        stats['transport_supported_models'] += int(selected['new_supported'])
        stats['old_retained_pixel_contributions'] += selected['old_count']
        stats['transport_pixel_contributions'] += selected['new_count']
        return selected, stats
    active = central & trusted & old_valid
    new_active = np.zeros(u.shape, bool)
    if new_valid:
        target_disagreement = np.max(np.abs(
            (_sample(left_model['profile'], u)+tau[..., None]*left_model['beta'])
            - (_sample(right_model['profile'], u)+tau[..., None]*right_model['beta'])), axis=2)
        new_active = (central & common[ui] & common[ui+1]
                      & (target_disagreement < 9))
    # Choose at each target, retaining the original valid contribution
    # wherever the new independent evidence does not support transport.
    old_active = active & ~new_active
    if not (old_active.any() or new_active.any()):
        return None, stats
    agreement_weight = np.clip(1-disagreement[ui]/12, 0, 1)
    ww = (spatial*agreement_weight*old_active).astype(np.float32)
    selected_prediction = (prediction if prediction is not None
                           else np.zeros((*u.shape, 3), np.float64))
    if new_active.any():
        new_ww = (spatial*np.clip(1-target_disagreement/12, 0, 1)*new_active).astype(np.float32)
        ww[new_active] = new_ww[new_active]
        selected_prediction = selected_prediction.copy()
        selected_prediction[new_active] = new_prediction[new_active]
    selected = dict(indices=np.arange(u.size), weights=ww.ravel(),
                    contributions=(selected_prediction.astype(np.float32)*ww[..., None]).reshape(-1, 3))
    stats['target_seconds'] += time.perf_counter()-target_started
    stats['supported_models'] += 1
    stats['old_supported_models'] += int(active.any())
    stats['transport_supported_models'] += int(new_active.any())
    stats['old_retained_pixel_contributions'] += int(old_active.sum())
    stats['transport_pixel_contributions'] += int(new_active.sum())
    return selected, stats


def _prepare_commit(accum, weight, selected):
    if selected is None:
        return None
    rows, columns = np.divmod(selected['indices'], weight.shape[1])
    # Complete both float32 additions before either caller buffer is modified.
    next_accum = accum[rows, columns] + selected['contributions']
    next_weight = weight[rows, columns] + selected['weights']
    return rows, columns, next_accum, next_weight


def _edge_scale(patch, mask, base_result, radius_factor, target=None, evidence=None):
    started = time.perf_counter()
    mask = np.asarray(mask) > 0
    if patch.shape != base_result.shape or mask.shape != patch.shape[:2]:
        raise ValueError('Image and mask dimensions do not match')
    h, w = mask.shape
    if target is None:
        target = mask
    if evidence is None:
        evidence = _prepare_evidence(patch, mask)
    known, gray, gx, gy, gradient_valid, energy, thickness = evidence
    target_count = cv2.integral(target.astype(np.uint8), sdepth=cv2.CV_32S)
    energy_count = cv2.integral((energy > 16).astype(np.uint8), sdepth=cv2.CV_32S)
    accum = np.zeros(patch.shape, np.float32)
    weight = np.zeros((h, w), np.float32)
    radius = int(np.clip(round(thickness*radius_factor), 32, 256))
    step = max(12, radius//2)
    stats = dict(windows=0, supported_models=0, radius=radius, curved_models=0,
                 old_supported_models=0, transport_supported_models=0,
                 old_valid_models=0, transport_valid_models=0,
                 old_retained_pixel_contributions=0, transport_pixel_contributions=0,
                 paired_profile_calls=0, full_profile_calls=0,
                 transport_rejections={}, old_rejections={},
                 profile_seconds=0., evidence_seconds=0.,
                 beta_count=0, beta_abs_sum=[0., 0., 0.], beta_abs_max=[0., 0., 0.])
    stats.update(observed_values=0, observation_seconds=0., target_seconds=0.,
                 evaluation_native_windows=0, evaluation_numpy_windows=0,
                 evaluation_fallback_windows=0, evaluation_fallback_reasons={},
                 evaluation_seconds=0.)
    stats.update(direction_windows=0, direction_native_windows=0,
                 direction_native_calls=0, direction_fallback_windows=0,
                 direction_fallback_reasons={}, direction_seconds=0.)
    for cy in range(step//2, h, step):
        for cx in range(step//2, w, step):
            x0, x1 = max(0, cx-radius), min(w, cx+radius+1)
            y0, y1 = max(0, cy-radius), min(h, cy+radius+1)
            mm = mask[y0:y1, x0:x1]
            target_tile = target[y0:y1, x0:x1]
            if (target_count[y1, x1]-target_count[y0, x1]
                    - target_count[y1, x0]+target_count[y0, x0]) == 0:
                continue
            ex = gx[y0:y1, x0:x1]
            ey = gy[y0:y1, x0:x1]
            valid = gradient_valid[y0:y1, x0:x1]
            ee = energy[y0:y1, x0:x1]
            if (energy_count[y1, x1]-energy_count[y0, x1]
                    - energy_count[y1, x0]+energy_count[y0, x0]) < 18:
                continue
            xx, yy, spatial = _coordinates(x0-cx, x1-cx, y0-cy, y1-cy, radius)
            vv = valid*spatial
            tensor = np.array([[np.sum(ex*ex*vv), np.sum(ex*ey*vv)],
                               [np.sum(ex*ey*vv), np.sum(ey*ey*vv)]])
            eigenvalues, vectors = np.linalg.eigh(tensor)
            coherence = (eigenvalues[1]-eigenvalues[0])/(eigenvalues.sum()+1e-8)
            stats['windows'] += 1
            if coherence < .7:
                continue
            nx, ny = vectors[:, 1]
            if coherence >= .94:
                direction_started = time.perf_counter()
                stats['direction_windows'] += 1
                nx, ny = _refine_direction(xx, yy, nx, ny, gray[y0:y1, x0:x1], valid, spatial, stats)
                stats['direction_seconds'] += time.perf_counter()-direction_started
            normal = xx*nx+yy*ny
            tangent = -xx*ny+yy*nx
            curve = (curved_normal(xx, yy, nx, ny, ex, ey, valid, spatial, radius)
                     if coherence < .94 else None)
            if curve is not None:
                normal, tangent = curve
            elif coherence < .94:
                continue
            offset = float(np.floor(normal.min())-1)
            u = normal-offset
            bins = int(np.ceil(u.max())+2)
            colors = patch[y0:y1, x0:x1].reshape(-1, 3).astype(np.float64)
            # A derivative needs a complete stencil; a color sample does not.
            # Keep narrow clean gaps available for the independent profiles.
            observed = ~mm
            all_weights = observed.astype(np.float64)*spatial
            # Fit the two sides independently. Matching profiles establish that
            # the orientation spans the gap, rather than ending at a text stroke.
            profile_started = time.perf_counter()
            left_model, right_model = native_transport.paired(
                u.ravel(), colors, all_weights.ravel(), tangent.ravel(),
                step*.25, radius, bins)
            stats['paired_profile_calls'] += 1
            stats['profile_seconds'] += time.perf_counter()-profile_started
            left, ls = left_model['p0'], left_model['support']
            right, rs = right_model['p0'], right_model['support']
            common = (ls > 2.5) & (rs > 2.5)
            if np.count_nonzero(common) < max(12, radius//2):
                _reject(stats, 'common_support')
                stats['old_rejections']['common_support'] = stats['old_rejections'].get('common_support', 0)+1
                continue
            disagreement = np.max(np.abs(left-right), axis=1)
            old_valid = np.percentile(disagreement[common], 90) <= 7
            if not old_valid:
                stats['old_rejections']['profile_disagreement'] = stats['old_rejections'].get('profile_disagreement', 0)+1
            evidence_started = time.perf_counter()
            tau = tangent/radius
            reason = ('forced_reject' if FORCE_TRANSPORT_REJECT else
                      _model_failure(left_model) or _model_failure(right_model))
            if reason is None and np.max(np.abs(left_model['beta']-right_model['beta'])) > 2:
                reason = 'beta_disagreement'
            if reason is None:
                at_zero = np.max(np.abs(left_model['profile']-right_model['profile']), axis=1)
                if np.percentile(at_zero[common], 90) > 7:
                    reason = 'profile_disagreement'
            if reason is None:
                reason = _cross_evidence(
                    left_model, right_model, u.ravel(), tau.ravel(), colors,
                    (observed & (tangent < -step*.25)).ravel(),
                    (observed & (tangent > step*.25)).ravel(), common,
                    (valid & (ee > 16)).ravel())
            new_valid = reason is None
            stats['evidence_seconds'] += time.perf_counter()-evidence_started
            if not (old_valid or new_valid):
                _reject(stats, reason)
                continue
            profile_started = time.perf_counter()
            full_model = native_transport.profile(u.ravel(), colors, all_weights.ravel(),
                                                  tau.ravel(), bins)
            stats['full_profile_calls'] += 1
            stats['profile_seconds'] += time.perf_counter()-profile_started
            evaluation_started = time.perf_counter()
            arguments = (patch[y0:y1, x0:x1], colors, u, tau, spatial, tangent,
                         observed, target_tile, step, bins, common, disagreement,
                         full_model, left_model, right_model, old_valid, new_valid, reason)
            accum_tile = accum[y0:y1, x0:x1]
            weight_tile = weight[y0:y1, x0:x1]
            try:
                selected, delta = _evaluate_window(
                    *arguments, fast=NATIVE_EVALUATION and native_evaluation.lib is not None)
                pending = _prepare_commit(accum_tile, weight_tile, selected)
            except Exception as error:
                # Neither kernel writes the accumulators. Recompute the entire
                # original R10 window before making one atomic logical commit.
                selected, delta = _evaluate_window(*arguments, fast=False)
                pending = _prepare_commit(accum_tile, weight_tile, selected)
                delta['evaluation_fallback_windows'] += 1
                name = type(error).__name__
                stats['evaluation_fallback_reasons'][name] = stats['evaluation_fallback_reasons'].get(name, 0)+1
            stats['evaluation_seconds'] += time.perf_counter()-evaluation_started
            for key, value in delta.items():
                if isinstance(value, dict):
                    for name, count in value.items():
                        stats[key][name] = stats[key].get(name, 0)+count
                elif key == 'beta_abs_max':
                    stats[key] = np.maximum(stats[key], value).tolist()
                elif key == 'beta_abs_sum':
                    stats[key] = (np.asarray(stats[key])+value).tolist()
                else:
                    stats[key] += value
            if pending is None:
                continue
            rows, columns, next_accum, next_weight = pending
            accum_tile[rows, columns] = next_accum
            weight_tile[rows, columns] = next_weight
            stats['curved_models'] += int(curve is not None)
    repaired = mask & (weight > .15)
    result = base_result.copy()
    result[repaired] = np.clip(np.rint(accum[repaired]/weight[repaired, None]), 0, 255).astype(np.uint8)
    if not np.array_equal(result[known], patch[known]):
        raise AssertionError('Edge correction escaped the supplied mask')
    stats.update(repaired_pixels=int(repaired.sum()), seconds=time.perf_counter()-started)
    return result, repaired, stats


def edge_continuity(patch, mask, base_result):
    mask = np.asarray(mask) > 0
    evidence = _prepare_evidence(patch, mask)
    large, accepted, coarse_stats = _edge_scale(patch, mask, base_result, 16, evidence=evidence)
    remaining = (np.asarray(mask) > 0) & ~accepted
    result, extra, fine_stats = _edge_scale(patch, mask, large, 4, remaining, evidence=evidence)
    repaired = accepted | extra
    stats = dict(repaired_pixels=int(repaired.sum()), coarse=coarse_stats, fine=fine_stats)
    return result, repaired, stats
