"""Estimate a premultiplied BGR overlay from real, aligned background estimates.

I = gain * B + offset, with one gain and three offsets at each template pixel.
No image files, generated images, tracking or video-specific coordinates are used.
Frame IDs are split before fitting; held-out frames never update fitted parameters.
Temporal correlation and erroneous background estimates can still invalidate the
model. Confidence measures consistency with supplied data, not hidden-pixel truth.
"""
import warnings

import numpy as np


def _stats(values, valid):
    count = valid.sum(axis=0)
    denominator = np.maximum(count, 1)[..., None]
    mean = np.sum(np.where(valid[..., None], values, 0), axis=0) / denominator
    variance = np.sum(np.where(valid[..., None], (values - mean) ** 2, 0), axis=0) / denominator
    low = np.min(np.where(valid[..., None], values, np.inf), axis=0, initial=np.inf)
    high = np.max(np.where(valid[..., None], values, -np.inf), axis=0, initial=-np.inf)
    span = np.max(np.where(count[..., None] > 0, high - low, 0), axis=-1)
    return count, np.sqrt(np.mean(variance, axis=-1)), span


def _residual(observed, background, gain, offset):
    return np.sqrt(np.mean((observed - gain[None, ..., None] * background - offset[None]) ** 2, axis=-1))


def _physical(gain, offset):
    gain = np.clip(gain, 0, 1)
    offset = np.minimum(np.maximum(offset, 0), (1 - gain)[..., None] * 255)
    return gain.astype(np.float32), offset.astype(np.float32)


def _weighted_fit(observed, background, weights, previous_gain):
    weight_sum = np.sum(weights, axis=0)
    normalizer = np.maximum(weight_sum, 1e-8)[..., None]
    mean_b = np.sum(weights[..., None] * background, axis=0) / normalizer
    mean_i = np.sum(weights[..., None] * observed, axis=0) / normalizer
    centered_b, centered_i = background - mean_b, observed - mean_i
    denominator = np.sum(weights[..., None] * centered_b ** 2, axis=(0, 3))
    numerator = np.sum(weights[..., None] * centered_b * centered_i, axis=(0, 3))
    gain = np.where(denominator > 1e-6, numerator / np.maximum(denominator, 1e-6), previous_gain)
    gain = np.clip(gain, 0, 1)
    return (*_physical(gain, mean_i - gain[..., None] * mean_b), denominator)


def _candidate_pairs(background, valid, limit, minimum_squared_contrast):
    """Select contrasting pairs from each pixel's own valid observations.

    At most limit candidates are scored per pixel. Pair discovery is quadratic in
    that pixel's valid observation count, not the full frame count. Scratch pair
    arrays cover only one pixel; no frames-squared by image-sized array is built.
    """
    _, height, width = valid.shape
    first = np.zeros((limit, height, width), np.int32)
    second = np.zeros_like(first)
    eligible_count = np.zeros((height, width), np.int32)
    selected_count = np.zeros_like(eligible_count)
    for y, x in np.argwhere(valid.sum(axis=0) >= 2):
        indices = np.flatnonzero(valid[:, y, x])
        a, b = np.triu_indices(len(indices), k=1)
        a, b = indices[a], indices[b]
        contrast = np.sum((background[a, y, x] - background[b, y, x]) ** 2, axis=-1)
        keep = np.flatnonzero(contrast >= minimum_squared_contrast)
        eligible_count[y, x] = len(keep)
        if not len(keep):
            continue
        count = min(limit, len(keep))
        # IDs were sorted before this function; selection is stable under input
        # permutation and depends only on background validity/contrast, not I.
        selection = keep[np.rint(np.linspace(0, len(keep) - 1, count)).astype(int)]
        first[:count, y, x], second[:count, y, x] = a[selection], b[selection]
        selected_count[y, x] = count
    count = int(selected_count.max(initial=0))
    # Padding is (frame 0, frame 0), with zero contrast, and cannot win a fit.
    return list(zip(first[:count], second[:count])), eligible_count, selected_count


def fit_overlay(observed, background, valid, frame_ids, *,
                min_train=3, min_heldout=2, heldout_fraction=.25,
                min_background_std=8., min_background_span=24.,
                min_heldout_span=16., inlier_rms=5., huber_delta=2.5,
                max_train_rms=4., max_heldout_rms=6., min_consensus_fraction=.6,
                min_heldout_inlier_fraction=.75, min_gain=.4, min_alpha=.02,
                min_confidence=.5, max_inverse_rms=12., max_pair_candidates=48):
    """Fit independent pixels in aligned N,H,W,3 BGR arrays on a 0..255 scale.

    valid is N,H,W. frame_ids is a unique integer N-vector of observation frames,
    not donor IDs. At least three training inliers and two held-out observations
    are required by default. Sorting IDs makes the split invariant to input order.
    Background estimator inputs should also avoid train/heldout information leaks;
    this module can enforce observation-frame separation only.

    Returns float32 gain(H,W), offset(H,W,3), alpha(H,W), color(H,W,3), confidence;
    bool support is the accepted inversion mask; bool identified includes valid
    opaque estimates; train/heldout count, span, RMS and gain_std_error maps; plus
    diagnostics with thresholds, disjoint frame IDs and nonexclusive rejection counts.
    offset is PREMULTIPLIED BGR, never unassociated watermark color.
    Unidentified gain/offset are reset to 1/0, but their metric maps remain available.
    """
    observed = np.asarray(observed, dtype=np.float32)
    background = np.asarray(background, dtype=np.float32)
    valid = np.array(valid, dtype=bool, copy=True)
    frame_ids = np.asarray(frame_ids)
    if observed.ndim != 4 or observed.shape[-1] != 3 or observed.shape != background.shape:
        raise ValueError("observed/background must have equal N,H,W,3 shapes")
    n, height, width, _ = observed.shape
    if n == 0 or height == 0 or width == 0 or valid.shape != observed.shape[:3]:
        raise ValueError("Nonempty inputs and an N,H,W valid array are required")
    if frame_ids.shape != (n,) or not np.issubdtype(frame_ids.dtype, np.integer):
        raise ValueError("frame_ids must be a unique integer vector")
    if len(np.unique(frame_ids)) != n:
        raise ValueError("Repeated observation-frame IDs are not independent observations")
    if not (min_train >= 3 and min_heldout >= 2 and 0 < heldout_fraction < 1
            and 0 < min_gain <= 1 and max_pair_candidates >= 1):
        raise ValueError("Invalid sample, holdout or stability settings")
    finite = np.isfinite(observed).all(axis=-1) & np.isfinite(background).all(axis=-1)
    in_range = (observed >= 0).all(axis=-1) & (observed <= 255).all(axis=-1)
    in_range &= (background >= 0).all(axis=-1) & (background <= 255).all(axis=-1)
    valid &= finite & in_range
    observed = np.where(valid[..., None], observed, 0)
    background = np.where(valid[..., None], background, 0)
    order = np.argsort(frame_ids)
    observed, background, valid, frame_ids = observed[order], background[order], valid[order], frame_ids[order]
    # Holdout positions are fixed by IDs/count, never by residuals or background.
    hold_count = min(n, max(min_heldout, int(np.ceil(n * heldout_fraction))))
    held = np.unique(np.rint(np.linspace(0, n - 1, hold_count)).astype(int))
    training = np.setdiff1d(np.arange(n), held)
    train_i, train_b, train_valid = observed[training], background[training], valid[training]
    held_i, held_b, held_valid = observed[held], background[held], valid[held]
    train_count, _, _ = _stats(train_b, train_valid)
    held_count, held_std, held_span = _stats(held_b, held_valid)
    gain = np.ones((height, width), np.float32)
    offset = np.zeros((height, width, 3), np.float32)
    best_count = np.zeros((height, width), np.int32)
    best_score = np.full((height, width), np.inf, np.float32)
    candidate_pairs, eligible_pair_count, selected_pair_count = _candidate_pairs(
        train_b, train_valid, max_pair_candidates, min_background_span ** 2)
    yy, xx = np.indices((height, width))
    for a, b in candidate_pairs:
        delta_b = train_b[a, yy, xx] - train_b[b, yy, xx]
        delta_i = train_i[a, yy, xx] - train_i[b, yy, xx]
        denominator = np.sum(delta_b ** 2, axis=-1)
        eligible = train_valid[a, yy, xx] & train_valid[b, yy, xx] & (denominator >= min_background_span ** 2)
        trial_gain = np.sum(delta_b * delta_i, axis=-1) / np.maximum(denominator, 1e-6)
        trial_gain = np.clip(trial_gain, 0, 1)
        trial_offset = ((train_i[a, yy, xx] + train_i[b, yy, xx])
                        - trial_gain[..., None] * (train_b[a, yy, xx] + train_b[b, yy, xx])) / 2
        trial_gain, trial_offset = _physical(trial_gain, trial_offset)
        residual = _residual(train_i, train_b, trial_gain, trial_offset)
        count = np.sum(train_valid & (residual <= inlier_rms), axis=0)
        score = np.sum(np.where(train_valid, np.minimum(residual, inlier_rms), 0), axis=0) / np.maximum(train_count, 1)
        better = eligible & ((count > best_count) | ((count == best_count) & (score < best_score)))
        gain[better], offset[better] = trial_gain[better], trial_offset[better]
        best_count[better], best_score[better] = count[better], score[better]
    denominator = np.zeros((height, width), np.float32)
    for _ in range(5):
        residual = _residual(train_i, train_b, gain, offset)
        inliers = train_valid & (residual <= inlier_rms)
        weights = inliers.astype(np.float32) * np.minimum(1, huber_delta / np.maximum(residual, 1e-6))
        gain, offset, denominator = _weighted_fit(train_i, train_b, weights, gain)
    train_residual = _residual(train_i, train_b, gain, offset)
    inliers = train_valid & (train_residual <= inlier_rms)
    inlier_count, background_std, background_span = _stats(train_b, inliers)
    train_rms = np.sqrt(np.sum(np.where(inliers, train_residual ** 2, 0), axis=0) / np.maximum(inlier_count, 1))
    train_rms = np.where(inlier_count > 0, train_rms, np.inf)
    held_residual = _residual(held_i, held_b, gain, offset)
    held_rms = np.sqrt(np.sum(np.where(held_valid, held_residual ** 2, 0), axis=0) / np.maximum(held_count, 1))
    held_rms = np.where(held_count > 0, held_rms, np.inf)
    held_inliers = np.sum(held_valid & (held_residual <= max_heldout_rms), axis=0)
    held_fraction = held_inliers / np.maximum(held_count, 1)
    gain_std_error = train_rms / np.sqrt(np.maximum(denominator, 1e-6))
    checks = {
        "enough_training_inliers": inlier_count >= min_train,
        "training_consensus": inlier_count >= np.ceil(train_count * min_consensus_fraction),
        "training_background_std": background_std >= min_background_std,
        "training_background_span": background_span >= min_background_span,
        "training_error": train_rms <= max_train_rms,
        "enough_heldout": held_count >= min_heldout,
        "heldout_background_span": held_span >= min_heldout_span,
        "heldout_error": held_rms <= max_heldout_rms,
        "heldout_inlier_fraction": held_fraction >= min_heldout_inlier_fraction,
    }
    identified = np.logical_and.reduce(list(checks.values()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        confidence = np.minimum.reduce([
            np.minimum(1, background_std / (2 * min_background_std)),
            np.minimum(1, held_span / (2 * min_heldout_span)),
            np.exp(-.5 * (train_rms / max_train_rms) ** 2),
            np.exp(-.5 * (held_rms / max_heldout_rms) ** 2),
            np.minimum(1, .05 / np.maximum(gain_std_error, 1e-6)),
        ])
    confidence = np.where(identified, np.nan_to_num(confidence), 0).astype(np.float32)
    alpha = 1 - gain
    inverse_rms = held_rms / np.maximum(gain, 1e-6)
    support = identified & (gain >= min_gain) & (alpha >= min_alpha)
    support &= (confidence >= min_confidence) & (inverse_rms <= max_inverse_rms)
    checks.update({"stable_gain": gain >= min_gain, "nonidentity_overlay": alpha >= min_alpha,
                   "confidence": confidence >= min_confidence, "predicted_inverse_error": inverse_rms <= max_inverse_rms})
    gain = np.where(identified, gain, 1).astype(np.float32)
    offset = np.where(identified[..., None], offset, 0).astype(np.float32)
    alpha = (1 - gain).astype(np.float32)
    color = np.divide(offset, np.maximum(alpha[..., None], 1e-6), out=np.zeros_like(offset), where=alpha[..., None] > 1e-6)
    parameters = {"min_train": min_train, "min_heldout": min_heldout, "heldout_fraction": heldout_fraction,
                  "min_background_std": min_background_std, "min_background_span": min_background_span,
                  "min_heldout_span": min_heldout_span, "inlier_rms": inlier_rms, "huber_delta": huber_delta,
                  "max_train_rms": max_train_rms, "max_heldout_rms": max_heldout_rms,
                  "min_consensus_fraction": min_consensus_fraction, "min_heldout_inlier_fraction": min_heldout_inlier_fraction,
                  "min_gain": min_gain, "min_alpha": min_alpha, "min_confidence": min_confidence,
                  "max_inverse_rms": max_inverse_rms, "max_pair_candidates": max_pair_candidates}
    return {"gain": gain, "offset": offset, "alpha": alpha, "color": color,
            "support": support, "identified": identified, "confidence": confidence,
            "train_count": train_count, "train_inlier_count": inlier_count,
            "background_std": background_std.astype(np.float32), "background_span": background_span.astype(np.float32),
            "train_rms": train_rms.astype(np.float32), "heldout_rms": held_rms.astype(np.float32),
            "heldout_count": held_count, "heldout_background_span": held_span.astype(np.float32),
            "heldout_inlier_fraction": held_fraction.astype(np.float32), "gain_std_error": gain_std_error.astype(np.float32),
            "predicted_inverse_rms": inverse_rms.astype(np.float32),
            "eligible_pair_count": eligible_pair_count, "selected_pair_count": selected_pair_count,
            "diagnostics": {"equation": "observed = shared_RGB_gain * background + premultiplied_BGR_offset",
                            "parameters": parameters, "frame_ids": frame_ids.tolist(),
                            "training_frame_ids": frame_ids[training].tolist(), "heldout_frame_ids": frame_ids[held].tolist(),
                            "heldout_used_for_fitting": False, "sample_pairs_evaluated": len(candidate_pairs),
                            "pair_strategy": "Per-pixel valid contrasting pairs; deterministic uniform selection, at most max_pair_candidates per pixel",
                            "pixels_with_eligible_pair": int((eligible_pair_count > 0).sum()),
                            "pixels_with_selected_pair": int((selected_pair_count > 0).sum()),
                            "pixels_with_all_eligible_pairs_evaluated": int(((eligible_pair_count > 0) & (eligible_pair_count <= max_pair_candidates)).sum()),
                            "selected_pairs_total": int(selected_pair_count.sum()),
                            "pixels": height * width, "identified_pixels": int(identified.sum()),
                            "invertible_pixels": int(support.sum()),
                            "identified_low_gain_pixels": int((identified & (gain < min_gain)).sum()),
                            "rejection_counts_nonexclusive": {key: int((~passed).sum()) for key, passed in checks.items()},
                            "limitations": ["Background estimates are not ground truth; common bias may satisfy this model.",
                                            "Frame-ID separation does not ensure statistical independence of correlated video frames or donors.",
                                            "Compression/resampling can violate pointwise alpha compositing.",
                                            "Confidence is consistency, not a guarantee of restored text or texture."]}}


def invert_overlay(observed, gain, offset, support, *, min_gain=.4, max_overflow=12.):
    """Return float32 reconstruction and the actual used mask, exact outside it."""
    observed = np.asarray(observed, dtype=np.float32)
    gain, offset, support = np.asarray(gain), np.asarray(offset), np.asarray(support, bool)
    if observed.ndim != 3 or observed.shape[-1] != 3 or gain.shape != observed.shape[:2]:
        raise ValueError("Expected H,W,3 observed and H,W gain")
    if offset.shape != observed.shape or support.shape != gain.shape:
        raise ValueError("Offset or support dimensions do not match")
    if not 0 < min_gain <= 1 or max_overflow < 0:
        raise ValueError("Invalid inversion stability settings")
    used = support & np.isfinite(gain) & (gain >= min_gain) & (gain <= 1)
    used &= np.isfinite(observed).all(axis=-1) & np.isfinite(offset).all(axis=-1)
    safe_gain = np.where(used, gain, 1)
    raw = (observed - np.where(used[..., None], offset, 0)) / safe_gain[..., None]
    used &= (raw >= -max_overflow).all(axis=-1) & (raw <= 255 + max_overflow).all(axis=-1)
    result = observed.copy()
    result[used] = np.clip(raw[used], 0, 255)
    return result, used
