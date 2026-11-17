"""Image-verified bidirectional bridging of bounded translation-track gaps.

Only original detector matches can anchor a gap. Motion predicts search centers;
it never supplies an accepted position. Each committed position must be observed
by two independently propagated image-template paths which close on both anchors.
"""
import time

import cv2
import numpy as np

from .track_logo import subpixel_peak


def _canonical(gray, position, shape):
    height, width = shape
    # getRectSubPix rounds the global center to float32. Preserve that rounding
    # before translating into a crop, so limiting the conversion is pixel exact.
    center = np.asarray(np.asarray(position) + [(width - 1) / 2, (height - 1) / 2], np.float32)
    left = center - [(width - 1) / 2, (height - 1) / 2]
    x0, y0 = max(0, int(np.floor(left[0]))), max(0, int(np.floor(left[1])))
    x1, y1 = min(gray.shape[1], int(np.ceil(left[0] + width))), min(gray.shape[0], int(np.ceil(left[1] + height)))
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Reference patch does not intersect the source frame')
    return cv2.getRectSubPix(gray[y0:y1, x0:x1].astype(np.float32), (width, height),
                             tuple(center - [x0, y0]))


def _gradient(image):
    smooth = cv2.GaussianBlur(image, (3, 3), .6)
    x = cv2.Sobel(smooth, cv2.CV_32F, 1, 0, ksize=3, scale=.125)
    y = cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3, scale=.125)
    return x, y, np.hypot(x, y)


def _ncc(image, template, weight):
    mass = float(weight.sum())
    shape = (image.shape[0] - template.shape[0] + 1,
             image.shape[1] - template.shape[1] + 1)
    if mass < 16:
        return np.full(shape, -1., np.float32)
    centered = template - float(np.sum(template * weight) / mass)
    energy = float(np.sum(centered * centered * weight))
    if energy <= mass:
        return np.full(shape, -1., np.float32)
    total = cv2.matchTemplate(image, weight, cv2.TM_CCORR)
    square = cv2.matchTemplate(image * image, weight, cv2.TM_CCORR)
    covariance = cv2.matchTemplate(image, centered * weight, cv2.TM_CCORR)
    variance = np.maximum(square - total * total / mass, 0.)
    score = covariance / np.sqrt(np.maximum(variance * energy, 1.e-6))
    score[variance <= mass] = -1.
    return np.clip(score, -1., 1.)


def _edge_recall(template, candidate, support):
    ax, ay, an = _gradient(template)
    bx, by, bn = _gradient(candidate)
    values = template[support > 0]
    contrast = float(np.percentile(values, 95)-np.percentile(values, 5))
    # A contrast-relative contour floor handles dark/translucent overlays while
    # the absolute floor excludes quantization noise. Recall itself stays .90.
    edges = (support > 0) & (an >= max(3., .10*contrast))
    same = (ax*bx+ay*by >= .8*an*bn) & (bn >= .2*an)
    return float(same[edges].mean()) if edges.sum() >= 24 else 0.


def _match(gray, template, support, predicted, search_margin):
    height, width = template.shape
    px, py = predicted
    x0, y0 = max(0, int(np.floor(px-search_margin))), max(0, int(np.floor(py-search_margin)))
    x1 = min(gray.shape[1], int(np.ceil(px+width+search_margin)))
    y1 = min(gray.shape[0], int(np.ceil(py+height+search_margin)))
    if x1-x0 < width or y1-y0 < height:
        return dict(accepted=False, reason='insufficient_search_area')
    response = _ncc(gray[y0:y1, x0:x1].astype(np.float32), template, support)
    _, confidence, _, location = cv2.minMaxLoc(response)
    position = subpixel_peak(response, location) + [x0, y0]
    sx, sy = location
    response[max(0, sy-4):sy+5, max(0, sx-4):sx+5] = -1
    peak_gap = float(confidence-response.max())
    candidate = _canonical(gray, position, template.shape)
    regions = []
    for ya, yb in ((0, height//2), (height//2, height)):
        for xa, xb in ((0, width//2), (width//2, width)):
            weight = np.zeros_like(support)
            weight[ya:yb, xa:xb] = support[ya:yb, xa:xb]
            if weight.sum() < 16:
                continue
            mean = float(np.sum(template*weight)/weight.sum())
            if np.sum((template-mean)**2*weight)/weight.sum() >= 9:
                regions.append(float(_ncc(candidate, template, weight)[0, 0]))
    recall = _edge_recall(template, candidate, support)
    accepted = confidence >= .70 and peak_gap >= .05 and len(regions) >= 2
    accepted = accepted and min(regions, default=-1.) >= .60 and recall >= .90
    return dict(accepted=bool(accepted), position=position.tolist(), confidence=float(confidence),
                peak_gap=peak_gap, region_scores=regions, edge_recall=recall,
                reason='observed_image' if accepted else 'image_evidence_rejected')


def bridge_tracking_gaps(records, frame_at, support, cut_prefix, fps, *,
                         max_gap_seconds=.6, max_gap_frames=24, search_margin=16,
                         max_position_error=.6, max_velocity_error=1.,
                         max_path_deviation=2., max_template_age=3):
    """Return copied records and diagnostics; the caller's records are immutable.

    ``frame_at(index)`` supplies original BGR or grayscale frames on demand.
    ``cut_prefix`` is the cumulative scene-cut count used by donor selection.
    Records must come from the original detector, not a previous bridge pass.
    At most one bounded gap and its anchors are retained, never the full video.
    Rejected/skipped observations cannot update either template path. Closure
    and agreement validate a path but do not authorize filling its missing frames.
    """
    if fps <= 0 or max_gap_seconds <= 0 or max_gap_frames < 1 or search_margin < 5:
        raise ValueError('Invalid bounded tracking configuration')
    if min(max_position_error, max_velocity_error, max_path_deviation) <= 0 or max_template_age < 1:
        raise ValueError('Invalid tracking evidence tolerances')
    if any(record.get('acceptance') == 'bridged' or
           (record['accepted'] and record.get('detector_accepted') is False)
           for record in records.values()):
        raise ValueError('Original detector records required; bridged records cannot become anchors')
    support = np.asarray(support, np.float32)
    if support.ndim != 2 or not np.isfinite(support).all() or (support < 0).any():
        raise ValueError('Invalid reference support')
    started = time.perf_counter()
    output = {index: dict(record, detector_accepted=bool(record['accepted']),
                          acceptance='matched' if record['accepted'] else 'rejected')
              for index, record in records.items()}
    limit = min(max_gap_frames, max(1, round(max_gap_seconds*fps)))
    if support.sum() < 24:
        # Sparse valid logos can still satisfy the original detector. Insufficient
        # corroborating evidence disables this optional recovery pass only.
        return output, dict(original_tracked_frames=sum(record['accepted'] for record in records.values()),
                            bridged_frames=[], bridged_count=0, gaps=[], max_gap_frames=limit,
                            skipped_reason='insufficient_reference_support',
                            seconds=time.perf_counter()-started)
    indices = sorted(records)
    reports, bridged = [], []
    cursor = 0
    while cursor < len(indices):
        if records[indices[cursor]]['accepted']:
            cursor += 1
            continue
        begin = cursor
        while cursor < len(indices) and not records[indices[cursor]]['accepted']:
            cursor += 1
        missing = indices[begin:cursor]
        left, right = missing[0]-1, missing[-1]+1
        detail = dict(frames=missing, anchors=[left, right], accepted_frames=[])

        def reject(reason):
            detail['reason'] = reason
            for index in missing:
                output[index]['bridge'] = dict(reason=reason, anchors=[left, right])
            reports.append(detail)

        if len(missing) > limit:
            reject('gap_too_long')
            continue
        context = (left-2, left, right, right+2)
        if (missing != list(range(left+1, right)) or
                any(index not in records or not records[index]['accepted'] for index in context)):
            reject('missing_original_anchors')
            continue
        if any(records[index]['confidence'] < .70 or records[index]['peak_gap'] < .05
               for index in (left, right)):
            reject('uncertain_original_anchors')
            continue
        if cut_prefix[left-2] != cut_prefix[right+2]:
            reject('scene_cut')
            continue
        points = [np.asarray(records[index]['anchor_subpixel_xy'], np.float64) for index in context]
        if any(point.shape != (2,) or not np.isfinite(point).all() for point in points):
            reject('invalid_original_positions')
            continue
        before, after = (points[1]-points[0])/2, (points[3]-points[2])/2
        velocity = (points[2]-points[1])/(right-left)
        velocity_error = max(np.linalg.norm(before-after), np.linalg.norm(before-velocity),
                             np.linalg.norm(after-velocity))
        detail['velocity_error'] = float(velocity_error)
        if velocity_error > max_velocity_error:
            reject('incompatible_anchor_motion')
            continue
        frames = {}
        for index in range(left, right+1):
            image = frame_at(index)
            frames[index] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        def propagate(start, end, step):
            position = np.asarray(records[start]['anchor_subpixel_xy'], np.float64)
            template = _canonical(frames[start], position, support.shape)
            original_template = template.copy()
            template_index = start
            observations = {}
            for index in range(start+step, end+step, step):
                if abs(index-template_index) > max_template_age:
                    observations[index] = dict(accepted=False, reason='template_chain_broken')
                    continue
                predicted = points[1]+velocity*(index-left)
                observation = _match(frames[index], template, support, predicted, search_margin)
                observation['template_frame'] = template_index
                if observation['accepted']:
                    deviation = float(np.linalg.norm(np.asarray(observation['position'])-predicted))
                    observation['path_deviation'] = deviation
                    candidate = _canonical(frames[index], observation['position'], support.shape)
                    original_recall = _edge_recall(original_template, candidate, support)
                    observation['original_anchor_edge_recall'] = original_recall
                    if deviation > max_path_deviation:
                        observation.update(accepted=False, reason='trajectory_deviation')
                    elif original_recall < .90:
                        # Adjacent agreement and endpoint closure alone can track
                        # a glyph that gradually morphs away and then returns.
                        observation.update(accepted=False, reason='original_anchor_shape_changed')
                    else:
                        position = np.asarray(observation['position'])
                        template = candidate
                        template_index = index
                observations[index] = observation
            final = observations[end]
            error = (float(np.linalg.norm(np.asarray(final['position'])-records[end]['anchor_subpixel_xy']))
                     if final['accepted'] else float('inf'))
            return observations, error

        forward, forward_error = propagate(left, right, 1)
        backward, backward_error = propagate(right, left, -1)
        detail['closure_error'] = [value if np.isfinite(value) else None
                                   for value in (forward_error, backward_error)]
        closed = max(forward_error, backward_error) <= max_position_error
        detail['observations'] = {}
        for index in missing:
            a, b = forward[index], backward[index]
            evidence = dict(forward=a, backward=b, anchors=[left, right])
            reason = 'endpoint_closure_failed' if not closed else 'bidirectional_image_evidence_rejected'
            if closed and a['accepted'] and b['accepted']:
                error = float(np.linalg.norm(np.asarray(a['position'])-b['position']))
                evidence['bidirectional_error'] = error
                reason = 'bidirectional_position_disagreement'
                if error <= max_position_error:
                    # Commit an actually observed peak, not a positional average.
                    chosen = min((a, b), key=lambda item: (-item['confidence'], item['template_frame']))
                    output[index].update(accepted=True, acceptance='bridged',
                                         anchor_subpixel_xy=chosen['position'], search='bidirectional_bridge')
                    evidence['observed_confidence'] = min(a['confidence'], b['confidence'])
                    evidence['observed_peak_gap'] = min(a['peak_gap'], b['peak_gap'])
                    detail['accepted_frames'].append(index)
                    bridged.append(index)
                    reason = 'bidirectional_observed_bridge'
            evidence['reason'] = reason
            output[index]['bridge'] = evidence
            detail['observations'][index] = evidence
        detail['reason'] = 'verified_bridge' if detail['accepted_frames'] else 'no_verified_observations'
        reports.append(detail)
    return output, dict(original_tracked_frames=sum(record['accepted'] for record in records.values()),
                        bridged_frames=bridged, bridged_count=len(bridged), gaps=reports,
                        max_gap_frames=limit, seconds=time.perf_counter()-started)
