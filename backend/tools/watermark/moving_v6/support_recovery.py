"""Recover rejected observations from immutable reference-image evidence.

An original nearby detection only bounds the search. Every accepted position
is a unique image peak, with independent regional and signed-contour checks.
Recovered records never become search anchors for another rejected frame.
"""
import numpy as np

from .tracking_bridge import _match


def recover_reference_support(records, frame_at, reference_gray, roi, support,
                              cut_prefix, fps, *, search_margin=16):
    x0, y0, x1, y1 = roi
    template = reference_gray[y0:y1, x0:x1].astype(np.float32)
    support = np.asarray(support, np.float32)
    if (template.shape != support.shape or not np.isfinite(support).all()
            or (support < 0).any()):
        raise ValueError('Reference support dimensions or mass are invalid')
    output = {index: dict(record) for index, record in records.items()}
    if support.sum() < 24:
        return output, dict(recovered_frames=[], recovered_count=0,
                            skipped_reason='insufficient_reference_support')
    originals = [index for index, record in records.items()
                 if record.get('detector_accepted', record['accepted'])]
    limit = max(1, round(.6 * fps))
    recovered = []
    for index, record in sorted(records.items()):
        if record['accepted']:
            continue
        anchors = [other for other in originals if abs(other-index) <= limit
                   and cut_prefix[other] == cut_prefix[index]]
        if not anchors:
            continue
        nearest = min(anchors, key=lambda other: (abs(other-index), other))
        observation = _match(frame_at(index), template, support,
                             records[nearest]['anchor_subpixel_xy'], search_margin)
        output[index]['support_evidence'] = dict(observation, anchor=nearest)
        if observation['accepted']:
            output[index].update(accepted=True, detector_accepted=False,
                                 acceptance='observed_support', search='local_support',
                                 anchor_subpixel_xy=observation['position'])
            recovered.append(index)
    return output, dict(recovered_frames=recovered, recovered_count=len(recovered))
