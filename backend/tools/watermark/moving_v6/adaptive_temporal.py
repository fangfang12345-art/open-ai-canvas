"""Select corroborating visible backgrounds within verified scene boundaries."""
import time

import cv2
import numpy as np

from .coherent_repair import repair_patch


def rank_donors(target, target_mask, pool, *, max_donors=12, min_gap=3):
    """Cheap coverage/context ranking; final registration must still validate it.

    Coverage counts are capped at two: a second independent observation is useful,
    but repeated near-identical masks should not crowd out a newly exposed region.
    Masks are conservative when reduced; only visible pixels affect context costs.
    """
    shape = (max(1, target.shape[1] // 4), max(1, target.shape[0] // 4))
    small = cv2.resize(target.astype(np.float32), shape, interpolation=cv2.INTER_AREA)
    target_blocked = cv2.dilate(target_mask.astype(np.uint8), np.ones((9, 9), np.uint8))
    blocked = cv2.resize(target_blocked.astype(np.float32), shape,
                         interpolation=cv2.INTER_AREA) > 0
    demand = target_mask > 0
    total = max(1, int(demand.sum()))
    ranked = []
    for donor in pool:
        if not donor.get('cut_ok', True) or donor.get('index') is None:
            continue
        dm = np.asarray(donor['mask'], bool)
        if dm.shape != target_mask.shape or donor['image'].shape != target.shape:
            raise ValueError('Donor shape disagrees with target')
        exclusion = cv2.dilate(dm.astype(np.uint8), np.ones((9, 9), np.uint8))
        small_mask = cv2.resize(exclusion.astype(np.float32), shape,
                                interpolation=cv2.INTER_AREA) > 0
        visible = ~(blocked | small_mask)
        if visible.sum() < 24:
            continue
        other = cv2.resize(donor['image'].astype(np.float32), shape,
                           interpolation=cv2.INTER_AREA)
        delta = np.mean(abs(other - small), axis=2)[visible]
        mean, bad = float(np.mean(delta)), float(np.mean(delta > 18))
        # Unregistered pixel differences cannot reject a moving background.
        # Scene boundaries are already checked, and the final registered local
        # fit plus independent-donor agreement still decide pixel eligibility.
        reveal = demand & (cv2.dilate(dm.astype(np.uint8), np.ones((5, 5), np.uint8)) == 0)
        ranked.append(dict(donor=donor, reveal=reveal, mean=mean, bad=bad,
                           uncovered=int(reveal.sum())))
    chosen, seen = [], set()
    counts = np.zeros(target_mask.shape, np.uint8)
    while ranked and len(chosen) < max_donors:
        best, best_score = None, -float('inf')
        for item in ranked:
            index = int(item['donor']['index'])
            if index in seen or any(abs(index - j) < min_gap for j in seen):
                continue
            benefit = float(np.sum((2 - counts[item['reveal']]) / 2)) / total
            fidelity = 1 / (1 + item['mean'] / 6 + item['bad'])
            # Retain a small value for motion-explained candidates even when
            # identity geometry predicts no new background exposure.
            score = (benefit + .08) * fidelity
            score -= .00015 * abs(item['donor'].get('distance', 0))
            if score > best_score:
                best, best_score = item, score
        if best is None:
            break
        ranked = [item for item in ranked if item is not best]
        chosen.append(best)
        seen.add(int(best['donor']['index']))
        counts[best['reveal']] = np.minimum(2, counts[best['reveal']] + 1)
    report = [dict(index=int(x['donor']['index']), distance=int(x['donor']['distance']),
                   coarse_context_mae=x['mean'], identity_uncovered_pixels=x['uncovered'])
              for x in chosen]
    return [x['donor'] for x in chosen], report


class AdaptiveTemporalRepair:
    """Use the immutable V4 decoded cache, masks and detected scene boundaries."""

    def __init__(self, cache, fps, *, search_seconds=4., max_donors=12):
        if fps <= 0 or search_seconds <= 0 or max_donors < 2:
            raise ValueError('Invalid adaptive temporal configuration')
        self.cache = cache
        self.fps = float(fps)
        self.horizon = max(3, round(search_seconds * fps))
        self.step = max(1, round(fps / 10))
        self.max_donors = int(max_donors)

    def repair(self, frame_index, base_frame):
        started = time.perf_counter()
        cache = self.cache
        if frame_index not in cache.masks:
            return base_frame.copy(), dict(roi=[0, 0, 0, 0], recovered_mask=np.zeros((0, 0), bool),
                                           coverage=0., recovered_pixels=0, reason='untracked_target')
        box, _ = cache.masks[frame_index]
        pad = cache.context_pad
        roi = [max(0, box[0]-pad), max(0, box[1]-pad),
               min(cache.width, box[2]+pad), min(cache.height, box[3]+pad)]
        x0, y0, x1, y1 = roi
        ux, uy, _, _ = cache.union
        target = base_frame[y0:y1, x0:x1]
        mask = cache._local_mask(frame_index, roi)
        first, last = max(0, frame_index-self.horizon), min(cache.frame_count, frame_index+self.horizon+1)
        indices = set(range(first, last, self.step))
        for offset in cache.offsets:
            indices.update((frame_index-offset, frame_index+offset))
        pool = []
        for index in sorted(indices):
            if (index == frame_index or not first <= index < last or index not in cache.masks
                    or abs(index-frame_index) < 3
                    or cache.cut_prefix[index] != cache.cut_prefix[frame_index]):
                continue
            image = np.asarray(cache.cache[index, y0-uy:y1-uy, x0-ux:x1-ux])
            pool.append(dict(index=index, distance=index-frame_index, cut_ok=True,
                             image=image, mask=cache._local_mask(index, roi)))
        donors, selection = rank_donors(target, mask, pool, max_donors=self.max_donors)
        selected_at = time.perf_counter()
        patch, diagnostic = repair_patch(target, mask, donors)
        result = base_frame.copy()
        result[y0:y1, x0:x1] = patch
        diagnostic.update(roi=roi, frame=int(frame_index), selection=selection,
                          selection_pool=len(pool), selection_seconds=selected_at-started,
                          adaptive_seconds=time.perf_counter()-started,
                          max_selected_distance=max((abs(x['distance']) for x in selection), default=0),
                          search_horizon_frames=self.horizon)
        return result, diagnostic
