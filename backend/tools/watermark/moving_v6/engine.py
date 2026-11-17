"""Source-derived contour identity, dynamic masks and evidence-gated repair.

Tracking and donor selection use independent evidence: overlay identity can
survive a background cut, while temporal background donors remain isolated.
No saved tracks, masks, frame caches or radiometric models are inputs.
"""
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import json
import threading

import cv2
import numpy as np

from media import Reader
from .adaptive_temporal import AdaptiveTemporalRepair
from .affine_overlay import fit_overlay, invert_overlay
from .spatial_continuity import repair as spatial_repair
from .track_logo import TemplateAnchorTracker, reference_support
from .tracking_bridge import bridge_tracking_gaps
from .support_recovery import recover_reference_support
from .contour_identity import verify_contour_tracks


class PlacedMasks:
    def __init__(self, records, support, width, height, max_bytes=8 * 1024 * 1024, max_entries=256):
        self.records, self.support = records, support
        self.width, self.height = width, height
        self.float_support = support.astype(np.float32)
        self.cached = OrderedDict()
        self.max_bytes, self.max_entries, self.used = max_bytes, max_entries, 0
        self.lock = threading.Lock()

    def __contains__(self, index):
        return index in self.records and self.records[index]["accepted"]

    def bounds(self, index):
        xy = self.records[index]["anchor_subpixel_xy"]
        h, w = self.support.shape
        # Include fractional interpolation support instead of clipping its halo.
        x0, y0 = max(0, int(np.floor(xy[0])) - 1), max(0, int(np.floor(xy[1])) - 1)
        x1, y1 = min(self.width, int(np.ceil(xy[0] + w)) + 1), min(self.height, int(np.ceil(xy[1] + h)) + 1)
        return [x0, y0, x1, y1]

    def __getitem__(self, index):
        with self.lock:
            if index in self.cached:
                self.cached.move_to_end(index)
                box, mask = self.cached[index]
                return list(box), mask
            box = self.bounds(index)
            x0, y0, x1, y1 = box
            xy = self.records[index]["anchor_subpixel_xy"]
            transform = np.float32([[1, 0, xy[0] - x0], [0, 1, xy[1] - y0]])
            mask = cv2.warpAffine(self.float_support, transform, (x1 - x0, y1 - y0),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > .01
            mask.setflags(write=False)
            if mask.nbytes <= self.max_bytes and self.max_entries > 0:
                while self.cached and (self.used + mask.nbytes > self.max_bytes
                                       or len(self.cached) >= self.max_entries):
                    self.used -= self.cached.popitem(last=False)[1][1].nbytes
                self.cached[index] = (tuple(box), mask)
                self.used += mask.nbytes
            return box, mask

    def close(self):
        with self.lock:
            self.cached.clear()
            self.used = 0


class FrameCache:
    """Cache original context pixels, retaining absolute NumPy slice semantics.

    A compact tracked union fits many donor frames in memory. For a trajectory
    spanning most of the image, cache only requested fixed-size tiles instead.
    Full-frame calibration reads never populate a full image of tiles.
    """
    def __init__(self, source, max_bytes=96 * 1024 * 1024, *, region=None,
                 min_frames=128, tile_size=128):
        self.reader = Reader(source)
        self.frames = OrderedDict()
        self.lock = threading.Lock()
        self.max_bytes, self.used = max(0, max_bytes), 0
        self.region = tuple(region) if region is not None else None
        self.min_frames, self.tile_size = max(1, min_frames), max(1, tile_size)
        self.shape = None

    def _remember(self, key, patch):
        if key in self.frames:
            self.frames.move_to_end(key)
            return
        if patch.nbytes > self.max_bytes:
            return
        while self.frames and self.used + patch.nbytes > self.max_bytes:
            self.used -= self.frames.popitem(last=False)[1].nbytes
        # Own the allocation: a view would pin the entire decoded source frame.
        self.frames[key] = patch.copy()
        self.used += patch.nbytes

    def _decode(self, index):
        frame = self.reader.frame(index)
        if self.shape is None:
            self.shape = frame.shape
            if self.region is not None:
                x0, y0, x1, y1 = self.region
                h, w = self.shape[:2]
                self.region = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                a, b, c, d = self.region
                if c <= a or d <= b or (c - a) * (d - b) * 3 > self.max_bytes // self.min_frames:
                    self.region = None
        if self.region is not None:
            x0, y0, x1, y1 = self.region
            self._remember(index, frame[y0:y1, x0:x1])
        return frame

    def __getitem__(self, key):
        index, rows, columns = key
        with self.lock:
            frame = self._decode(index) if self.shape is None else None
            ys = range(self.shape[0])[rows]
            xs = range(self.shape[1])[columns]
            if not len(ys) or not len(xs):
                return np.empty((len(ys), len(xs), self.shape[2]), np.uint8)
            x0, x1 = min(xs[0], xs[-1]), max(xs[0], xs[-1]) + 1
            y0, y1 = min(ys[0], ys[-1]), max(ys[0], ys[-1]) + 1
            if self.region is not None:
                a, b, c, d = self.region
                if a <= x0 and b <= y0 and c >= x1 and d >= y1:
                    if index not in self.frames:
                        self._decode(index)
                    self.frames.move_to_end(index)
                    patch = self.frames[index][y0-b:y1-b, x0-a:x1-a]
                    return patch[::ys.step, ::xs.step].copy()
                # A full-frame or out-of-union request must read actual source
                # pixels; padding a crop would silently corrupt calibration.
                frame = self._decode(index) if frame is None else frame
                return frame[rows, columns].copy()
            size = self.tile_size
            tiles = [(index, x, y) for y in range(y0 // size, (y1-1) // size + 1)
                     for x in range(x0 // size, (x1-1) // size + 1)]
            if all(tile in self.frames for tile in tiles):
                patch = np.empty((y1-y0, x1-x0, self.shape[2]), np.uint8)
                for tile in tiles:
                    _, x, y = tile
                    a, b = x * size, y * size
                    c, d = min(x1, a + size), min(y1, b + size)
                    xa, ya = max(x0, a), max(y0, b)
                    patch[ya-y0:d-y0, xa-x0:c-x0] = self.frames[tile][ya-b:d-b, xa-a:c-a]
                    self.frames.move_to_end(tile)
                return patch[::ys.step, ::xs.step].copy()
            frame = self._decode(index) if frame is None else frame
            if not (x0 == y0 == 0 and x1 == self.shape[1] and y1 == self.shape[0]):
                for tile in tiles:
                    _, x, y = tile
                    self._remember(tile, frame[y*size:(y+1)*size, x*size:(x+1)*size])
            return frame[rows, columns].copy()

    def close(self):
        with self.lock:
            self.reader.close()
            self.frames.clear()
            self.used = 0


class RepairCache:
    """Bounded, one-use temporal results from the calibration sample frames."""
    def __init__(self, max_bytes=24 * 1024 * 1024, max_entries=24):
        self.max_bytes, self.used = max(0, max_bytes), 0
        self.max_entries = max(0, max_entries)
        self.frames = OrderedDict()
        self.lock = threading.Lock()

    def put(self, index, repaired, diagnostic):
        x0, y0, x1, y1 = diagnostic["roi"]
        patch = repaired[y0:y1, x0:x1]
        recovered = diagnostic["recovered_mask"]
        size = patch.nbytes + recovered.nbytes
        with self.lock:
            if index in self.frames:
                old = self.frames.pop(index)
                self.used -= old[1].nbytes + old[2].nbytes
            if size > self.max_bytes or not self.max_entries:
                return
            while self.frames and (self.used + size > self.max_bytes or len(self.frames) >= self.max_entries):
                old = self.frames.popitem(last=False)[1]
                self.used -= old[1].nbytes + old[2].nbytes
            self.frames[index] = ((x0, y0, x1, y1), patch.copy(), recovered.copy())
            self.used += size

    def take(self, index, original):
        with self.lock:
            entry = self.frames.pop(index, None)
            if entry is None:
                return None
            roi, patch, recovered = entry
            self.used -= patch.nbytes + recovered.nbytes
        result = original.copy()
        x0, y0, x1, y1 = roi
        result[y0:y1, x0:x1] = patch
        return result, dict(roi=list(roi), recovered_mask=recovered)

    def close(self):
        with self.lock:
            self.frames.clear()
            self.used = 0


class MovingRepair:
    def __init__(self, source, info, first, count, reference, roi, progress):
        self.width, self.height, self.frame_count = info["width"], info["height"], info["frames"]
        self.fps = info["fps"]
        self.context_pad = 48
        self.offsets = (4, 8, 12, 18, 24, 36, 48)
        self.union = [0, 0, self.width, self.height]
        x0, y0, x1, y1 = roi
        if (x1 - x0) * (y1 - y0) > 196608 or max(x1 - x0, y1 - y0) > 1024:
            raise ValueError("Moving V6 reference ROI is limited to 196608 pixels and 1024 pixels per side")
        horizon = round(4 * self.fps)
        begin = max(0, first - horizon)
        end = min(self.frame_count, first + count + horizon)
        if end - begin > 100000:
            raise ValueError("Moving V6 supports at most 100000 context frames per job; use a shorter interval")
        reader = Reader(source)
        records = {}
        # Cuts are calculated from adjacent source frames, independently of
        # tracking acceptance. Unknown tracks are never clean donor evidence.
        cuts = np.zeros(end - begin, np.int32)
        visited, total = 0, end - begin
        try:
            reference_frame = reader.frame(reference)
            reference_gray = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2GRAY)
            patch = reference_frame[y0:y1, x0:x1]
            support, details = reference_support(patch, dilation=2)
            occupancy = float(support.mean())
            if details["border_noise_p95"] > 24 or not .005 <= occupancy <= .8:
                raise ValueError("Reference ROI needs a clearly separated logo with a near-uniform border; choose another reference frame/ROI")
            self.mask_details = details
            self.support = support > 0

            def track_frame(index, frame, tracker, previous_thumb, reverse=False):
                nonlocal visited
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                record = tracker.update(gray)
                records[index] = record
                thumb = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
                if previous_thumb is not None:
                    delta = np.mean(abs(thumb - previous_thumb), axis=2)
                    cuts[(index + 1 if reverse else index) - begin] = int(delta.mean() > 22 and (delta > 30).mean() > .32)
                visited += 1
                if visited % max(1, round(self.fps)) == 0:
                    progress(5 + 18 * visited / total, "tracking_moving_mask")
                return thumb

            # A distant reference supplies appearance, never a reason to decode
            # every intervening frame. Reacquire globally inside this segment's
            # bounded context, retaining the stricter global confidence floor.
            anchor = min(max(reference, begin), end - 1)
            anchor_frame = reference_frame if anchor == reference else reader.frame(anchor)
            anchor_gray = cv2.cvtColor(anchor_frame, cv2.COLOR_BGR2GRAY)
            tracker = TemplateAnchorTracker(reference_gray, roi, search_margin=max(16, round(self.width / 80)))
            tracker.lost = anchor != reference
            previous = None
            for index in range(anchor, end):
                previous = track_frame(index, anchor_frame if index == anchor else reader.frame(index), tracker, previous)
            tracker = TemplateAnchorTracker(reference_gray, roi, search_margin=max(16, round(self.width / 80)))
            tracker.lost = anchor != reference
            tracker.update(anchor_gray)
            previous = cv2.resize(anchor_frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
            # Reverse tracking decodes small chronological blocks, avoiding an
            # all-video grayscale cache and per-frame backwards keyframe seeks.
            cursor = anchor
            block_size = max(1, min(24, (64 * 1024 * 1024) // (self.width * self.height * 3)))
            while cursor > begin:
                block_start = max(begin, cursor - block_size)
                block = [(index, reader.frame(index)) for index in range(block_start, cursor)]
                for index, frame in reversed(block):
                    previous = track_frame(index, frame, tracker, previous, reverse=True)
                cursor = block_start
            cut_prefix = {begin + index: int(value) for index, value in enumerate(np.cumsum(cuts))}
            records, self.bridge_report = bridge_tracking_gaps(
                records, reader.frame, self.support, cut_prefix, self.fps,
                search_margin=max(16, round(self.width / 80)))
            records, self.support_report = recover_reference_support(
                records, lambda index: cv2.cvtColor(reader.frame(index), cv2.COLOR_BGR2GRAY),
                reference_gray, roi, self.support, cut_prefix, self.fps,
                search_margin=max(16, round(self.width / 80)))
            records, self.contour_report = verify_contour_tracks(
                records, reader.frame, reference_frame, roi, self.support,
                search_margin=max(16, round(self.width / 80)), fps=self.fps)
        finally:
            reader.close()
        self.records = records
        self.cut_prefix = cut_prefix
        self.tracking_context = dict(firstFrame=begin, endFrameExclusive=end,
                                     decodedFrames=end - begin, referenceFrame=reference,
                                     reacquired=reference != anchor)
        self.masks = PlacedMasks(records, self.support, self.width, self.height)
        accepted = sum(index in self.masks for index in range(first, first + count))
        self.accepted = accepted
        self.original_accepted = sum(records[index]["accepted"] and records[index]["acceptance"] == "matched"
                                     for index in range(first, first + count))
        self.bridged = sum(records[index]["acceptance"] == "bridged"
                           for index in range(first, first + count))
        self.support_recovered = sum(records[index]["acceptance"] == "observed_support"
                                     for index in range(first, first + count))
        self.contour_recovered = sum(records[index]["acceptance"] == "observed_contour"
                                     for index in range(first, first + count))
        self.identity_rejected = sum(records[index].get("pre_identity_accepted", False)
                                     and not records[index]["accepted"]
                                     for index in range(first, first + count))
        self.output_first, self.output_count = first, count
        tracked = [self.width, self.height, 0, 0]
        for index in records:
            if index in self.masks:
                a, b, c, d = self.masks.bounds(index)
                tracked = [min(tracked[0], a), min(tracked[1], b),
                           max(tracked[2], c), max(tracked[3], d)]
        region = [max(0, tracked[0] - self.context_pad), max(0, tracked[1] - self.context_pad),
                  min(self.width, tracked[2] + self.context_pad), min(self.height, tracked[3] + self.context_pad)]
        self.cache = FrameCache(source, region=region, min_frames=2 * horizon + 1)
        self.calibrated = RepairCache()
        self.adaptive = AdaptiveTemporalRepair(self, self.fps, search_seconds=4., max_donors=12)
        self.changed = self.temporal_pixels = self.inverse_pixels = self.fallback_pixels = 0
        self.spatial_totals = dict(flat_pixels=0, edge_pixels=0, seam_pixels=0, texture_pixels=0,
                                   excluded_reference_pixels=0, unsupported_pixels=0)
        try:
            self.model = self._calibrate(first, count, progress)
        except BaseException:
            self.close()
            raise
        progress(40, "moving_model_ready")

    def _local_mask(self, index, roi):
        x0, y0, x1, y1 = roi
        result = np.zeros((y1 - y0, x1 - x0), bool)
        if index not in self.masks:
            return np.ones_like(result)
        box, mask = self.masks[index]
        a, b, c, d = max(x0, box[0]), max(y0, box[1]), min(x1, box[2]), min(y1, box[3])
        if c > a and d > b:
            result[b - y0:d - y0, a - x0:c - x0] = mask[b - box[1]:d - box[1], a - box[0]:c - box[0]]
        return result

    def _canonical(self, image, index, nearest=False):
        x, y = self.records[index]["anchor_subpixel_xy"]
        transform = np.float32([[1, 0, -x], [0, 1, -y]])
        return cv2.warpAffine(image, transform, self.support.shape[::-1],
                              flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)

    def _calibrate(self, first, count, progress):
        indices = [int(i) for i in np.unique(np.linspace(first, first + count - 1, min(24, count)).round().astype(int))
                   if int(i) in self.masks]
        observations, backgrounds, validity = [], [], []
        for offset, index in enumerate(indices):
            original = self.cache[index, slice(None), slice(None)]
            repaired, diagnostic = self.adaptive.repair(index, original)
            self.calibrated.put(index, repaired, diagnostic)
            x0, y0, x1, y1 = diagnostic["roi"]
            recovered = np.zeros((self.height, self.width), np.uint8)
            recovered[y0:y1, x0:x1] = diagnostic["recovered_mask"]
            # Only corroborated visible source pixels calibrate inversion.
            # Spatial fill is never presented as observed background.
            valid = self._canonical(recovered.astype(np.float32), index) >= .999
            observations.append(self._canonical(original, index))
            backgrounds.append(self._canonical(repaired, index))
            validity.append(valid & self.support)
            progress(23 + 16 * (offset + 1) / max(1, len(indices)), "calibrating_moving_inverse")
        if len(indices) < 5:
            return None
        return fit_overlay(np.stack(observations), np.stack(backgrounds), np.stack(validity), np.array(indices))

    def repair(self, index, original):
        if index not in self.masks:
            return original.copy(), (0, 0, 0, 0, {})
        roi, mask = self.masks[index]
        temporal = self.calibrated.take(index, original)
        restored, diagnostic = temporal if temporal is not None else self.adaptive.repair(index, original)
        x0, y0, x1, y1 = roi
        a, b, c, d = diagnostic["roi"]
        recovered = diagnostic["recovered_mask"][y0 - b:y1 - b, x0 - a:x1 - a]
        remaining = mask & ~recovered
        spatial_stats = {}
        if remaining.any():
            xa, ya, xb, yb = max(0, x0 - 12), max(0, y0 - 12), min(self.width, x1 + 12), min(self.height, y1 + 12)
            local_mask = np.zeros((yb - ya, xb - xa), np.uint8)
            local_mask[y0 - ya:y1 - ya, x0 - xa:x1 - xa] = remaining
            evidence_mask = np.zeros_like(local_mask)
            evidence_mask[y0 - ya:y1 - ya, x0 - xa:x1 - xa] = mask
            # Fit only original visible observations. Previously recovered
            # temporal pixels stay intact and cannot become synthetic evidence.
            patch, spatial_stats = spatial_repair(original[ya:yb, xa:xb], local_mask,
                                                  evidence_mask=evidence_mask)
            selected = local_mask > 0
            restored[ya:yb, xa:xb][selected] = patch[selected]
        inverse_count = 0
        if self.model is not None and self.model["support"].any():
            px, py = self.records[index]["anchor_subpixel_xy"]
            matrix = np.float32([[1, 0, px - x0], [0, 1, py - y0]])
            size = (x1 - x0, y1 - y0)

            def warp(field):
                return cv2.warpAffine(field.astype(np.float32), matrix, size, flags=cv2.INTER_LINEAR)

            gain = 1 - warp(1 - self.model["gain"])
            support = (warp(self.model["support"]) >= .999) & mask
            corrected, used = invert_overlay(original[y0:y1, x0:x1], gain, warp(self.model["offset"]), support)
            restored[y0:y1, x0:x1][used] = np.rint(corrected[used]).astype(np.uint8)
            inverse_count = int(used.sum())
        # Construct output from the source and the permitted local pixels. This
        # bounds change accounting to the ROI and makes mask containment a write
        # boundary, including if a future donor solver returns unrelated edits.
        output = original.copy()
        patch = restored[y0:y1, x0:x1]
        changed = np.any(patch[mask] != original[y0:y1, x0:x1][mask], axis=1)
        output[y0:y1, x0:x1][mask] = patch[mask]
        return output, (int(changed.sum()), int(recovered.sum()), inverse_count,
                        int(remaining.sum()), spatial_stats)

    def frames(self, reader, first, count, workers):
        # V6 pools contain multiple registered donor patches. Bound parallelism
        # to two to keep memory bounded independently of user encoder settings.
        with ThreadPoolExecutor(max_workers=min(2, workers)) as pool:
            pending = deque()

            def finish():
                result, counts = pending.popleft().result()
                self.changed += counts[0]
                self.temporal_pixels += counts[1]
                self.inverse_pixels += counts[2]
                self.fallback_pixels += counts[3]
                for key in self.spatial_totals:
                    self.spatial_totals[key] += counts[4].get(key, 0)
                return result

            for index in range(first, first + count):
                pending.append(pool.submit(self.repair, index, reader.frame(index)))
                if len(pending) >= min(2, workers):
                    yield finish()
            while pending:
                yield finish()

    def _untracked_frame_details(self, limit=100, max_bytes=32 * 1024):
        frames = []
        total = 0
        used = 128
        for index in range(self.output_first, self.output_first + self.output_count):
            record = self.records[index]
            if record['accepted']:
                continue
            total += 1
            if len(frames) >= limit:
                continue
            contour = record.get('contour_evidence', {})
            reasons = {name: evidence['reason'] for name, evidence in (
                ('contour', contour), ('temporalRetry', contour.get('temporal_retry', {})),
                ('bridge', record.get('bridge', {})), ('support', record.get('support_evidence', {})))
                if 'reason' in evidence}
            detail = dict(frameIndex=index, timeSeconds=index / self.fps,
                          acceptance=record.get('acceptance'), reasons=reasons,
                          detectorConfidence=record.get('confidence'), detectorPeakGap=record.get('peak_gap'))
            size = len(json.dumps(detail).encode('utf-8')) + 2
            if used + size <= max_bytes:
                frames.append(detail)
                used += size
        return dict(total=total, truncated=total > len(frames), frameIndexBase=0, frames=frames)

    def report(self):
        return dict(algorithm="V6 contour identity + registered donors + held-out inversion + spatial exemplar continuity",
                    changedPixels=self.changed, trackedFrames=self.accepted,
                    originalTrackedFrames=self.original_accepted, bridgedFrames=self.bridged,
                    supportRecoveredFrames=self.support_recovered,
                    contourRecoveredFrames=self.contour_recovered, identityRejectedFrames=self.identity_rejected,
                    trackingContext=self.tracking_context,
                    preservedUncertainFrames=self.output_count - self.accepted,
                    passthrough=self.changed == 0,
                    untrackedFrames=self.output_count - self.accepted,
                    untrackedFrameDetails=self._untracked_frame_details(),
                    trackingCoverage=self.accepted / self.output_count,
                    temporalRecoveredPixels=self.temporal_pixels, inversePixels=self.inverse_pixels,
                    spatialFallbackPixels=self.fallback_pixels, referenceMask=self.mask_details,
                    spatialContinuity=self.spatial_totals,
                    calibratedInversePixels=int(self.model["support"].sum()) if self.model is not None else 0,
                    limitations=["Requires a stable translation-only logo and a near-uniform reference ROI border.",
                                 "Uncertain tracked frames are preserved, and never used as clean donor evidence.",
                                 "V6 estimates missing background using evidence-gated spatial continuity; hidden texture is not ground truth.",
                                 "Large holes with no validated planar background may use spatial exemplar synthesis, not observed background truth.",
                                 "Inversion is disabled per pixel when held-out calibration cannot establish its gain.",
                                 "Occlusion, rotation, scaling, changing glyphs, and indistinguishable scene detail are unsupported."])

    def close(self):
        self.cache.close()
        self.calibrated.close()
        self.masks.close()
