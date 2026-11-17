"""Explicit fixed overlays share source-reference identities across segments."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import cv2
import numpy as np

from media import Reader
from moving_v6.contour_identity import ContourIdentity, _edges
from .calibrate_fixed import detect_persistent_light

VERSION = 'fixed-r16-regions-time-v2'


def region_repair_task(frame, box, mask):
    # Validation imports this module before native preparation. Resolve the
    # solver only inside workers, after the parent's prepared-library env exists.
    from .engine import repair_task
    return repair_task(frame, box, mask)


def boundary(seconds, fps):
    return math.floor(seconds * fps + .5)


def validate_regions(regions, info):
    if not isinstance(regions, list) or not 1 <= len(regions) <= 4:
        raise ValueError('fixedRegions must contain one to four regions')
    result = []
    for number, value in enumerate(regions):
        if not isinstance(value, dict) or not isinstance(value.get('roi'), dict):
            raise ValueError('Each fixed region requires roi and absolute reference/activity times')
        raw = [value['roi'].get(key) for key in ('x', 'y', 'width', 'height')]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v != int(v) for v in raw):
            raise ValueError('Fixed region ROI must use integer source pixels')
        x, y, w, h = map(int, raw)
        if min(x, y) < 0 or min(w, h) < 4 or x+w > info['width'] or y+h > info['height']:
            raise ValueError('Fixed region ROI must lie inside the source video')
        times = [value.get(key) for key in ('referenceSeconds', 'startSeconds', 'durationSeconds')]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in times):
            raise ValueError('Fixed region times must be nonnegative finite seconds')
        reference, start, duration = times
        end = start + duration if duration else info['duration']
        if start >= end or end > info['duration'] + 1e-6 or not start <= reference < end:
            raise ValueError('Fixed region reference must lie inside its source activity interval')
        first = boundary(start, info['fps'])
        last = min(info['frames'], boundary(end, info['fps']))
        ref = boundary(reference, info['fps'])
        if last-first < 3 or not first <= ref < last:
            raise ValueError('Fixed region activity must contain three frames and its reference frame')
        box = [x, y, x+w, y+h]
        for old in result:
            a, b, c, d = old['roi']
            if x < c and x+w > a and y < d and y+h > b:
                raise ValueError('Fixed region ROIs must not overlap')
        result.append(dict(region=number+1, roi=box, first=first, end=last, reference=ref))
    return result


def calibrate(reader, regions, info, progress):
    models = []
    for region in regions:
        x0, y0, x1, y1 = region['roi']
        radius = max(1, boundary(.5, info['fps']))
        lo, hi = max(region['first'], region['reference']-radius), min(region['end']-1, region['reference']+radius)
        indices = np.unique(np.linspace(lo, hi, min(9, hi-lo+1)).round().astype(int))
        samples = [reader.frame(int(i))[y0:y1, x0:x1].copy() for i in indices]
        support, core, _, statistics = detect_persistent_light(np.stack(samples))
        reference = reader.frame(region['reference'])[y0:y1, x0:x1].copy()
        model = dict(region, support=support, core=core, reference_image=reference,
                     calibration=dict(statistics, sampleFrames=indices.tolist()))
        # Reference selection is explicit. A noisy surrounding background cannot
        # justify a photometric identity and must not silently weaken its bounds.
        verifier = identity(model)
        if not verifier.available:
            raise ValueError(f"Fixed region {region['region']} reference is not identifiable; select a complete overlay with a quiet reference border")
        evidence = verifier.match(reference, predicted=[0, 0], margin=0, unsigned=True)
        if not evidence['accepted']:
            raise ValueError(f"Fixed region {region['region']} reference identity failed: {evidence['reason']}")
        models.append(model)
        progress(5 + 30*len(models)/len(regions), 'calibrating_fixed_references')
    return models


def identity(model):
    image = model['reference_image']
    return FixedIdentity(image, model['support'])


def missing_stroke_size(missing, reference_edge):
    # One-pixel gaps may connect an interrupted stroke, but matched neighboring
    # edges are not missing evidence and must not increase its measured size.
    joined = cv2.dilate(missing, np.ones((3, 3), np.uint8)) & reference_edge
    _, labels = cv2.connectedComponents(joined.astype(np.uint8), connectivity=8)
    sizes = np.bincount(labels[missing > 0])
    return int(sizes[1:].max(initial=0))


class FixedIdentity(ContourIdentity):
    """Verify one declared position without searching for replacement objects.

    Codec filtering and antialiasing change boundary samples across backgrounds.
    A one-pixel raster envelope is permitted for photometric bounds, while the
    observed edge must still follow the reference and cannot lose a connected
    stroke. The envelope cannot explain a missing stroke interior or added ink.
    """
    def __init__(self, image, support):
        super().__init__(image, [0, 0, image.shape[1], image.shape[0]], support)
        kernel = np.ones((3, 3), np.uint8)
        self.low = cv2.erode(self.low, kernel)
        self.high = cv2.dilate(self.high, kernel)

    def match(self, image, predicted=None, margin=0, unsigned=True):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edge, bins, _ = _edges(gray, self.threshold)
        distances = np.zeros(self.mass, np.float32)
        expected_bins = self.bins[self.ys, self.xs]
        for k in range(6):
            selected = expected_bins % 6 == k
            if not selected.any():
                continue
            compatible = edge & (((bins-k) % 6 <= 1) | ((k-bins) % 6 <= 1))
            field = cv2.distanceTransform((~compatible).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            distances[selected] = field[self.ys[selected], self.xs[selected]]
        coverage = float(np.mean(distances <= 1.5)) if self.mass else 0.
        cost = float(np.mean(np.minimum(distances, 4))) if self.mass else 4.
        missing = np.zeros(self.edge.shape, np.uint8)
        missing[self.ys[distances > 1.5], self.xs[distances > 1.5]] = 1
        missing_component = missing_stroke_size(missing, self.edge)
        pixel = self._pixel_identity(image, np.array([0., 0.]))
        extra = self._extra_contours(gray, np.array([0., 0.]))
        observed = coverage >= .6 and cost <= 1.8 and missing_component < 6
        accepted = observed and pixel['accepted'] and extra['extra_component'] < 6
        reason = ('fixed_reference_identity' if accepted else 'missing_stroke' if not pixel['accepted']
                  else 'extra_stroke' if extra['extra_component'] >= 6 else 'missing_reference_contour')
        return dict(accepted=bool(accepted), reason=reason, contour_coverage=coverage,
                    contour_cost=cost, missing_contour_component=missing_component,
                    pixel_identity=pixel, extra_contours=extra)


def load_models(source, info, regions, cache_directory, progress):
    source = Path(source)

    def signature():
        stat = source.stat()
        return [VERSION, str(source.resolve()), stat.st_dev, stat.st_ino, stat.st_size,
                stat.st_mtime_ns, stat.st_ctime_ns, info['width'], info['height'], info['rate'], regions]

    def build():
        reader = Reader(source)
        try:
            return calibrate(reader, regions, info, progress)
        finally:
            reader.close()

    if not cache_directory:
        return build()
    try:
        import fcntl
    except ImportError:
        return build()
    source_signature = signature()
    key = hashlib.sha256(json.dumps(source_signature, sort_keys=True).encode()).hexdigest()
    directory = Path(cache_directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'.watermark-fixed-{key}.npz'
    with path.with_suffix('.lock').open('a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            try:
                with np.load(path, allow_pickle=False) as cached:
                    meta = json.loads(str(cached['metadata']))
                    if meta['signature'] == source_signature and signature() == source_signature:
                        models = []
                        for i, model in enumerate(meta['models']):
                            models.append(dict(model, support=cached[f'support_{i}'].copy(),
                                               core=cached[f'core_{i}'].copy(),
                                               reference_image=cached[f'reference_{i}'].copy()))
                        return models
            except (OSError, ValueError, KeyError):
                pass
        models = build()
        if signature() != source_signature:
            raise ValueError('Source changed while calibrating fixed references')
        arrays, metadata = {}, []
        for i, model in enumerate(models):
            metadata.append({k: v for k, v in model.items() if k not in ('support', 'core', 'reference_image')})
            for key, field in (('support', 'support'), ('core', 'core'), ('reference', 'reference_image')):
                arrays[f'{key}_{i}'] = model[field]
        arrays['metadata'] = np.array(json.dumps(dict(signature=source_signature, models=metadata)))
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return models


class MultiFixedRepair:
    def __init__(self, source, info, first, count, regions, progress, cache_directory=None):
        self.info = info
        self.regions = validate_regions(regions, info)
        self.models = load_models(source, info, self.regions, cache_directory, progress)
        self.identities = [identity(model) for model in self.models]
        self.changed = self.targeted = 0
        self.active = self.inactive = 0
        self.region_counts = [dict(region=m['region'], activeFrames=0, inactiveFrames=0) for m in self.models]
        self.solver_totals = dict(robustNativeFrames=0, robustNumpyFrames=0, robustFallbackFrames=0, affineFallbackFrames=0)
        progress(40, 'fixed_references_ready')

    def masks(self, original, index):
        results = []
        for model, verifier, counts in zip(self.models, self.identities, self.region_counts):
            if not model['first'] <= index < model['end']:
                counts['inactiveFrames'] += 1
                self.inactive += 1
                continue
            x0, y0, x1, y1 = model['roi']
            evidence = verifier.match(original[y0:y1, x0:x1], predicted=[0, 0], margin=0, unsigned=True)
            if not evidence['accepted']:
                raise ValueError(f"Fixed region {model['region']} identity unverified at frame {index} ({index/self.info['fps']:.6f}s): {evidence['reason']}; adjust its reference or active interval")
            counts['activeFrames'] += 1
            self.active += 1
            results.append((model['roi'], model['support']))
        return results

    def frames(self, reader, first, count, workers):
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending = deque()

            def finish():
                original, jobs = pending.popleft()
                for future, box, crop in jobs:
                    patch, changed, targeted, solvers = future.result()
                    x0, y0, x1, y1 = box
                    xa, ya = crop
                    # Context crops can overlap even though user ROIs cannot.
                    # Publish only each ROI so another region's result survives.
                    original[y0:y1, x0:x1] = patch[y0-ya:y1-ya, x0-xa:x1-xa]
                    self.changed += changed
                    self.targeted += targeted
                    for key, value in solvers.items():
                        self.solver_totals[key] += value
                return original

            for index in range(first, first+count):
                original = reader.frame(index)
                jobs = []
                for box, mask in self.masks(original, index):
                    x0, y0, x1, y1 = box
                    xa, ya = max(0, x0-12), max(0, y0-12)
                    xb, yb = min(original.shape[1], x1+12), min(original.shape[0], y1+12)
                    future = pool.submit(region_repair_task, np.ascontiguousarray(original[ya:yb, xa:xb]),
                                         (x0-xa, y0-ya, x1-xa, y1-ya), mask)
                    jobs.append((future, box, (xa, ya)))
                pending.append((original, jobs))
                if len(pending) >= workers*2:
                    yield finish()
            while pending:
                yield finish()

    def report(self):
        return dict(algorithm='R16 explicit regions + shared fixed reference identity + robust spatial continuation',
                    fixedReferenceVersion=VERSION, changedPixels=self.changed, targetPixels=self.targeted,
                    activeRegionFrames=self.active, inactiveRegionFrames=self.inactive,
                    fixedRegions=[dict(counts, roi=model['roi'], referenceFrame=model['reference'],
                                       activeStartFrame=model['first'], activeEndFrame=model['end'],
                                       calibration=model['calibration'])
                                  for model, counts in zip(self.models, self.region_counts)],
                    solverUsage=self.solver_totals,
                    limitations=['Frames outside explicitly configured activity intervals are preserved without target verification.',
                                 'Active frames must match the shared bright-overlay reference; uncertainty fails the segment.',
                                 'Opaque hidden content is estimated spatially, not recovered as ground truth.'])

    def close(self):
        pass
