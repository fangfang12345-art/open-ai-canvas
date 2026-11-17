"""Research-only reference-position activity evidence for automatic proposals.

This module never authorizes removal and never equates an unmatched anchor with
global target absence. Sparse samples propose local transitions; dense audits
only cover the explicit frame windows recorded in the output.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from media import Reader


def contrast_field(gray):
    values = gray.astype(np.float32)
    return values - cv2.GaussianBlur(values, (0, 0), 3.)


class ComponentEvidence:
    """Measure a discovered reference at its original position, without motion."""

    def __init__(self, reference, support):
        if reference.ndim != 2 or support.shape != reference.shape or not support.any():
            raise ValueError('Reference and nonempty support must have equal grayscale dimensions')
        self.reference = reference.astype(np.uint8)
        self.support = support.astype(bool)
        self.ring = cv2.dilate(self.support.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
        self.template = contrast_field(self.reference)
        self.edge = cv2.Canny(self.reference, 40, 100).astype(bool) & self.ring
        self.energy = float(np.sum(self.template[self.ring] ** 2))

    def observe(self, gray):
        if gray.shape != self.reference.shape:
            raise ValueError('Candidate crop and automatic reference dimensions differ')
        values = contrast_field(gray)[self.ring]
        template = self.template[self.ring]
        correlation = float(np.dot(values, template) / max(
            math.sqrt(float(np.dot(values, values)) * self.energy), 1e-9))
        amplitude = float(np.dot(values, template) / max(self.energy, 1e-9))
        edge = cv2.Canny(gray, 40, 100).astype(bool)
        distance = cv2.distanceTransform((~edge).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        coverage = float(np.mean(distance[self.edge] <= 2)) if self.edge.any() else 0.
        # These are local matching thresholds, not foreground ownership proof.
        # Low opacity, motion and changed scene structure all remain unknown.
        state = 'present' if correlation >= .78 and coverage >= .78 and amplitude >= .45 else 'unknown'
        evidence = ('matched' if state == 'present' else 'no_reference_evidence'
                    if correlation < .25 and coverage < .4 else 'inconclusive')
        return dict(state=state, anchorEvidence=evidence, scope='reference_position_only',
                    correlation=correlation, amplitude=amplitude, coverage=coverage)


def sampled_match_runs(observations, total_frames):
    """Describe runs of sampled matches; intervening frames remain unverified."""
    result, begin = [], None
    for offset, item in enumerate(observations):
        if item['state'] == 'present' and begin is None:
            begin = offset
        if begin is not None and (item['state'] != 'present' or offset == len(observations)-1):
            end = offset if item['state'] != 'present' else offset+1
            first, last = observations[begin]['frame'], observations[end-1]['frame']
            left = observations[begin-1]['frame'] if begin else None
            right = observations[end]['frame'] if end < len(observations) else None
            result.append(dict(firstObservedFrame=first, lastObservedFrame=last,
                               neighboringUnmatchedFrames=[left, right],
                               unverifiedStartBracket=[left+1 if left is not None else 0, first],
                               unverifiedEndBracket=[last+1, right if right is not None else total_frames],
                               status='runs_of_sampled_matches'))
            begin = None
    return result


def refine_transitions(observations, observe, total_frames, radius):
    """Bisection proposes boundaries; dense windows retain any extra switches.

    The full-resolution verifier can disagree with the reduced sparse cache.
    Such endpoints are reported rather than manufactured into a boundary.
    """
    if radius < 0:
        raise ValueError('Audit radius must be nonnegative')
    measured, transitions = {}, []

    def measure(index):
        if not 0 <= index < total_frames:
            raise ValueError('Activity frame is outside the source')
        if index not in measured:
            measured[index] = dict(frame=index, **observe(index))
        return measured[index]

    for one, two in zip(observations, observations[1:]):
        if one['state'] == two['state']:
            continue
        left, right = one['frame'], two['frame']
        coarse = [left, right]
        initial_left, initial_right = measure(left)['state'], measure(right)['state']
        if initial_left == initial_right:
            transitions.append(dict(sparseBracket=coarse, reason='resolution_dependent_evidence',
                                    endpointObservations=[measure(left), measure(right)]))
            continue
        while right-left > 1:
            middle = (left+right)//2
            if measure(middle)['state'] == initial_left:
                left = middle
            else:
                right = middle
        start, end = max(coarse[0], left-radius), min(coarse[1]+1, right+radius+1)
        dense = [measure(index) for index in range(start, end)]
        boundaries = [dict(lastPreviousFrame=a['frame'], firstNextFrame=b['frame'],
                           fromState=a['state'], toState=b['state'])
                      for a, b in zip(dense, dense[1:]) if a['state'] != b['state']]
        transitions.append(dict(sparseBracket=coarse, reason='reference_position_transition_only',
                                bisectionProposal=[left, right], denseCoverage=[start, end],
                                boundaries=boundaries, observations=dense))
    return dict(transitions=transitions, fullResolutionFrames=sorted(measured))


def load_models(discovery):
    """Consume only discovery coordinates/references, never an annotation file."""
    info = discovery['info']
    with np.load(discovery['samplesPath'], allow_pickle=False) as cache:
        scale = float(cache['scale'])
        indices, gray = cache['frameIndices'].copy(), cache['gray'].copy()
        if 'key' in cache and json.loads(str(cache['key']))['source'] != discovery['source']:
            raise ValueError('Sparse cache and discovery source signatures differ')
    if not math.isfinite(scale) or not 0 < scale <= 1:
        raise ValueError('Invalid discovery cache scale')
    if (len(indices) != len(gray) or not len(indices) or np.any(np.diff(indices) <= 0)
            or indices[0] < 0 or indices[-1] >= info['frames']):
        raise ValueError('Sparse frame indices must be unique, sorted and in the source')
    expected = (round(info['height']*scale), round(info['width']*scale))
    if gray.ndim != 3 or gray.shape[1:] != expected:
        raise ValueError('Discovery cache dimensions do not match the source')
    models = []
    for candidate in discovery['candidates']:
        reference = cv2.imread(candidate['referencePath'], cv2.IMREAD_GRAYSCALE)
        support = np.load(candidate['supportPath'], allow_pickle=False).astype(bool)
        x, y, z, w = candidate['roi']
        if reference is None or reference.shape != (w-y, z-x) or support.shape != reference.shape:
            raise ValueError('Discovery reference/support must be equal candidate crops')
        if not 0 <= x < z <= info['width'] or not 0 <= y < w <= info['height']:
            raise ValueError('Discovery candidate is outside the source')
        for component in candidate['components']:
            a, b, c, d = component['roi']
            if not x <= a < c <= z or not y <= b < d <= w:
                raise ValueError('Discovery component is outside its candidate')
            pad = max(4, round(min(c-a, d-b)*.08))
            a, b, c, d = max(x, a-pad), max(y, b-pad), min(z, c+pad), min(w, d+pad)
            crop, mask = reference[b-y:d-y, a-x:c-x], support[b-y:d-y, a-x:c-x]
            sa, sb, sc, sd = [round(value*scale) for value in (a, b, c, d)]
            if sc <= sa or sd <= sb:
                continue
            small_crop = cv2.resize(crop, (sc-sa, sd-sb), interpolation=cv2.INTER_AREA)
            small_mask = cv2.resize(mask.astype(np.uint8), (sc-sa, sd-sb),
                                    interpolation=cv2.INTER_NEAREST).astype(bool)
            if not small_mask.any():
                continue
            verifier = ComponentEvidence(small_crop, small_mask)
            observations = [dict(frame=int(index), **verifier.observe(image[sb:sd, sa:sc]))
                            for index, image in zip(indices, gray)]
            models.append(dict(id=f"{candidate['id']}/{component['id']}", roi=[a, b, c, d],
                               reference=crop, support=mask, observations=observations,
                               referenceFrame=candidate['referenceFrame'], status='candidate_requires_identity'))
    return models, indices


def transition_candidates(models, limit):
    selected = []
    for model in models:
        if sum(item['state'] == 'present' for item in model['observations']) < 5:
            continue
        a, b, c, d = model['roi']
        if any(sum(abs(x-y) for x, y in zip(model['roi'], old['roi'])) < .2*((c-a)+(d-b))
               for old in selected):
            continue
        if len(selected) >= limit:
            break
        selected.append(model)
    return selected


def run(discovery_path, output, candidate_limit=8, audit_radius_seconds=.5):
    if candidate_limit < 0 or not math.isfinite(audit_radius_seconds) or audit_radius_seconds < 0:
        raise ValueError('Invalid activity audit budget')
    payload = discovery_path.read_bytes()
    data = json.loads(payload)
    info, source = data['info'], data['source']
    source_path = Path(source['path'])
    stat = source_path.stat()
    if stat.st_size != source['size'] or stat.st_mtime_ns != source['mtimeNs']:
        raise ValueError('Source changed since automatic discovery')
    models, indices = load_models(data)
    output.mkdir(parents=True, exist_ok=True)
    serializable = []
    for model in models:
        entry = {key: value for key, value in model.items() if key not in ('reference', 'support')}
        entry['sampledMatchRuns'] = sampled_match_runs(model['observations'], info['frames'])
        serializable.append(entry)
    provenance = dict(discoverySHA256=hashlib.sha256(payload).hexdigest(),
                      source=source, status='candidate_requires_identity', scope='reference_position_only')
    (output/'activity.json').write_text(json.dumps(dict(
        **provenance, fps=info['fps'], frames=info['frames'], sparseFrameIndices=indices.tolist(), models=serializable), indent=2))
    selected = transition_candidates(models, candidate_limit)
    refined = dict(**provenance, candidateBudget=candidate_limit, auditRadiusSeconds=audit_radius_seconds,
                   models=[], limitations=[
                       'Matching scene text does not prove overlay ownership or authorize removal.',
                       'Unknown at a reference position is not global target absence.',
                       'Unobserved frames remain unverified; sparse sampling can miss short events.',
                       'Bisection proposes local transitions, not monotonic activity or global bounds.'])
    reader = Reader(source_path)
    try:
        for model in selected:
            a, b, c, d = model['roi']
            verifier = ComponentEvidence(model['reference'], model['support'])

            def observe(index):
                crop = reader.frame(index)[b:d, a:c]
                return verifier.observe(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))

            result = refine_transitions(model['observations'], observe, info['frames'],
                                        round(info['fps']*audit_radius_seconds))
            refined['models'].append(dict(id=model['id'], roi=model['roi'], status='candidate_requires_identity',
                                           coverage=dict(sparseFrames=indices.tolist(),
                                                         fullResolutionFrames=result['fullResolutionFrames']),
                                           transitions=result['transitions']))
            (output/'activity-refined.json').write_text(json.dumps(refined, indent=2))
            print(json.dumps(dict(stage='activity', candidate=model['id'],
                                  transitions=len(result['transitions']),
                                  fullResolutionObservations=len(result['fullResolutionFrames']))), flush=True)
    finally:
        reader.close()
    (output/'activity-refined.json').write_text(json.dumps(refined, indent=2))
    return refined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--discovery', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-limit', type=int, default=8)
    parser.add_argument('--audit-radius-seconds', type=float, default=.5)
    args = parser.parse_args()
    run(args.discovery, args.output, args.candidate_limit, args.audit_radius_seconds)


if __name__ == '__main__':
    main()
