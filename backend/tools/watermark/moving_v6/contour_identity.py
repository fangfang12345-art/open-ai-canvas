"""Observe translated watermark contours and verify their photometric identity.

Directional distance fields avoid treating a changing background as part of the
template. They only propose a position. A separate pixel feasibility test vetoes
missing strokes, including on positions accepted by the older NCC detector.
Background cuts do not reset an overlay's identity; donor cut isolation remains
the responsibility of the repair engine.
"""
import cv2
import numpy as np

from .track_logo import subpixel_peak


def _edges(gray, threshold):
    smooth = cv2.GaussianBlur(gray, (3, 3), .6)
    dx = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    edge = cv2.Canny(smooth, threshold, 2 * threshold, L2gradient=True) > 0
    bins = np.rint(np.arctan2(dy, dx) * (6 / np.pi)).astype(np.int8) % 12
    return edge, bins, np.hypot(dx, dy) * .125


def _largest_component(binary):
    _, _, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    return int(max(stats[1:, cv2.CC_STAT_AREA], default=0))


class ContourIdentity:
    def __init__(self, reference, roi, support):
        x0, y0, x1, y1 = roi
        self.template = reference[y0:y1, x0:x1].astype(np.float32)
        self.height, self.width = self.template.shape[:2]
        self.support = np.asarray(support, bool)
        border = np.concatenate([self.template[0], self.template[-1],
                                 self.template[:, 0], self.template[:, -1]])
        self.background = np.median(border, axis=0)
        self.noise = float(np.percentile(np.max(abs(border - self.background), axis=1), 95))
        gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        _, _, magnitude = _edges(gray, 20)
        selected = magnitude[y0:y1, x0:x1][self.support]
        self.threshold = max(6., min(20., float(np.percentile(selected, 90)) if selected.size else 6.))
        edge, bins, _ = _edges(gray, self.threshold)
        self.edge = edge[y0:y1, x0:x1] & self.support
        self.bins = bins[y0:y1, x0:x1]
        self.ys, self.xs = np.where(self.edge)
        self.mass = len(self.xs)
        self.kernels = [(self.edge & (self.bins == k)).astype(np.float32) for k in range(12)]
        # p = beta*b + (1-beta)*f, 0 <= b,f <= 255. The largest
        # feasible beta gives the widest possible interval on a new background.
        beta = np.minimum(self.template / np.maximum(self.background, 1.),
                          (255 - self.template) / np.maximum(255 - self.background, 1.))
        beta = np.minimum(1., np.min(beta, axis=2))[..., None]
        self.standard_low = self.template - beta * self.background
        self.standard_high = self.template + beta * (255 - self.background)
        # An embossed overlay can contain additive dark/bright offsets, which
        # are not fixed-alpha compositing. Their union is still bounded and
        # cannot explain a dark hole in a sufficiently bright required stroke.
        delta = self.template - self.background
        self.low = np.maximum(0., delta)
        self.high = np.minimum(255., 255 + delta)
        self.tolerance = 12. + self.noise
        self.available = self.mass >= 24 and self.support.sum() >= 24 and self.noise <= 24

    def _pixel_identity(self, image, position):
        ix, iy = np.floor(position).astype(int)
        dx, dy = position - [ix, iy]
        transform = np.float32([[1, 0, dx], [0, 1, dy]])
        shape = (self.width, self.height)
        candidate = image[iy:iy + self.height, ix:ix + self.width].astype(np.float32)
        if candidate.shape != self.template.shape:
            return dict(accepted=False, reason='incomplete_identity_patch')

        def violations(low, high):
            lower = cv2.warpAffine(low, transform, shape, borderValue=(0, 0, 0))
            upper = cv2.warpAffine(high, transform, shape, borderValue=(255, 255, 255))
            error = np.maximum(np.max(lower - candidate, axis=2), np.max(candidate - upper, axis=2))
            bad = error > self.tolerance
            return dict(pixels=int(bad.sum()), largest_component=_largest_component(bad),
                        maximum_excess=float(max(0., error.max() - self.tolerance)))

        standard = violations(self.standard_low, self.standard_high)
        bounded = violations(self.low, self.high)
        allowed = max(4, int(np.ceil(self.support.sum() * .0005)))
        return dict(accepted=bounded['largest_component'] < allowed,
                    reason='bounded_blend_consistent' if bounded['largest_component'] < allowed else 'missing_stroke',
                    standard_alpha=standard, alpha_or_additive=bounded,
                    tolerance=self.tolerance, rejected_component_size=allowed)

    def _extra_contours(self, gray, position, edge_threshold=None):
        """Check added strokes when the surrounding background is demonstrably quiet.

        Contours touching the ROI border can be scene edges and are excluded.
        Only a component connected to observed reference contours can constitute
        an added stroke; unrelated scene objects cannot fail this check.
        """
        ix, iy = np.floor(position).astype(int)
        patch = gray[iy:iy + self.height, ix:ix + self.width]
        edge, _, magnitude = _edges(patch, self.threshold if edge_threshold is None else edge_threshold)
        transform = np.float32([[1, 0, position[0] - ix], [0, 1, position[1] - iy]])
        expected = cv2.warpAffine(self.edge.astype(np.uint8), transform,
                                  (self.width, self.height), flags=cv2.INTER_NEAREST) > 0
        support = cv2.warpAffine(self.support.astype(np.uint8), transform,
                                 (self.width, self.height), flags=cv2.INTER_NEAREST) > 0
        background = magnitude[~cv2.dilate(support.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)]
        quiet = float(np.percentile(background, 75)) if background.size else None
        if quiet is None or quiet > 1.5:
            return dict(checked=False, reason='textured_background', background_gradient_p75=quiet,
                        extra_component=0)
        distance = cv2.distanceTransform((~expected).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(edge.astype(np.uint8), connectivity=8)
        largest = 0
        continuations = []
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if x == 0 or y == 0 or x + w == self.width or y + h == self.height:
                continue
            component = labels == label
            matched = component & (distance <= 1.5)
            if matched.sum() < max(8, area * .5):
                continue
            extra = component & (distance > 1.5)
            if extra.sum() > max(4, area * .05):
                parts, part_labels = cv2.connectedComponents(extra.astype(np.uint8), connectivity=8)
                for part in range(1, parts):
                    excess = part_labels == part
                    continuation = self._background_continuation(
                        gray, position, excess, self.threshold if edge_threshold is None else edge_threshold)
                    if continuation is not None:
                        extra[excess] = False
                        continuations.append(continuation)
                largest = max(largest, _largest_component(cv2.dilate(extra.astype(np.uint8), np.ones((3, 3), np.uint8)) & component))
        return dict(checked=True, reason='quiet_background_topology', background_gradient_p75=quiet,
                    extra_component=largest, background_continuations=continuations)

    def _background_continuation(self, gray, position, extra, threshold):
        """Verify an excess edge continues into visible scene outside the logo.

        Canny can split a scene edge where it crosses the overlay. First trace
        its fitted direction, then directions observed outside the whole logo.
        Known reference support can occlude the edge; outside it, more than two
        raster positions without an aligned observation stop the trace. A nearby
        parallel line cannot supply a detached continuation. The added-stroke
        rejection size is unchanged.
        """
        ys, xs = np.where(extra)
        if len(xs) < 4:
            return None
        points = np.float32(np.column_stack([xs, ys]))
        vx, vy, mx, my = cv2.fitLine(points, cv2.DIST_L2, 0, .01, .01).ravel()
        direction = np.array([vx, vy])
        center = np.array([mx, my])
        along = (points - center) @ direction
        across = (points - center) @ np.array([-vy, vx])
        if np.ptp(along) < 4 or np.percentile(abs(across), 95) > .75:
            return None
        ix, iy = np.floor(position).astype(int)
        pad = max(12, min(64, max(self.width, self.height) // 4))
        x0, y0 = max(0, ix-pad), max(0, iy-pad)
        x1, y1 = min(gray.shape[1], ix+self.width+pad), min(gray.shape[0], iy+self.height+pad)
        edge, bins, _ = _edges(gray[y0:y1, x0:x1], threshold)
        transform = np.float32([[1, 0, position[0]-x0], [0, 1, position[1]-y0]])
        support = cv2.warpAffine(self.support.astype(np.float32), transform, (x1-x0, y1-y0)) > 0
        sy, sx = np.where(support)
        if not len(sx):
            return None
        bounds = [int(sx.min()), int(sy.min()), int(sx.max()), int(sy.max())]
        center += [ix-x0, iy-y0]
        points += [ix-x0, iy-y0]

        def trace(axis, origin, source):
            normal_bin = int(round(np.arctan2(axis[0], -axis[1]) * 6 / np.pi)) % 6
            compatible = edge & (((bins-normal_bin) % 6 <= 1) | ((normal_bin-bins) % 6 <= 1))
            distance = cv2.distanceTransform((~compatible).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            projected = (points-origin) @ axis
            for sign, endpoint in ((1, np.ceil(projected.max())), (-1, -np.floor(projected.min()))):
                missing = 0
                witnesses = []
                previous = None
                for step in range(int(endpoint)+1, int(endpoint)+2*max(self.width, self.height)+1):
                    qx, qy = np.rint(origin + sign*step*axis).astype(int)
                    if qx < 0 or qy < 0 or qx >= x1-x0 or qy >= y1-y0:
                        break
                    if previous == (qx, qy):
                        continue
                    previous = (qx, qy)
                    if support[qy, qx]:
                        missing = 0
                        continue
                    if distance[qy, qx] > 1.5:
                        missing += 1
                        if missing > 2:
                            break
                        continue
                    missing = 0
                    if qx < bounds[0] or qy < bounds[1] or qx > bounds[2] or qy > bounds[3]:
                        witnesses.append([int(qx+x0), int(qy+y0)])
                        if len(witnesses) >= 6:
                            return dict(extra_pixels=int(extra.sum()), direction=axis.tolist(),
                                        direction_source=source, visible_scene_points=witnesses,
                                        fit_p95=float(np.percentile(abs(across), 95)))
            return None

        observed = trace(direction, center, 'extra_component')
        if observed is not None:
            return observed
        # A one-pixel raster step biases a short edge's fitted angle. Estimate
        # alternate directions from observed scene outside the entire logo,
        # never from extrapolation or nearby glyph pixels. The scene line must
        # intersect every excess pixel within the existing raster tolerance and
        # pass the unchanged visible-gap and external-witness checks.
        scene = edge.astype(np.uint8)
        scene[bounds[1]:bounds[3]+1, bounds[0]:bounds[2]+1] = 0
        lines = cv2.HoughLinesP(scene, 1, np.pi / 180, 6, minLineLength=6, maxLineGap=2)
        if lines is None:
            return None
        lines = sorted(lines[:, 0], key=lambda line: -float(np.linalg.norm(line[2:]-line[:2])))
        for line in lines[:32]:
            a, b = line[:2].astype(np.float32), line[2:].astype(np.float32)
            axis = (b-a) / np.linalg.norm(b-a)
            normal = np.array([-axis[1], axis[0]])
            residual = (points-a) @ normal
            if np.max(abs(residual)) > 1.5:
                continue
            origin = center - ((center-a) @ normal) * normal
            observed = trace(axis, origin, 'external_scene_line')
            if observed is not None:
                observed.update(scene_line=(line.reshape(2, 2)+[x0, y0]).tolist(),
                                extra_line_distance_max=float(np.max(abs(residual))))
                return observed
        return None

    def match(self, image, predicted=None, margin=16, *, edge_threshold=None, unsigned=False):
        threshold = self.threshold if edge_threshold is None else edge_threshold
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ih, iw = gray.shape
        if predicted is None:
            x0 = y0 = 0
            x1, y1 = iw, ih
        else:
            px, py = predicted
            x0, y0 = max(0, int(np.floor(px - margin))), max(0, int(np.floor(py - margin)))
            x1, y1 = min(iw, int(np.ceil(px + self.width + margin))), min(ih, int(np.ceil(py + self.height + margin)))
        if x1 - x0 < self.width or y1 - y0 < self.height:
            return dict(accepted=False, reason='insufficient_search_area')
        edge, bins, _ = _edges(gray[y0:y1, x0:x1], threshold)
        response = np.zeros((edge.shape[0] - self.height + 1, edge.shape[1] - self.width + 1), np.float32)
        fields = []
        for k, kernel in enumerate(self.kernels):
            period = 6 if unsigned else 12
            candidate = edge & (((bins - k) % period <= 1) | ((k - bins) % period <= 1))
            distance = cv2.distanceTransform((~candidate).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
            fields.append(distance)
            if kernel.any():
                response += cv2.matchTemplate(np.minimum(distance, 4), kernel, cv2.TM_CCORR)
        response *= -1 / max(1, self.mass)
        remaining = response.copy()
        proposals = []
        for _ in range(8):
            _, score, _, location = cv2.minMaxLoc(remaining)
            if score <= -3.999 and proposals:
                break
            position = subpixel_peak(response, location) + [x0, y0]
            sx, sy = location
            remaining[max(0, sy - 4):sy + 5, max(0, sx - 4):sx + 5] = -4
            distance = np.array([fields[int(self.bins[y, x])][y + sy, x + sx]
                                 for y, x in zip(self.ys, self.xs)])
            coverage = float(np.mean(distance <= 1.5))
            identity = self._pixel_identity(image, position)
            extra = self._extra_contours(gray, position, threshold)
            observed = coverage >= .60 and -score <= 1.8
            valid = observed and identity['accepted'] and extra['extra_component'] < 6
            reason = ('observed_contour_identity' if valid else
                      'missing_stroke' if not identity['accepted'] else
                      'extra_stroke' if extra['extra_component'] >= 6 else 'contour_evidence_rejected')
            proposals.append(dict(accepted=bool(valid), reason=reason, position=position.tolist(),
                                  contour_cost=float(-score), contour_coverage=coverage,
                                  reference_edges=self.mass, pixel_identity=identity, extra_contours=extra,
                                  clipped_peak=bool((sx == 0 and x0 > 0) or (sy == 0 and y0 > 0)
                                                    or (sx == response.shape[1] - 1 and x1 < iw)
                                                    or (sy == response.shape[0] - 1 and y1 < ih))))
        valid = [candidate for candidate in proposals if candidate['accepted']]
        winner = dict(valid[0] if valid else proposals[0])
        # Invalid glyphs are not identity competitors. Unexamined peaks remain
        # competitors, so exhausting the bounded search cannot manufacture a
        # uniqueness margin or silently discard a second identical watermark.
        competing = max(float(remaining.max()), -valid[1]['contour_cost'] if len(valid) > 1 else -4.)
        gap = -winner['contour_cost'] - competing
        if winner['accepted'] and gap < .8:
            winner.update(accepted=False, reason='ambiguous_contours')
        if winner['accepted'] and winner['clipped_peak']:
            winner.update(accepted=False, reason='search_boundary')
        winner.update(contour_peak_gap=float(gap), search='global' if predicted is None else 'local',
                      candidate_edge_threshold=float(threshold), orientation='unsigned' if unsigned else 'signed',
                      candidates=[dict(position=c['position'], cost=c['contour_cost'],
                                       coverage=c['contour_coverage'], reason=c['reason']) for c in proposals])
        return winner


def verify_contour_tracks(records, frame_at, reference, roi, support, *, search_margin=16, fps=30.):
    model = ContourIdentity(reference, roi, support)
    output = {index: dict(record) for index, record in records.items()}
    report = dict(recovered_frames=[], rejected_frames=[], relocated_frames=[],
                  temporal_recovered_frames=[], checked_frames=0)
    if not model.available:
        report['skipped_reason'] = 'insufficient_reference_contours'
        return output, report
    anchors = [index for index, record in records.items() if record['accepted']]
    for index, record in sorted(records.items()):
        image = frame_at(index)
        predicted = record.get('anchor_subpixel_xy') or record.get('candidate_subpixel_xy')
        if predicted is None and anchors:
            nearest = min(anchors, key=lambda other: (abs(other - index), other))
            predicted = records[nearest]['anchor_subpixel_xy']
        attempts = []
        # Weak edges and per-region polarity changes affect proposal extraction,
        # not the immutable pixel identity or the uniqueness acceptance gates.
        # A failed local proposal always permits global image reacquisition.
        modes = [(model.threshold, False)]
        if model.threshold > 6:
            modes.append((6., False))
        modes.append((6., True))
        for threshold, unsigned in modes:
            evidence = model.match(image, predicted, search_margin, edge_threshold=threshold, unsigned=unsigned)
            if not evidence['accepted'] and predicted is not None:
                local = evidence
                evidence = model.match(image, edge_threshold=threshold, unsigned=unsigned)
                evidence['local_rejection'] = dict(reason=local['reason'], position=local.get('position'))
            attempts.append(dict(reason=evidence['reason'], position=evidence.get('position'),
                                 candidate_edge_threshold=float(threshold), unsigned=unsigned))
            if evidence['accepted']:
                break
        evidence['proposal_attempts'] = attempts
        result = output[index]
        result['contour_evidence'] = evidence
        result['pre_identity_accepted'] = bool(record['accepted'])
        report['checked_frames'] += 1
        if evidence['accepted']:
            position = evidence['position']
            previous = record.get('anchor_subpixel_xy')
            separation = float(np.linalg.norm(np.asarray(previous) - position)) if previous is not None else None
            # NCC has finer photometric phase information when its fit remains
            # strong. Identity validation does not need to replace an already
            # accurate subpixel registration with a quantized contour estimate.
            if record['accepted'] and record.get('confidence', 0) >= .90 and separation <= .5:
                original_identity = model._pixel_identity(image, np.asarray(previous))
                original_extra = model._extra_contours(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), np.asarray(previous))
                evidence['original_position_identity'] = original_identity
                if original_identity['accepted'] and original_extra['extra_component'] < 6:
                    position = previous
                    evidence['retained_original_position'] = True
            result['anchor_subpixel_xy'] = position
            result['accepted'] = True
            relocated = record['accepted'] and separation > .5 and 'local_rejection' in evidence
            if not record['accepted'] or relocated:
                result.update(acceptance='observed_contour', search='contour_identity')
                report['recovered_frames'].append(index)
                if relocated:
                    report['relocated_frames'].append(index)
        else:
            result.update(accepted=False, anchor_subpixel_xy=None, acceptance='identity_rejected')
            if record['accepted']:
                report['rejected_frames'].append(index)
    # Freeze the independently observed anchors before retrying short gaps.
    # Two endpoints constrain a search region only; every intermediate frame
    # must still produce its own unique peak and pass the same identity gates.
    observed = sorted(index for index, record in output.items() if record['accepted'])
    for left, right in zip(observed, observed[1:]):
        if right-left <= 1 or right-left > max(2, round(fps)):
            continue
        start = np.asarray(output[left]['anchor_subpixel_xy'])
        end = np.asarray(output[right]['anchor_subpixel_xy'])
        for index in range(left+1, right):
            if index not in output:
                continue
            predicted = start + (end-start) * (index-left) / (right-left)
            image = frame_at(index)
            for threshold, unsigned in modes:
                evidence = model.match(image, predicted, search_margin,
                                       edge_threshold=threshold, unsigned=unsigned)
                if evidence['accepted']:
                    break
            evidence['temporal_bracket'] = dict(left=left, right=right,
                                                predicted_position=predicted.tolist(), margin=search_margin)
            result = output[index]
            if not evidence['accepted']:
                result['contour_evidence']['temporal_retry'] = evidence
                continue
            evidence['unconstrained_rejection'] = result['contour_evidence']
            result.update(accepted=True, anchor_subpixel_xy=evidence['position'],
                          acceptance='observed_contour', search='contour_identity', contour_evidence=evidence)
            report['recovered_frames'].append(index)
            report['temporal_recovered_frames'].append(index)
            if index in report['rejected_frames']:
                report['rejected_frames'].remove(index)
    return output, report
