"""Identity, ambiguity and reacquisition checks with known overlay pixels."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moving_v6.contour_identity import ContourIdentity, verify_contour_tracks, _edges
from moving_v6.track_logo import reference_support
from test_support_recovery import make_frame


class ContourIdentityTruth(unittest.TestCase):
    roi = (30, 30, 94, 70)

    def model(self, reference):
        return ContourIdentity(reference, self.roi, reference_support(reference[30:70, 30:94], 2)[0])

    def test_colors_opacities_and_subpixels_on_changed_background(self):
        for color, background in [((235, 235, 235), 45), ((5, 5, 5), 130), ((30, 170, 240), 45)]:
            for opacity in (1., .65, .3):
                model = self.model(make_frame(color=color, alpha=opacity, background=background))
                for xy in ((31., 29.), (38.25, 34.5), (42.75, 26.25)):
                    result = model.match(make_frame(xy, color=color, alpha=opacity, background=90), xy)
                    self.assertTrue(result['accepted'], (color, opacity, xy, result))
                    self.assertLess(np.linalg.norm(np.asarray(result['position']) - xy), .4)

    def test_missing_strokes_absence_and_occlusion_are_rejected(self):
        model = self.model(make_frame())
        frames = [make_frame(text=''), make_frame(occluded=True)]
        frames.extend(make_frame(text=text) for text in
                      ('L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L8', 'L9', 'I7', 'T7', 'F7', 'J7', '17', 'Z7', 'LZ', 'L?'))
        for frame in frames:
            self.assertFalse(model.match(frame, [30, 30])['accepted'])

    def test_extra_strokes_are_not_hidden_by_high_contour_recall(self):
        for original, changed in [('F', 'E'), ('C', 'G')]:
            result = self.model(make_frame(text=original)).match(make_frame(text=changed), [30, 30])
            self.assertGreater(result['contour_coverage'], .9)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['reason'], 'extra_stroke')

    def test_scene_edge_crossing_sparse_overlay_has_external_continuation(self):
        mask = np.zeros((104, 176), np.uint8)
        cv2.circle(mask, (50, 50), 10, 255, 2, cv2.LINE_AA)
        cv2.polylines(mask, [np.array([[70, 39], [87, 50], [70, 61]], np.int32)],
                      True, 255, 2, cv2.LINE_AA)
        alpha = mask[..., None] / 255 * .65
        reference = np.repeat(np.rint(45 * (1-alpha) + 235 * alpha).astype(np.uint8), 3, axis=2)
        model = self.model(reference)
        for height in (44, 46):
            background = np.full_like(reference, 45)
            cv2.fillPoly(background, [np.array([[0, height+12], [175, height-12],
                                                [175, 103], [0, 103]], np.int32)], (105,)*3)
            frame = np.rint(background * (1-alpha) + 235 * alpha).astype(np.uint8)
            with patch.object(model, '_background_continuation', return_value=None):
                self.assertEqual(model.match(frame, [30, 30])['reason'], 'extra_stroke')
            result = model.match(frame, [30, 30])
            self.assertTrue(result['accepted'], result)
            self.assertLess(np.linalg.norm(np.asarray(result['position']) - [30, 30]), .4)
            for evidence in result['extra_contours']['background_continuations']:
                self.assertGreaterEqual(len(evidence['visible_scene_points']), 6)
            self.assertTrue(result['extra_contours']['background_continuations'])

    def test_unrelated_parallel_lines_do_not_hide_genuinely_added_strokes(self):
        for original, changed in [('F', 'E'), ('C', 'G')]:
            model = self.model(make_frame(text=original))
            for height in (52, 56, 60, 64):
                for start in (62, 70):
                    frame = make_frame(text=changed)
                    cv2.line(frame, (start, height), (140, height), (100,)*3, 1)
                    result = model.match(frame, [30, 30])
                    self.assertFalse(result['accepted'], (original, height, start, result))
                    self.assertEqual(result['reason'], 'extra_stroke')
                    self.assertEqual(result['extra_contours']['background_continuations'], [])

    def scene_line_fixture(self):
        reference = np.full((160, 440, 3), 45, np.uint8)
        cv2.putText(reference, 'L7    M5', (35, 102), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, (235,)*3, 3, cv2.LINE_AA)
        roi = (30, 40, 400, 125)
        support = reference_support(reference[40:125, 30:400], 2)[0]
        gray = np.full((160, 440), 45, np.uint8)
        for x in range(gray.shape[1]):
            gray[95 + (1 if 180 <= x < 186 else 0):, x] = 105
        edge, _, _ = _edges(gray, 20)
        extra = np.zeros(support.shape, bool)
        extra[53:58, 150:160] = edge[93:98, 180:190]
        return reference, roi, support, gray, extra

    def test_scene_direction_resolves_short_raster_edge_extrapolation(self):
        reference, roi, support, gray, extra = self.scene_line_fixture()
        for rotated in (False, True):
            if rotated:
                reference = cv2.rotate(reference, cv2.ROTATE_90_CLOCKWISE)
                gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
                support = cv2.rotate(support, cv2.ROTATE_90_CLOCKWISE)
                extra = cv2.rotate(extra.astype(np.uint8), cv2.ROTATE_90_CLOCKWISE) > 0
                roi = (35, 30, 120, 400)
            model = ContourIdentity(reference, roi, support)
            position = np.array(roi[:2], dtype=float)
            with patch('cv2.HoughLinesP', return_value=None):
                self.assertIsNone(model._background_continuation(gray, position, extra, 20))
            evidence = model._background_continuation(gray, position, extra, 20)
            self.assertIsNotNone(evidence)
            self.assertEqual(evidence['direction_source'], 'external_scene_line')
            self.assertLessEqual(evidence['extra_line_distance_max'], 1.5)
            self.assertGreaterEqual(len(evidence['visible_scene_points']), 6)

    def test_external_collinear_segment_cannot_bridge_visible_blank_gap(self):
        reference, roi, support, gray, extra = self.scene_line_fixture()
        model = ContourIdentity(reference, roi, support)
        disconnected = np.full_like(gray, 45)
        disconnected[:, 178:193] = gray[:, 178:193]
        disconnected[:, 320:] = gray[:, 320:]
        self.assertIsNone(model._background_continuation(disconnected, np.array(roi[:2], float), extra, 20))

    def test_external_parallel_segment_cannot_absorb_short_added_edge(self):
        reference, roi, support, gray, extra = self.scene_line_fixture()
        model = ContourIdentity(reference, roi, support)
        for offset in (-6, -3, 3, 6):
            nearby = np.full_like(gray, 45)
            nearby[:, 178:193] = gray[:, 178:193]
            nearby[95+offset:, 320:] = 105
            self.assertIsNone(model._background_continuation(nearby, np.array(roi[:2], float), extra, 20))

    def test_duplicate_identity_peaks_remain_ambiguous(self):
        model = self.model(make_frame())
        frame = np.maximum(make_frame((20, 30)), make_frame((92, 30)))
        result = model.match(frame)
        self.assertFalse(result['accepted'])
        self.assertEqual(result['reason'], 'ambiguous_contours')

    def test_random_frames_cannot_gain_identity(self):
        model = self.model(make_frame())
        for seed in range(30):
            random = np.random.default_rng(seed).integers(0, 256, (104, 176, 3), np.uint8)
            self.assertFalse(model.match(random)['accepted'], seed)

    def test_false_local_glyph_reacquires_distant_true_identity(self):
        reference = cv2.copyMakeBorder(make_frame(), 0, 0, 0, 160, cv2.BORDER_CONSTANT, value=(45, 45, 45))
        frame = cv2.copyMakeBorder(make_frame(text='L2'), 0, 0, 0, 160, cv2.BORDER_CONSTANT, value=(45, 45, 45))
        frame[30:70, 220:284] = make_frame()[30:70, 30:94]
        support = reference_support(reference[30:70, 30:94], 2)[0]
        records = {0: dict(accepted=True, detector_accepted=True, acceptance='matched',
                           anchor_subpixel_xy=[30., 30.], confidence=.95)}
        result, report = verify_contour_tracks(records, lambda index: frame, reference, self.roi, support)
        self.assertTrue(result[0]['accepted'], result)
        self.assertLess(np.linalg.norm(np.asarray(result[0]['anchor_subpixel_xy']) - [220, 30]), .4)
        self.assertEqual(result[0]['acceptance'], 'observed_contour')
        self.assertEqual(report['relocated_frames'], [0])
        self.assertEqual(records[0]['anchor_subpixel_xy'], [30., 30.])

    def test_high_confidence_registration_is_preserved_after_identity_check(self):
        reference = make_frame()
        records = {0: dict(accepted=True, detector_accepted=True, acceptance='matched',
                           anchor_subpixel_xy=[30.02, 29.99], confidence=.99)}
        support = reference_support(reference[30:70, 30:94], 2)[0]
        result, _ = verify_contour_tracks(records, lambda index: reference, reference, self.roi, support)
        self.assertTrue(result[0]['accepted'])
        self.assertEqual(result[0]['anchor_subpixel_xy'], [30.02, 29.99])
        self.assertTrue(result[0]['contour_evidence']['retained_original_position'])

    def test_sparse_reference_skips_optional_contours(self):
        reference = make_frame(text='')
        records = {0: dict(accepted=False, acceptance='rejected', anchor_subpixel_xy=None)}
        result, report = verify_contour_tracks(records, lambda index: self.fail('Unexpected read'),
                                               reference, self.roi, np.zeros((40, 64), bool))
        self.assertFalse(result[0]['accepted'])
        self.assertEqual(report['skipped_reason'], 'insufficient_reference_contours')

    def test_clipped_local_peak_reacquires_its_unclipped_global_position(self):
        reference = make_frame()
        support = reference_support(reference[30:70, 30:94], 2)[0]
        records = {0: dict(accepted=False, anchor_subpixel_xy=None, candidate_subpixel_xy=[98, 30])}
        result, _ = verify_contour_tracks(records, lambda index: make_frame((81, 30)), reference, self.roi, support)
        self.assertTrue(result[0]['accepted'], result)
        self.assertLess(np.linalg.norm(np.asarray(result[0]['anchor_subpixel_xy']) - [81, 30]), .4)
        self.assertEqual(result[0]['contour_evidence']['search'], 'global')

    def test_gray_overlay_can_reverse_polarity_within_one_frame(self):
        reference = make_frame(color=(128,)*3, alpha=.65, background=45)
        support = reference_support(reference[30:70, 30:94], 2)[0]
        mask = np.zeros((104, 176), np.uint8)
        cv2.putText(mask, 'L7', (38, 59), cv2.FONT_HERSHEY_SIMPLEX, .8, 255, 2, cv2.LINE_AA)
        alpha = mask[..., None] / 255 * .65
        records = {0: dict(accepted=False, anchor_subpixel_xy=None, candidate_subpixel_xy=[30, 30])}
        for split in (43, 49, 55):
            background = np.full_like(reference, 45)
            background[:, split:] = 210
            frame = np.rint(background * (1-alpha) + 128*alpha).astype(np.uint8)
            result, _ = verify_contour_tracks(records, lambda index: frame, reference, self.roi, support)
            evidence = result[0]['contour_evidence']
            self.assertTrue(result[0]['accepted'], evidence)
            self.assertEqual(evidence['orientation'], 'unsigned')
            self.assertGreater(evidence['contour_coverage'], .95)
            self.assertLess(np.linalg.norm(np.asarray(result[0]['anchor_subpixel_xy']) - [30, 30]), .2)

    def test_weak_and_unsigned_retries_do_not_promote_absence_or_changed_identity(self):
        reference = make_frame()
        support = reference_support(reference[30:70, 30:94], 2)[0]
        frames = [make_frame(text=text) for text in ('', 'L1', 'L2', 'L3', 'L8', 'I7', 'F7', 'LZ')]
        frames.append(make_frame(occluded=True))
        frames.extend(np.random.default_rng(seed).integers(0, 256, reference.shape, np.uint8) for seed in range(10))
        records = {i: dict(accepted=True, anchor_subpixel_xy=[30, 30], confidence=.95) for i in range(len(frames))}
        result, _ = verify_contour_tracks(records, lambda index: frames[index], reference, self.roi, support)
        self.assertFalse(any(record['accepted'] for record in result.values()))

    def test_brackets_constrain_image_search_without_filling_absence(self):
        reference = cv2.copyMakeBorder(make_frame(), 0, 0, 0, 160, cv2.BORDER_CONSTANT, value=(45,)*3)
        support = reference_support(reference[30:70, 30:94], 2)[0]
        duplicate = reference.copy()
        duplicate[30:70, 220:284] = reference[30:70, 30:94]
        missing = cv2.copyMakeBorder(make_frame(text=''), 0, 0, 0, 160, cv2.BORDER_CONSTANT, value=(45,)*3)
        changed = cv2.copyMakeBorder(make_frame(text='L2'), 0, 0, 0, 160, cv2.BORDER_CONSTANT, value=(45,)*3)
        for middle, expected in ((duplicate, True), (missing, False), (changed, False)):
            frames = [reference, middle, reference]
            records = {i: dict(accepted=i != 1, anchor_subpixel_xy=[30., 30.] if i != 1 else None,
                               candidate_subpixel_xy=[150., 0.], confidence=.99) for i in range(3)}
            result, report = verify_contour_tracks(records, lambda i: frames[i], reference, self.roi, support)
            self.assertEqual(result[1]['accepted'], expected, result[1])
            if expected:
                self.assertEqual(report['temporal_recovered_frames'], [1])
                self.assertLess(np.linalg.norm(np.asarray(result[1]['anchor_subpixel_xy']) - [30, 30]), .4)
            else:
                self.assertEqual(report['temporal_recovered_frames'], [])


if __name__ == '__main__':
    unittest.main()
