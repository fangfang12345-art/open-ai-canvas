"""Independent truth and adversarial negatives for bounded image track bridges."""
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from moving_v6.track_logo import reference_support
from moving_v6.tracking_bridge import _canonical, bridge_tracking_gaps


def make_sequence(color=(235, 235, 235), alpha=1., *, missing=(), replacement=None,
                  occluded=(), duplicate=False, count=24, background_kind='flat'):
    shape = (104, 176)
    base = np.full((*shape, 3), 45., np.float32)
    mask = np.zeros((40, 64), np.uint8)
    cv2.putText(mask, 'L7', (8, 29), cv2.FONT_HERSHEY_SIMPLEX, .8, 255, 2, cv2.LINE_AA)
    frames, positions = [], []
    for index in range(count):
        xy = np.array([28.25 + .35*index, 30.25 + .12*index])
        positions.append(xy)
        background = base.copy()
        background += 8*np.sin(index*.2)
        if background_kind == 'texture':
            rows, columns = np.indices(shape)
            texture = 10*np.sin((columns+index)*.4)*np.cos((rows-index*.2)*.3)
            background += texture[..., None]
        elif background_kind == 'text':
            cv2.putText(background, 'NEWS', (20-index//3, 57),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (105.,)*3, 1, cv2.LINE_AA)
        cv2.line(background, (100+index, 0), (100+index, 103), (180.,)*3, 9)
        cv2.putText(background, 'SCENE', (5, 95), cv2.FONT_HERSHEY_SIMPLEX, .6, (110.,)*3, 1)
        layer = mask.copy()
        if replacement is not None and index in missing:
            layer[:] = 0
            cv2.putText(layer, replacement, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, .8, 255, 2, cv2.LINE_AA)
        elif index in missing:
            layer[:] = 0
        if index in occluded:
            layer[:, 25:] = 0
        placed = cv2.warpAffine(layer.astype(np.float32)/255,
                                np.float32([[1, 0, xy[0]], [0, 1, xy[1]]]), shape[::-1])
        if duplicate:
            placed = np.maximum(placed, cv2.warpAffine(layer.astype(np.float32)/255,
                np.float32([[1, 0, xy[0]+12], [0, 1, xy[1]]]), shape[::-1]))
        blend = placed[..., None]*alpha
        frames.append(np.rint(background*(1-blend)+np.asarray(color)*blend).astype(np.uint8))
    reference = np.full((40, 64, 3), 45., np.float32)
    blend = mask[..., None].astype(np.float32)/255*alpha
    reference = np.rint(reference*(1-blend)+np.asarray(color)*blend).astype(np.uint8)
    support, _ = reference_support(reference, dilation=2)
    return frames, positions, support


def records_for(positions, gap):
    return {index: dict(accepted=index not in gap, anchor_subpixel_xy=xy.tolist() if index not in gap else None,
                        confidence=.98 if index not in gap else .4, peak_gap=.5,
                        search='local' if index not in gap else 'global')
            for index, xy in enumerate(positions)}


class TrackingBridgeTruth(unittest.TestCase):
    def test_bounded_patch_sampling_matches_full_frame_float_conversion(self):
        rng = np.random.default_rng(317)
        frame = rng.integers(0, 256, (720, 1280), np.uint8)
        shape = (56, 202)
        positions = [(0., 0.), (1078., 664.), (.125, 600.375)]
        positions += list(rng.uniform([0, 0], [1078, 664], (100, 2)))
        for position in positions:
            expected = cv2.getRectSubPix(frame.astype(np.float32), shape[::-1],
                tuple(np.asarray(position)+[(shape[1]-1)/2, (shape[0]-1)/2]))
            np.testing.assert_array_equal(_canonical(frame, position, shape), expected)

    def bridge(self, frames, positions, support, gap, cuts=None, **kwargs):
        records = records_for(positions, gap)
        cuts = np.zeros(len(frames), np.int32) if cuts is None else cuts
        result, report = bridge_tracking_gaps(records, frames.__getitem__, support, cuts, 30., **kwargs)
        self.assertTrue(all(not records[index]['accepted'] for index in gap))
        for index in gap:
            self.assertFalse(result[index]['detector_accepted'])
            if result[index]['accepted']:
                self.assertEqual(result[index]['acceptance'], 'bridged')
                self.assertLess(np.linalg.norm(np.asarray(result[index]['anchor_subpixel_xy'])-positions[index]), .4)
        return result, report

    def test_bilateral_observations_bridge_colors_opacities_and_long_short_gap(self):
        gap = list(range(5, 18))
        for color in ((235, 235, 235), (5, 5, 5), (30, 170, 240)):
            for alpha in (1., .65):
                frames, positions, support = make_sequence(color, alpha)
                result, report = self.bridge(frames, positions, support, gap)
                self.assertEqual(report['bridged_frames'], gap, (color, alpha, report))

    def test_scene_cut_and_direction_change_reject_before_decoding(self):
        frames, positions, support = make_sequence()
        gap = [7, 8]
        cuts = np.zeros(len(frames), np.int32)
        cuts[8:] = 1
        result, report = self.bridge(frames, positions, support, gap, cuts)
        self.assertEqual(report['bridged_count'], 0)
        self.assertEqual(report['gaps'][0]['reason'], 'scene_cut')
        positions[11] = positions[9]-[4., 0.]
        result, report = self.bridge(frames, positions, support, gap)
        self.assertEqual(report['bridged_count'], 0)
        self.assertEqual(report['gaps'][0]['reason'], 'incompatible_anchor_motion')

    def test_missing_occluded_and_changed_glyphs_cannot_become_observations(self):
        gap = [7, 8]
        cases = [dict(missing=gap), dict(occluded=gap)]
        cases += [dict(missing=gap, replacement=text) for text in ('L2', 'F7', 'J7', 'L?', 'L1')]
        for case in cases:
            frames, positions, support = make_sequence(**case)
            result, report = self.bridge(frames, positions, support, gap)
            self.assertEqual(report['bridged_count'], 0, (case, report))

    def test_gradually_changed_glyph_cannot_pass_by_template_drift_and_closure(self):
        gap = list(range(5, 18))
        frames, positions, support = make_sequence()
        changed, _, _ = make_sequence(missing=range(24), replacement='L2')
        for index in gap:
            strength = min((index-4)/5, (18-index)/5, 1.)
            frames[index] = np.rint(frames[index]*(1-strength)+changed[index]*strength).astype(np.uint8)
        _, report = self.bridge(frames, positions, support, gap)
        self.assertEqual(report['bridged_count'], 0)

    def test_image_evidence_on_moving_texture_and_body_text_backgrounds(self):
        gap = [7, 8]
        for background in ('texture', 'text'):
            frames, positions, support = make_sequence(background_kind=background)
            _, report = self.bridge(frames, positions, support, gap)
            self.assertEqual(report['bridged_frames'], gap, (background, report))

    def test_unrelated_texture_and_repeated_candidates_cannot_bridge(self):
        gap = [7, 8]
        rng = np.random.default_rng(20261117)
        for _ in range(24):
            frames, positions, support = make_sequence()
            for index in gap:
                frames[index] = rng.integers(0, 256, frames[index].shape, np.uint8)
            _, report = self.bridge(frames, positions, support, gap)
            self.assertEqual(report['bridged_count'], 0)

        frames, positions, support = make_sequence()
        # Two identical isolated copies produce two equally supported peaks.
        for index in gap:
            frames[index][:] = 45
            template = frames[6][30:70, 30:94]
            frames[index][30:70, 30:94] = template
            frames[index][30:70, 102:166] = template
        _, report = self.bridge(frames, positions, support, gap, search_margin=80)
        self.assertEqual(report['bridged_count'], 0)

    def test_uncertain_endpoint_and_unbounded_gap_preserve_original_rejection(self):
        frames, positions, support = make_sequence()
        gap = [7, 8]
        records = records_for(positions, gap)
        records[6]['confidence'] = .65
        result, report = bridge_tracking_gaps(records, frames.__getitem__, support, np.zeros(24), 30.)
        self.assertEqual(report['bridged_count'], 0)
        self.assertEqual(report['gaps'][0]['reason'], 'uncertain_original_anchors')
        _, report = self.bridge(frames, positions, support, list(range(3, 22)))
        self.assertEqual(report['bridged_count'], 0)
        self.assertEqual(report['gaps'][0]['reason'], 'gap_too_long')

    def test_reusing_bridged_results_cannot_promote_them_to_original_anchors(self):
        frames, positions, support = make_sequence()
        result, report = self.bridge(frames, positions, support, [7, 8])
        self.assertEqual(report['bridged_count'], 2)
        with self.assertRaisesRegex(ValueError, 'Original detector records required'):
            bridge_tracking_gaps(result, frames.__getitem__, support, np.zeros(24), 30.)

    def test_missing_records_and_unbounded_ends_never_interpolate(self):
        frames, positions, support = make_sequence()
        records = records_for(positions, [7, 8])
        del records[7]
        result, report = bridge_tracking_gaps(records, frames.__getitem__, support, np.zeros(24), 30.)
        self.assertEqual(report['bridged_count'], 0)
        for gap in ([0, 1], [22, 23]):
            _, report = self.bridge(frames, positions, support, gap)
            self.assertEqual(report['bridged_count'], 0)


if __name__ == '__main__':
    unittest.main()
