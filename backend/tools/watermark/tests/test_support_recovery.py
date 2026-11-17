"""Ground truth and adversarial checks for immutable-reference reacquisition."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moving_v6.track_logo import reference_support
from moving_v6.support_recovery import recover_reference_support
from moving_v6.tracking_bridge import bridge_tracking_gaps


def make_frame(xy=(30, 30), *, text='L7', color=(235, 235, 235), alpha=1.,
               background=45., occluded=False):
    image = np.full((104, 176, 3), background, np.float32)
    mask = np.zeros((40, 64), np.uint8)
    cv2.putText(mask, text, (8, 29), cv2.FONT_HERSHEY_SIMPLEX, .8, 255, 2, cv2.LINE_AA)
    if occluded:
        mask[:, 25:] = 0
    placed = cv2.warpAffine(mask.astype(np.float32)/255,
                           np.float32([[1, 0, xy[0]], [0, 1, xy[1]]]), (176, 104))
    blend = placed[..., None]*alpha
    return np.rint(image*(1-blend)+np.asarray(color)*blend).astype(np.uint8)


class SupportRecoveryTruth(unittest.TestCase):
    def test_sparse_support_skips_recovery_without_decoding_or_promoting_rejections(self):
        records = {
            0: dict(accepted=True, anchor_subpixel_xy=[3., 3.]),
            1: dict(accepted=False, anchor_subpixel_xy=None),
        }
        reference = np.zeros((20, 20), np.uint8)

        def unavailable(index):
            self.fail('Sparse support should not trigger optional image decoding')

        for pixels in (0, 22):
            support = np.zeros((10, 10), np.uint8)
            support.flat[:pixels] = 1
            bridged, bridge_report = bridge_tracking_gaps(records, unavailable, support, [0, 0], 30.)
            recovered, recovery_report = recover_reference_support(
                bridged, unavailable, reference, (0, 0, 10, 10), support, [0, 0], 30.)
            self.assertTrue(recovered[0]['accepted'])
            self.assertFalse(recovered[1]['accepted'])
            self.assertEqual(recovered[1]['acceptance'], 'rejected')
            self.assertNotIn('detector_accepted', records[0])
            self.assertEqual(bridge_report['bridged_count'], 0)
            self.assertEqual(recovery_report['recovered_count'], 0)
            for report in (bridge_report, recovery_report):
                self.assertEqual(report['skipped_reason'], 'insufficient_reference_support')

    def recover(self, frame, reference=None, *, cuts=None, records=None, margin=16):
        reference = make_frame() if reference is None else reference
        support, _ = reference_support(reference[30:70, 30:94], dilation=2)
        records = records or {
            0: dict(accepted=True, detector_accepted=True, anchor_subpixel_xy=[30., 30.]),
            1: dict(accepted=False, detector_accepted=False, anchor_subpixel_xy=None),
        }
        result, report = recover_reference_support(
            records, lambda index: cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), (30, 30, 94, 70), support,
            np.zeros(100, np.int32) if cuts is None else cuts, 30., search_margin=margin)
        self.assertFalse(records[1]['accepted'])
        return result[1], report

    def test_colors_opacities_subpixels_and_changed_background(self):
        for color in ((235, 235, 235), (5, 5, 5), (30, 170, 240)):
            for alpha in (1., .65):
                reference = make_frame(color=color, alpha=alpha)
                for xy in ((31., 29.), (38.25, 34.5), (42.75, 26.25)):
                    record, report = self.recover(make_frame(xy, color=color, alpha=alpha), reference)
                    self.assertEqual(report['recovered_count'], 1, (color, alpha, xy, record))
                    self.assertLess(np.linalg.norm(np.asarray(record['anchor_subpixel_xy'])-xy), .4)
                    self.assertEqual(record['acceptance'], 'observed_support')
                    self.assertFalse(record['detector_accepted'])
        record, _ = self.recover(make_frame((37.25, 31.5), background=115.))
        self.assertTrue(record['accepted'])

    def test_absent_occluded_and_similar_glyphs_are_rejected(self):
        cases = [make_frame(text=''), make_frame(occluded=True)]
        cases += [make_frame(text=text) for text in
                  ('L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L8', 'L9',
                   'I7', 'T7', 'F7', 'J7', '17', 'Z7', 'LZ', 'L?')]
        for frame in cases:
            record, report = self.recover(frame)
            self.assertEqual(report['recovered_count'], 0, record)

    def test_random_backgrounds_never_gain_evidence(self):
        for seed in range(100):
            frame = np.random.default_rng(seed).integers(0, 256, (104, 176, 3), np.uint8)
            record, report = self.recover(frame)
            self.assertEqual(report['recovered_count'], 0, (seed, record))

    def test_duplicate_peaks_are_ambiguous(self):
        frame = make_frame((20, 30))
        other = make_frame((92, 30))
        frame = np.maximum(frame, other)
        record, report = self.recover(frame, margin=80)
        self.assertEqual(report['recovered_count'], 0, record)

    def test_cuts_missing_anchors_and_recovered_anchors_cannot_promote(self):
        frame = make_frame()
        cuts = np.zeros(100, np.int32)
        cuts[1:] = 1
        _, report = self.recover(frame, cuts=cuts)
        self.assertEqual(report['recovered_count'], 0)
        for accepted in (False, True):
            records = {0: dict(accepted=accepted, detector_accepted=False, anchor_subpixel_xy=[30., 30.]),
                       1: dict(accepted=False, detector_accepted=False, anchor_subpixel_xy=None)}
            _, report = self.recover(frame, records=records)
            self.assertEqual(report['recovered_count'], 0)


if __name__ == '__main__':
    unittest.main()
