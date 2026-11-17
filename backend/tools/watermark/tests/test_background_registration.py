import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moving_v6.adaptive_temporal import rank_donors
from moving_v6.background_registration import descriptor_transforms, translation_transforms
from moving_v6.coherent_repair import repair_patch


class BackgroundRegistrationTests(unittest.TestCase):
    def scene(self):
        rng = np.random.default_rng(713)
        noise = rng.integers(0, 256, (180, 520, 3), dtype=np.uint8)
        return cv2.GaussianBlur(noise, (5, 5), .8)

    def pan(self):
        scene = self.scene()
        clean = scene[:, 120:400].copy()
        mask = np.zeros(clean.shape[:2], bool)
        mask[67:108, 104:150] = True
        target = clean.copy()
        target[mask] = (235, 242, 248)
        donors = []
        for index, shift in ((3, 61), (7, 78)):
            image = scene[:, 120-shift:400-shift].copy()
            image[mask] = (235, 242, 248)
            donors.append(dict(image=image, mask=mask.copy(), index=index, distance=-index))
        return clean, target, mask, donors

    def test_large_camera_pan_recovers_observed_texture(self):
        clean, target, mask, donors = self.pan()
        result, report = repair_patch(target, mask, donors)
        recovered = report['recovered_mask']
        self.assertGreater(recovered.sum(), mask.sum() * .85)
        self.assertLess(np.abs(result[recovered].astype(float) - clean[recovered]).mean(), 2.)
        np.testing.assert_array_equal(result[~mask], target[~mask])

    def test_proposals_do_not_read_either_watermark(self):
        _, target, mask, donors = self.pan()
        changed_target, changed_donor = target.copy(), donors[0]['image'].copy()
        changed_target[mask] = (0, 255, 0)
        changed_donor[mask] = (255, 0, 255)
        for propose in (descriptor_transforms, translation_transforms):
            before = propose(target, donors[0]['image'], mask, mask)
            after = propose(changed_target, changed_donor, mask, mask)
            self.assertEqual(len(before), len(after))
            for first, second in zip(before, after):
                np.testing.assert_array_equal(first, second)

    def test_changed_foreground_cannot_borrow_stale_background(self):
        _, target, mask, donors = self.pan()
        # The new object is independently visible around the whole hole.
        target[40:135, 76:178] = (12, 35, 240)
        target[mask] = (235, 242, 248)
        result, report = repair_patch(target, mask, donors)
        self.assertEqual(report['recovered_pixels'], 0)
        np.testing.assert_array_equal(result, target)

    def test_unknown_masks_and_scene_cuts_remain_unusable(self):
        _, target, mask, donors = self.pan()
        donors[0]['cut_ok'] = False
        donors[1]['mask'] = None
        result, report = repair_patch(target, mask, donors)
        self.assertEqual(report['recovered_pixels'], 0)
        np.testing.assert_array_equal(result, target)

    def test_coarse_unregistered_difference_cannot_drop_camera_pan(self):
        _, target, mask, donors = self.pan()
        candidates, _ = rank_donors(target, mask, donors)
        self.assertEqual(len(candidates), 2)

    def test_unbounded_edge_slide_cannot_register_by_discarding_structure(self):
        target = np.full((150, 180, 3), 50, np.uint8)
        donor = target.copy()
        mask = np.zeros(target.shape[:2], bool)
        # Required horizontal motion exceeds the search range. A flat overlap
        # must not authorize a smaller displacement that merely clips the edge.
        target[:, :40] = 235
        donor[:, :125] = 235
        self.assertEqual(translation_transforms(target, donor, mask, mask), [])
        target[90:, :40] = 50
        donor[120:, :125] = 50
        self.assertEqual(translation_transforms(target, donor, mask, mask), [])

    def test_corner_motion_is_observable_but_flat_field_is_not(self):
        target = np.full((150, 180, 3), 50, np.uint8)
        donor = target.copy()
        mask = np.zeros(target.shape[:2], bool)
        self.assertEqual(translation_transforms(target, donor, mask, mask), [])
        target[:70, :35] = 235
        donor[:35, :35] = 235
        proposals = translation_transforms(target, donor, mask, mask)
        self.assertTrue(any(abs(p[0, 2]) < .1 and abs(p[1, 2] - 35) < .1
                            for p in proposals))


if __name__ == '__main__':
    unittest.main()
