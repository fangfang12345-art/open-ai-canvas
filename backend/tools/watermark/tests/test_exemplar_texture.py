import sys
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moving_v6.exemplar_texture import synthesize


class ExemplarTextureTests(unittest.TestCase):
    def texture(self):
        rng = np.random.default_rng(61)
        tile = rng.integers(35, 215, (12, 12, 3), dtype=np.uint8)
        tile = cv2.GaussianBlur(tile, (3, 3), .6, borderType=cv2.BORDER_REFLECT)
        return np.tile(tile, (10, 14, 1))

    def test_patch_sized_holes_preserve_observed_repeated_structure(self):
        truth = self.texture()
        for size in (24, 36):
            with self.subTest(size=size):
                mask = np.zeros(truth.shape[:2], bool)
                mask[42:42+size, 54:54+size] = True
                image = truth.copy()
                image[mask] = 255
                repaired, selected, report = synthesize(image, mask, mask)
                self.assertEqual(selected.sum(), mask.sum())
                self.assertLess(np.abs(repaired[mask].astype(float)-truth[mask]).mean(), 1.)
                self.assertGreater(report['patches'], 0)
                np.testing.assert_array_equal(repaired[~mask], image[~mask])

    def test_narrow_holes_do_not_add_unobservable_noise(self):
        image = self.texture()
        for height in (2, 4, 8, 12):
            with self.subTest(height=height):
                mask = np.zeros(image.shape[:2], bool)
                mask[44:44+height, 55:115] = True
                repaired, selected, report = synthesize(image, mask, mask)
                self.assertEqual(selected.sum(), 0)
                self.assertEqual(report['narrow_components'], 1)
                np.testing.assert_array_equal(repaired, image)

    def test_partial_writes_keep_the_geometry_of_all_missing_observations(self):
        truth = self.texture()
        evidence = np.zeros(truth.shape[:2], bool)
        evidence[42:78, 54:90] = True
        image = truth.copy()
        image[evidence] = 255
        complete, full_selected, _ = synthesize(image, evidence, evidence)
        self.assertEqual(full_selected.sum(), evidence.sum())
        for step in (2, 5, 11):
            with self.subTest(step=step):
                write = evidence.copy()
                write[:, ::step] = False
                # Non-write excluded RGB can be a temporal estimate or any
                # uncertain value; it must supply neither evidence nor truth.
                observed = image.copy()
                observed[evidence & ~write] = (17, 239, 83)
                repaired, selected, report = synthesize(observed, write, evidence)
                self.assertEqual(selected.sum(), write.sum())
                self.assertEqual(report['narrow_components'], 0)
                np.testing.assert_array_equal(repaired[write], complete[write])
                np.testing.assert_array_equal(repaired[~write], observed[~write])

    def test_excluded_pixels_never_become_patch_sources(self):
        image = self.texture()
        write = np.zeros(image.shape[:2], bool)
        write[40:76, 50:86] = True
        evidence = write.copy()
        evidence[18:35, 15:145] = True
        first, second = image.copy(), image.copy()
        first[evidence] = (0, 255, 0)
        second[evidence] = (255, 0, 255)
        a, a_selected, _ = synthesize(first, write, evidence)
        b, b_selected, _ = synthesize(second, write, evidence)
        np.testing.assert_array_equal(a_selected, b_selected)
        np.testing.assert_array_equal(a[write], b[write])
        np.testing.assert_array_equal(a[~write], first[~write])
        np.testing.assert_array_equal(b[~write], second[~write])

    def test_no_clean_source_and_clipped_edges_are_preserved(self):
        image = self.texture()
        write = np.ones(image.shape[:2], bool)
        repaired, selected, _ = synthesize(image, write, write)
        self.assertFalse(selected.any())
        np.testing.assert_array_equal(repaired, image)

    def test_spatially_held_out_planes_keep_interpolation_at_multiple_scales(self):
        yy, xx = np.indices((120, 168))
        truth = np.stack([35 + .61*xx, 190 - .37*yy, 40 + .28*xx + .19*yy], axis=2)
        truth = np.rint(truth).astype(np.uint8)
        for size in (24, 36, 48):
            with self.subTest(size=size):
                mask = np.zeros(truth.shape[:2], bool)
                mask[40:40+size, 50:50+size] = True
                observed = truth.copy()
                observed[mask] = (5, 250, 90)
                result, selected, report = synthesize(observed, mask, mask)
                self.assertFalse(selected.any())
                self.assertEqual(report['smooth_components'], 1)
                np.testing.assert_array_equal(result, observed)

    def test_structural_boundary_cannot_pass_the_smooth_plane_holdout(self):
        yy, xx = np.indices((120, 168))
        truth = np.where((yy > .37*xx + 22)[..., None], [45, 170, 90], [210, 50, 30]).astype(np.uint8)
        mask = np.zeros(truth.shape[:2], bool)
        mask[37:73, 50:86] = True
        observed = truth.copy()
        observed[mask] = 255
        _, selected, report = synthesize(observed, mask, mask)
        self.assertEqual(report['smooth_components'], 0)
        self.assertEqual(int(selected.sum()), int(mask.sum()))


if __name__ == '__main__':
    unittest.main()
