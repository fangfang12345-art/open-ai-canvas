"""Ground-truth regressions for source-independent moving spatial repair."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moving_v6.spatial_fallback import spatial_fallback


def error(result, truth, mask):
    return float(np.abs(result[mask].astype(np.float64) - truth[mask]).mean())


class MovingSpatialRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from native_build import prepare
        prepare()
        from moving_v6.spatial_continuity import repair
        cls.repair = staticmethod(repair)

    def setUp(self):
        self.mask = np.zeros((96, 128), bool)
        self.mask[35:61, 49:80] = True

    def run_case(self, truth, overlay=(240, 240, 240), mask=None):
        mask = self.mask if mask is None else mask
        observed = truth.copy()
        observed[mask] = overlay
        before = observed.copy()
        old, _ = spatial_fallback(observed, mask)
        repaired, diagnostics = self.repair(observed, mask)
        np.testing.assert_array_equal(observed, before)
        np.testing.assert_array_equal(repaired[~mask], observed[~mask])
        self.assertEqual(diagnostics["write_pixels"], int(mask.sum()))
        return old, repaired, diagnostics

    def test_light_dark_and_colored_affine_backgrounds(self):
        yy, xx = np.indices(self.mask.shape)
        dark = np.stack([20 + xx * .4 + yy * .1, 25 + xx * .1 + yy * .2,
                         30 + xx * .25 + yy * .1], axis=2).astype(np.uint8)
        colored = np.stack([30 + xx * .5, 180 - yy * .3, 40 + yy * .6], axis=2).astype(np.uint8)
        for truth in (dark, 255 - dark, colored):
            with self.subTest(background=tuple(truth[0, 0])):
                old, repaired, _ = self.run_case(truth)
                self.assertLess(error(repaired, truth, self.mask), .3)
                self.assertLessEqual(error(repaired, truth, self.mask), error(old, truth, self.mask) + .1)

    def test_color_edges_and_observed_thin_lines_continue_across_gap(self):
        yy, xx = np.indices(self.mask.shape)
        for direction in (yy >= 48, xx >= 64, yy >= xx * .4 + 20):
            truth = np.where(direction[..., None], [180, 50, 90], [35, 180, 220]).astype(np.uint8)
            old, repaired, diagnostics = self.run_case(truth)
            self.assertLess(error(repaired, truth, self.mask), .35 * error(old, truth, self.mask))
            self.assertGreater(diagnostics["edge_pixels"], 0)
        for thickness in (1, 3, 5):
            for invert in (False, True):
                with self.subTest(thickness=thickness, invert=invert):
                    truth = np.full((96, 128, 3), 220, np.uint8)
                    cv2.line(truth, (0, 48), (127, 48), (20, 20, 20), thickness)
                    if invert:
                        truth = 255 - truth
                    old, repaired, _ = self.run_case(truth, overlay=(30, 180, 60))
                    self.assertLess(error(repaired, truth, self.mask), .7 * error(old, truth, self.mask))

    def test_isoluminant_colored_edge_is_not_treated_as_grayscale_truth(self):
        yy, _ = np.indices(self.mask.shape)
        truth = np.where((yy >= 48)[..., None], [0, 0, 255], [0, 130, 0]).astype(np.uint8)
        old, repaired, _ = self.run_case(truth)
        self.assertLess(error(repaired, truth, self.mask), error(old, truth, self.mask))

    def test_halo_exclusion_is_symmetric_and_never_expands_writes(self):
        cases = [((30, 40, 50), (230, 240, 250)),
                 ((210, 220, 230), (10, 20, 30)),
                 ((120, 30, 210), (20, 180, 60))]
        ring = (cv2.dilate(self.mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & ~self.mask
        for background, overlay in cases:
            with self.subTest(background=background, overlay=overlay):
                truth = np.full((96, 128, 3), background, np.uint8)
                observed = truth.copy()
                observed[ring] = ((np.array(background) + np.array(overlay)) / 2).astype(np.uint8)
                observed[self.mask] = overlay
                repaired, diagnostics = self.repair(observed, self.mask)
                self.assertLess(error(repaired, truth, self.mask), .1)
                self.assertGreater(diagnostics["excluded_reference_pixels"], 0)
                np.testing.assert_array_equal(repaired[~self.mask], observed[~self.mask])

    def test_ambiguous_text_endpoints_keep_conservative_fallback_for_both_polarities(self):
        for invert in (False, True):
            for overlay in ((240, 240, 240), (10, 10, 10), (20, 200, 80)):
                with self.subTest(invert=invert, overlay=overlay):
                    truth = np.full((96, 128, 3), 220, np.uint8)
                    cv2.putText(truth, "CONTENT", (5, 56), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (20, 20, 20), 2, cv2.LINE_AA)
                    if invert:
                        truth = 255 - truth
                    old, repaired, diagnostics = self.run_case(truth, overlay=overlay)
                    self.assertLessEqual(error(repaired, truth, self.mask), error(old, truth, self.mask) + .1)
                    self.assertGreater(diagnostics["unsupported_components"], 0)

    def test_mixed_edge_and_text_holes_require_independent_component_support(self):
        truth = np.full((128, 256, 3), 220, np.uint8)
        truth[65:, :112] = 20
        cv2.putText(truth, "DATA", (133, 71), cv2.FONT_HERSHEY_SIMPLEX,
                    .8, (20, 20, 20), 2, cv2.LINE_AA)
        mask = np.zeros(truth.shape[:2], bool)
        mask[47:81, 37:75] = True
        mask[52:76, 162:198] = True
        old, repaired, diagnostics = self.run_case(truth, mask=mask)
        xx = np.indices(mask.shape)[1]
        edge, text = mask & (xx < 112), mask & (xx > 112)
        self.assertLess(error(repaired, truth, edge), .5 * error(old, truth, edge))
        self.assertLessEqual(error(repaired, truth, text), error(old, truth, text) + .1)
        self.assertEqual(diagnostics["unsupported_components"], 1)

    def test_excluded_prior_estimates_never_become_independent_observations(self):
        yy, xx = np.indices(self.mask.shape)
        truth = np.stack([20 + xx, 30 + yy, 40 + xx // 2], axis=2).astype(np.uint8)
        write = self.mask.copy()
        write[45:] = False
        first, second = truth.copy(), truth.copy()
        first[self.mask] = (250, 0, 180)
        second[self.mask] = (0, 240, 30)
        a, _ = self.repair(first, write, evidence_mask=self.mask)
        b, _ = self.repair(second, write, evidence_mask=self.mask)
        np.testing.assert_array_equal(a[write], b[write])
        np.testing.assert_array_equal(a[~write], first[~write])
        np.testing.assert_array_equal(b[~write], second[~write])
        self.assertLess(error(a, truth, write), 2)

    def test_repeated_texture_near_image_border_needs_majority_support(self):
        yy, xx = np.indices((96, 192))
        mask = np.zeros((96, 192), bool)
        mask[65:94, 30:160] = True
        for period in (12, 24):
            with self.subTest(period=period):
                values = 80 + 25 * np.sin(xx / period * 2 * np.pi)
                values += 8 * np.sin(yy / (period * 1.3) * 2 * np.pi)
                truth = np.repeat(values.astype(np.uint8)[..., None], 3, axis=2)
                old, repaired, diagnostics = self.run_case(truth, mask=mask)
                self.assertLessEqual(error(repaired, truth, mask), error(old, truth, mask) + .1)
                self.assertEqual(diagnostics["unsupported_pixels"], int(mask.sum()))

    def test_bricks_checkerboard_and_noisy_edges_keep_unsupported_texture(self):
        yy, xx = np.indices((96, 192))
        mask_image = np.zeros((96, 192), np.uint8)
        cv2.putText(mask_image, "LABEL", (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 8)
        mask = mask_image > 0
        random = np.random.default_rng(27)
        row = yy // 16
        column = (xx + (row % 2) * 16) // 32
        colors = random.integers(70, 155, (7, 8))
        bricks = colors[row, column].astype(np.float64)
        mortar = (yy % 16 <= 1) | ((xx + (row % 2) * 16) % 32 <= 1)
        bricks[mortar] = 40
        bricks += random.normal(0, 3, bricks.shape)
        checker = np.where(((xx // 12) + (yy // 12)) % 2, 180, 50)
        noisy_edge = np.where(yy > 64, 150, 75) + random.normal(0, 8, yy.shape)
        for name, values in (("bricks", bricks), ("checker", checker), ("noisy_edge", noisy_edge)):
            with self.subTest(texture=name):
                truth = np.repeat(np.clip(values, 0, 255).astype(np.uint8)[..., None], 3, axis=2)
                old, repaired, diagnostics = self.run_case(truth, mask=mask)
                self.assertLessEqual(error(repaired, truth, mask), error(old, truth, mask) + .1)
                self.assertGreater(diagnostics["unsupported_pixels"], 0)

    def test_empty_mask_and_noncontiguous_clipped_masks_preserve_other_pixels(self):
        random = np.random.default_rng(987)
        image = random.integers(0, 256, (40, 80, 3), dtype=np.uint8)[:, ::2]
        empty = np.zeros(image.shape[:2], bool)
        repaired, diagnostics = self.repair(image, empty)
        np.testing.assert_array_equal(repaired, image)
        self.assertEqual(diagnostics["changed_pixels"], 0)
        mask = empty.copy()
        mask[:5, :7] = True
        repaired, _ = self.repair(image, mask)
        np.testing.assert_array_equal(repaired[~mask], image[~mask])


if __name__ == "__main__":
    unittest.main()
