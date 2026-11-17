"""Integration boundaries between observed donors and spatial estimates."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moving_v6.engine import MovingRepair, RepairCache


class MovingCompositionRegression(unittest.TestCase):
    def test_sparse_valid_logo_retains_original_tracking_without_optional_recovery(self):
        source = np.full((72, 96, 3), 45, np.uint8)
        source[31, 41:43] = 235

        class Reader:
            def __init__(self, source):
                pass

            def frame(self, index):
                return source.copy()

            def close(self):
                pass

        with patch('moving_v6.engine.Reader', Reader), patch.object(MovingRepair, '_calibrate', return_value=None):
            repairer = MovingRepair('virtual', dict(width=96, height=72, frames=4, fps=30.),
                                    0, 4, 0, (32, 22, 52, 42), lambda *args: None)
            try:
                self.assertEqual(int(repairer.support.sum()), 22)
                self.assertEqual(repairer.accepted, 4)
                self.assertEqual(repairer.original_accepted, 4)
                self.assertEqual(repairer.bridged, 0)
                self.assertEqual(repairer.support_recovered, 0)
                for report in (repairer.bridge_report, repairer.support_report):
                    self.assertEqual(report['skipped_reason'], 'insufficient_reference_support')
            finally:
                repairer.close()

    def test_spatial_upgrade_preserves_verified_donors_and_unmasked_content(self):
        from native_build import prepare
        prepare()
        yy, xx = np.indices((64, 80))
        original = np.stack((30 + xx, 30 + yy, 40 + xx // 2), axis=2).astype(np.uint8)
        box = [25, 20, 55, 44]
        mask = np.zeros((24, 30), bool)
        mask[3:21, 3:27] = True
        original[20:44, 25:55][mask] = 240
        recovered = np.zeros_like(mask)
        recovered[8:15, 3:27] = True
        temporal = original.copy()
        # A corroborated donor may expose a dark stroke completely invisible
        # in the current frame. Spatial fitting must not erase that observation.
        temporal[20:44, 25:55][recovered] = (8, 16, 24)

        class Donors:
            def repair(self, index, frame):
                return temporal.copy(), dict(roi=box, recovered_mask=recovered.copy())

        repairer = MovingRepair.__new__(MovingRepair)
        repairer.width, repairer.height = 80, 64
        repairer.masks = {0: (box, mask)}
        repairer.calibrated = RepairCache()
        repairer.adaptive = Donors()
        repairer.model = None
        repaired, counts = repairer.repair(0, original)
        np.testing.assert_array_equal(repaired[20:44, 25:55][recovered],
                                      temporal[20:44, 25:55][recovered])
        full_mask = np.zeros(original.shape[:2], bool)
        full_mask[20:44, 25:55] = mask
        np.testing.assert_array_equal(repaired[~full_mask], original[~full_mask])
        self.assertEqual(counts[1], int(recovered.sum()))
        self.assertEqual(counts[3], int((mask & ~recovered).sum()))
        self.assertGreater(counts[4]['write_pixels'], 0)

    def test_rejected_track_never_changes_source(self):
        repairer = MovingRepair.__new__(MovingRepair)
        repairer.masks = {}
        source = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
        result, counts = repairer.repair(7, source)
        np.testing.assert_array_equal(result, source)
        self.assertEqual(sum(counts[:4]), 0)
        self.assertFalse(np.shares_memory(result, source))

    def test_write_boundary_ignores_unrelated_solver_edits(self):
        source = np.full((72, 96, 3), 45, np.uint8)
        box = [25, 20, 55, 44]
        mask = np.zeros((24, 30), bool)
        mask[3:21, 3:27] = True

        class Donors:
            def repair(self, index, frame):
                return np.full_like(frame, 110), dict(roi=box, recovered_mask=mask.copy())

        repairer = MovingRepair.__new__(MovingRepair)
        repairer.width, repairer.height = 96, 72
        repairer.masks = {0: (box, mask)}
        repairer.calibrated = RepairCache()
        repairer.adaptive = Donors()
        repairer.model = None
        output, counts = repairer.repair(0, source)
        expected = source.copy()
        expected[20:44, 25:55][mask] = 110
        np.testing.assert_array_equal(output, expected)
        self.assertEqual(counts[0], int(mask.sum()))


if __name__ == '__main__':
    unittest.main()
