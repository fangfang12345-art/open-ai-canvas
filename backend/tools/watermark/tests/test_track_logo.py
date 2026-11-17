"""Motion prediction at image boundaries must retain global reacquisition."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moving_v6.track_logo import TemplateAnchorTracker


class TrackerBoundaryRegression(unittest.TestCase):
    roi = (30, 30, 94, 70)

    @staticmethod
    def frame(position):
        image = np.full((320, 320), 45, np.uint8)
        patch = np.full((40, 64), 45, np.uint8)
        cv2.putText(patch, 'L7', (8, 29), cv2.FONT_HERSHEY_SIMPLEX, .8, 235, 2, cv2.LINE_AA)
        x, y = position
        image[y:y+40, x:x+64] = patch
        return image

    def approach_boundary(self, axis):
        tracker = TemplateAnchorTracker(self.frame((30, 30)), self.roi)
        sequence = [30, 46, 67, 92, 121, 153, 187, 223, 256]
        if axis == 1:
            sequence.append(280)
        for coordinate in sequence:
            position = [30, 30]
            position[axis] = coordinate
            result = tracker.update(self.frame(position))
            self.assertTrue(result['accepted'], result)
            self.assertEqual(result['search'], 'local')
            self.assertLess(np.linalg.norm(np.asarray(result['anchor_subpixel_xy']) - position), .1)
        self.assertGreater(tracker.velocity[axis], tracker.search_margin + 1)
        return tracker, position

    def test_right_boundary_stop_reacquires_complete_visible_target(self):
        tracker, position = self.approach_boundary(0)
        result = tracker.update(self.frame(position))
        self.assertTrue(result['accepted'], result)
        self.assertEqual(result['search'], 'global')
        self.assertLess(np.linalg.norm(np.asarray(result['anchor_subpixel_xy']) - position), .1)
        np.testing.assert_array_equal(tracker.velocity, [0., 0.])

    def test_bottom_boundary_stop_reacquires_complete_visible_target(self):
        tracker, position = self.approach_boundary(1)
        result = tracker.update(self.frame(position))
        self.assertTrue(result['accepted'], result)
        self.assertEqual(result['search'], 'global')
        self.assertLess(np.linalg.norm(np.asarray(result['anchor_subpixel_xy']) - position), .1)
        np.testing.assert_array_equal(tracker.velocity, [0., 0.])

    def test_missing_boundary_target_is_rejected_after_global_search(self):
        tracker, _ = self.approach_boundary(0)
        result = tracker.update(np.full((320, 320), 45, np.uint8))
        self.assertFalse(result['accepted'], result)
        self.assertEqual(result['search'], 'global')
        self.assertIsNone(result['anchor_subpixel_xy'])

    def test_unrelated_match_error_is_not_swallowed(self):
        tracker = TemplateAnchorTracker(self.frame((30, 30)), self.roi)
        with self.assertRaises(cv2.error):
            tracker.update(self.frame((30, 30)).astype(np.float32))


if __name__ == '__main__':
    unittest.main()
