"""Source-only proposal invariants, without claiming foreground ownership."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_overlay.discover import proposals, temporal_windows


class AutoDiscoveryTests(unittest.TestCase):
    def test_windows_include_short_and_full_lifetime(self):
        windows = temporal_windows(96)
        self.assertIn((0, 96), windows)
        self.assertTrue(any(a <= 31 and b <= 36 and b-a == 3 for a,b in windows))

    def test_intermediate_short_overlay_is_not_diluted_by_full_video(self):
        random = np.random.default_rng(723)
        frames = random.integers(70,100,(36,120,240),np.uint8)
        for frame in frames[12:18]:
            cv2.putText(frame,'Q4',(75,75),cv2.FONT_HERSHEY_SIMPLEX,1.1,235,2,cv2.LINE_AA)
        found = proposals(frames)
        matches = [c for c in found if c['box'][0] < 90 and c['box'][2] > 110
                   and c['box'][1] < 60 and c['box'][3] > 68]
        self.assertTrue(matches)
        self.assertTrue(any(all(12 <= i < 18 for i in c['present']) for c in matches))

    def test_dark_shape_at_new_position_is_proposed(self):
        frames = np.full((12,128,256),180,np.uint8)
        for frame in frames:
            cv2.putText(frame,'B8',(135,100),cv2.FONT_HERSHEY_SIMPLEX,1.15,35,2,cv2.LINE_AA)
        found = proposals(frames)
        self.assertTrue(any(c['box'][0] > 120 and c['box'][3] > 95 for c in found))

    def test_uniform_video_has_no_candidate(self):
        frames = np.full((12,128,256),180,np.uint8)
        self.assertEqual(proposals(frames), [])

    def test_repeated_scene_text_remains_a_proposal_not_an_ownership_claim(self):
        # These pixels are observationally identical whether drawn on a wall
        # or composited. Discovery must expose this ambiguity to later stages.
        frames = np.full((12,128,256),180,np.uint8)
        for frame in frames:
            cv2.putText(frame,'ROOM',(25,80),cv2.FONT_HERSHEY_SIMPLEX,.8,35,2,cv2.LINE_AA)
        found = proposals(frames)
        self.assertTrue(found)
        self.assertTrue(all('accepted' not in candidate for candidate in found))


if __name__ == '__main__':
    unittest.main()
