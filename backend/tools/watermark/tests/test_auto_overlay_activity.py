"""Local activity evidence must not become global absence or erase authority."""
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from auto_overlay.activity import ComponentEvidence, refine_transitions, sampled_match_runs


class AutoActivityTests(unittest.TestCase):
    def setUp(self):
        self.background = np.full((96, 256), 30, np.uint8)
        self.reference = self.background.copy()
        cv2.putText(self.reference, 'Q4', (28, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 240, 3, cv2.LINE_AA)
        self.support = self.reference != self.background
        self.verifier = ComponentEvidence(self.reference, self.support)

    def test_low_opacity_is_unknown_not_absent(self):
        faded = (self.background*.75+self.reference*.25).astype(np.uint8)
        result = self.verifier.observe(faded)
        self.assertEqual(result['state'], 'unknown')
        self.assertEqual(result['scope'], 'reference_position_only')
        self.assertNotIn('accepted', result)

    def test_same_watermark_elsewhere_is_not_global_absence(self):
        moved = cv2.warpAffine(self.reference, np.float32([[1, 0, 120], [0, 1, 0]]),
                               (256, 96), borderValue=30)
        result = self.verifier.observe(moved)
        self.assertEqual(result['state'], 'unknown')
        self.assertEqual(result['anchorEvidence'], 'no_reference_evidence')

    def test_blank_and_changed_shape_never_authorize_erasing(self):
        wrong = self.background.copy()
        cv2.putText(wrong, 'Y9', (28, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 240, 3, cv2.LINE_AA)
        for image in (wrong, self.background):
            result = self.verifier.observe(image)
            self.assertEqual(result['state'], 'unknown')
            self.assertNotIn('mask', result)

    def test_observationally_identical_scene_text_keeps_local_scope(self):
        result = self.verifier.observe(self.reference)
        self.assertEqual(result['state'], 'present')
        self.assertEqual(result['scope'], 'reference_position_only')
        self.assertNotIn('accepted', result)

    def test_dense_audit_keeps_non_monotonic_switches(self):
        def observe(index):
            return dict(state='present' if index in (10, 11) or index >= 14 else 'unknown')
        sparse = [dict(frame=0, state='unknown'), dict(frame=20, state='present')]
        result = refine_transitions(sparse, observe, 21, radius=6)
        transition = result['transitions'][0]
        self.assertEqual([x['firstNextFrame'] for x in transition['boundaries']], [10, 12, 14])
        self.assertEqual(transition['denseCoverage'], [3, 17])
        self.assertNotIn('absent', str(result))

    def test_resolution_disagreement_is_reported_without_fake_boundary(self):
        sparse = [dict(frame=0, state='unknown'), dict(frame=20, state='present')]
        result = refine_transitions(sparse, lambda index: dict(state='present'), 21, radius=3)
        self.assertEqual(result['fullResolutionFrames'], [0, 20])
        self.assertEqual(result['transitions'][0]['reason'], 'resolution_dependent_evidence')
        self.assertNotIn('boundaries', result['transitions'][0])

    def test_sparse_run_does_not_claim_intervening_frames_verified(self):
        sparse = [dict(frame=0, state='unknown'), dict(frame=20, state='present'),
                  dict(frame=40, state='present'), dict(frame=60, state='unknown')]
        result = sampled_match_runs(sparse, 80)[0]
        self.assertEqual(result['status'], 'runs_of_sampled_matches')
        self.assertEqual(result['neighboringUnmatchedFrames'], [0, 60])
        self.assertEqual(result['unverifiedStartBracket'], [1, 20])
        self.assertNotIn('accepted', result)


if __name__ == '__main__':
    unittest.main()
