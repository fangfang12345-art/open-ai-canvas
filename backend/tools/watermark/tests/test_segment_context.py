"""Long-source segment boundaries and distant reference context."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moving_v6.engine import MovingRepair
from runner import frame_boundary


class SegmentContextRegression(unittest.TestCase):
    def test_absolute_boundaries_have_no_gaps_at_fractional_rates(self):
        for fps in (24000/1001, 30000/1001, 60000/1001, 25.05):
            boundaries = [frame_boundary(index*10., fps) for index in range(61)]
            counts = [end-start for start, end in zip(boundaries, boundaries[1:])]
            self.assertEqual(sum(counts), frame_boundary(600., fps))
            self.assertLessEqual(max(counts)-min(counts), 1)
        self.assertEqual(frame_boundary(10., 25.05), 251)

    def test_distant_reference_decodes_only_segment_context(self):
        first, count, fps = 1_000_000, 120, 12.
        begin, end = first-48, first+count+48
        seen = []

        class Reader:
            def __init__(self, source):
                pass

            def frame(self, index):
                seen.append(index)
                if index != reference and not begin <= index < end:
                    raise AssertionError('Decoded intervening video outside segment context')
                image = np.full((120, 160, 3), 38, np.uint8)
                x = 12 if index == reference else 70
                cv2.rectangle(image, (x+5, 27), (x+14, 39), (235,)*3, -1)
                cv2.rectangle(image, (x+14, 34), (x+22, 39), (235,)*3, -1)
                return image

            def close(self):
                pass

        for reference in (0, 1_999_999):
            seen.clear()
            with patch('moving_v6.engine.Reader', Reader), patch.object(MovingRepair, '_calibrate', return_value=None):
                repairer = MovingRepair('virtual', dict(width=160, height=120, frames=2_000_000, fps=fps),
                                        first, count, reference, (12, 22, 40, 46), lambda *args: None)
                try:
                    self.assertEqual(repairer.accepted, count)
                    self.assertEqual(set(repairer.records), set(range(begin, end)))
                    self.assertEqual(len(repairer.cut_prefix), end-begin)
                    self.assertLessEqual(len(set(seen)), end-begin+1)
                    self.assertLessEqual(len(seen), 2*(end-begin)+2)
                    self.assertTrue(repairer.tracking_context['reacquired'])
                finally:
                    repairer.close()

    def test_segment_without_verified_logo_preserves_source_and_reports_it(self):
        from test_support_recovery import make_frame
        reference = make_frame()
        blank = np.full_like(reference, 45)

        class Reader:
            def __init__(self, source):
                pass

            def frame(self, index):
                return reference.copy() if index == 0 else blank.copy()

            def close(self):
                pass

        with patch('moving_v6.engine.Reader', Reader):
            repairer = MovingRepair('virtual', dict(width=176, height=104, frames=600, fps=30.),
                                    300, 30, 0, (30, 30, 94, 70), lambda *args: None)
            try:
                self.assertEqual(repairer.accepted, 0)
                output = list(repairer.frames(Reader('virtual'), 300, 30, 2))
                self.assertEqual(len(output), 30)
                for frame in output:
                    np.testing.assert_array_equal(frame, blank)
                report = repairer.report()
                self.assertTrue(report['passthrough'])
                self.assertEqual(report['untrackedFrames'], 30)
                self.assertEqual(report['changedPixels'], 0)
                details = report['untrackedFrameDetails']
                self.assertEqual(details['total'], 30)
                self.assertFalse(details['truncated'])
                self.assertEqual(details['frameIndexBase'], 0)
                self.assertEqual([frame['frameIndex'] for frame in details['frames']], list(range(300, 330)))
                for frame in details['frames']:
                    index = frame['frameIndex']
                    self.assertEqual(frame['timeSeconds'], index / 30.)
                    self.assertEqual(frame['reasons']['contour'], repairer.records[index]['contour_evidence']['reason'])
                limited = repairer._untracked_frame_details(limit=2)
                self.assertEqual(limited['total'], 30)
                self.assertTrue(limited['truncated'])
                self.assertEqual([frame['frameIndex'] for frame in limited['frames']], [300, 301])
                # The service reads completed JSON with a 64 KiB line limit and
                # the indented report with a 256 KiB limit.
                record = dict(repairer.records[300])
                record['contour_evidence'] = dict(reason='ambiguous_contours',
                                                  temporal_retry=dict(reason='insufficient_search_area'))
                record['bridge'] = dict(reason='scene_cut')
                record['support_evidence'] = dict(reason='insufficient_reference_support')
                repairer.records = {index: dict(record) for index in range(300, 1500)}
                repairer.output_count = 1200
                report = repairer.report()
                details = report['untrackedFrameDetails']
                self.assertEqual(details['total'], 1200)
                self.assertTrue(details['truncated'])
                self.assertLessEqual(len(details['frames']), 100)
                self.assertLessEqual(len(json.dumps(details).encode('utf-8')), 32 * 1024)
                completed = dict(progress=100, stage='completed', report=report)
                self.assertLess(len(json.dumps(completed).encode('utf-8')), 64 * 1024)
                self.assertLess(len(json.dumps(report, indent=2).encode('utf-8')), 256 * 1024)
                self.assertEqual(repairer._untracked_frame_details(max_bytes=128)['frames'], [])
            finally:
                repairer.close()

    def test_fractional_short_tail_shares_the_preceding_boundary(self):
        fps = 30000/1001
        total = 601
        previous_end = frame_boundary(20., fps)
        tail_start = frame_boundary(20., fps)
        tail_end = min(total, frame_boundary((total/fps), fps))
        self.assertEqual(previous_end, tail_start)
        self.assertEqual(tail_end-tail_start, 2)


if __name__ == '__main__':
    unittest.main()
