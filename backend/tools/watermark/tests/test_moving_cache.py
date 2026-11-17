"""Exact source slicing and bounded ownership for the moving donor caches."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from moving_v6.adaptive_temporal import AdaptiveTemporalRepair
from moving_v6.engine import FrameCache, MovingRepair, PlacedMasks, RepairCache


class ArrayReader:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []
        self.closed = False

    def frame(self, index):
        self.calls.append(index)
        return self.frames[index].copy()

    def close(self):
        self.closed = True


class FrameCacheTests(unittest.TestCase):
    def setUp(self):
        self.frames = np.random.default_rng(28).integers(0, 256, (80, 60, 100, 3), dtype=np.uint8)
        self.reader = ArrayReader(self.frames)

    def cache(self, **kwargs):
        with patch("moving_v6.engine.Reader", return_value=self.reader):
            result = FrameCache("unused", **kwargs)
        self.addCleanup(result.close)
        return result

    def assert_owned_budget(self, cache):
        self.assertEqual(cache.used, sum(frame.nbytes for frame in cache.frames.values()))
        self.assertLessEqual(cache.used, cache.max_bytes)
        self.assertTrue(all(frame.base is None for frame in cache.frames.values()))

    def test_union_retains_a_donor_window_without_pinning_full_frames(self):
        cache = self.cache(region=(30, 20, 50, 30), min_frames=80, max_bytes=80 * 20 * 10 * 3)
        for _ in range(2):
            for index in range(80):
                np.testing.assert_array_equal(cache[index, 22:28, 32:48], self.frames[index, 22:28, 32:48])
        self.assertEqual(len(self.reader.calls), 80)
        self.assert_owned_budget(cache)
        result = cache[0, 22:28, 32:48]
        result[:] = 0
        np.testing.assert_array_equal(cache[0, 22:28, 32:48], self.frames[0, 22:28, 32:48])

    def test_absolute_slices_outside_union_and_image_edges_match_numpy(self):
        cache = self.cache(region=(30, 20, 50, 30), min_frames=1, max_bytes=2000)
        slices = [(slice(None), slice(None)), (slice(-4, 200), slice(-20, 300)),
                  (slice(29, 19, -2), slice(49, 29, -3)), (slice(None, None, -2), slice(None, None, -3)),
                  (slice(21, 29, 3), slice(32, 49, 4)), (slice(500, 900), slice(None)),
                  (slice(18, 32), slice(28, 52)), (slice(20, 30), slice(30, 50))]
        for rows, columns in slices:
            with self.subTest(rows=rows, columns=columns):
                np.testing.assert_array_equal(cache[3, rows, columns], self.frames[3, rows, columns])
                self.assert_owned_budget(cache)
        self.assertEqual(cache.frames[3].shape, (10, 20, 3))

    def test_clips_cached_union_to_actual_source_dimensions(self):
        cache = self.cache(region=(-20, -10, 200, 300), min_frames=1, max_bytes=20000)
        np.testing.assert_array_equal(cache[2, -25:90, -40:200], self.frames[2, -25:90, -40:200])
        self.assertEqual(cache.region, (0, 0, 100, 60))
        self.assert_owned_budget(cache)

    def test_wide_trajectory_uses_tiles_and_reuses_overlapping_queries(self):
        cache = self.cache(region=(0, 0, 100, 60), min_frames=80, max_bytes=3000, tile_size=8)
        np.testing.assert_array_equal(cache[4, 9:23, 17:31], self.frames[4, 9:23, 17:31])
        self.assertIsNone(cache.region)
        np.testing.assert_array_equal(cache[4, 22:8:-2, 30:16:-3], self.frames[4, 22:8:-2, 30:16:-3])
        self.assertEqual(self.reader.calls, [4])
        np.testing.assert_array_equal(cache[4, 53:90, 93:200], self.frames[4, 53:90, 93:200])
        self.assertEqual(self.reader.calls, [4, 4])
        self.assert_owned_budget(cache)

    def test_tile_eviction_and_oversized_tile_do_not_exceed_budget(self):
        for budget in (0, 1, 8 * 8 * 3, 1000):
            cache = self.cache(max_bytes=budget, tile_size=8)
            for index in range(5):
                for rows, columns in ((slice(2, 40, 2), slice(4, 60, 3)),
                                      (slice(55, 20, -3), slice(99, 0, -7))):
                    np.testing.assert_array_equal(cache[index, rows, columns], self.frames[index, rows, columns])
                    self.assert_owned_budget(cache)

    def test_full_frame_reads_do_not_evict_the_tile_working_set(self):
        cache = self.cache(max_bytes=2000, tile_size=8)
        cache[1, 8:16, 16:24]
        keys, used = list(cache.frames), cache.used
        np.testing.assert_array_equal(cache[2, :, :], self.frames[2])
        self.assertEqual(list(cache.frames), keys)
        self.assertEqual(cache.used, used)
        cache[1, 8:16, 16:24]
        self.assertEqual(self.reader.calls, [1, 2])

    def test_concurrent_queries_and_close_preserve_exact_pixels_and_accounting(self):
        cache = self.cache(region=(30, 20, 50, 30), min_frames=1, max_bytes=2000)
        indices = [index % 7 for index in range(40)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda index: cache[index, 20:30, 30:50], indices))
        for index, result in zip(indices, results):
            np.testing.assert_array_equal(result, self.frames[index, 20:30, 30:50])
        self.assert_owned_budget(cache)
        cache.close()
        self.assertEqual(cache.used, 0)
        self.assertFalse(cache.frames)
        self.assertTrue(self.reader.closed)


class PlacedMaskCacheTests(unittest.TestCase):
    def test_cached_masks_match_uncached_fractional_warp_and_remain_bounded(self):
        support = np.zeros((9, 13), bool)
        support[2:8, 2:11] = True
        records = {i: dict(accepted=True, anchor_subpixel_xy=(i * .8 - 1, i * .4 - 1)) for i in range(40)}
        masks = PlacedMasks(records, support, 100, 60, max_bytes=1000, max_entries=3)
        for index in list(records) + list(reversed(records)):
            box, actual = masks[index]
            x, y = records[index]["anchor_subpixel_xy"]
            expected = cv2.warpAffine(support.astype(np.float32),
                                     np.float32([[1, 0, x-box[0]], [0, 1, y-box[1]]]),
                                     (box[2]-box[0], box[3]-box[1]),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT) > .01
            np.testing.assert_array_equal(actual, expected)
            self.assertFalse(actual.flags.writeable)
            self.assertLessEqual(masks.used, 1000)
            self.assertLessEqual(len(masks.cached), 3)
        with patch("moving_v6.engine.cv2.warpAffine", side_effect=AssertionError("cached mask was recomputed")):
            masks[0]
        masks.close()
        self.assertEqual(masks.used, 0)


class RepairCacheTests(unittest.TestCase):
    def test_reconstitutes_exact_temporal_output_once_and_owns_only_the_roi(self):
        original = np.full((60, 100, 3), 25, np.uint8)
        repaired = original.copy()
        repaired[20:30, 30:50] = 170
        recovered = np.ones((10, 20), bool)
        cache = RepairCache(max_bytes=1000)
        cache.put(8, repaired, dict(roi=[30, 20, 50, 30], recovered_mask=recovered))
        self.assertEqual(cache.used, 800)
        self.assertIsNone(cache.frames[8][1].base)
        repaired[:] = 0
        recovered[:] = False
        result, diagnostic = cache.take(8, original)
        expected = original.copy()
        expected[20:30, 30:50] = 170
        np.testing.assert_array_equal(result, expected)
        self.assertTrue(diagnostic["recovered_mask"].all())
        self.assertEqual(cache.used, 0)
        self.assertIsNone(cache.take(8, original))

    def test_eviction_duplicate_replacement_and_oversized_results_are_bounded(self):
        image = np.zeros((10, 20, 3), np.uint8)
        diagnostic = dict(roi=[0, 0, 20, 10], recovered_mask=np.zeros((10, 20), bool))
        cache = RepairCache(max_bytes=1600, max_entries=2)
        for index in (1, 2, 3, 3):
            cache.put(index, image, diagnostic)
            self.assertLessEqual(cache.used, 1600)
        self.assertEqual(list(cache.frames), [2, 3])
        cache.put(4, np.zeros((100, 100, 3), np.uint8),
                  dict(roi=[0, 0, 100, 100], recovered_mask=np.zeros((100, 100), bool)))
        self.assertEqual(list(cache.frames), [2, 3])
        cache.close()
        self.assertEqual(cache.used, 0)
        self.assertFalse(cache.frames)


class TemporalCacheEquivalenceTests(unittest.TestCase):
    def test_donor_selection_and_repair_match_eager_source_pixels(self):
        yy, xx = np.indices((64, 100))
        clean = np.stack((30 + xx, 30 + yy, 40 + (xx // 8) * 6), axis=2).astype(np.uint8)
        records = {index: dict(accepted=True, anchor_subpixel_xy=(20 + index, 25.25)) for index in range(24)}
        support = np.ones((8, 10), bool)
        masks = PlacedMasks(records, support, 100, 64)
        source = []
        for index in records:
            frame = clean.copy()
            (x0, y0, x1, y1), mask = masks[index]
            frame[y0:y1, x0:x1][mask] = 240
            source.append(frame)
        source = np.stack(source)

        def owner(frame_cache):
            result = MovingRepair.__new__(MovingRepair)
            result.width, result.height, result.frame_count = 100, 64, len(source)
            result.masks, result.cache = masks, frame_cache
            result.union, result.context_pad = [0, 0, 100, 64], 12
            result.offsets = (4, 8, 12, 18, 24, 36, 48)
            result.cut_prefix = np.zeros(len(source), np.int32)
            return result

        reader = ArrayReader(source)
        with patch("moving_v6.engine.Reader", return_value=reader):
            bounded = FrameCache("unused", region=(6, 10, 68, 48), min_frames=1)
        self.addCleanup(bounded.close)
        eager_repair = AdaptiveTemporalRepair(owner(source), 12)
        cached_repair = AdaptiveTemporalRepair(owner(bounded), 12)
        expected, expected_report = eager_repair.repair(12, source[12])
        actual, actual_report = cached_repair.repair(12, source[12])
        self.assertGreater(actual_report["selection_pool"], 0)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual_report["recovered_mask"], expected_report["recovered_mask"])
        self.assertEqual(actual_report["selection"], expected_report["selection"])
        self.assertEqual(actual_report["selection_pool"], expected_report["selection_pool"])


if __name__ == "__main__":
    unittest.main()
