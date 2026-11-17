"""Synthetic regressions and real CLI/FFmpeg smoke tests; no external assets."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from media import executable, probe
from moving_v6.coherent_repair import repair_patch
from moving_v6.affine_overlay import fit_overlay


class FixedRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from native_build import prepare
        prepare()
        from fixed_r16.engine import repair_task
        from spatial_repair import local_spatial
        import robust_continuation
        cls.repair_task = staticmethod(repair_task)
        cls.local_spatial = staticmethod(local_spatial)
        cls.solver = robust_continuation

    def test_r16_four_round_reference_and_native_agree(self):
        yy, xx = np.indices((20, 28))
        truth = np.stack([20 + xx * 4, 30 + yy * 5, 40 + xx + yy * 2], axis=2).astype(np.uint8)
        unknown = np.zeros(truth.shape[:2], bool)
        unknown[7:14, 10:17] = True
        observed, base = truth.copy(), truth.copy()
        observed[unknown] = 240
        base[unknown] = 75
        fields = [np.full(unknown.shape, value) for value in (.7, .4, 1., 0.)]
        expected, report, _ = self.solver.solve_with_fields(
            observed, base, unknown, np.zeros_like(unknown), *fields, native=False)
        self.assertEqual(report["robust_pcg_rounds"], 4)
        np.testing.assert_array_equal(expected[~unknown], observed[~unknown])
        self.assertTrue(np.any(expected[unknown] != base[unknown]))
        if self.solver.NATIVE is not None:
            actual, report, _ = self.solver.solve_with_fields(
                observed, base, unknown, np.zeros_like(unknown), *fields, native=True)
            self.assertEqual(report["robust_pcg_rounds"], 4)
            np.testing.assert_array_equal(actual, expected)

    def test_local_ipc_crop_matches_full_frame_before_encoding(self):
        yy, xx = np.indices((80, 112))
        original = np.stack([20 + xx, 20 + yy, 60 + (xx // 10) * 8], axis=2).astype(np.uint8)
        box = (40, 25, 66, 48)
        mask = np.zeros((23, 26), bool)
        mask[5:18, 7:20] = True
        original[25:48, 40:66][mask] = 240
        expected = self.local_spatial(original, box, mask.astype(np.uint8), diagnostics=True)[0]
        patch = original[13:60, 28:78].copy()
        repaired = self.repair_task(patch, (12, 12, 38, 35), mask)[0]
        actual = original.copy()
        actual[13:60, 28:78] = repaired
        np.testing.assert_array_equal(actual, expected)


class TemporalRegression(unittest.TestCase):
    def setUp(self):
        self.clean = np.full((96, 128, 3), 38, np.uint8)
        cv2.line(self.clean, (15, 48), (110, 48), (220, 220, 220), 3)
        self.mask = np.zeros((96, 128), bool)
        self.mask[35:61, 49:80] = True
        self.target = self.clean.copy()
        self.target[self.mask] = (90, 10, 190)

    def donor(self, index, **extra):
        return dict(image=self.clean.copy(), mask=np.zeros_like(self.mask), index=index,
                    distance=index - 20, cut_ok=True, **extra)

    def test_two_independent_donors_restore_actual_hidden_line(self):
        result, report = repair_patch(self.target, self.mask, [self.donor(10), self.donor(30)])
        np.testing.assert_array_equal(result, self.clean)
        self.assertEqual(report["recovered_pixels"], int(self.mask.sum()))

    def test_missing_mask_scene_cut_and_duplicate_donors_cannot_vote(self):
        cases = [[self.donor(10), self.donor(10)], [self.donor(10)],
                 [self.donor(10), dict(self.donor(30), mask=None)],
                 [self.donor(10), dict(self.donor(30), cut_ok=False)]]
        for donors in cases:
            result, report = repair_patch(self.target, self.mask, donors)
            np.testing.assert_array_equal(result, self.target)
            self.assertEqual(report["recovered_pixels"], 0)

    def test_disagreeing_donors_do_not_average_hidden_content(self):
        changed = self.donor(30)
        changed["image"][self.mask] = (0, 0, 0)
        result, report = repair_patch(self.target, self.mask, [self.donor(10), changed])
        np.testing.assert_array_equal(result[~self.mask], self.target[~self.mask])
        self.assertFalse(report["recovered_mask"][48, 64])

    def test_affine_fit_requires_background_variation_and_holdout(self):
        backgrounds = np.broadcast_to(np.linspace(20, 200, 16)[:, None, None, None], (16, 4, 4, 3)).copy()
        observations = .65 * backgrounds + 70
        model = fit_overlay(observations, backgrounds, np.ones((16, 4, 4), bool), np.arange(16))
        self.assertTrue(model["support"].all())
        np.testing.assert_allclose(model["gain"], .65, atol=1e-4)
        self.assertFalse(model["diagnostics"]["heldout_used_for_fitting"])
        constant = np.full_like(backgrounds, 40)
        rejected = fit_overlay(.65 * constant + 70, constant, np.ones((16, 4, 4), bool), np.arange(16))
        self.assertFalse(rejected["support"].any())


class CLISmoke(unittest.TestCase):
    def make_source(self, path, moving=False, mark_value=235):
        frames = []
        for index in range(36):
            frame = np.full((120, 160, 3), 38, np.uint8)
            cv2.line(frame, (0, 94), (159, 94), (100, 70, 30), 2)
            x = 12 + index * 2 if moving else 12
            cv2.rectangle(frame, (x + 5, 27), (x + 14, 39), (mark_value,) * 3, -1)
            cv2.rectangle(frame, (x + 14, 34), (x + 22, 39), (mark_value,) * 3, -1)
            frames.append(frame)
        command = [executable("ffmpeg"), "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                   "-s", "160x120", "-r", "12", "-i", "pipe:0", "-f", "lavfi", "-i",
                   "sine=frequency=440:sample_rate=48000", "-map", "0:v", "-map", "1:a",
                   "-c:v", "libx264", "-crf", "0", "-pix_fmt", "yuv444p", "-c:a", "aac",
                   "-t", "3", "-y", str(path)]
        subprocess.run(command, input=np.stack(frames).tobytes(), check=True, capture_output=True)

    def test_fixed_gray_rejection_explains_scope_without_claiming_detected_motion(self):
        from fixed_r16.calibrate_fixed import detect_persistent_light
        with self.assertRaisesRegex(ValueError, "No persistent bright neutral watermark candidates"):
            detect_persistent_light(np.full((3, 24, 28, 3), 50, np.uint8))
        with tempfile.TemporaryDirectory(prefix="watermark-fixed-scope-") as temporary:
            directory = Path(temporary)
            source, output = directory / "source.mp4", directory / "output.mp4"
            self.make_source(source, mark_value=70)
            request = dict(input=str(source), branch="fixed-r16", workers=1,
                           roi=dict(x=12, y=22, width=28, height=24))
            request_path = directory / "request.json"
            request_path.write_text(json.dumps(request))
            result = subprocess.run([sys.executable, "-B", str(ROOT / "runner.py"), "--request",
                                     str(request_path), "--output", str(output)], capture_output=True,
                                    text=True, timeout=120)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            self.assertIn("未可靠识别位置固定的亮色水印", result.stderr)
            self.assertIn("水印会移动时请选择移动 V6", result.stderr)
            self.assertNotIn("检测到移动", result.stderr)
            self.assertNotIn(temporary, result.stderr)

    def test_both_branches_trim_keep_audio_and_emit_real_algorithm_statistics(self):
        with tempfile.TemporaryDirectory(prefix="watermark-smoke-") as temporary:
            directory = Path(temporary)
            for branch in ("fixed-r16", "moving-v6"):
                source = directory / f"{branch}-source.mp4"
                output = directory / f"{branch}-output.mp4"
                self.make_source(source, moving=branch == "moving-v6")
                request = dict(input=str(source), branch=branch, startSeconds=.5, durationSeconds=2,
                               referenceSeconds=0, workers=2, target="top-left",
                               roi=dict(x=12, y=22, width=28, height=24))
                request_path = directory / "request.json"
                request_path.write_text(json.dumps(request))
                result = subprocess.run([sys.executable, "-B", str(ROOT / "runner.py"), "--request",
                                         str(request_path), "--output", str(output)], capture_output=True,
                                        text=True, timeout=180)
                self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
                events = [json.loads(line) for line in result.stdout.splitlines()]
                self.assertEqual(events[-1]["progress"], 100)
                self.assertEqual(events[-1]["stage"], "completed")
                self.assertEqual([event["progress"] for event in events], sorted(event["progress"] for event in events))
                report = json.loads(output.with_suffix(".report.json").read_text())
                self.assertGreater(report["changedPixels"], 0)
                self.assertEqual(report["frames"], 24)
                if branch == "moving-v6":
                    self.assertGreater(report["temporalRecoveredPixels"], 0)
                    self.assertGreaterEqual(report["trackedFrames"], 20)
                    self.assertEqual(report["untrackedFrames"], report["frames"] - report["trackedFrames"])
                    self.assertEqual(report["trackingCoverage"], report["trackedFrames"] / report["frames"])
                data = json.loads(subprocess.check_output([executable("ffprobe"), "-v", "error",
                                                          "-show_streams", "-of", "json", str(output)]))
                self.assertEqual(sum(s["codec_type"] == "audio" for s in data["streams"]), 1)
                self.assertAlmostEqual(probe(output)["duration"], 2, places=2)

    def test_zero_duration_reaches_end_and_invalid_reference_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="watermark-tail-") as temporary:
            directory = Path(temporary)
            source = directory / "source.mp4"
            self.make_source(source)
            request = dict(input=str(source), branch="fixed-r16", startSeconds=2, durationSeconds=0,
                           workers=1, roi=dict(x=12, y=22, width=28, height=24))
            path = directory / "request.json"
            path.write_text(json.dumps(request))
            output = directory / "output.mp4"
            result = subprocess.run([sys.executable, "-B", str(ROOT / "runner.py"), "--request", str(path),
                                     "--output", str(output)], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual(json.loads(output.with_suffix(".report.json").read_text())["frames"], 12)
            request["referenceSeconds"] = 4
            path.write_text(json.dumps(request))
            invalid = directory / "invalid.mp4"
            result = subprocess.run([sys.executable, "-B", str(ROOT / "runner.py"), "--request", str(path),
                                     "--output", str(invalid)], capture_output=True, text=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(invalid.exists())
            self.assertTrue(result.stderr.splitlines()[-1].startswith("ERROR: "))
            self.assertNotIn(temporary, result.stderr)

    def test_check_works_without_site_packages(self):
        result = subprocess.run([sys.executable, "-S", "-B", str(ROOT / "runner.py"), "--check"],
                                capture_output=True, text=True, check=True)
        self.assertFalse(json.loads(result.stdout)["branches"]["moving-v6"]["available"])

    def test_tail_reference_tracks_backward_into_earlier_output(self):
        with tempfile.TemporaryDirectory(prefix="watermark-reverse-") as temporary:
            directory = Path(temporary)
            source, output = directory / "source.mp4", directory / "output.mp4"
            self.make_source(source, moving=True)
            request = dict(input=str(source), branch="moving-v6", startSeconds=0, durationSeconds=1,
                           referenceSeconds=2.999, workers=1, roi=dict(x=82, y=22, width=28, height=24))
            path = directory / "request.json"
            path.write_text(json.dumps(request))
            result = subprocess.run([sys.executable, "-B", str(ROOT / "runner.py"), "--request", str(path),
                                     "--output", str(output)], capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(output.with_suffix(".report.json").read_text())
            self.assertEqual(report["frames"], 12)
            self.assertEqual(report["trackedFrames"], 12)
            self.assertGreater(report["temporalRecoveredPixels"], 0)

    def test_untracked_moving_segment_completes_with_explicit_passthrough(self):
        with tempfile.TemporaryDirectory(prefix="watermark-untracked-") as temporary:
            directory = Path(temporary)
            marked, source, output = [directory/name for name in ('marked.mp4', 'source.mp4', 'output.mp4')]
            self.make_source(marked)
            subprocess.run([executable('ffmpeg'), '-v', 'error', '-i', str(marked),
                            '-vf', "drawbox=x=0:y=20:w=160:h=30:color=0x262626:t=fill:enable='gte(t,0.5)'",
                            '-c:v', 'libx264', '-crf', '0', '-c:a', 'copy', str(source)],
                           check=True, capture_output=True)
            request = dict(input=str(source), branch='moving-v6', startSeconds=1., durationSeconds=1.,
                           referenceSeconds=0., workers=1, roi=dict(x=12, y=22, width=28, height=24),
                           validationCacheDir=str(directory/'validation'))
            request_path = directory/'request.json'
            request_path.write_text(json.dumps(request))
            result = subprocess.run([sys.executable, '-B', str(ROOT/'runner.py'), '--request',
                                     str(request_path), '--output', str(output)], capture_output=True,
                                    text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            report = json.loads(output.with_suffix('.report.json').read_text())
            self.assertTrue(report['passthrough'])
            self.assertEqual(report['untrackedFrames'], 12)
            self.assertEqual(report['trackedFrames'], 0)
            self.assertEqual(report['changedPixels'], 0)
            self.assertAlmostEqual(probe(output)['duration'], 1., places=2)


if __name__ == "__main__":
    unittest.main()
