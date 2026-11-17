"""Shared fixed-reference identity, activity boundaries and output isolation."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fixed_r16.regions import MultiFixedRepair, calibrate, identity, load_models, missing_stroke_size, validate_regions
from fixed_r16.calibrate_fixed import detect_persistent_light
from media import executable, probe
from test_support_recovery import make_frame


def fixture(text='L7', background=45):
    return make_frame(text=text, background=background)[20:80, 20:105].copy()


def model(image):
    support, core, _, statistics = detect_persistent_light(np.stack([image]*3))
    return dict(region=1, first=0, end=10, reference=5, roi=[0, 0, image.shape[1], image.shape[0]],
                support=support, core=core, reference_image=image, calibration=statistics)


class ArrayReader:
    def __init__(self, frames):
        self.frames = frames
        self.calls = []

    def frame(self, index):
        self.calls.append(index)
        return self.frames[index].copy()

    def close(self):
        pass


class FixedRegionTests(unittest.TestCase):
    info = dict(width=320, height=104, fps=10., rate='10', frames=100, duration=10.)
    region = dict(roi=dict(x=20, y=20, width=85, height=60), referenceSeconds=.5,
                  startSeconds=0, durationSeconds=1.)

    def test_validation_rejects_overlap_empty_and_outside_reference(self):
        for value in ([], [self.region]*2, [dict(self.region, referenceSeconds=1.)],
                      [dict(self.region, durationSeconds=.1)],
                      [dict(self.region, roi=dict(x=300, y=0, width=85, height=60))]):
            with self.assertRaises(ValueError):
                validate_regions(value, self.info)

    def test_absolute_activity_boundaries_use_half_up_rounding(self):
        info = dict(self.info, fps=30000/1001, frames=300, duration=10.01)
        actual = validate_regions([dict(self.region, referenceSeconds=1., startSeconds=.05, durationSeconds=1.)], info)[0]
        self.assertEqual((actual['first'], actual['end']), (1, 31))

    def test_reference_shape_survives_changed_background_and_codec_noise(self):
        verifier = identity(model(fixture()))
        for background in (0, 45, 90, 130):
            image = fixture(background=background)
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            self.assertTrue(ok)
            self.assertTrue(verifier.match(image)['accepted'])
            self.assertTrue(verifier.match(cv2.imdecode(encoded, cv2.IMREAD_COLOR))['accepted'])

    def test_missing_changed_and_extra_strokes_cannot_be_substituted(self):
        for original, others in [('L7', ['', 'L2', 'L1', 'L8', 'F7']), ('F', ['E']), ('C', ['G'])]:
            verifier = identity(model(fixture(original)))
            for changed in others:
                self.assertFalse(verifier.match(fixture(changed))['accepted'], (original, changed))

    def test_missing_stroke_counts_only_unobserved_edges(self):
        reference = np.zeros((12, 24), np.uint8)
        reference[5, 2:22] = 1
        missing = np.zeros_like(reference)
        missing[5, 6:11] = 1
        self.assertEqual(missing_stroke_size(missing, reference), 5)
        missing[5, 12:16] = 1
        self.assertEqual(missing_stroke_size(missing, reference), 9)
        self.assertEqual(missing_stroke_size(np.zeros_like(missing), reference), 0)

    def test_validation_before_prepare_keeps_direct_and_worker_native_bindings(self):
        script = '''
import json, multiprocessing, sys
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np
from fixed_r16.regions import region_repair_task, validate_regions

def main():
    assert 'fixed_r16.engine' not in sys.modules
    from native_build import prepare
    prepared = prepare()
    if 'native_robust_continuation' not in prepared.get('kernels', []):
        print(json.dumps({'skip': True}))
        return
    yy, xx = np.mgrid[:48, :64]
    image = np.repeat((30 + xx + yy * 2).astype(np.uint8)[..., None], 3, axis=2)
    mask = np.zeros((32, 48), bool)
    mask[8:24, 12:36] = True
    image[16:32, 20:44] = 240
    args = (image, (8, 8, 56, 40), mask)
    results = {}
    # The child must inherit the prepared environment, whether it shares the
    # parent's modules or imports them again in a fresh interpreter.
    for method in multiprocessing.get_all_start_methods():
        if method not in ('fork', 'spawn'):
            continue
        with ProcessPoolExecutor(1, mp_context=multiprocessing.get_context(method)) as pool:
            results[method] = pool.submit(region_repair_task, *args).result()[-1]
    results['direct'] = region_repair_task(*args)[-1]
    print(json.dumps(results))

if __name__ == '__main__':
    main()
'''
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'native_order.py'
            path.write_text(script)
            env = dict(os.environ, PYTHONPATH=str(ROOT))
            env.pop('CANVAS_WATERMARK_NATIVE_LIBRARY_DIR', None)
            process = subprocess.run([sys.executable, '-B', str(path)], env=env,
                                     capture_output=True, text=True, timeout=120)
        self.assertEqual(process.returncode, 0, process.stderr)
        results = json.loads(process.stdout)
        if results.get('skip'):
            self.skipTest('Optional native robust compiler unavailable')
        for mode, usage in results.items():
            self.assertEqual(usage['robustNativeFrames'], 1, (mode, usage))
            self.assertEqual(usage['robustNumpyFrames'], 0, (mode, usage))

    def test_active_failure_names_absolute_frame_region_and_time(self):
        m = model(fixture())
        with patch('fixed_r16.regions.load_models', return_value=[m]):
            repair = MultiFixedRepair('unused', dict(self.info, width=85, height=60), 0, 10,
                                      [dict(self.region, roi=dict(x=0,y=0,width=85,height=60))], lambda *a: None)
        with self.assertRaisesRegex(ValueError, r'region 1.*frame 2 \(0.200000s\)'):
            repair.masks(fixture(''), 2)
        self.assertEqual(repair.active, 0)

    def test_inactive_interval_preserves_original_and_reports_it(self):
        m = model(fixture())
        with patch('fixed_r16.regions.load_models', return_value=[m]):
            repair = MultiFixedRepair('unused', dict(self.info, width=85, height=60), 20, 2,
                                      [dict(self.region, roi=dict(x=0,y=0,width=85,height=60))], lambda *a: None)
        reader = ArrayReader([fixture('')]*22)
        actual = list(repair.frames(reader, 20, 2, 1))
        for frame in actual:
            np.testing.assert_array_equal(frame, fixture(''))
        self.assertEqual(repair.report()['inactiveRegionFrames'], 2)
        self.assertEqual(repair.report()['changedPixels'], 0)

    def test_calibration_reads_reference_window_not_segment_or_remote_context(self):
        image = make_frame()
        info = dict(self.info, width=image.shape[1], height=image.shape[0], frames=10000, duration=1000.)
        regions = validate_regions([self.region], info)
        reader = ArrayReader([image]*10)
        calibrate(reader, regions, info, lambda *a: None)
        self.assertTrue(all(0 <= i < 10 for i in reader.calls))
        self.assertIn(5, reader.calls)

    def test_cache_is_shared_and_invalidated_by_source_or_region_identity(self):
        image = make_frame()
        info = dict(self.info, width=image.shape[1], height=image.shape[0])
        regions = validate_regions([self.region], info)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source.dat'
            source.write_bytes(b'first')
            with patch('fixed_r16.regions.Reader', side_effect=lambda _: ArrayReader([image]*100)) as reader:
                first = load_models(source, info, regions, directory, lambda *a: None)
                second = load_models(source, info, regions, directory, lambda *a: None)
                self.assertEqual(reader.call_count, 1)
                np.testing.assert_array_equal(first[0]['support'], second[0]['support'])
                source.write_bytes(b'second source')
                load_models(source, info, regions, directory, lambda *a: None)
                self.assertEqual(reader.call_count, 2)
                changed = [dict(regions[0], reference=4)]
                load_models(source, info, changed, directory, lambda *a: None)
                self.assertEqual(reader.call_count, 3)

    def test_two_regions_write_only_union_and_preserve_input(self):
        from native_build import prepare
        prepare()
        a = fixture()
        image = np.full((84, 194, 3), 45, np.uint8)
        image[12:72, 10:95] = a
        image[12:72, 99:184] = a
        models = [dict(model(a), region=i+1, roi=[x,12,x+85,72]) for i,x in enumerate((10,99))]
        regions = [dict(self.region, roi=dict(x=x,y=12,width=85,height=60)) for x in (10,99)]
        info = dict(self.info, width=194, height=84)
        with patch('fixed_r16.regions.load_models', return_value=models):
            repair = MultiFixedRepair('unused', info, 0, 1, regions, lambda *a: None)
        before = image.copy()
        output = list(repair.frames(ArrayReader([image]), 0, 1, 1))[0]
        union = np.zeros(image.shape[:2], bool)
        for m in models:
            x,y,xx,yy = m['roi']
            union[y:yy,x:xx] |= m['support']
        np.testing.assert_array_equal(image, before)
        np.testing.assert_array_equal(output[~union], before[~union])
        for x in (10,99):
            self.assertTrue(np.any(output[12:72,x:x+85] != before[12:72,x:x+85]))

    def test_real_cli_preserves_frames_and_reports_two_activity_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory/'source.mkv'
            images = []
            for index in range(8):
                image = np.full((104,320,3),45,np.uint8)
                image[20:80,20:105] = fixture()
                if 2 <= index < 6:
                    image[20:80,200:285] = fixture()
                images.append(image)
            subprocess.run([executable('ffmpeg'),'-v','error','-f','rawvideo','-pixel_format','bgr24',
                            '-video_size','320x104','-framerate','10','-i','pipe:0','-c:v','ffv1',str(source)],
                           input=np.stack(images).tobytes(),check=True)
            regions = [dict(self.region, referenceSeconds=.3, durationSeconds=0),
                       dict(self.region, roi=dict(x=200,y=20,width=85,height=60),
                            referenceSeconds=.3,startSeconds=.2,durationSeconds=.4)]
            request = directory/'request.json'
            request.write_text(json.dumps(dict(input=str(source),branch='fixed-r16',workers=1,
                                               fixedRegions=regions,validationCacheDir=str(directory))))
            output = directory/'output.mp4'
            result = subprocess.run([sys.executable,'-B',str(ROOT/'runner.py'),'--request',str(request),
                                     '--output',str(output)],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report = json.loads(output.with_suffix('.report.json').read_text())
            self.assertEqual(probe(output)['frames'],8)
            self.assertEqual(report['activeRegionFrames'],12)
            self.assertEqual(report['inactiveRegionFrames'],4)
            self.assertLess(len(result.stdout.splitlines()[-1].encode()),65536)


if __name__ == '__main__':
    unittest.main()
