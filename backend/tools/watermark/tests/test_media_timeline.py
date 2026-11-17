"""Real FFmpeg regressions for bounded, timestamp-based video decoding."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import media
from runner import frame_boundary


def run(arguments, **kwargs):
    return subprocess.run([media.executable('ffmpeg'), '-nostdin', '-v', 'error', '-y',
                           *arguments], check=True, capture_output=True, **kwargs)


class TimestampTimeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix='watermark-time-')
        cls.directory = Path(cls.temporary.name)
        yy, xx = np.indices((32, 48))
        cls.frames = np.stack([np.stack(((xx+i) % 256, (yy+i*7) % 256,
                                          (xx+yy+i*19) % 256), axis=2).astype(np.uint8)
                               for i in range(660)])
        for name, filters in [('cfr', []), ('vfr', ['-vf', "select='not(mod(n,2))+not(mod(n,3))'"] )]:
            run(['-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '48x32', '-r', '30000/1001',
                 '-i', 'pipe:0', *filters, '-fps_mode', 'vfr', '-c:v', 'libx264rgb', '-crf', '0',
                 '-g', '45', '-movflags', '+faststart', str(cls.directory/f'{name}.mp4')],
                input=cls.frames.tobytes())
        for extension in ('mp4', 'mkv'):
            run(['-i', str(cls.directory/'vfr.mp4'), '-c', 'copy', '-output_ts_offset', '5',
                 str(cls.directory/f'offset.{extension}')])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def oracle(self, source, info):
        raw = run(['-copyts', '-i', str(source), '-map', '0:v:0', '-vf',
                   f"setpts=PTS-({info['sourceTimestampOrigin']})/TB,fps={info['rate']}:start_time=0:round=near",
                   '-frames:v', str(info['frames']), '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']).stdout
        return np.frombuffer(raw, np.uint8).reshape(-1, 32, 48, 3)

    def test_real_vfr_random_references_and_10_second_parts_equal_whole_timeline(self):
        for filename in ('cfr.mp4', 'vfr.mp4', 'offset.mp4', 'offset.mkv'):
            with self.subTest(source=filename):
                source = self.directory/filename
                info = media.probe(source)
                whole = self.oracle(source, info)
                self.assertEqual(len(whole), info['frames'])
                reader = media.Reader(source)
                try:
                    for index in [0, 1, 33, 255, 256, 257, 300, info['frames']-1, 2]:
                        np.testing.assert_array_equal(reader.frame(index), whole[index])
                finally:
                    reader.close()
                boundaries = [0, frame_boundary(10., info['fps']),
                              frame_boundary(20., info['fps']), info['frames']]
                parts = []
                for first, end in zip(boundaries, boundaries[1:]):
                    reader = media.Reader(source, info, end=end)
                    try:
                        parts.extend(reader.frame(i) for i in range(first, end))
                    finally:
                        reader.close()
                np.testing.assert_array_equal(np.stack(parts), whole)
                if filename == 'cfr.mp4':
                    np.testing.assert_array_equal(whole, self.frames)
                if filename == 'offset.mkv':
                    self.assertAlmostEqual(info['duration'], media.probe(self.directory/'vfr.mp4')['duration'], places=2)
                    self.assertEqual(info['sourceTimestampOrigin'], 5.)

    def test_vfr_reference_is_time_not_source_frame_number(self):
        source = self.directory/'vfr.mp4'
        info = media.probe(source)
        timestamps = json.loads(subprocess.check_output([
            media.executable('ffprobe'), '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', str(source)]))['frames']
        deltas = np.diff([float(frame['best_effort_timestamp_time']) for frame in timestamps])
        self.assertGreater(len(set(np.round(deltas, 5))), 1)
        raw = run(['-i', str(source), '-fps_mode', 'passthrough', '-f', 'rawvideo',
                   '-pix_fmt', 'bgr24', 'pipe:1']).stdout
        ordinal = np.frombuffer(raw, np.uint8).reshape(-1, 32, 48, 3)
        normalized = self.oracle(source, info)
        different = np.flatnonzero(np.any(ordinal != normalized, axis=(1, 2, 3)))
        self.assertGreater(len(different), 10)
        index = int(different[len(different)//2])
        reader = media.Reader(source)
        try:
            np.testing.assert_array_equal(reader.frame(index), normalized[index])
            self.assertFalse(np.array_equal(reader.frame(index), ordinal[index]))
        finally:
            reader.close()

    def test_probe_does_not_scan_frames_and_does_not_trust_source_nb_frames(self):
        payload = dict(streams=[dict(codec_type='video', width=48, height=32, avg_frame_rate='30',
                                    duration='20.1', nb_frames='300', start_time='2.5')], format={})
        with patch.object(media.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, json.dumps(payload))) as call:
            info = media.probe('metadata-only.mp4')
        self.assertEqual(info['frames'], 603)
        self.assertNotIn('-show_frames', call.call_args.args[0])
        self.assertNotIn('-show_entries', call.call_args.args[0])
        payload['streams'][0].pop('duration')
        payload['format'] = dict(duration='22.6', format_name='unknown')
        with patch.object(media.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, json.dumps(payload))):
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                media.probe('ambiguous-duration.mp4')

    def test_truncated_decode_is_failure_instead_of_successful_short_output(self):
        source = self.directory/'cfr.mp4'
        broken = self.directory/'truncated.mp4'
        data = source.read_bytes()
        broken.write_bytes(data[:len(data)*2//3])
        info = media.probe(broken)
        reader = media.Reader(broken, info)
        try:
            with self.assertRaisesRegex(RuntimeError, 'decode|Decoder stopped'):
                for index in range(info['frames']):
                    reader.frame(index)
        finally:
            reader.close()


if __name__ == '__main__':
    unittest.main()
