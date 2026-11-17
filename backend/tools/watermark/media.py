"""Bounded timestamp decoding on a shared CFR grid and audio-preserving encoding."""
from __future__ import annotations

from fractions import Fraction
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np


TIMELINE_MODE = 'timestamp-resampled-cfr-v1'


def executable(name):
    value = os.environ.get(f"CANVAS_{name.upper()}_PATH") or shutil.which(name)
    if not value:
        raise RuntimeError(f"Required executable is unavailable: {name}")
    return value


def probe(source):
    result = subprocess.run([
        executable("ffprobe"), "-v", "error", "-show_streams", "-show_format",
        "-of", "json", str(source),
    ], check=True, capture_output=True, text=True)
    data = json.loads(result.stdout)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError("Input has no video stream")
    fps = Fraction(video.get("avg_frame_rate", "0/1"))
    origin = float(video.get('start_time') or 0)
    duration = video.get('duration')
    if duration is None and video.get('duration_ts') is not None and video.get('time_base'):
        duration = int(video['duration_ts']) * Fraction(video['time_base'])
    if duration is None:
        container = data.get('format', {})
        if set(container.get('format_name', '').split(',')) & {'matroska', 'webm'}:
            # Matroska's DURATION tag and format duration are end timestamps,
            # including a nonzero stream origin, rather than elapsed lengths.
            tag = video.get('tags', {}).get('DURATION')
            if tag:
                hours, minutes, seconds = map(float, tag.split(':'))
                duration = hours * 3600 + minutes * 60 + seconds - origin
            else:
                duration = float(container.get('duration', 0)) - origin
        elif origin == 0:
            duration = container.get('duration', 0)
        else:
            raise ValueError('Video duration metadata is ambiguous for a nonzero timestamp origin')
    duration = float(duration)
    width, height = int(video["width"]), int(video["height"])
    if (fps <= 0 or not math.isfinite(origin) or not math.isfinite(duration)
            or duration <= 0 or min(width, height) < 16):
        raise ValueError("Invalid input dimensions, frame rate, or duration")
    rotation = next((float(s.get("rotation", 0)) for s in video.get("side_data_list", [])
                     if "rotation" in s), float(video.get("tags", {}).get("rotate", 0)))
    if rotation % 360:
        raise ValueError("Rotated display metadata is unsupported; normalize orientation before processing")
    return dict(width=width, height=height, fps=float(fps), rate=str(fps), duration=duration,
                frames=math.floor(duration * float(fps) + .5), sourceTimestampOrigin=origin,
                timelineMode=TIMELINE_MODE)


class Reader:
    """Read output-grid index i at time i/fps, never source frame ordinal i.

    A random seek retains the preceding keyframe and the fps filter sees global
    timestamps. Thus references, context, and separately decoded segments all
    select the same source observation. Sequential reads reuse a bounded pipe.
    """
    def __init__(self, source, info=None, end=None):
        self.source = str(source)
        self.info = info if info is not None else probe(source)
        self.end = self.info['frames'] if end is None else min(end, self.info['frames'])
        self.process = self.log = None
        self.position = self.block_end = -1

    def _start(self, index):
        self.close()
        info = self.info
        count = min(256, self.end - index)
        origin = info.get('sourceTimestampOrigin', 0.)
        # Accurate seek would discard the previous observation needed by the
        # fps filter. A one-grid-step guard also covers all-intra VFR inputs.
        seek = origin + max(0., (index - 1) / info['fps'])
        filters = (f"setpts=PTS-({origin})/TB,"
                   f"fps=fps={info['rate']}:start_time={index / info['fps']}:round=near")
        command = [executable('ffmpeg'), '-nostdin', '-v', 'error', '-xerror',
                   '-err_detect', 'explode', '-threads', '1', '-copyts', '-seek_timestamp', '1',
                   '-ss', str(seek), '-noaccurate_seek', '-noautorotate', '-i', self.source,
                   '-map', '0:v:0', '-an', '-sn', '-dn', '-filter_threads', '1', '-vf', filters,
                   '-frames:v', str(count), '-fps_mode', 'passthrough', '-threads', '1',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1']
        self.log = tempfile.TemporaryFile()
        self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=self.log)
        self.position, self.block_end = index, index + count

    def _verify_finished(self):
        code = self.process.wait()
        self.log.seek(0)
        if code != 0 or self.log.read(1):
            raise RuntimeError('FFmpeg could not decode the requested video interval')

    def frame(self, index):
        if not isinstance(index, (int, np.integer)) or not 0 <= index < self.end:
            raise ValueError('Requested frame lies outside the normalized video timeline')
        if index != self.position or index >= self.block_end:
            self._start(index)
        size = self.info['width'] * self.info['height'] * 3
        raw = self.process.stdout.read(size)
        if len(raw) != size:
            self._verify_finished()
            raise RuntimeError(f'Decoder stopped before normalized frame {index}')
        self.position = index + 1
        if self.position == self.block_end:
            self._verify_finished()
        elif self.process.poll() not in (None, 0):
            self._verify_finished()
        return np.frombuffer(raw, np.uint8).reshape(self.info['height'], self.info['width'], 3).copy()

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                # An abandoned raw-video pipe may be blocked in write(); no
                # trailer is needed, so do not wait for FFmpeg to flush it.
                self.process.kill()
            self.process.wait()
            self.process.stdout.close()
            self.log.close()
            self.process = self.log = None
        self.position = self.block_end = -1


class Encoder:
    def __init__(self, source, output, info, first, count, workers):
        self.path = Path(output)
        self.log = self.path.with_suffix(".ffmpeg.log").open("wb")
        command = [executable("ffmpeg"), "-nostdin", "-v", "error", "-n",
                   "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size",
                   f'{info["width"]}x{info["height"]}', "-framerate", info["rate"], "-i", "pipe:0",
                   "-seek_timestamp", "1", "-ss", str(info.get('sourceTimestampOrigin', 0.) + first / info["fps"]),
                   "-i", str(source),
                   "-map", "0:v:0", "-map", "1:a?", "-map_metadata", "1",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                   "-threads", str(workers), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                   "-t", str(count / info["fps"]), "-movflags", "+faststart", str(self.path)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=self.log)

    def write(self, frame):
        self.process.stdin.write(memoryview(np.ascontiguousarray(frame)).cast("B"))

    def finish(self, success=True):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        if not success and self.process.poll() is None:
            self.process.terminate()
        code = self.process.wait()
        self.log.close()
        if success and code != 0:
            raise RuntimeError("FFmpeg could not encode the processed video and audio")
