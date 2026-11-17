"""One local-file CLI for the fixed R16 and moving V6 algorithm branches."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True


def progress(value, stage, **details):
    print(json.dumps(dict(progress=round(float(value), 2), stage=stage, **details), allow_nan=False), flush=True)


def number(request, key, default=0):
    value = request.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{key} must be a finite number")
    return value


def frame_boundary(seconds, fps):
    # Nonnegative half-up rounding matches the backend's math.Round boundaries.
    return math.floor(seconds * fps + .5)


def execute(request, output):
    import cv2
    from media import Encoder, Reader, probe

    if not isinstance(request, dict):
        raise ValueError("Request must be a JSON object")
    source_value = request.get("input")
    if not isinstance(source_value, str) or not source_value:
        raise ValueError("input must be a local file path")
    source = Path(source_value).resolve(strict=True)
    if not source.is_file():
        raise ValueError("input must be a regular local file")
    output = Path(output).resolve()
    if output.exists() or output.with_suffix(".report.json").exists() or output == source:
        raise ValueError("Output must be a new path distinct from the input")
    branch = request.get("branch")
    if branch not in ("fixed-r16", "moving-v6"):
        raise ValueError("branch must be fixed-r16 or moving-v6")
    fixed_regions = request.get('fixedRegions')
    if fixed_regions is not None and branch != 'fixed-r16':
        raise ValueError('fixedRegions is only supported by fixed-r16')
    start = number(request, "startSeconds")
    duration = number(request, "durationSeconds")
    reference = number(request, "referenceSeconds", start)
    workers = number(request, "workers", 2)
    if start < 0 or duration < 0 or reference < 0 or workers != int(workers) or not 1 <= workers <= 16:
        raise ValueError("Invalid processing interval, reference time, or workers (1..16)")
    workers = int(workers)
    cv2.setNumThreads(1)
    progress(0, "probing")
    info = probe(source)
    if info["width"] % 2 or info["height"] % 2:
        raise ValueError("H.264 output requires even input dimensions")
    if start >= info["duration"] or reference >= info["duration"]:
        raise ValueError("Start/reference time must lie inside the source video")
    available = info["duration"] - start
    if duration > available + 1 / info["fps"]:
        raise ValueError("Requested duration extends past the source video")
    first = frame_boundary(start, info["fps"])
    end = min(info["frames"], frame_boundary(start + (duration or available), info["fps"]))
    count = end - first
    if count < 1:
        raise ValueError("Processing interval must contain at least one frame")
    roi_value = request.get("roi") if fixed_regions is None else None
    if roi_value is not None:
        if not isinstance(roi_value, dict):
            raise ValueError("roi must contain x, y, width, height")
        values = [number(roi_value, key, -1) for key in ("x", "y", "width", "height")]
        if any(value != int(value) for value in values):
            raise ValueError("ROI coordinates must be integer source pixels")
        x, y, width, height = map(int, values)
        if x < 0 or y < 0 or min(width, height) < 4 or x + width > info["width"] or y + height > info["height"]:
            raise ValueError("ROI must lie inside the original video")
        roi = [x, y, x + width, y + height]
    elif fixed_regions is None:
        if branch == "moving-v6":
            raise ValueError("moving-v6 requires a reference ROI")
        targets = {"top-left": (0, 0, .5, .25), "top-right": (.5, 0, 1, .25),
                   "bottom-left": (0, .75, .5, 1), "bottom-right": (.5, .75, 1, 1)}
        target = request.get("target", "top-left")
        if target not in targets:
            raise ValueError("Unknown fixed watermark target")
        roi = [round(value * (info["width"] if i % 2 == 0 else info["height"]))
               for i, value in enumerate(targets[target])]
    else:
        from fixed_r16.regions import validate_regions
        validate_regions(fixed_regions, info)
        roi = None
    cache_directory = request.get('validationCacheDir')
    if cache_directory is not None and (not isinstance(cache_directory, str) or not cache_directory):
        raise ValueError('validationCacheDir must be a nonempty local directory path')
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="watermark-", dir=output.parent) as temporary:
        if branch == "fixed-r16":
            from native_build import prepare
            progress(3, "preparing_native_kernels")
            native = prepare()
            if fixed_regions is None:
                from fixed_r16.engine import FixedRepair
                repairer = FixedRepair(source, info, first, count, roi, progress)
            else:
                from fixed_r16.regions import MultiFixedRepair
                repairer = MultiFixedRepair(source, info, first, count, fixed_regions, progress, cache_directory)
        else:
            from native_build import prepare
            progress(3, "preparing_native_kernels")
            native = prepare()
            from moving_v6.engine import MovingRepair
            reference_frame = min(info["frames"] - 1, frame_boundary(reference, info["fps"]))
            repairer = MovingRepair(source, info, first, count, reference_frame, roi, progress)
        reader = Reader(source, info, end=end)
        writer = None
        report = dict(branch=branch, sourceDimensions=[info["width"], info["height"]],
                      startSeconds=first / info["fps"], durationSeconds=count / info["fps"],
                      frames=count, fps=info["fps"], outputFps=info['fps'], roi=roi,
                      timelineMode=info['timelineMode'], sourceTimestampOrigin=info['sourceTimestampOrigin'])
        try:
            writer = Encoder(source, Path(temporary) / "result.mp4", info, first, count, workers)
            # Algorithm workers operate on independent frames; decode and encode
            # stay ordered and at most two frames per worker are in flight.
            for offset, result in enumerate(repairer.frames(reader, first, count, workers)):
                writer.write(result)
                if offset % max(1, round(info["fps"])) == 0 or offset == count - 1:
                    progress(40 + 55 * (offset + 1) / count, "repairing")
            progress(97, "encoding")
            writer.finish()
            writer = None
            report.update(repairer.report())
            report["native"] = native
            if not report.get("changedPixels", 0) and branch != 'moving-v6' and fixed_regions is None:
                raise ValueError("No verified watermark pixels were repaired; adjust the region/reference")
            report_path = output.with_suffix(".report.json")
            temporary_report = Path(temporary) / "report.json"
            temporary_report.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            os.replace(temporary_report, report_path)
            os.replace(Path(temporary) / "result.mp4", output)
            progress(100, "completed", report=report)
        finally:
            reader.close()
            repairer.close()
            if writer is not None:
                writer.finish(success=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        import importlib
        import shutil
        missing = []
        for name in ("numpy", "cv2"):
            try:
                importlib.import_module(name)
            except (ImportError, OSError):
                missing.append(name)
        for name in ("ffmpeg", "ffprobe"):
            value = os.environ.get(f"CANVAS_{name.upper()}_PATH") or shutil.which(name)
            if not value or not os.access(value, os.X_OK):
                missing.append(name)
        if sys.version_info < (3, 10):
            missing.append("Python >= 3.10")
        reason = "Missing runtime dependencies: " + ", ".join(missing) if missing else ""
        print(json.dumps({"branches": {name: {"available": not missing, "reason": reason}
                                      for name in ("fixed-r16", "moving-v6")}}))
        return 0
    if args.request is None or args.output is None:
        parser.error("--request and --output are required unless --check is used")
    try:
        execute(json.loads(args.request.read_text()), args.output)
    except Exception as error:
        # Deliberately omit local paths, command arguments, and tracebacks from
        # the process contract consumed by the web backend.
        origin = error.__traceback__
        while origin.tb_next is not None:
            origin = origin.tb_next
        owned_error = Path(origin.tb_frame.f_code.co_filename).resolve().is_relative_to(Path(__file__).resolve().parent)
        message = str(error) if owned_error and isinstance(error, (ValueError, RuntimeError)) else type(error).__name__
        if message == "No persistent bright neutral watermark candidates":
            message = ("未可靠识别位置固定的亮色水印。请检查所选角落；水印会移动时请选择移动 V6。"
                       "低对比度灰色水印也可能无法标定。")
        failed_region = re.fullmatch(r'Fixed region (\d+) identity unverified at frame (\d+) \(([\d.]+)s\): (\w+); adjust its reference or active interval', message)
        if failed_region:
            region, frame, seconds, reason = failed_region.groups()
            reasons = dict(missing_stroke='参考字形笔画不完整', extra_stroke='出现参考以外的笔画',
                           missing_reference_contour='参考轮廓证据不足')
            message = (f'固定区域 {region} 在原片第 {frame} 帧（{seconds} 秒）未通过身份验证：'
                       f'{reasons.get(reason, reason)}。请检查参考画面与活动时间，当前片段未发布。')
        bad_reference = re.match(r'Fixed region (\d+) reference (?:is not identifiable|identity failed)', message)
        if bad_reference:
            message = (f'固定区域 {bad_reference.group(1)} 的参考画面无法可靠标定。'
                       '请选择包含完整水印、四周背景较均匀的参考画面，并收紧区域范围。')
        message = " ".join(message.split())[:400]
        print(json.dumps(dict(error="watermark_processing_failed", message=message)), file=sys.stderr, flush=True)
        print("ERROR: " + message, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
