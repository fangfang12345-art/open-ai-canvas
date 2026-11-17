"""Build optional R16 kernels into a platform/source-specific external cache."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile

CORE = Path(__file__).resolve().parent / "fixed_r16" / "v6_core"


def prepare():
    if sys.platform not in ("darwin", "linux") or os.environ.get("CANVAS_WATERMARK_DISABLE_NATIVE") == "1":
        return {"enabled": False, "reason": "Native kernels disabled or unsupported; using NumPy"}
    compiler = shutil.which(os.environ.get("CXX", "c++"))
    if not compiler:
        return {"enabled": False, "reason": "C++17 compiler unavailable; using original NumPy implementations"}
    sources = sorted(CORE.glob("native_*.cpp"))
    signature = hashlib.sha256((sys.platform + platform.machine() + compiler).encode())
    for source in sources:
        signature.update(source.read_bytes())
    root = Path(os.environ.get("CANVAS_WATERMARK_NATIVE_CACHE", Path(tempfile.gettempdir()) / "yingce-watermark-native"))
    directory = root / signature.hexdigest()[:20]
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["CANVAS_WATERMARK_NATIVE_LIBRARY_DIR"] = str(directory)
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    built, failed = [], []
    for source in sources:
        if source.stem == "native_background" and sys.platform != "darwin":
            continue  # Optional BLAS optimization requires Apple's Accelerate.
        output = directory / (source.stem + suffix)
        if not output.exists():
            temporary = directory / (source.stem + f".{os.getpid()}.tmp" + suffix)
            flags = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
            command = [compiler, "-std=c++17", "-O3", "-ffp-contract=off", "-fno-fast-math",
                       *flags, str(source), "-o", str(temporary)]
            if source.stem == "native_background":
                command.extend(["-framework", "Accelerate", "-Wno-deprecated-declarations"])
            try:
                subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                os.replace(temporary, output)
            except (OSError, subprocess.SubprocessError):
                temporary.unlink(missing_ok=True)
                failed.append(source.stem)
                continue
        built.append(source.stem)
    return {"enabled": bool(built), "kernels": built, "unavailableKernels": failed,
            "backgroundImplementation": "Accelerate" if "native_background" in built else "NumPy"}


if __name__ == "__main__":
    import json
    print(json.dumps(prepare()))
