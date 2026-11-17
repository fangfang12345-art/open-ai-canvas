"""Load native kernels from the runtime cache, never from source-tree binaries."""
import os
from pathlib import Path
import sys


def library_path(name):
    directory = os.environ.get("CANVAS_WATERMARK_NATIVE_LIBRARY_DIR")
    if not directory or os.environ.get("CANVAS_WATERMARK_DISABLE_NATIVE") == "1":
        return Path("/nonexistent/yingce-watermark-native") / name
    return Path(directory) / (name + (".dylib" if sys.platform == "darwin" else ".so"))
