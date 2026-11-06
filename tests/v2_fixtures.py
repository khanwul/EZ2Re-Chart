"""Procedural lossless VFR frames; no external media needed."""

from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np


def fixture(root: Path, *, translated=(0, 0), weak=False):
    template = np.random.default_rng(42).integers(80, 240, (7, 16, 3), dtype=np.uint8)
    cv2.imwrite(str(root / "note.png"), template)
    profile = '''schema_version = 1
id = "synthetic"
game = "synthetic"
mode = "2k"
resolution = [96, 96]
judge_y = 72.0
effect_top = 65.0
beat_roi = [5, 80, 10, 85]
[alignment]
band = [20, 70, 61, 76]
max_shift_px = 8
'''
    for kind in ("tap", "head", "tail"):
        profile += f'''[calibration.{kind}]
detection_reference = "template_top"
status = "unverified"
version = "test"
'''
    for left in (20, 40):
        profile += f'''[[lanes]]
roi = [{left}, 0, {left + 20}, 70]
match_roi = [{left - 2}, 0, {left + 22}, 70]
channel = "gray"
[lanes.templates]
tap = "note.png"
head = "note.png"
tail = "note.png"
'''
    (root / "profile.toml").write_text(profile)
    (root / "algo.toml").write_text("schema_version = 1\nchunk_frames = 2\n")
    song = root / "song.toml"
    song.write_text('''schema_version = 1
[input]
video = "video.mkv"
profile = "profile.toml"
[metadata]
title = "Synthetic"
difficulty = "SHD"
[timing]
mode = "fixed"
bpm = "191.910"
[algorithm]
config = "algo.toml"
''')
    pts = [1000, 1017, 1033, 1083, 1100, 1133]
    dx, dy = translated
    with av.open(str(root / "video.mkv"), "w") as container:
        stream = container.add_stream("ffv1", rate=60)
        stream.width = stream.height = 96
        stream.pix_fmt = "bgr0"
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for i, timestamp in enumerate(pts):
            image = np.zeros((96, 96, 3), np.uint8)
            image[70 + dy:76 + dy, 20 + dx:61 + dx] = (220, 120, 20)
            y = [10, 18, 26, 26, 42, 55][i] + dy
            patch = (template * 0.22).astype(np.uint8) if weak else template
            image[y:y + 7, 22 + dx:38 + dx] = patch
            image[80 + dy:85 + dy, 5 + dx:10 + dx, 1] = 100 if i % 2 else 10
            frame = av.VideoFrame.from_ndarray(image, format="bgr24")
            frame.pts, frame.time_base = timestamp, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return song, pts
