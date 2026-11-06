"""Streaming decoder preserving original PTS, time base and full-video origin."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from pathlib import Path

import av
import numpy as np

VERSION = "pts-1"


class CaptureError(ValueError):
    pass


@dataclass(frozen=True)
class Stamp:
    frame_id: int
    pts: int | None
    time_base: Fraction
    timestamp_ms: float
    source: str
    duration_ms: float | None


@dataclass
class Frame:
    stamp: Stamp
    bgr: np.ndarray


class TimestampClock:
    """Missing timestamps are explicit; present but decreasing PTS are errors."""
    def __init__(self, rate: Fraction | None, policy="strict"):
        if policy not in {"strict", "fps_fallback"}:
            raise CaptureError("unknown timestamp policy")
        self.rate, self.policy = rate, policy
        self.origin: Fraction | None = None
        self.first_pts: int | None = None
        self.first_time_base: Fraction | None = None
        self.previous: Fraction | None = None
        self.missing_origin = False
        self.fallback_count = 0

    def stamp(self, index, pts, time_base, duration=None) -> Stamp:
        base = Fraction(time_base) if time_base is not None else Fraction(0)
        present = pts is not None and base > 0
        if index == 0:
            self.first_pts, self.first_time_base = pts, base
            self.origin = pts * base if present else None
            self.missing_origin = not present
        if not present or self.missing_origin:
            if self.policy != "fps_fallback" or self.rate is None or self.rate <= 0:
                raise CaptureError(f"missing PTS/time base at frame {index}; explicit fps_fallback required")
            value = (Fraction(index, 1) / self.rate if self.missing_origin
                     else (self.previous or Fraction(0)) + 1 / self.rate)
            source = "fps_fallback"
            self.fallback_count += 1
        else:
            value = pts * base - self.origin
            source = "container_pts"
        if self.previous is not None and value < self.previous:
            raise CaptureError(f"nonmonotonic PTS at frame {index}")
        self.previous = value
        return Stamp(index, pts, base, float(value * 1000), source,
                     float(duration * base * 1000) if duration and base > 0 else None)


class VideoReader:
    def __init__(self, path: Path, *, start_ms=0.0, end_ms=None, timestamp_policy="strict"):
        if (not math.isfinite(start_ms) or start_ms < 0 or end_ms is not None
                and (not math.isfinite(end_ms) or end_ms <= start_ms)):
            raise CaptureError("invalid analysis interval")
        self.path = Path(path)
        self.start_ms, self.end_ms = start_ms, end_ms
        self.timestamp_policy = timestamp_policy
        self.metadata = {}

    def __iter__(self):
        if not self.path.is_file():
            raise CaptureError(f"video not found: {self.path}")
        count, selected = 0, 0
        try:
            with av.open(str(self.path)) as container:
                if not container.streams.video:
                    raise CaptureError("input contains no video stream")
                stream = container.streams.video[0]
                stream.codec_context.thread_count = 1
                clock = TimestampClock(stream.average_rate, self.timestamp_policy)
                self.metadata = {
                    "decoder": "PyAV", "decoder_version": av.__version__,
                    "libraries": {k: list(v) for k, v in av.library_versions.items()},
                    "stream_index": stream.index, "width": stream.width, "height": stream.height,
                    "average_rate": str(stream.average_rate), "codec": stream.codec_context.name,
                    "timestamp_policy": self.timestamp_policy,
                }
                for index, frame in enumerate(container.decode(stream)):
                    if frame.is_corrupt:
                        raise CaptureError(f"corrupt decoded frame {index}")
                    stamp = clock.stamp(index, frame.pts, frame.time_base, frame.duration)
                    count += 1
                    if index == 0:
                        base = clock.first_time_base
                        self.metadata.update(first_pts=clock.first_pts,
                            first_time_base=[base.numerator, base.denominator],
                            origin_status="fps_fallback" if clock.missing_origin else "container_pts")
                    if self.end_ms is not None and stamp.timestamp_ms >= self.end_ms:
                        break
                    if stamp.timestamp_ms < self.start_ms:
                        continue
                    selected += 1
                    yield Frame(stamp, frame.to_ndarray(format="bgr24"))
                self.metadata.update(decoded_frames=count, selected_frames=selected,
                                     fallback_frames=clock.fallback_count)
        except av.FFmpegError as exc:
            raise CaptureError(f"video decode failed: {exc}") from exc
        if not selected:
            raise CaptureError("no frames in the requested analysis interval")


def decoder_parameters() -> dict:
    return {"decoder": "PyAV", "version": av.__version__,
            "libraries": {k: list(v) for k, v in av.library_versions.items()},
            "stream": "first_video", "pixel_format": "bgr24", "thread_count": 1}


def validate_timeline(chunk: dict, metadata: dict):
    """Validate raw PTS against derived milliseconds, including fallback flags."""
    ids = chunk["frame_id"]
    n = len(ids)
    for key in ("pts", "pts_valid", "timestamp_ms", "duration_ms", "fps_fallback"):
        if chunk[key].shape != (n,):
            raise ValueError(f"invalid timeline shape: {key}")
    if (ids.dtype.kind not in "iu" or chunk["pts"].dtype.kind not in "iu"
            or chunk["time_base"].dtype.kind not in "iu"
            or chunk["pts_valid"].dtype.kind != "b" or chunk["fps_fallback"].dtype.kind != "b"
            or chunk["time_base"].shape != (n, 2)
            or np.any(chunk["time_base"][:, 0] < 0) or np.any(chunk["time_base"][:, 1] <= 0)
            or not np.isfinite(chunk["timestamp_ms"]).all()
            or not np.isfinite(chunk["duration_ms"]).all()
            or np.any((chunk["duration_ms"] < 0) & (chunk["duration_ms"] != -1))):
        raise ValueError("invalid capture timeline values")
    original = metadata["first_pts"]
    origin = original * Fraction(*metadata["first_time_base"]) if original is not None else None
    for i in range(n):
        if chunk["fps_fallback"][i]:
            if metadata["timestamp_policy"] != "fps_fallback":
                raise ValueError("undeclared FPS fallback")
        else:
            if origin is None or not chunk["pts_valid"][i] or chunk["time_base"][i, 0] <= 0:
                raise ValueError("PTS source is missing its origin or time base")
            base = Fraction(*map(int, chunk["time_base"][i]))
            expected = float((int(chunk["pts"][i]) * base - origin) * 1000)
            if abs(expected - chunk["timestamp_ms"][i]) > 1e-6:
                raise ValueError("derived milliseconds disagree with original PTS")
