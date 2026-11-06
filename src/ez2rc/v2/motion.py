"""Boundary for common scroll estimation; implementation follows observation validation."""

from dataclasses import dataclass
from typing import Iterable, Protocol

from .contracts import ChunkBoundary, StageRef


@dataclass(frozen=True)
class ScrollPoint:
    frame_id: int
    timestamp_ms: float
    scroll_px: float
    error_px: float
    repeated_group: int | None
    observation_ids: tuple[str, ...]


class MotionEstimator(Protocol):
    def estimate(self, observations: StageRef, boundary: ChunkBoundary | None = None
                 ) -> Iterable[ScrollPoint]: ...
