"""Boundary for weighted visual evidence in common scroll coordinates."""

from dataclasses import dataclass
from typing import Iterable, Protocol

from .contracts import ChunkBoundary, StageRef


@dataclass(frozen=True)
class Evidence:
    id: str
    lane: int
    coordinate_px: float
    position_error_px: float
    scores: tuple[float, float, float, float]
    visible_weight: float
    occluded_weight: float
    observation_ids: tuple[str, ...]


class EvidenceAccumulator(Protocol):
    def accumulate(self, observations: StageRef, motion: StageRef,
                   boundary: ChunkBoundary | None = None) -> Iterable[Evidence]: ...
