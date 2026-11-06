"""Boundary for per-lane interval dynamic programming, independent of video I/O."""

from dataclasses import dataclass
from typing import Iterable, Protocol

from .contracts import ChunkBoundary, Event
from .fusion import Evidence


@dataclass(frozen=True)
class IntervalCandidate:
    id: str
    lane: int
    type: str
    start_px: float | None
    end_px: float | None
    cost: float
    endpoint_observation_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


class IntervalDecoder(Protocol):
    def decode(self, evidence: Iterable[Evidence], boundary: ChunkBoundary | None = None
               ) -> tuple[tuple[Event, ...], ChunkBoundary]: ...
