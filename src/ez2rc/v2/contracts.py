"""Decoder-independent values shared across the seven pipeline boundaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import math
from typing import Literal

SCHEMA_VERSION = 1
STAGES = ("capture", "observation", "motion", "fusion", "decode", "timing", "export")


def finite_number(value) -> bool:
    return type(value) in {int, float} and math.isfinite(value)


def decimal_bpm(value: str) -> Fraction:
    if not isinstance(value, str):
        raise ValueError("bpm must be a positive decimal string")
    try:
        number = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("invalid decimal bpm") from exc
    if not number.is_finite() or number <= 0:
        raise ValueError("bpm must be positive and finite")
    return Fraction(number)


def ratio(value: Fraction) -> list[int]:
    return [value.numerator, value.denominator]


def read_ratio(value: list[int]) -> Fraction:
    if (not isinstance(value, list) or len(value) != 2
            or any(type(v) is not int for v in value) or value[1] <= 0
            or math.gcd(*value) != 1):
        raise ValueError("beat must be a reduced integer pair with positive denominator")
    return Fraction(*value)


def exact_ms(value) -> Fraction:
    """Interpret numeric milliseconds as decimal, retaining rational inputs."""
    if isinstance(value, Fraction):
        return value
    if not finite_number(value):
        raise ValueError('invalid exact millisecond value')
    return Fraction(str(value))


def origin_schema(value) -> int:
    return 2 if isinstance(value, Fraction) else 1


def origin_payload(value, key='beat_zero_ms') -> dict:
    """Schema 2 separates an exact pair from an unresolved numeric field."""
    if value is not None:
        exact_ms(value)
    if isinstance(value, Fraction):
        return {key: None, key + '_exact': ratio(value)}
    return {key: value}


def read_origin(value, schema_version, key='beat_zero_ms'):
    """Reject ambiguous dual representations and schema downgrades."""
    if type(schema_version) is not int or schema_version not in (1, 2):
        raise ValueError('unsupported origin schema')
    numeric = value[key]
    if key + '_exact' in value:
        if schema_version != 2 or numeric is not None:
            raise ValueError('exact origin requires schema 2 and a null numeric origin')
        return read_ratio(value[key + '_exact'])
    if numeric is not None:
        if type(numeric) not in (int, float):
            raise ValueError('numeric origin must be a number or null')
        exact_ms(numeric)
    return numeric


@dataclass(frozen=True)
class TimeInterval:
    lower_ms: float
    upper_ms: float

    def __post_init__(self):
        if (not finite_number(self.lower_ms) or not finite_number(self.upper_ms)
                or self.upper_ms < self.lower_ms):
            raise ValueError("invalid time uncertainty interval")


@dataclass(frozen=True)
class Note:
    id: str
    lane: int
    type: Literal["tap", "longnote"]
    start_beat: Fraction
    end_beat: Fraction | None = None

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id or type(self.lane) is not int or self.lane < 0:
            raise ValueError("invalid note identity or lane")
        if not isinstance(self.start_beat, Fraction):
            raise ValueError("start_beat must be a Fraction")
        if self.type == "tap":
            if self.end_beat is not None:
                raise ValueError("tap end must be null")
        elif self.type == "longnote":
            if not isinstance(self.end_beat, Fraction) or self.end_beat <= self.start_beat:
                raise ValueError("longnote end must follow its start")
        else:
            raise ValueError("unknown note type")


@dataclass(frozen=True)
class Event:
    id: str
    lane: int
    status: Literal["resolved", "ambiguous", "incomplete"]
    type: Literal["tap", "longnote"] | None = None
    start_ms: float | None = None
    end_ms: float | None = None
    start_interval: TimeInterval | None = None
    end_interval: TimeInterval | None = None
    observation_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    selected_cost: float | None = None
    alternative_cost: float | None = None
    reason: str | None = None
    start_beat: Fraction | None = None
    end_beat: Fraction | None = None
    beat_coordinate: str | None = None

    def __post_init__(self):
        if not isinstance(self.id, str) or not self.id or type(self.lane) is not int or self.lane < 0:
            raise ValueError("invalid event identity or lane")
        if self.status not in {"resolved", "ambiguous", "incomplete"}:
            raise ValueError("invalid event status")
        if self.type not in {None, "tap", "longnote"}:
            raise ValueError("invalid event type")
        for value in (self.start_ms, self.end_ms, self.selected_cost, self.alternative_cost):
            if value is not None and not finite_number(value):
                raise ValueError("non-finite event value")
        musical = self.beat_coordinate == 'first_barline_zero'
        if self.beat_coordinate not in (None, 'first_barline_zero'):
            raise ValueError('unknown event beat coordinate')
        if musical:
            if self.status != 'resolved' or self.type is None:
                raise ValueError('exact musical event requires a resolved physical type')
            Note(self.id,self.lane,self.type,self.start_beat,self.end_beat)
            if any(v is not None for v in (self.start_ms,self.end_ms,self.start_interval,self.end_interval)):
                raise ValueError('relative musical event must keep capture time separate')
        elif self.start_beat is not None or self.end_beat is not None:
            raise ValueError('event beats require a coordinate convention')
        if self.status == "resolved" and (self.type is None or self.start_ms is None and not musical):
            raise ValueError("resolved event requires type and start time")
        if self.type == "tap" and (self.end_ms is not None or self.end_interval is not None):
            raise ValueError("tap cannot have an end")
        if self.status == "resolved" and self.type == "longnote" and self.end_ms is None and not musical:
            raise ValueError("resolved longnote requires an end")
        if self.start_ms is not None and self.end_ms is not None and self.end_ms <= self.start_ms:
            raise ValueError("event end must follow its start")
        for value, interval in ((self.start_ms, self.start_interval), (self.end_ms, self.end_interval)):
            if interval is not None and (value is None or not interval.lower_ms <= value <= interval.upper_ms):
                raise ValueError("endpoint must lie inside its uncertainty interval")
        if self.status != "resolved" and not self.reason:
            raise ValueError("unresolved event requires a reason")
        if any(not isinstance(ref, str) or not ref for ref in (*self.observation_ids, *self.candidate_ids)):
            raise ValueError("evidence and candidate references must be nonempty IDs")


@dataclass(frozen=True)
class Metadata:
    game: str
    mode: str
    lane_count: int
    title: str
    difficulty: str

    def __post_init__(self):
        if (type(self.lane_count) is not int or self.lane_count < 1
                or not all(isinstance(v, str) and v.strip()
                           for v in (self.game, self.mode, self.title, self.difficulty))):
            raise ValueError("invalid chart metadata")


@dataclass(frozen=True)
class Timing:
    bpm: str
    beat_zero_ms: float | Fraction | None = None
    measure_beats: Fraction | None = None
    status: Literal["resolved", "unresolved"] = "unresolved"
    reason: str | None = "beat origin and meter have not been determined"
    coordinate: str = 'recording_timestamp_ms'

    def __post_init__(self):
        decimal_bpm(self.bpm)
        if self.coordinate not in ('recording_timestamp_ms','first_barline_zero'):
            raise ValueError('unknown musical timing coordinate')
        if self.coordinate == 'first_barline_zero' and self.beat_zero_ms is not None:
            raise ValueError('relative musical origin cannot supply an absolute recording timestamp')
        if self.beat_zero_ms is not None:
            exact_ms(self.beat_zero_ms)
        if self.measure_beats is not None and (
                not isinstance(self.measure_beats, Fraction) or self.measure_beats <= 0):
            raise ValueError("invalid measure length")
        if self.status == "resolved":
            if (self.beat_zero_ms is None and self.coordinate == 'recording_timestamp_ms'
                    or self.measure_beats is None or self.reason is not None):
                raise ValueError("resolved timing requires origin, meter and no unresolved reason")
        elif self.status != "unresolved" or not self.reason:
            raise ValueError("unresolved timing requires a reason")


@dataclass(frozen=True)
class Chart:
    metadata: Metadata
    timing: Timing
    notes: tuple[Note, ...] = ()
    unresolved: tuple[Event, ...] = ()
    pending_stages: tuple[str, ...] = ()

    def __post_init__(self):
        ids = [n.id for n in self.notes] + [e.id for e in self.unresolved]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate or both resolved and unresolved note ID")
        if any(n.lane >= self.metadata.lane_count for n in (*self.notes, *self.unresolved)):
            raise ValueError("lane outside chart mode")
        if any(e.status == "resolved" for e in self.unresolved):
            raise ValueError("resolved event in unresolved list")
        if any(s not in STAGES for s in self.pending_stages):
            raise ValueError("unknown pending stage")

    @property
    def status(self) -> str:
        return ("complete" if self.timing.status == "resolved" and not self.unresolved
                and not self.pending_stages else "partial")


@dataclass(frozen=True)
class StageRef:
    stage: str
    cache_id: str
    algorithm_version: str


@dataclass
class ChunkBoundary:
    """Continuation state; a chunk edge is never a physical note endpoint."""
    scroll_px: float
    scroll_error_px: float
    open_candidate_ids: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    state: dict = field(default_factory=dict)
