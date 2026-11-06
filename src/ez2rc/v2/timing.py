"""Exact fixed-BPM arithmetic. Origin estimation and grid inference are separate."""

from fractions import Fraction
from typing import Protocol

from .contracts import Chart, Event, Metadata, Timing, decimal_bpm


def beat_to_ms(beat: Fraction, timing: Timing) -> Fraction:
    if timing.beat_zero_ms is None:
        raise ValueError("beat origin is unresolved")
    return Fraction(str(timing.beat_zero_ms)) + beat * 60000 / decimal_bpm(timing.bpm)


def ms_to_beat(timestamp_ms: Fraction, timing: Timing) -> Fraction:
    if timing.beat_zero_ms is None:
        raise ValueError("beat origin is unresolved")
    return (timestamp_ms - Fraction(str(timing.beat_zero_ms))) * decimal_bpm(timing.bpm) / 60000


class TimingResolver(Protocol):
    def resolve(self, events: tuple[Event, ...], metadata: Metadata, bpm: str) -> Chart: ...
