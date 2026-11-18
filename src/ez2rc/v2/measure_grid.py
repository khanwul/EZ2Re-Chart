"""Observed measure boundaries and column positions for video drafts."""

from bisect import bisect_left, bisect_right
from fractions import Fraction

from .contracts import read_ratio


class MeasureGrid:
    def __init__(self, projection):
        self.starts = tuple(read_ratio(b) for b in projection.get('barline_beats', ()))
        if self.starts and (len(self.starts) < 2 or self.starts[0] != 0 or
                            any(b <= a for a, b in zip(self.starts, self.starts[1:]))):
            raise ValueError('invalid observed measure boundaries')
        measure = projection.get('measure_beats')
        self.length = read_ratio(measure) if measure is not None else None
        if self.length is not None and self.length <= 0:
            raise ValueError('invalid measure length')
        gaps = projection.get('uncertain_measure_spans', ())
        self.first_gap = min((read_ratio(a) for a, _ in gaps), default=None)

    @property
    def numbered(self):
        return bool(self.starts) or self.length is not None

    def boundary(self, index):
        # Extend the layout for notes outside the observed range. Only visible
        # boundaries receive measure lines and numbers in a newly measured draft.
        if not self.starts:
            return index * (self.length or Fraction(4))
        if index < 0:
            return index * self.starts[1]
        if index >= len(self.starts):
            return self.starts[-1] + (index-len(self.starts)+1) * (self.starts[-1]-self.starts[-2])
        return self.starts[index]

    def index(self, beat):
        if not self.starts:
            return beat // (self.length or Fraction(4))
        if beat < 0:
            return beat // self.starts[1]
        if beat >= self.starts[-1]:
            return len(self.starts)-1 + (beat-self.starts[-1]) // (self.starts[-1]-self.starts[-2])
        return bisect_right(self.starts, beat)-1

    def column(self, beat, count, *, end=False):
        block = self.index(beat) // count
        if end and self.boundary(block*count) == beat:
            block -= 1
        return block

    def visible(self, index):
        return not self.starts or 0 <= index < len(self.starts)

    def label(self, index):
        if self.first_gap is not None and self.boundary(index) > self.first_gap:
            return '?'
        return str(index + 1)

    def observed_between(self, low, high):
        return range(bisect_left(self.starts, low), bisect_right(self.starts, high))
