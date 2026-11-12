"""Finite, configurable musical-grid hypotheses over exact beat domains.

Enumerating a unique grid candidate does not validate its input uncertainty or
resolve a competing physical interpretation. Keep those decisions separate.
"""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product

from .contracts import ratio, read_ratio

VERSION = 'bounded-rhythm-hypotheses-1'
DEFAULT_SUBDIVISIONS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


@dataclass(frozen=True)
class RhythmGrid:
    subdivisions: tuple[int, ...] = DEFAULT_SUBDIVISIONS

    def __post_init__(self):
        if (not isinstance(self.subdivisions, (tuple, list)) or not self.subdivisions
                or any(type(v) is not int or v < 1 for v in self.subdivisions)
                or len(set(self.subdivisions)) != len(self.subdivisions)):
            raise ValueError('rhythm subdivisions must be distinct positive integers')
        object.__setattr__(self, 'subdivisions', tuple(sorted(self.subdivisions)))

    def snapshot(self):
        return {'subdivisions': list(self.subdivisions)}


def grid_candidates(domain, grid):
    """Union of the declared grids, preserving which grids support each beat."""
    if domain is None:
        return None
    lower, upper = read_ratio(domain['lower']), read_ratio(domain['upper'])
    if (lower > upper or type(domain['lower_inclusive']) is not bool
            or type(domain['upper_inclusive']) is not bool):
        raise ValueError('invalid rhythm beat domain')
    beats = {}
    for subdivision in grid.subdivisions:
        a, b = lower * subdivision, upper * subdivision
        first, last = -(-a.numerator // a.denominator), b.numerator // b.denominator
        if not domain['lower_inclusive'] and a.denominator == 1:
            first += 1
        if not domain['upper_inclusive'] and b.denominator == 1:
            last -= 1
        for tick in range(first, last + 1):
            beat = Fraction(tick, subdivision)
            beats.setdefault(beat, []).append(subdivision)
    return [{'beat': ratio(beat), 'subdivisions': divisions} for beat, divisions in sorted(beats.items())]


def rhythm_candidates(relative, grid):
    """Attach exact candidate notes to conditional spatial candidates.

    These are outer possibilities. A positive duration and a unique marginal
    grid assignment do not certify a jointly feasible physical reconstruction.
    """
    if relative is None:
        return {'format': 'ez2rc.rhythm-candidates', 'schema_version': 1, 'algorithm_version': VERSION,
                'grid': grid.snapshot(), 'coordinate': 'first_barline_zero', 'status': 'unavailable',
                'accepted_note_count': 0, 'candidates': [], 'reasons': ['spatial_reference_unavailable']}
    if relative['format'] != 'ez2rc.relative-beats' or relative['schema_version'] != 1:
        raise ValueError('unsupported relative beat input')
    output = []
    for candidate in relative['candidates']:
        points, reasons = {}, list(candidate['reasons'])
        for side, point in candidate['endpoints'].items():
            projection = point['relative_beat']
            domain = projection['beat_range'] if projection is not None else None
            possibilities = grid_candidates(domain, grid)
            status = ('unavailable' if possibilities is None else 'conflict' if not possibilities else
                      'unique' if len(possibilities) == 1 else 'ambiguous')
            points[side] = {'endpoint_id': point['endpoint_id'], 'status': status,
                            'beat_range': domain, 'candidates': possibilities}
            if status != 'unique':
                reasons.append(side + '_rhythm_' + status)
        notes = None
        if all(p['candidates'] is not None for p in points.values()):
            starts = points['start']['candidates']
            if candidate['type'] == 'tap':
                notes = [{'start_beat': p['beat'], 'end_beat': None} for p in starts]
            else:
                ends = points['end']['candidates']
                duration = candidate['duration_beat_range']
                bounds = (read_ratio(duration['lower']), read_ratio(duration['upper'])) if duration else None
                notes = []
                for start, end in product(starts, ends):
                    delta = read_ratio(end['beat']) - read_ratio(start['beat'])
                    compatible = (bounds is None or
                        ((delta > bounds[0] or delta == bounds[0] and duration['lower_inclusive']) and
                         (delta < bounds[1] or delta == bounds[1] and duration['upper_inclusive'])))
                    if delta > 0 and compatible:
                        notes.append({'start_beat': start['beat'], 'end_beat': end['beat']})
        status = ('unavailable' if notes is None else 'conflict' if not notes else
                  'unique' if len(notes) == 1 else 'ambiguous')
        if notes == []:
            reasons.append('no_compatible_grid_note')
        output.append({'event_id': candidate['event_id'], 'candidate_id': candidate['candidate_id'],
                       'lane': candidate['lane'], 'type': candidate['type'], 'pixel_status': candidate['pixel_status'],
                       'status': status, 'endpoints': points, 'note_candidates': notes,
                       'reasons': sorted(set(reasons)), 'accepted': False})
    return {'format': 'ez2rc.rhythm-candidates', 'schema_version': 1, 'algorithm_version': VERSION,
            'grid': grid.snapshot(), 'coordinate': 'first_barline_zero',
            'status': 'conditional', 'accepted_note_count': 0, 'candidates': output,
            'semantics': 'Union of explicitly supported grids. Unique means one remaining grid label within the supplied outer domains; calibration, shared feasibility, origin and physical alternatives still require validation.'}
