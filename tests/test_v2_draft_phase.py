"""Draft positions on procedurally drawn scrolling edges and a pulsing indicator."""
from copy import deepcopy
from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np

from ez2rc.v2.contracts import ratio, read_ratio
from ez2rc.v2.draft_phase import (
    _conflicting_starts, _indicator_ranges, _phase_estimators, _relative_bounds, _track, recover_positions,
)
from ez2rc.v2.endpoint_ranges import closed


class DraftPhaseTest(unittest.TestCase):
    def test_bracketing_pulses_recover_fractional_beat(self):
        lights = {f: 195.5 - 12 * (f % 10) for f in range(160)}
        times = {f: 50 * f for f in lights}
        tracks, assignments = [], []
        for start in range(0, 150, 20):
            marker = str(start)
            assignments.append({'marker_id': marker, 'beat': [start // 10, 1]})
            rows = [{'frame_id': f, 'y_px': (f - start) * 10} for f in range(start, start + 14)]
            tracks.append({'id': marker, 'observations': rows})
        grid = {'bpm': '120', 'hypotheses': [{'assignments': assignments}], 'tracks': tracks}
        estimates = _phase_estimators(grid, lights, times, 100)
        samples = np.array([(f, f * 10 - 555) for f in range(58, 70)])
        self.assertEqual(len(estimates), 2)
        for estimate in estimates.values():
            self.assertAlmostEqual(estimate(samples), 5.55, delta=0.005)
            self.assertIsNone(estimate(samples - [58, 0]))
        self.assertFalse(_phase_estimators(grid, dict.fromkeys(lights, 180), times, 100))
        shifted = {f: ms + (1000 if 58 <= f < 70 else 0) for f, ms in times.items()}
        for estimate in _phase_estimators(grid, lights, shifted, 100).values():
            self.assertIsNone(estimate(samples))

    def test_tracks_exclude_branches_and_disagreeing_edges(self):
        points = {f: {10*f, 10*f+1} for f in range(10)}
        track = _track(points, 1)
        self.assertEqual(track[0, 1], 0.5)
        self.assertIsNone(_track({**points, **{f+20: ys for f, ys in points.items()}}, 1))
        noisy = track.copy()
        noisy[3, 1] += 10
        self.assertIsNone(_relative_bounds(track, noisy, [Fraction(80)], Fraction(1)))

    def test_phase_envelope_requires_agreement_with_independent_taps(self):
        tracks = {str(i): i for i in range(41)}
        controls = {str(i): Fraction(i) for i in range(40)}
        methods = {'stable': lambda x: x + 0.012, 'drifting': lambda x: x + x / 200}
        bounds = _indicator_ranges(methods, tracks, controls, {'40': None}, 0.001)
        self.assertEqual(set(bounds['40']), {'stable'})
        interval = bounds['40']['stable']['range']
        self.assertLess(read_ratio(interval['lower']), 40)
        self.assertGreater(read_ratio(interval['upper']), 40)
        self.assertEqual(_indicator_ranges(methods, tracks, {'0': Fraction(0)}, {'40': None}, 0.001), {})

    def test_conflicting_new_starts_are_both_withheld(self):
        path = np.array([(f, 8*f) for f in range(12)])
        tracks = {'first': path, 'second': path, 'later': path + [20, 0]}
        values = {eid: (Fraction(i+1),) for i, eid in enumerate(tracks)}
        self.assertEqual(_conflicting_starts(values, tracks, [Fraction(256)], 1), {'first', 'second'})

    def test_shared_edges_restore_missing_values_and_keep_shape_review(self):
        shapes = [dict.fromkeys(('tap', 'head', 'tail'), (12, 6))]
        anchor_beat = Fraction(3)
        def point(offset, known=False):
            p = {'offset_px': [offset, 1], 'beat_range': None,
                 'reasons': ['no_observed_path_to_numbered_barline']}
            if known:
                p.update(beat_range=closed(anchor_beat, anchor_beat), candidate_ticks=[576], reasons=[])
            return p
        anchor = {'id': 'a', 'lane': 0, 'type': 'tap', 'start_beat': ratio(anchor_beat),
                  'end_beat': None, 'issues': [], 'status': 'draft',
                  'evidence': {'start': point(11, True), 'pixel_status': 'resolved'}}
        hold = {'id': 'h', 'lane': 0, 'type': 'longnote', 'start_beat': None, 'end_beat': None,
                'issues': ['pixel_ambiguous', 'start_unavailable', 'end_unavailable'],
                'status': 'unresolved', 'evidence': {'pixel_status': 'ambiguous',
                                                      'start': point(11), 'end': point(0)}}
        base = np.array([(f, 900 + f) for f in range(20)])
        tracks = {'a0': base, 'h0': base - [0, 128], 'h1': base - [0, 333]}
        rows = [dict(id='a', start='a0', end=None, selected=True),
                dict(id='h', start='h0', end='h1', selected=True)]
        def recover(notes, distances):
            with patch('ez2rc.v2.draft_phase._read_tracks', return_value=(distances, {}, {})):
                recover_positions(notes, rows, None, {}, shapes, {'hypotheses': []}, [[1024, 1]]*2, 1)
        original = deepcopy(anchor)
        boundary = deepcopy(hold)
        boundary['evidence']['hold_resolution'] = True
        recover([anchor, boundary], tracks)
        self.assertIsNone(boundary['start_beat'])
        unknown = deepcopy(hold)
        recover([anchor, unknown], {k: v for k, v in tracks.items() if k != 'a0'})
        self.assertIsNone(unknown['start_beat'])
        recover([anchor, hold], tracks)
        self.assertEqual(anchor, original)
        self.assertEqual(read_ratio(hold['start_beat']), Fraction(25, 8))
        self.assertEqual(read_ratio(hold['end_beat']), 3 + Fraction(333, 1024))
        self.assertEqual(hold['issues'], ['pixel_ambiguous'])
        self.assertEqual(hold['status'], 'unresolved')
        self.assertIn('a0', hold['evidence']['end']['recovery']['shared_endpoints'])
