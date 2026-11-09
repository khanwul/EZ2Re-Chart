"""Independent procedural marker clocks, ambiguity and missing evidence."""
from fractions import Fraction
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ez2rc.v2.motion import STATUS
from ez2rc.v2.reference import phase_candidates, reference_candidates
from ez2rc.v2.runner import execute
from v2_fixtures import fixture


def marker(mid, lower, upper=None):
    return {'id': mid, 'timestamp_interval_ms': [lower, lower if upper is None else upper]}


def evidence(brightness, *, scroll=None, times=None, ids=None, groups=None, bars=(), states=None, chunk=4):
    n = len(brightness)
    ids = np.arange(n) if ids is None else np.array(ids)
    times = np.arange(n) * 50. if times is None else np.array(times, float)
    scroll = np.arange(n) * 10. if scroll is None else np.array(scroll, float)
    groups = ids if groups is None else np.array(groups)
    bars = np.array(bars, float).reshape(-1, 3)
    features, movement = [], []
    for start in range(0, n, chunk):
        stop = min(n, start + chunk)
        frame_ids = ids[start:stop]
        features.append({'frame_id': frame_ids, 'timestamp_ms': times[start:stop],
                         'repeat_candidate_group': groups[start:stop],
                         'beat_brightness': np.array(brightness[start:stop], float),
                         'barlines': bars[np.isin(bars[:, 0], frame_ids)]})
        movement.append({'frame_id': frame_ids, 'timestamp_ms': times[start:stop],
                         'scroll_px': scroll[start:stop], 'error_px': np.zeros(len(frame_ids)),
                         'status': np.array([STATUS[s] for s in states[start:stop]]) if states else
                                   np.full(len(frame_ids), STATUS['observed'])})
    obs = SimpleNamespace(id='observations', manifest={'metadata': {'profile': {'judge_y': 80.}}},
                          chunks=lambda: iter(features))
    moved = SimpleNamespace(id='motion', manifest={'key': {'parents': ['observations'],
                            'parameters': {'match_tolerance_px': 2.}}}, chunks=lambda: iter(movement))
    return obs, moved


class CyclicPhaseTest(unittest.TestCase):
    def test_wrapped_window_is_one_hypothesis_with_two_arcs(self):
        result = phase_candidates([marker('a', 490., 510.), marker('b', 990., 1010.)], Fraction(500))
        self.assertEqual(result['status'], 'supported')
        hypothesis, = result['hypotheses']
        self.assertEqual(hypothesis['distinct_cycles_min'], 2)
        self.assertEqual([(s['lower'], s['upper']) for s in hypothesis['phase_intervals_ms']], [(0., 10.), (490., 500.)])
        self.assertFalse(hypothesis['phase_intervals_ms'][-1]['upper_inclusive'])

    def test_conflicting_phases_and_isolated_evidence_survive(self):
        markers = [marker('a', 90., 110.), marker('b', 590., 610.),
                   marker('c', 240., 260.), marker('d', 740., 760.), marker('noise', 400.)]
        result = phase_candidates(markers, Fraction(500))
        self.assertEqual(result['status'], 'ambiguous')
        self.assertEqual([h['support_ids'] for h in result['hypotheses']], [['a', 'b'], ['c', 'd'], ['noise']])
        self.assertEqual(result['hypotheses'][0]['conflicting_ids'], ['c', 'd', 'noise'])

    def test_missing_cycles_do_not_determine_absolute_origin_or_meter(self):
        result = phase_candidates([marker('first', 100.), marker('later', 1600.)], Fraction(500))
        self.assertEqual(result['status'], 'supported')
        self.assertEqual(result['hypotheses'][0]['distinct_cycles_min'], 2)
        self.assertEqual(result['hypotheses'][0]['phase_intervals_ms'][0]['lower'], 100.)

    def test_same_cycle_and_full_period_windows_do_not_establish_recurrence(self):
        result = phase_candidates([marker('a', 90., 110.), marker('b', 95., 115.), marker('wide', 0., 500.)], Fraction(500))
        self.assertEqual(result['status'], 'insufficient')
        self.assertEqual(result['hypotheses'][0]['distinct_cycles_max'], 1)
        self.assertEqual(result['excluded_markers'], [{'id': 'wide', 'reason': 'window_covers_full_period'}])

    def test_closed_boundary_and_negative_decimal_times_are_exact(self):
        result = phase_candidates([marker('a', -.3), marker('b', .2)], Fraction(1, 10))
        self.assertEqual(result['status'], 'supported')
        self.assertEqual(result['period_ms'], [1, 10])
        self.assertEqual(result['hypotheses'][0]['phase_intervals_ms'], [
            {'lower': 0., 'upper': 0., 'lower_inclusive': True, 'upper_inclusive': True}])
        result = phase_candidates([marker('a', 90., 100.), marker('b', 600., 610.)], Fraction(500))
        self.assertEqual(result['hypotheses'][0]['phase_intervals_ms'][0]['lower'], 100.)
        self.assertEqual(result['hypotheses'][0]['phase_intervals_ms'][0]['upper'], 100.)

    def test_malformed_windows_and_periods_are_rejected(self):
        for markers, period in (([marker('a', 0.), marker('a', 500.)], Fraction(500)),
                                ([marker('a', float('nan'))], Fraction(500)),
                                ([marker('a', 2., 1.)], Fraction(500)),
                                ([], Fraction(0)), ([], 500)):
            with self.subTest(markers=markers, period=period), self.assertRaises(ValueError):
                phase_candidates(markers, period)


class VisualReferenceTest(unittest.TestCase):
    def test_plateaus_form_one_peak_across_chunks_and_ignore_lane_repeats(self):
        args = {'times': [0, 100, 200, 300, 400, 500, 600, 700], 'groups': [0] * 8}
        result = reference_candidates(*evidence([0, 20, 20, 0, 0, 20, 20, 0], chunk=2, **args), '150')
        markers = result['beat_signal']['markers']
        self.assertEqual(len(markers), 2)
        self.assertEqual(markers[0]['frame_ids'], [1, 2])
        self.assertEqual(markers[0]['timestamp_interval_ms'], [0, 300])
        self.assertEqual(result['beat_signal']['phase']['status'], 'supported')
        other = reference_candidates(*evidence([0, 20, 20, 0, 0, 20, 20, 0], chunk=7, **args), '150')
        self.assertEqual(result, other)
        self.assertIsNone(result['inferred_beat_zero_ms'])
        self.assertEqual(result['status'], 'unresolved')

    def test_gaps_equal_timestamps_and_missing_neighbors_keep_incomplete_peaks(self):
        for args in ({'ids': [0, 1, 4, 5]}, {'times': [0, 100, 100, 200]}):
            result = reference_candidates(*evidence([0, 20, 20, 0], **args), '120')
            markers = result['beat_signal']['markers']
            self.assertEqual(len(markers), 2)
            self.assertTrue(all(m['timestamp_interval_ms'] is None for m in markers))
            self.assertEqual(result['beat_signal']['phase']['status'], 'insufficient')
        result = reference_candidates(*evidence([40] * 5), '120')
        self.assertEqual(result['beat_signal']['markers'][0]['reasons'],
                         ['missing_rising_neighbor', 'missing_falling_neighbor'])

    def test_moving_bands_require_supplied_measure_and_never_infer_zero(self):
        bars = [(i, i * 10 + 80 - q, 70) for i in range(31) for q in (100, 200, 300)
                if 0 <= i * 10 + 80 - q < 75]
        inputs = evidence([0] * 31, bars=bars)
        missing = reference_candidates(*inputs, '120')
        self.assertEqual(missing['barlines']['phase']['reason'], 'measure_length_not_supplied')
        result = reference_candidates(*inputs, '120', beat_zero_ms=123., measure_beats=Fraction(1))
        self.assertEqual(result['barlines']['phase']['status'], 'supported')
        self.assertEqual(len(result['barlines']['markers']), 3)
        self.assertEqual([m['projection']['nominal_timestamp_ms'] for m in result['barlines']['markers']], [500., 1000., 1500.])
        self.assertEqual(result['supplied_reference']['beat_zero_ms'], 123.)
        self.assertIsNone(result['inferred_beat_zero_ms'])
        self.assertIsNone(result['inferred_measure_beats'])

    def test_barline_repeats_static_bands_and_association_conflicts_are_retained(self):
        repeated = reference_candidates(*evidence([0] * 3, scroll=[40, 40, 50], groups=[0, 0, 2],
                                      bars=[(0, 40, 70), (1, 40, 70), (2, 50, 70)]), '120')
        self.assertEqual(repeated['barlines']['markers'][0]['independent_frame_groups'], 2)
        static = reference_candidates(*evidence([0] * 3, bars=[(i, 40, 70) for i in range(3)]), '120')
        self.assertTrue(all('no_observed_motion' in m['reasons'] for m in static['barlines']['markers']))
        conflict = reference_candidates(*evidence([0, 0], scroll=[0, 5], bars=[(0, 40, 70), (0, 44, 70), (1, 47, 70)]), '120')
        self.assertEqual(len(conflict['barlines']['markers']), 3)
        self.assertTrue(all('ambiguous_barline_association' in m['reasons'] for m in conflict['barlines']['markers']))

    def test_uncertain_motion_and_missing_frames_break_tracks(self):
        for args in ({'states': ['observed', 'ambiguous', 'observed']}, {'ids': [0, 2, 4]}):
            ids = args.get('ids', [0, 1, 2])
            result = reference_candidates(*evidence([0] * 3, bars=[(fid, 20 + i * 10, 70) for i, fid in enumerate(ids)], **args), '120')
            self.assertEqual(len(result['barlines']['markers']), 3)
            self.assertTrue(all(m['timestamp_interval_ms'] is None for m in result['barlines']['markers']))

    def test_parent_timeline_and_signal_validation(self):
        obs, moved = evidence([0, 10, 0])
        moved.manifest['key']['parents'] = ['wrong']
        with self.assertRaisesRegex(ValueError, 'lineage'):
            reference_candidates(obs, moved, '120')
        for values in ([0, float('nan'), 0], [0, 256, 0]):
            with self.assertRaisesRegex(ValueError, 'signals'):
                reference_candidates(*evidence(values), '120')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            reference_candidates(*evidence([0, 0], bars=[(0, 20, 30), (0, 20, 40)]), '120')

    def test_pipeline_preserves_explicit_reference_and_reconstructs_without_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song, _ = fixture(root)
            song.write_text(song.read_text().replace('bpm = "191.910"',
                            'bpm = "191.910"\nbeat_zero_ms = 123.0\nmeasure_beats = [3, 1]'))
            code, first = execute(song, output_root=root / 'out')
            self.assertEqual(code, 2)
            before = json.loads((first / 'artifacts/reference-candidates.json').read_text())
            self.assertEqual(before['supplied_reference']['beat_zero_ms'], 123.)
            self.assertIsNone(before['inferred_beat_zero_ms'])
            (root / 'video.mkv').unlink()
            with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no video decoding')):
                code, second = execute(song, output_root=root / 'out', observation_cache=first / 'artifacts/observations')
            self.assertEqual(code, 2)
            self.assertEqual(before, json.loads((second / 'artifacts/reference-candidates.json').read_text()))
            chart = json.loads((second / 'chart.json').read_text())
            self.assertEqual(chart['timing']['beat_zero_ms'], 123.)
            self.assertEqual(chart['notes'], [])
            events = json.loads((second / 'artifacts/events.json').read_text())
            self.assertEqual(events['reference_candidates_ref'], 'artifacts/reference-candidates.json')
            self.assertIn('reference-candidates.json', (second / 'report.html').read_text())


if __name__ == '__main__':
    unittest.main()
