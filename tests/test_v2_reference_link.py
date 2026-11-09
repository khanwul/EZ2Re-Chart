"""Procedural absolute clocks, shared marker offsets, and conflicting anchors."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ez2rc.v2.config import BeatAnchor, ConfigError, MarkerCalibration, load_song
from ez2rc.v2.contracts import decimal_bpm, ratio
from ez2rc.v2.reference import VERSION
from ez2rc.v2.reference_link import link_reference_origin
from ez2rc.v2.runner import execute
from test_v2_timing_resolver import moving_taps


def marker(name, lo, hi=None):
    return {'id': name, 'timestamp_interval_ms': [lo, lo if hi is None else hi]}


def candidates(beats=(), bars=(), *, bpm='120', zero=None, measure=None):
    return {'format': 'ez2rc.reference-candidates', 'schema_version': 1, 'algorithm_version': VERSION,
            'observation_cache_id': 'obs', 'motion_cache_id': 'motion',
            'supplied_reference': {'bpm': bpm, 'beat_zero_ms': zero, 'measure_beats': ratio(measure) if measure else None},
            'beat_signal': {'markers': list(beats)}, 'barlines': {'markers': list(bars)}}


def calibration(offset=0., error=0., meaning='integer_beat'):
    return MarkerCalibration('verified', 'procedural-clock', meaning, offset, error)


def anchor(name='a', beat=Fraction(0), time=100., error=0., status='verified'):
    return BeatAnchor(name, beat, time, error, status, 'procedural-anchor')


class ReferenceLinkTest(unittest.TestCase):
    def test_channel_offsets_have_independent_signs_and_preserve_raw_input(self):
        raw = candidates([marker('a', 109, 111), marker('b', 609, 611)],
                         [marker('x', 79, 81), marker('y', 2079, 2081)], measure=Fraction(4))
        before = deepcopy(raw)
        result = link_reference_origin(raw, marker_calibrations={
            'beat_signal': calibration(-10, 2), 'barlines': calibration(20, 4, 'measure_boundary')},
            beat_anchors=[anchor(beat=Fraction(2), time=1100, error=5)])
        self.assertEqual(raw, before)
        for channel, expected in [('beat_signal', (97, 103)), ('barlines', (95, 105))]:
            hypothesis, = result['channels'][channel]['hypotheses']
            window, = hypothesis['corrected_periodic_windows_ms']
            self.assertEqual((window['lower'], window['upper']), ([expected[0], 1], [expected[1], 1]))
            self.assertTrue(all(c['status'] == 'compatible' for c in hypothesis['origin_comparisons']))
        self.assertEqual(result['anchor_intersection_ms']['lower'], [95, 1])
        self.assertIsNone(result['inferred_beat_zero_ms'])

    def test_decimal_bpm_and_fractional_negative_beat_use_exact_arithmetic(self):
        bpm = '123.456'
        a = anchor(beat=Fraction(-7, 3), time=-12.3, error=.2)
        result = link_reference_origin(candidates(bpm=bpm), beat_anchors=[a])
        center = Fraction('-12.3') + Fraction(7, 3) * 60000 / decimal_bpm(bpm)
        self.assertEqual(result['anchor_intersection_ms']['lower'], ratio(center - Fraction('.2')))
        self.assertEqual(result['anchor_intersection_ms']['upper'], ratio(center + Fraction('.2')))

    def test_nonterminating_period_exact_boundary_can_link(self):
        raw = candidates([marker('a', 312500), marker('b', 625000)], bpm='123.456')
        result = link_reference_origin(raw, marker_calibrations={'beat_signal': calibration(-.1)},
                                       beat_anchors=[anchor(beat=Fraction(643), time=312499.9)])
        hypothesis, = result['channels']['beat_signal']['hypotheses']
        self.assertEqual(result['anchor_intersection_ms']['lower'], [-1, 10])
        self.assertTrue(all(c['status'] == 'compatible' for c in hypothesis['origin_comparisons']))
        self.assertEqual(hypothesis['corrected_periodic_windows_ms'][0]['lower'], [-1, 10])

    def test_recurrence_does_not_reduce_common_offset_error(self):
        for count in (2, 20):
            raw = candidates([marker(str(i), 100 + i * 500) for i in range(count)])
            result = link_reference_origin(raw, marker_calibrations={'beat_signal': calibration(0, 7)})
            hypothesis, = result['channels']['beat_signal']['hypotheses']
            self.assertEqual(hypothesis['corrected_periodic_windows_ms'][0]['lower'], [93, 1])
            self.assertEqual(hypothesis['corrected_periodic_windows_ms'][0]['upper'], [107, 1])
            self.assertEqual(hypothesis['link_status'], 'absolute_origin_unavailable')

    def test_shared_error_cannot_manufacture_recurrence(self):
        raw = candidates([marker('a', 0), marker('b', 250)])
        result = link_reference_origin(raw, marker_calibrations={'beat_signal': calibration(0, 300)},
                                       beat_anchors=[anchor()])
        channel = result['channels']['beat_signal']
        self.assertEqual(channel['status'], 'insufficient')
        self.assertEqual(len(channel['hypotheses']), 2)
        self.assertTrue(all(h['status'] == 'insufficient_recurrence' for h in channel['hypotheses']))
        self.assertIn('clock_error_covers_full_period', channel['reasons'])
        self.assertTrue(all('phase_intervals_ms' not in h for h in channel['hypotheses']))

    def test_anchor_conflicts_and_supplied_origin_are_all_retained(self):
        raw = candidates(zero=0.)
        result = link_reference_origin(raw, beat_anchors=[anchor('a', time=0, error=1),
            anchor('b', beat=Fraction(2), time=1100, error=1), anchor('c', time=0, status='unverified')])
        self.assertEqual(result['anchor_status'], 'conflicting')
        self.assertIsNone(result['anchor_intersection_ms'])
        self.assertEqual([a['conflicting_anchor_ids'] for a in result['anchors']], [['b'], ['a'], []])
        self.assertEqual(result['supplied_origin_comparisons'], [
            {'anchor_id': 'a', 'status': 'compatible'}, {'anchor_id': 'b', 'status': 'conflict'}])
        self.assertEqual(result['supplied_reference']['beat_zero_ms'], 0.)
        self.assertEqual(result['anchors'][2]['reasons'], ['anchor_unverified'])

    def test_closed_anchor_intersection_and_disjoint_phase_candidates(self):
        raw = candidates([marker('a', 100), marker('b', 600), marker('c', 250), marker('d', 750)])
        result = link_reference_origin(raw, marker_calibrations={'beat_signal': calibration()},
                                       beat_anchors=[anchor('a', time=90, error=10), anchor('b', time=110, error=10)])
        self.assertEqual(result['anchor_intersection_ms']['lower'], [100, 1])
        self.assertEqual(result['anchor_intersection_ms']['upper'], [100, 1])
        hypotheses = result['channels']['beat_signal']['hypotheses']
        self.assertEqual(len(hypotheses), 2)
        self.assertEqual([h['origin_comparisons'][-1]['status'] for h in hypotheses], ['compatible', 'conflict'])
        self.assertEqual(hypotheses[1]['support_ids'], ['c', 'd'])

    def test_wrapped_phase_keeps_half_open_endpoint_and_negative_shifts(self):
        raw = candidates([marker('a', 490, 510), marker('b', 990, 1010)], zero=0.)
        result = link_reference_origin(raw, marker_calibrations={'beat_signal': calibration()})
        hypothesis, = result['channels']['beat_signal']['hypotheses']
        comparison, = hypothesis['origin_comparisons']
        self.assertEqual(comparison['matches'], [{'window_index': 0, 'integer_shifts': {'first': 0, 'last': 0}}])
        raw['supplied_reference']['beat_zero_ms'] = -5.
        result = link_reference_origin(raw, marker_calibrations={'beat_signal': calibration()})
        self.assertEqual(result['channels']['beat_signal']['hypotheses'][0]['origin_comparisons'][0]['matches'],
                         [{'window_index': 1, 'integer_shifts': {'first': -1, 'last': -1}}])

    def test_wide_anchor_keeps_all_cycle_lifts_without_enumeration(self):
        raw = candidates([marker('a', 100), marker('b', 600)])
        result = link_reference_origin(raw, marker_calibrations={'beat_signal': calibration()},
                                       beat_anchors=[anchor(time=100, error=1e20)])
        shifts = result['channels']['beat_signal']['hypotheses'][0]['origin_comparisons'][0]['matches'][0]['integer_shifts']
        self.assertEqual(shifts, {'first': -200000000000000000, 'last': 200000000000000000})
        self.assertLess(len(json.dumps(result)), 10000)

    def test_missing_unverified_channel_and_unknown_measure_remain_unavailable(self):
        raw = candidates([marker('a', 100), marker('b', 600)])
        result = link_reference_origin(raw, marker_calibrations={
            'beat_signal': MarkerCalibration('unverified', 'not-calibrated', 'integer_beat'),
            'barlines': calibration(0, 0, 'measure_boundary')}, beat_anchors=[anchor(status='unverified')])
        self.assertEqual(result['anchor_status'], 'unavailable')
        self.assertEqual(result['channels']['beat_signal']['status'], 'unavailable')
        self.assertEqual(result['channels']['barlines']['reasons'], ['measure_length_not_supplied'])
        self.assertIsNone(result['inferred_measure_beats'])

    def test_excluded_markers_and_isolated_candidates_survive(self):
        raw = candidates([marker('a', 100), marker('b', 600), marker('wide', 0, 500),
                          {'id': 'missing', 'timestamp_interval_ms': None}, marker('isolated', 250)])
        result = link_reference_origin(raw, marker_calibrations={'beat_signal': calibration()}, beat_anchors=[anchor()])
        channel = result['channels']['beat_signal']
        self.assertEqual(len(channel['excluded_markers']), 2)
        self.assertEqual(channel['status'], 'ambiguous')
        self.assertEqual(channel['hypotheses'][1]['status'], 'insufficient_recurrence')


class ReferenceConfigTest(unittest.TestCase):
    base = '''schema_version = 1
[input]
video = "unused.mkv"
profile = "unused.toml"
[metadata]
title = "Procedural"
difficulty = "HD"
[timing]
mode = "fixed"
bpm = "120"
'''
    marker = '''[timing.marker_calibrations.beat_signal]
status = "verified"
version = "procedural"
meaning = "integer_beat"
offset_ms = -12.5
error_ms = 1.0
'''
    anchor = '''[[timing.beat_anchors]]
id = "known-beat"
beat = [-3, 2]
timestamp_ms = -100.0
error_ms = 2.0
status = "verified"
version = "procedural"
'''

    def load(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'song.toml'; path.write_text(self.base + text)
            return load_song(path)

    def test_config_roundtrip_preserves_versions_and_exact_beats(self):
        config = self.load(self.marker + self.anchor)
        snapshot = json.loads(json.dumps(config.snapshot()))
        self.assertEqual(snapshot['beat_anchors'][0]['beat'], [-3, 2])
        self.assertEqual(snapshot['marker_calibrations']['beat_signal']['offset_ms'], -12.5)
        self.assertEqual(self.load('').beat_anchors, ())

    def test_invalid_or_unversioned_reference_claims_are_rejected(self):
        bad = [self.anchor + self.anchor, self.anchor.replace('[-3, 2]', '[2, 2]'),
               self.anchor.replace('[-3, 2]', '[true, 1]'), self.anchor.replace('-100.0', 'nan'),
               self.anchor.replace('error_ms = 2.0', 'error_ms = -1'), self.anchor.replace('version = "procedural"', ''),
               self.marker.replace('integer_beat', 'measure_boundary'), self.marker.replace('beat_signal', 'unknown'),
               self.marker.replace('verified', 'unverified'), self.marker.replace('-12.5', 'true'),
               self.marker.replace('error_ms = 1.0', ''), self.marker.replace('error_ms = 1.0', 'error_ms = inf'),
               'beat_anchors = 3', self.marker + 'extra = 1']
        for text in bad:
            with self.subTest(text=text), self.assertRaises(ConfigError):
                self.load(text)

    def test_direct_api_validates_types_channels_and_candidate_version(self):
        for options in ({'beat_anchors': [anchor(), anchor()]}, {'beat_anchors': [object()]},
                        {'marker_calibrations': {'barlines': calibration()}},
                        {'marker_calibrations': {'unknown': calibration()}}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                link_reference_origin(candidates(), **options)
        bad = candidates(); bad['algorithm_version'] = 'future'
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            link_reference_origin(bad)


class ReferenceLinkPipelineTest(unittest.TestCase):
    def test_reference_only_reconstruction_keeps_notes_events_and_all_visual_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_taps(root)
            text = song.read_text().replace('bpm = "191.910"', 'bpm = "120"\nbeat_zero_ms = 0.0\nmeasure_beats = [4, 1]\n'
                '[timing.judgment_calibration]\nstatus = "verified"\nversion = "procedural"\noffset_ms = 7.5\nerror_ms = 0.25')
            song.write_text(text)
            code, first = execute(song, output_root=root / 'out')
            self.assertIn(code, (0, 2))
            load = lambda run, name: json.loads((run / name).read_text())
            chart = load(first, 'chart.json')
            self.assertGreater(len(chart['notes']), 0)
            (root / 'video.mkv').unlink()
            song.write_text(text + '\n' + ReferenceConfigTest.marker + ReferenceConfigTest.anchor)
            with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no video decoding')):
                code, second = execute(song, output_root=root / 'out', observation_cache=first / 'artifacts/observations')
            self.assertIn(code, (0, 2))
            for name in ('chart.json', 'artifacts/events.json', 'artifacts/time-projection.json',
                         'artifacts/timing-resolution.json', 'artifacts/reference-candidates.json'):
                self.assertEqual(load(first, name), load(second, name))
            linked = load(second, 'artifacts/reference-origin-link.json')
            self.assertEqual(linked['supplied_origin_comparisons'][0]['status'], 'conflict')
            self.assertEqual(linked['reference_candidates_sha256'],
                             hashlib.sha256((second / linked['reference_candidates_ref']).read_bytes()).hexdigest())
            self.assertEqual(load(second, 'artifacts/events.json')['reference_origin_link_ref'], 'artifacts/reference-origin-link.json')
            self.assertIn('reference-origin-link.json', (second / 'report.html').read_text())
            stages = load(second, 'run.json')['stages']
            self.assertTrue(all(stages[s]['reused'] for s in ('capture', 'observation', 'motion', 'fusion', 'decode')))


if __name__ == '__main__':
    unittest.main()
