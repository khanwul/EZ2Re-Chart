"""Procedural exact clocks, schema boundaries, and rational reconstruction."""
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ez2rc.v2.cli import main
from ez2rc.v2.config import ConfigError, load_song
from ez2rc.v2.contracts import (Chart, Timing, origin_payload, read_origin, ratio)
from ez2rc.v2.export import chart_payload, read_chart
from ez2rc.v2.origin import adopt_reference_origin
from ez2rc.v2.reference import reference_candidates
from ez2rc.v2.reference_link import link_reference_origin
from ez2rc.v2.runner import execute
from ez2rc.v2.storage import read_json, write_json
from ez2rc.v2.timing import beat_to_ms, ms_to_beat, resolve_pixel_timing
from test_v2_origin_adoption import REQUEST, REQUEST_TOML
from test_v2_reference_candidates import evidence
from test_v2_reference_link import anchor, calibration, candidates, marker
from test_v2_timing_resolver import META, VERIFIED, moving_taps, timing_input


def rational_candidates(zero, **kwargs):
    raw = candidates(**kwargs)
    raw['schema_version'] = 2
    raw['supplied_reference'].update(origin_payload(zero))
    return raw


class RationalOriginTest(unittest.TestCase):
    def test_roundtrip_arithmetic_never_converts_rational_origin_to_float(self):
        for zero in (Fraction(-1000, 3), Fraction(1, 7), Fraction(10**400, 3), Fraction(1, 10**400)):
            timing = Timing('123.456', zero, Fraction(7, 2), 'resolved', None)
            for beat in (Fraction(0), Fraction(-11, 7), Fraction(12345, 17)):
                self.assertEqual(ms_to_beat(beat_to_ms(beat, timing), timing), beat)
            self.assertEqual(beat_to_ms(Fraction(0), timing), zero)

    def test_schema_one_numeric_and_unknown_origins_keep_their_representation(self):
        for zero in (None, 0., -12.5):
            payload = chart_payload(Chart(META, Timing('120', zero)))
            self.assertEqual(payload['schema_version'], 1)
            self.assertEqual(payload['timing']['beat_zero_ms'], zero)
            self.assertNotIn('beat_zero_ms_exact', payload['timing'])
            self.assertEqual(read_origin(payload['timing'], 1), zero)

    def test_schema_two_chart_roundtrip_supports_exact_zero_and_large_denominators(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.json'
            for zero in (Fraction(0), Fraction(-1000, 3), Fraction(10**400, 3), Fraction(1, 10**400)):
                payload = chart_payload(Chart(META, Timing('180', zero, Fraction(4), 'resolved', None)))
                write_json(path, payload)
                restored = read_chart(path)
                self.assertEqual(restored, payload)
                self.assertEqual(restored['schema_version'], 2)
                self.assertIsNone(restored['timing']['beat_zero_ms'])
                self.assertEqual(read_origin(restored['timing'], 2), zero)
                with self.assertRaisesRegex(ValueError, 'schema'):
                    read_json(path, 'ez2rc.chart')

    def test_invalid_fraction_dual_representation_and_schema_downgrade_are_rejected(self):
        payload = chart_payload(Chart(META, Timing('180', Fraction(1, 3), Fraction(4), 'resolved', None)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'chart.json'
            for pair in ([2, 6], [1, 0], [1, -3], [True, 3], [1., 3], [1], '1/3', None):
                value = deepcopy(payload); value['timing']['beat_zero_ms_exact'] = pair
                write_json(path, value)
                with self.subTest(pair=pair), self.assertRaises(ValueError):
                    read_chart(path)
            for field, bad in (('schema_version', 1), ('schema_version', True), ('schema_version', 99),
                               ('beat_zero_ms', 0.), ('beat_zero_ms', True)):
                value = deepcopy(payload)
                target = value if field == 'schema_version' else value['timing']
                target[field] = bad
                write_json(path, value)
                with self.subTest(field=field, bad=bad), self.assertRaises(ValueError):
                    read_chart(path)

    def test_rational_timing_and_note_duration_remain_exact(self):
        data = timing_input()
        zero = Fraction(-1000, 3)
        chart, events, report = resolve_pixel_timing(*data, META, '180', beat_zero_ms=zero,
                                                   measure_beats=Fraction(4), judgment_calibration=VERIFIED)
        self.assertEqual(chart.status, 'complete')
        self.assertEqual(chart.notes[0].start_beat, Fraction(529, 400))
        self.assertEqual(chart.notes[1].end_beat - chart.notes[1].start_beat, Fraction(6, 25))
        self.assertEqual(report['schema_version'], 2)
        self.assertEqual(read_origin(json.loads(json.dumps(report))['reference'], 2), zero)
        self.assertEqual(events[0].start_ms, 107.5)
        self.assertTrue(all(e.start_interval is None for e in events))

    def test_exact_origin_does_not_resolve_missing_calibration_or_meter(self):
        for options in ({'judgment_calibration': None, 'measure_beats': Fraction(4)},
                        {'judgment_calibration': VERIFIED, 'measure_beats': None}):
            chart, _, _ = resolve_pixel_timing(*timing_input(), META, '180',
                                               beat_zero_ms=Fraction(-1000, 3), **options)
            self.assertEqual(chart.status, 'partial')
            self.assertFalse(chart.notes)

    def test_supplied_rational_origin_survives_candidates_and_link_without_rounding(self):
        zero = Fraction(-1000, 3)
        raw = reference_candidates(*evidence([0., 2., 0., 2., 0.]), '180', beat_zero_ms=zero)
        self.assertEqual(read_origin(json.loads(json.dumps(raw))['supplied_reference'], 2), zero)
        raw = rational_candidates(zero, beats=[marker('p', 0), marker('q', 1000)], bpm='180')
        linked = link_reference_origin(raw, beat_anchors=[anchor(beat=Fraction(1), time=0)],
                                       marker_calibrations={'beat_signal': calibration()})
        self.assertEqual(linked['schema_version'], 2)
        self.assertEqual(linked['supplied_origin_comparisons'][0]['status'], 'compatible')
        result = adopt_reference_origin(linked, REQUEST)
        self.assertEqual(result['status'], 'confirmed_supplied')
        self.assertEqual(read_origin(result, 2, 'effective_beat_zero_ms'), zero)
        rounded = candidates(bpm='180', zero=float(zero))
        rounded_link = link_reference_origin(rounded, beat_anchors=[anchor(beat=Fraction(1), time=0)])
        self.assertEqual(rounded_link['supplied_origin_comparisons'][0]['status'], 'conflict')

    def test_supplied_fraction_is_authoritative_on_conflicting_adoption(self):
        zero = Fraction(1, 3)
        linked = link_reference_origin(rational_candidates(zero), beat_anchors=[anchor()])
        for request, status in ((None, 'not_requested'), (REQUEST, 'blocked')):
            result = adopt_reference_origin(linked, request)
            self.assertEqual(result['status'], status)
            self.assertEqual(result['origin_source'], 'supplied')
            self.assertEqual(read_origin(result, 2, 'effective_beat_zero_ms'), zero)
        self.assertIn('supplied_origin_conflict', result['reasons'])

    def test_rational_adoption_requires_request_and_still_blocks_intervals_and_alternatives(self):
        raw = candidates(bpm='180', beats=[marker('a', 0), marker('b', 1000),
                                          marker('c', 100), marker('d', 1100)])
        exact = anchor(beat=Fraction(1), time=0)
        linked = link_reference_origin(raw, beat_anchors=[exact])
        self.assertIsNone(adopt_reference_origin(linked)['effective_beat_zero_ms'])
        adopted = adopt_reference_origin(linked, REQUEST)
        self.assertEqual(adopted['status'], 'adopted')
        self.assertEqual(adopted['effective_beat_zero_ms_exact'], [-1000, 3])
        interval = link_reference_origin(raw, beat_anchors=[anchor(beat=Fraction(1), time=0, error=.1)])
        self.assertEqual(adopt_reference_origin(interval, REQUEST)['reasons'], ['anchor_origin_is_interval'])
        ambiguous = link_reference_origin(raw, beat_anchors=[exact], marker_calibrations={'beat_signal': calibration()})
        self.assertEqual(adopt_reference_origin(ambiguous, REQUEST)['status'], 'blocked')

    def test_diagnostics_reject_unknown_schema_and_rational_schema_downgrade(self):
        for version in (1, True, 3):
            raw = rational_candidates(Fraction(1, 3)); raw['schema_version'] = version
            with self.assertRaises(ValueError):
                link_reference_origin(raw)
            linked = link_reference_origin(rational_candidates(Fraction(1, 3)))
            linked['schema_version'] = version
            with self.assertRaises(ValueError):
                adopt_reference_origin(linked, REQUEST)


class RationalOriginConfigTest(unittest.TestCase):
    base = '''schema_version = 1
[input]
video = "unused.mkv"
profile = "unused.toml"
[metadata]
title = "Procedural"
difficulty = "HD"
[timing]
mode = "fixed"
bpm = "180"
'''

    def test_exact_origin_config_and_json_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'song.toml'
            for zero in (Fraction(-1000, 3), Fraction(0), Fraction(1, 7)):
                path.write_text(self.base + f'beat_zero_ms_exact = {ratio(zero)}\n')
                song = load_song(path)
                self.assertEqual(song.beat_zero_ms, zero)
                self.assertEqual(read_origin(json.loads(json.dumps(song.snapshot())), 2), zero)

    def test_malformed_or_duplicate_origin_settings_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'song.toml'
            for extra in ('beat_zero_ms_exact = [2, 6]', 'beat_zero_ms_exact = [1, 0]',
                          'beat_zero_ms_exact = [1, -3]', 'beat_zero_ms_exact = [true, 3]',
                          'beat_zero_ms_exact = [1.0, 3]', 'beat_zero_ms_exact = "1/3"',
                          'beat_zero_ms_exact = [1, 3]\nbeat_zero_ms = 0.0'):
                path.write_text(self.base + extra)
                with self.subTest(extra=extra), self.assertRaises(ConfigError):
                    load_song(path)


class RationalOriginPipelineTest(unittest.TestCase):
    def test_exact_anchor_extract_and_rational_supplied_reconstruction_use_same_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_taps(root)
            base = song.read_text().replace('bpm = "191.910"', 'bpm = "180"\nmeasure_beats = [4, 1]\n'
                '[timing.judgment_calibration]\nstatus = "verified"\nversion = "procedural"\noffset_ms = 7.5\nerror_ms = 0.25')
            anchors = '''[[timing.beat_anchors]]
id = "origin"
beat = [1, 1]
timestamp_ms = 0.0
error_ms = 0.0
status = "verified"
version = "procedural"
'''
            song.write_text(base + '\n' + anchors + REQUEST_TOML)
            code, first = execute(song, output_root=root / 'out')
            self.assertIn(code, (0, 2))
            load = lambda run, name: json.loads((run / name).read_bytes())
            before = read_chart(first / 'chart.json')
            self.assertEqual(before['schema_version'], 2)
            self.assertTrue(before['notes'])
            self.assertEqual(read_origin(before['timing'], 2), Fraction(-1000, 3))
            adoption = load(first, 'artifacts/origin-adoption.json')
            self.assertEqual(adoption['effective_beat_zero_ms_exact'], [-1000, 3])
            self.assertEqual(load(first, 'run.json')['stages']['export']['schema_version'], 2)
            (root / 'video.mkv').unlink()
            for name, origin_line, shift in (
                    ('same', 'beat_zero_ms_exact = [-1000, 3]', Fraction(0)),
                    ('shifted', 'beat_zero_ms_exact = [-333, 1]', Fraction(1, 1000)),
                    ('numeric', 'beat_zero_ms = 0.0', Fraction(1))):
                song.write_text(base.replace('bpm = "180"', 'bpm = "180"\n' + origin_line))
                with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no video decoding')):
                    code, run = execute(song, output_root=root / 'out', observation_cache=first / 'artifacts/observations')
                self.assertIn(code, (0, 2))
                chart = read_chart(run / 'chart.json')
                self.assertEqual(chart['schema_version'], 1 if name == 'numeric' else 2)
                for a, b in zip(before['notes'], chart['notes'], strict=True):
                    self.assertEqual(Fraction(*a['start_beat']) - Fraction(*b['start_beat']), shift)
                self.assertEqual(load(first, 'artifacts/events.json')['events'], load(run, 'artifacts/events.json')['events'])
                self.assertEqual(load(first, 'artifacts/time-projection.json'), load(run, 'artifacts/time-projection.json'))
                stages = load(run, 'run.json')['stages']
                self.assertTrue(all(stages[s]['reused'] for s in ('capture', 'observation', 'motion', 'fusion', 'decode')))
                linked = load(run, 'artifacts/reference-origin-link.json')
                self.assertEqual(linked['schema_version'], chart['schema_version'])
                self.assertEqual(linked['reference_candidates_sha256'],
                                 hashlib.sha256((run / linked['reference_candidates_ref']).read_bytes()).hexdigest())
                if name == 'same':
                    self.assertEqual(before, chart)
                    self.assertEqual(load(run, 'run.json')['schema_version'], 2)
                    with patch('ez2rc.v2.cli.execute', return_value=(code, run)), redirect_stdout(io.StringIO()):
                        self.assertEqual(main(['extract', str(song)]), code)


if __name__ == '__main__':
    unittest.main()
