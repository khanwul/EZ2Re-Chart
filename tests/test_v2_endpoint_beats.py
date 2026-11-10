"""Procedural joins of endpoint domains and declared shared timing terms."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from ez2rc.v2.beat_ranges import project_beat_ranges
from ez2rc.v2.config import ConfigError, JudgmentCalibration, RangeUncertainty, load_song
from ez2rc.v2.contracts import Event, decimal_bpm, read_ratio
from ez2rc.v2.endpoint_ranges import project_endpoint_time_ranges
from ez2rc.v2.runner import execute
from ez2rc.v2.storage import read_json, write_json
from ez2rc.v2.timing import resolve_pixel_timing
from test_v2_beat_ranges import INTERVAL_TOML, decision, window
from test_v2_endpoint_ranges import inputs as endpoint_inputs
from test_v2_origin_adoption import ANCHOR_TOML
from test_v2_reference_link import candidates
import test_v2_shared_judgment as shared_judgment
from test_v2_shared_judgment import UNKNOWN_TOML
from test_v2_timing_projection import clock
from test_v2_timing_resolver import META, moving_taps

MODEL = RangeUncertainty('procedural-join', 'shared_all_events', 'unknown',
                         endpoint_component='calibration_and_localization')
REQUEST = 'endpoint_component = "calibration_and_localization"\n'


def inputs(*, qs=(5., 15., 25.), error=3., bpm='120', resolution_options=None, **kwargs):
    kwargs.setdefault('values', clock([0, 10, 20, 30], times=[0, 20, 50, 90]))
    obs, moved, projection = endpoint_inputs(qs, **kwargs)
    lane = projection['lanes'][0]
    for point, kind in zip(lane['endpoints'], ('tap', 'head', 'tail')):
        point['kind'] = kind
    lane['candidates'] = [
        {'id': 'c0', 'type': 'tap', 'start': 'p0', 'end': None, 'selected': True, 'pixel_status': 'resolved'},
        {'id': 'c1', 'type': 'longnote', 'start': 'p1', 'end': 'p2', 'selected': True, 'pixel_status': 'resolved'}]
    events = tuple(Event(f'e{i}', 0, 'incomplete', c['type'], candidate_ids=(c['id'],), reason='pending')
                   for i, c in enumerate(lane['candidates']))
    pixels = [{'id': e.id, 'lane': 0, 'status': 'resolved', 'selected_intervals': [deepcopy(c)]}
              for e, c in zip(events, lane['candidates'])]
    endpoints = project_endpoint_time_ranges(obs, moved, projection)
    options = {'measure_beats': Fraction(4), 'judgment_calibration': JudgmentCalibration('verified', 'procedural', 7.5, error),
               **(resolution_options or {})}
    chart, events, resolution = resolve_pixel_timing(events, pixels, projection, META, bpm, **options)
    origin = decision(candidates(bpm=bpm, measure=options['measure_beats']))
    return chart, events, resolution, origin, pixels, projection, endpoints


def project(data, model=MODEL):
    _, events, resolution, origin, pixels, projection, endpoints = data
    return project_beat_ranges(events, resolution, origin, uncertainty_model=model,
                              pixels=pixels, projection=projection, endpoint_ranges=endpoints)


class EndpointBeatTest(unittest.TestCase):
    def test_exact_join_applies_nominal_offset_once_and_retains_named_sources(self):
        data = inputs(); before = deepcopy(data)
        result = project(data)
        self.assertEqual(result['schema_version'], 3)
        tap, ln = result['events']
        self.assertEqual(tap['start_capture_time_range_ms'], window(Fraction('8.5'), Fraction('11.5')))
        self.assertEqual(tap['start_judgment_time_range_ms'], window(16, 19))
        self.assertEqual(tap['start_beat_range'], window(Fraction(-97, 500), Fraction(-68, 500)))
        self.assertEqual(ln['duration_beat_range'], window(Fraction(119, 2000), Fraction(161, 2000)))
        self.assertIsNone(ln['duration_beats']); self.assertEqual(ln['nominal_duration_beats'], [7, 100])
        self.assertEqual(ln['selected_candidate_id'], 'c1'); self.assertEqual(ln['end_endpoint_id'], 'p2')
        legacy = project_beat_ranges(data[1], data[2], data[3], uncertainty_model=replace(MODEL, endpoint_component=None))
        self.assertEqual(tap['start_nominal_time_beat_range'], legacy['events'][0]['start_beat_range'])
        self.assertEqual(result['shared_origin'], legacy['shared_origin'])
        self.assertEqual(result['shared_judgment'], legacy['shared_judgment'])
        self.assertEqual(data, before); self.assertEqual(data[0].status, 'partial'); self.assertFalse(data[0].notes)

    def test_same_clock_cancels_only_shared_shift_not_endpoint_width(self):
        data = inputs(error=10)
        result = project(data, replace(MODEL, origin_relation='same_clock_shift', origin_reference_ms=Fraction(100)))
        self.assertEqual(result['events'][0]['start_beat_range'], window(Fraction(-21, 125), Fraction(-81, 500)))
        self.assertNotEqual(result['events'][0]['start_beat_range']['lower'], result['events'][0]['start_beat_range']['upper'])
        self.assertEqual(result['events'][1]['duration_beat_range'], project(data)['events'][1]['duration_beat_range'])
        blocked = project(data, replace(MODEL, origin_relation='same_clock_shift', origin_reference_ms=Fraction(101)))
        self.assertEqual(blocked['bounded_event_count'], 0)
        self.assertIn('shared_clock_bounds_mismatch', blocked['events'][0]['reasons'])

    def test_finite_joint_domain_oracle_without_independence_assumption(self):
        rng = random.Random(681)
        for _ in range(60):
            bpm = rng.choice(('120', '123.456', '180'))
            error = rng.randrange(0, 17) / 4
            data = inputs(error=error, bpm=bpm, calibration=rng.randrange(0, 9) / 4, localization=rng.randrange(0, 9) / 4)
            result = project(data); scale = decimal_bpm(bpm) / 60000
            for row in result['events']:
                for side in ('start', 'end'):
                    value = row[side + '_capture_time_range_ms']
                    if value is None: continue
                    values = [(read_ratio(value[edge]) + Fraction('7.5') + Fraction(str(delta)) - origin) * scale
                              for edge in ('lower', 'upper') for delta in (-error, error) for origin in (90, 110)]
                    self.assertEqual(row[side + '_beat_range'], window(min(values), max(values)))
            ln = result['events'][1]
            differences = [(read_ratio(ln['end_capture_time_range_ms'][end]) - read_ratio(ln['start_capture_time_range_ms'][start])) * scale
                           for start in ('lower', 'upper') for end in ('lower', 'upper')]
            self.assertEqual(ln['duration_beat_range'], window(min(differences), max(differences)))

    def test_nonpositive_duration_bounds_are_not_clipped(self):
        result = project(inputs(qs=(5., 9.75, 10.25), calibration=1., localization=1.))
        ln = result['events'][1]
        self.assertLess(read_ratio(ln['duration_beat_range']['lower']), 0)
        self.assertEqual(ln['diagnostics'], ['duration_may_be_nonpositive'])
        self.assertEqual(ln['status'], 'bounded'); self.assertIsNone(ln['duration_beats'])

    def test_zero_errors_keep_exact_inverse_instead_of_float_nominal_rounding(self):
        data = inputs(qs=(1., 1.5, 2.), calibration=0., localization=0., error=0.,
                      values=clock([0, 3], times=[0, 1000]))
        result = project(data)
        self.assertEqual(result['events'][0]['start_capture_time_range_ms'], window(Fraction(1000, 3), Fraction(1000, 3)))
        self.assertEqual(result['events'][0]['start_beat_range'],
                         window((Fraction(1000, 3) + Fraction('7.5') - 110) / 500,
                                (Fraction(1000, 3) + Fraction('7.5') - 90) / 500))
        self.assertEqual(result['events'][1]['duration_beat_range'], window(Fraction(1, 3), Fraction(1, 3)))

    def test_unknown_endpoint_error_withholds_combined_ranges_but_preserves_nominal(self):
        result = project(inputs(localization=None))
        self.assertEqual(result['bounded_event_count'], 0)
        for row in result['events']:
            self.assertIsNone(row['start_beat_range']); self.assertIsNotNone(row['start_nominal_time_beat_range'])
            self.assertIn('start_localization_error_unknown', row['reasons'])
        partial = project(inputs(qs=(0., 15., 25.)))
        self.assertEqual(partial['bounded_event_count'], 1)
        self.assertIn('start_outside_observed_scroll_range', partial['events'][0]['reasons'])

    def test_existing_time_and_measure_gates_remain(self):
        for options in ({'resolution_options': {'judgment_calibration': None}},
                        {'resolution_options': {'measure_beats': None}}, {'verified': False},
                        {'qs': (10., 15., 25.), 'values': clock([0, 10, 10, 30], times=[0, 20, 30, 90])}):
            data = inputs(**options); result = project(data)
            self.assertIsNone(result['events'][0]['start_beat_range'])
        data = list(inputs()); data[3] = decision(anchors=[])
        self.assertEqual(project(data)['bounded_event_count'], 0)

    def test_competing_or_missing_selected_endpoints_never_gain_ranges(self):
        for mode in ('ambiguous', 'missing_end', 'multiple'):
            data = list(inputs())
            if mode == 'ambiguous':
                data[4][1]['status'] = 'ambiguous'
                data[5]['lanes'][0]['candidates'][1]['pixel_status'] = 'ambiguous'
                data[6]['lanes'][0]['candidates'][1]['pixel_status'] = 'ambiguous'
            elif mode == 'missing_end':
                data[4][1]['selected_intervals'][0]['end'] = None
                data[5]['lanes'][0]['candidates'][1]['end'] = None
                data[6]['lanes'][0]['candidates'][1]['end'] = None
            else:
                data[4][1]['selected_intervals'].append(deepcopy(data[4][0]['selected_intervals'][0]))
            data[0], data[1], data[2] = resolve_pixel_timing(data[1], data[4], data[5], META, '120',
                measure_beats=Fraction(4), judgment_calibration=JudgmentCalibration('verified', 'procedural', 7.5, 3.))
            result = project(data)
            self.assertEqual(result['events'][1]['status'], 'unavailable')
            self.assertIsNone(result['events'][1]['start_beat_range'])
            self.assertIsNone(result['events'][1]['duration_beat_range'])

    def test_parent_selection_lane_kind_and_time_mismatches_are_rejected(self):
        for mode in ('parent', 'duplicate', 'candidate', 'lane', 'point_kind', 'accepted_time', 'pixel_status', 'missing_parent'):
            data = list(inputs()); point = data[6]['lanes'][0]['endpoints'][0]
            if mode == 'parent': data[6]['decode_cache_id'] = 'other'
            if mode == 'duplicate': data[6]['lanes'][0]['endpoints'].append(deepcopy(point))
            if mode == 'candidate': data[4][0]['selected_intervals'][0]['start'] = 'p1'
            if mode == 'lane': data[4][0]['lane'] = 1
            if mode == 'point_kind': point['kind'] = 'head'
            if mode == 'accepted_time': data[5]['lanes'][0]['endpoints'][0]['nominal_timestamp_ms'] += 1
            if mode == 'pixel_status': data[4][0]['status'] = 'ambiguous'
            if mode == 'missing_parent': data[6] = None
            with self.subTest(mode=mode), self.assertRaises(ValueError): project(data)

    def test_invalid_closed_bounds_and_component_claims_are_rejected(self):
        for mode in ('open', 'reversed', 'unreduced', 'pixel', 'error', 'status', 'support'):
            data = inputs(); component = data[6]['lanes'][0]['endpoints'][0]['components']['calibration_and_localization']
            if mode == 'open': component['time_range_ms']['lower_inclusive'] = False
            if mode == 'reversed': component['time_range_ms'] = window(10, 9)
            if mode == 'unreduced': component['time_range_ms']['lower'] = [2, 2]
            if mode == 'pixel': component['pixel_range_px'] = window(0, 1)
            if mode == 'error': component['error_px_exact'] = [9, 1]
            if mode == 'status': component['status'] = 'unavailable'
            if mode == 'support': component['time_range_ms'] = window(-1000, 1000)
            with self.subTest(mode=mode), self.assertRaises(ValueError): project(data)

    def test_request_and_schema_are_explicit_and_old_results_stay_unchanged(self):
        data = inputs(); _, events, resolution, origin, pixels, projection, endpoints = data
        with self.assertRaises(ValueError):
            project_beat_ranges(events, resolution, origin, uncertainty_model=replace(MODEL, endpoint_component=None), endpoint_ranges=endpoints)
        with self.assertRaises(ValueError): project_beat_ranges(events, resolution, origin, uncertainty_model=MODEL)
        result = project(data)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'range.json'; write_json(path, result)
            with self.assertRaises(ValueError): read_json(path, 'ez2rc.beat-ranges')
            self.assertEqual(read_json(path, 'ez2rc.beat-ranges', schema_versions=(3,)), result)
        legacy = project_beat_ranges(events, resolution, origin, uncertainty_model=replace(MODEL, endpoint_component=None))
        self.assertEqual(legacy['schema_version'], 2)
        self.assertNotIn('endpoint_component', legacy['uncertainty_model']['request'])

    def test_explicit_complete_component_config_and_legacy_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'song.toml'
            base = shared_judgment.SharedJudgmentConfigTest.base + UNKNOWN_TOML
            path.write_text(base + REQUEST)
            self.assertEqual(load_song(path).range_uncertainty.endpoint_component, 'calibration_and_localization')
            self.assertEqual(load_song(path).snapshot()['range_uncertainty']['endpoint_component'], 'calibration_and_localization')
            for value in ('calibration', 'localization', 'independent', ''):
                path.write_text(base + REQUEST.replace('calibration_and_localization', value))
                with self.subTest(value=value), self.assertRaises(ConfigError): load_song(path)
            path.write_text(base)
            self.assertNotIn('endpoint_component', load_song(path).snapshot()['range_uncertainty'])


class EndpointBeatPipelineTest(unittest.TestCase):
    def test_opt_in_and_removal_reuse_cache_and_preserve_unknown_location(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_taps(root)
            base = song.read_text().replace('bpm = "191.910"', 'bpm = "120"\nmeasure_beats = [4, 1]\n'
                '[timing.judgment_calibration]\nstatus = "verified"\nversion = "procedural"\noffset_ms = 7.5\nerror_ms = 2.0')
            base += '\n' + ANCHOR_TOML.replace('error_ms = 0.0', 'error_ms = 2.0') + INTERVAL_TOML + UNKNOWN_TOML
            song.write_text(base)
            code, first = execute(song, output_root=root / 'out'); self.assertEqual(code, 2)
            load = lambda run, name: json.loads((run / name).read_bytes())
            before = load(first, 'artifacts/beat-ranges.json'); self.assertGreater(before['bounded_event_count'], 0)
            (root / 'video.mkv').unlink()
            for enabled in (True, False):
                song.write_text(base + (REQUEST if enabled else ''))
                with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no video decoding')):
                    code, run = execute(song, output_root=root / 'out', observation_cache=first / 'artifacts/observations')
                self.assertEqual(code, 2)
                for name in ('chart.json', 'artifacts/events.json', 'artifacts/timing-resolution.json', 'artifacts/time-projection.json',
                             'artifacts/origin-adoption.json', 'artifacts/endpoint-time-ranges.json'):
                    self.assertEqual(load(first, name), load(run, name))
                result = load(run, 'artifacts/beat-ranges.json')
                if enabled:
                    self.assertEqual(result['schema_version'], 3); self.assertEqual(result['bounded_event_count'], 0)
                    self.assertTrue(any('start_localization_error_unknown' in row['reasons'] for row in result['events']))
                    for key in ('timing_resolution', 'origin_adoption', 'time_projection', 'endpoint_time_ranges'):
                        self.assertEqual(result[key + '_sha256'], hashlib.sha256((run / result[key + '_ref']).read_bytes()).hexdigest())
                else: self.assertEqual(result, before)
                stages = load(run, 'run.json')['stages']
                self.assertEqual(stages['timing']['beat_ranges_version'], result['algorithm_version'])
                self.assertTrue(all(stages[s]['reused'] for s in ('capture', 'observation', 'motion', 'fusion', 'decode')))


if __name__ == '__main__':
    unittest.main()
