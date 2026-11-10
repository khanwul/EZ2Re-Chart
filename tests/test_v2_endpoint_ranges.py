"""Procedural pixel intervals and exact inverse-curve coverage checks."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ez2rc.v2.endpoint_ranges import EndpointTimeMap, project_endpoint_time_ranges
from ez2rc.v2.contracts import read_ratio
from ez2rc.v2.runner import execute
from ez2rc.v2.storage import canonical, read_json, write_json
from ez2rc.v2.timing import project_video_times
from test_v2_timing_projection import clock
from test_v2_timing_resolver import moving_taps


def bounds(result, key='time_range_ms'):
    return tuple(read_ratio(result[key][side]) for side in ('lower', 'upper'))


def inputs(qs=(5.,), *, calibration=.5, localization=.25, verified=True, values=None):
    cal = {'status': 'verified' if verified else 'unverified', 'version': 'procedural',
           **({'offset_px': 0., 'error_px': calibration} if verified else {})}
    obs = SimpleNamespace(id='obs', manifest={'metadata': {
        'position_error_px': localization,
        'profile': {'calibration': {kind: deepcopy(cal) for kind in ('tap', 'head', 'tail')}}}})
    moved = SimpleNamespace(id='motion', manifest={'key': {'parents': ['obs']}},
                            chunks=lambda: iter([clock([0, 10, 30], times=[100, 110, 150]) if values is None else values]))
    points = [{'id': f'p{i}', 'kind': 'tap', 'q_px': q, 'observed_range_px': [-1000, 1000],
               'localization_error_px': localization, 'calibration_error_px': calibration if verified else None,
               'shared_motion_error_px': 50., 'fusion_error_px': 80.} for i, q in enumerate(qs)]
    lane = {'lane': 0, 'endpoints': points,
            'candidates': [{'id': f'c{i}', 'type': 'tap', 'start': p['id'], 'end': None,
                            'selected': i == 0, 'competitive': True} for i, p in enumerate(points)],
            'events': [{'status': 'ambiguous', 'candidate_ids': [f'c{i}' for i in range(len(points))]}]}
    decoded = SimpleNamespace(id='decode', manifest={'key': {'parents': ['obs', 'motion', 'fusion']}},
                              chunks=lambda: iter([{'payload': np.frombuffer(canonical(lane), np.uint8)}]))
    return obs, moved, project_video_times(obs, moved, decoded)


class EndpointTimeMapTest(unittest.TestCase):
    def test_variable_speed_crosses_each_bracket_with_exact_arithmetic(self):
        curve = EndpointTimeMap(clock([0, 10, 30], times=[100, 110, 150]))
        result = curve.project(5, 20)
        self.assertEqual(bounds(result), (105, 130))
        self.assertEqual(result['source_frame_id_range'], [0, 2])
        self.assertEqual(result['source_sample_count'], 3)
        fractional = EndpointTimeMap(clock([0, 3], times=[0, 1000])).project(1, 2)
        self.assertEqual(bounds(fractional), (Fraction(1000, 3), Fraction(2000, 3)))

    def test_holds_preserve_full_inverse_even_at_interval_edges(self):
        curve = EndpointTimeMap(clock([0, 8, 8, 8, 16], states=['origin', 'observed', 'repeated', 'repeated', 'observed']))
        for lower, upper, expected in ((8, 8, (10, 30)), (8, 12, (10, 35)), (4, 8, (5, 30)), (4, 12, (5, 35))):
            self.assertEqual(bounds(curve.project(lower, upper)), expected)
        self.assertEqual(bounds(EndpointTimeMap(clock([2], times=[123])).project(2, 2)), (123, 123))

    def test_entire_support_must_be_valid_including_interior_faults(self):
        for values, reason in (
            (clock([0, 10, 20, 30, 40], states=['origin', 'observed', 'ambiguous', 'observed', 'observed']), 'uncertain_motion'),
            (clock([0, 10, 20, 30, 40], ids=[0, 1, 3, 4, 5]), 'missing_frames'),
            (clock([0, 10, 20, 30, 40], times=[0, 10, 20, 20, 40]), 'nonincreasing_sample_times')):
            curve = EndpointTimeMap(values)
            self.assertIsNotNone(curve.nominal.project(5)['nominal_timestamp_ms'])
            self.assertIsNotNone(curve.nominal.project(35)['nominal_timestamp_ms'])
            result = curve.project(5, 35)
            self.assertEqual(result['status'], 'unavailable')
            self.assertIn(reason, result['reasons'])
            self.assertIsNone(result['time_range_ms'])
            self.assertEqual(result['source_sample_count'], 5)

    def test_outside_domain_is_not_clipped_and_unrelated_faults_do_not_block(self):
        curve = EndpointTimeMap(clock([0, 10, 20], ids=[0, 2, 3], states=['origin', 'observed', 'observed']))
        for lower, upper in ((-1, 10), (10, 21), (-1, 21)):
            result = curve.project(lower, upper)
            self.assertEqual(result['reasons'], ['outside_observed_scroll_range'])
            self.assertIsNone(result['time_range_ms'])
        self.assertEqual(bounds(curve.project(10, 20)), (10, 20))
        self.assertEqual(bounds(curve.project(10, 10)), (10, 10))

    def test_inverse_enclosure_matches_piecewise_segment_oracle(self):
        rng = random.Random(691)
        for _ in range(100):
            scroll, times = [0], [rng.randrange(-20, 20)]
            for _ in range(8):
                scroll.append(scroll[-1] + rng.randrange(0, 5))
                times.append(times[-1] + rng.randrange(1, 10))
            lower, upper = sorted(Fraction(rng.randrange(scroll[-1] * 4 + 1), 4) for _ in range(2))
            expected = []
            for q0, q1, t0, t1 in zip(scroll, scroll[1:], times, times[1:]):
                if q0 == q1:
                    if lower <= q0 <= upper: expected.extend((t0, t1))
                elif max(lower, q0) <= min(upper, q1):
                    expected.extend(Fraction(t0) + (q - q0) * (t1 - t0) / (q1 - q0)
                                    for q in (max(lower, q0), min(upper, q1)))
            result = EndpointTimeMap(clock(scroll, times=times)).project(lower, upper)
            self.assertEqual(bounds(result), (min(expected), max(expected)))

    def test_invalid_queries_and_source_mutations(self):
        values = clock([0, 10]); curve = EndpointTimeMap(values)
        values['scroll_px'][1] = 100
        self.assertEqual(bounds(curve.project(4, 6)), (4, 6))
        for lower, upper in ((2, 1), (True, 1), (0, float('inf')), (float('nan'), 1), ('0', 1)):
            with self.assertRaises(ValueError): curve.project(lower, upper)


class EndpointRangesTest(unittest.TestCase):
    def test_components_preserve_unknown_dependence_and_diagnostic_evidence(self):
        obs, moved, projection = inputs((10.,))
        before = deepcopy(projection)
        result = project_endpoint_time_ranges(obs, moved, projection)
        point = result['lanes'][0]['endpoints'][0]
        self.assertEqual(bounds(point['components']['calibration']), (Fraction('109.5'), 111))
        self.assertEqual(bounds(point['components']['localization']), (Fraction('109.75'), Fraction('110.5')))
        self.assertEqual(bounds(point['components']['calibration_and_localization']), (Fraction('109.25'), Fraction('111.5')))
        self.assertEqual(point['unpropagated_evidence']['observed_range_px'], [-1000, 1000])
        self.assertEqual(point['unpropagated_evidence']['endpoint_motion_error_px'], 50.)
        self.assertEqual(result['lanes'][0]['candidates'][0]['pixel_status'], 'ambiguous')
        self.assertEqual(result['endpoint_status_counts'], {'bounded': 1, 'unavailable': 0})
        self.assertEqual(projection, before)

    def test_unknown_location_is_not_zero_and_calibration_stays_separate(self):
        result = project_endpoint_time_ranges(*inputs(localization=None))
        point = result['lanes'][0]['endpoints'][0]
        self.assertEqual(point['status'], 'unavailable')
        self.assertEqual(point['components']['calibration']['status'], 'bounded')
        for name in ('localization', 'calibration_and_localization'):
            self.assertEqual(point['components'][name]['reasons'], ['localization_error_unknown'])
            self.assertIsNone(point['components'][name]['time_range_ms'])
        zero = project_endpoint_time_ranges(*inputs(calibration=0., localization=0.))
        self.assertEqual(bounds(zero['lanes'][0]['endpoints'][0]['components']['calibration_and_localization']), (105, 105))

    def test_unverified_unlocalized_and_outside_ranges_stay_unavailable(self):
        for data, reason in ((inputs(verified=False), 'endpoint_calibration_unverified'),
                             (inputs((None,)), 'endpoint_unlocalized'),
                             (inputs((0.,)), 'outside_observed_scroll_range')):
            result = project_endpoint_time_ranges(*data)
            point = result['lanes'][0]['endpoints'][0]
            self.assertEqual(point['status'], 'unavailable')
            self.assertIn(reason, point['components']['calibration_and_localization']['reasons'])
            self.assertIsNone(point['components']['calibration_and_localization']['time_range_ms'])

    def test_parent_tampering_duplicate_ids_and_bad_errors_are_rejected(self):
        for mode in ('motion', 'observation', 'calibration', 'location', 'time', 'duplicate', 'version'):
            obs, moved, projection = inputs()
            point = projection['lanes'][0]['endpoints'][0]
            if mode == 'motion': moved.manifest['key']['parents'] = ['other']
            if mode == 'observation': projection['observation_cache_id'] = 'other'
            if mode == 'calibration': point['calibration_error_px'] = 9
            if mode == 'location': point['localization_error_px'] = 9
            if mode == 'time': point['nominal_timestamp_ms'] += 1
            if mode == 'duplicate': projection['lanes'][0]['endpoints'].append(deepcopy(point))
            if mode == 'version': projection['algorithm_version'] = 'unknown'
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                project_endpoint_time_ranges(obs, moved, projection)
        for error in (-1, float('inf'), float('nan'), True):
            with self.subTest(error=error), self.assertRaises(ValueError):
                project_endpoint_time_ranges(*inputs(localization=error))

    def test_closed_exact_json_ranges_round_trip_and_holds_are_only_candidate_evidence(self):
        data = inputs((8.,), calibration=0., localization=0.,
                      values=clock([0, 8, 8, 16], states=['origin', 'observed', 'repeated', 'observed']))
        result = project_endpoint_time_ranges(*data)
        point = result['lanes'][0]['endpoints'][0]
        self.assertEqual(point['nominal_projection_status'], 'held')
        self.assertEqual(bounds(point['components']['calibration_and_localization']), (10, 20))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ranges.json'; write_json(path, result)
            self.assertEqual(read_json(path, 'ez2rc.endpoint-time-ranges'), result)


class EndpointRangesPipelineTest(unittest.TestCase):
    def test_reconstruction_exports_same_ranges_and_hashes_without_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_taps(root)
            code, first = execute(song, output_root=root / 'out'); self.assertEqual(code, 2)
            load = lambda run, name: json.loads((run / name).read_bytes())
            name = 'artifacts/endpoint-time-ranges.json'
            before = load(first, name)
            self.assertGreater(before['calibration_component_bounded_count'], 0)
            self.assertEqual(before['endpoint_status_counts']['bounded'], 0)
            (root / 'video.mkv').unlink()
            with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no video decoding')):
                code, run = execute(song, output_root=root / 'out', observation_cache=first / 'artifacts/observations')
            self.assertEqual(code, 2); self.assertEqual(before, load(run, name))
            for key in ('time_projection', 'motion'):
                self.assertEqual(before[key + '_sha256'], hashlib.sha256((run / before[key + '_ref']).read_bytes()).hexdigest())
            for other in ('chart.json', 'artifacts/events.json', 'artifacts/time-projection.json', 'artifacts/timing-resolution.json', 'artifacts/beat-ranges.json'):
                self.assertEqual(load(first, other), load(run, other))
            self.assertEqual(load(run, 'chart.json')['notes'], [])
            self.assertEqual(load(run, 'artifacts/events.json')['endpoint_time_ranges_ref'], name)
            self.assertIn(name, (run / 'report.html').read_text())
            stages = load(run, 'run.json')['stages']
            self.assertEqual(stages['timing']['endpoint_time_ranges_ref'], name)
            self.assertTrue(all(stages[s]['reused'] for s in ('capture', 'observation', 'motion', 'fusion', 'decode')))


if __name__ == '__main__':
    unittest.main()
