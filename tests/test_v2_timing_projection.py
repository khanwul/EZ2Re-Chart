"""Procedural scroll curves with holds, uncertain links and irregular timestamps."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ez2rc.v2.motion import STATUS
from ez2rc.v2.runner import execute
from ez2rc.v2.storage import canonical
from ez2rc.v2.timing import ScrollTimeMap, project_video_times
from v2_fixtures import fixture


def clock(scroll, *, times=None, ids=None, states=None):
    count = len(scroll)
    return {'frame_id': np.arange(count) if ids is None else np.array(ids),
            'timestamp_ms': np.arange(count) * 10. if times is None else np.array(times, float),
            'scroll_px': np.array(scroll, float),
            'status': np.array([STATUS['origin'], *[STATUS['observed']] * (count - 1)] if states is None else [STATUS[s] for s in states]),
            'error_px': np.arange(count) * .5}


class ScrollTimeMapTest(unittest.TestCase):
    def test_variable_speed_and_pts_use_each_observed_bracket(self):
        curve = ScrollTimeMap(clock([0, 10, 30, 45], times=[100, 110, 150, 160]))
        for q, expected, bracket in ((5, 105, [100, 110]), (20, 130, [110, 150]), (37.5, 155, [150, 160])):
            result = curve.project(q)
            self.assertEqual(result['nominal_timestamp_ms'], expected)
            self.assertEqual(result['bracket_ms'], bracket)
            self.assertEqual(result['status'], 'interpolated')
        self.assertEqual(curve.project(30)['status'], 'sample')
        self.assertEqual(curve.project(30)['source_frame_ids'], [2])

    def test_held_display_retains_full_time_range(self):
        curve = ScrollTimeMap(clock([0, 8, 8, 8, 16], states=['origin', 'observed', 'repeated', 'repeated', 'observed']))
        held = curve.project(8)
        self.assertEqual(held['status'], 'held')
        self.assertIsNone(held['nominal_timestamp_ms'])
        self.assertEqual(held['bracket_ms'], [10, 30])
        self.assertEqual(held['source_frame_ids'], [1, 2, 3])
        self.assertEqual(curve.project(12)['nominal_timestamp_ms'], 35)

    def test_uncertain_motion_does_not_produce_precise_interpolation(self):
        for status in ('single_lane', 'ambiguous', 'extrapolated', 'unresolved'):
            with self.subTest(status=status):
                curve = ScrollTimeMap(clock([0, 10, 20], states=['origin', status, 'observed']))
                for q in (5, 10, 15):
                    result = curve.project(q)
                    self.assertEqual(result['status'], 'unavailable')
                    self.assertIsNone(result['nominal_timestamp_ms'])
                    self.assertIn('uncertain_motion', result['reasons'])

    def test_missing_frame_and_zero_duration_motion_keep_diagnostic_brackets(self):
        for values, reason in ((clock([0, 20], ids=[0, 2]), 'missing_frames'),
                               (clock([0, 20], times=[50, 50]), 'nonincreasing_sample_times')):
            result = ScrollTimeMap(values).project(10)
            self.assertEqual(result['status'], 'unavailable')
            self.assertIn(reason, result['reasons'])
            self.assertIsNone(result['nominal_timestamp_ms'])

    def test_observed_range_is_not_extrapolated_and_origin_is_not_beat_zero(self):
        curve = ScrollTimeMap(clock([0, 10], times=[1250, 1260]))
        self.assertEqual(curve.project(0)['nominal_timestamp_ms'], 1250)
        self.assertEqual(curve.project(10)['nominal_timestamp_ms'], 1260)
        for q in (-.01, 10.01):
            result = curve.project(q)
            self.assertEqual(result['status'], 'outside_range')
            self.assertIsNone(result['nominal_timestamp_ms'])
            self.assertIsNone(result['bracket_ms'])

    def test_invalid_curves_and_queries_are_rejected(self):
        for values in (clock([]), clock([0, -1]), clock([0, 1], times=[2, 1]),
                       clock([0, float('nan')]), clock([0, 1], ids=[0, 0])):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ScrollTimeMap(values)
        values = clock([0, 1]); values['frame_id'] = np.array([1, 0], np.uint64)
        with self.assertRaises(ValueError):
            ScrollTimeMap(values)
        curve = ScrollTimeMap(clock([0, 1]))
        for q in (float('nan'), float('inf'), True):
            with self.subTest(q=q), self.assertRaises(ValueError):
                curve.project(q)

    def test_source_mutation_does_not_change_prepared_curve(self):
        values = clock([0, 10])
        curve = ScrollTimeMap(values)
        values['scroll_px'][1] = 100
        self.assertEqual(curve.project(5)['nominal_timestamp_ms'], 5)

    def test_unlocalized_endpoint_and_verified_offset_do_not_resolve_timing(self):
        obs = SimpleNamespace(id='obs', manifest={'metadata': {'profile': {'calibration': {
            kind: {'status': 'verified'} for kind in ('tap', 'head', 'tail')}}}})
        moved = SimpleNamespace(id='motion', manifest={'key': {'parents': ['obs']}},
                                chunks=lambda: iter([clock([0, 20])]))
        endpoints = [{'id': f'e{i}', 'kind': 'tap' if i == 0 else 'head', 'q_px': q,
                      'observed_range_px': [9, 11], 'localization_error_px': None,
                      'calibration_error_px': .5, 'shared_motion_error_px': 2., 'fusion_error_px': 3.}
                     for i, q in enumerate((10., None))]
        lane = {'lane': 0, 'endpoints': endpoints,
                'candidates': [{'id': f'c{i}', 'type': 'tap' if i == 0 else 'longnote',
                                'start': e['id'], 'end': None, 'competitive': True, 'selected': i == 0}
                               for i, e in enumerate(endpoints)],
                'events': [{'status': 'ambiguous', 'candidate_ids': ['c0', 'c1']}]}
        decoded = SimpleNamespace(id='decode', manifest={'key': {'parents': ['obs', 'motion', 'fusion']}},
                                  chunks=lambda: iter([{'payload': np.frombuffer(canonical(lane), np.uint8)}]))
        result = project_video_times(obs, moved, decoded)
        self.assertEqual(result['status'], 'provisional')
        self.assertEqual(result['timing_status'], 'pending')
        self.assertEqual(result['endpoint_status_counts'], {'interpolated': 1, 'unlocalized': 1})
        self.assertEqual(result['lanes'][0]['endpoints'][0]['nominal_timestamp_ms'], 5)
        self.assertIsNone(result['lanes'][0]['endpoints'][1]['nominal_timestamp_ms'])
        self.assertEqual(result['lanes'][0]['candidates'][0]['pixel_status'], 'ambiguous')
        moved.manifest['key']['parents'] = ['other-observations']
        with self.assertRaisesRegex(ValueError, 'lineage'):
            project_video_times(obs, moved, decoded)

    def test_pipeline_exports_same_projections_when_reconstructed_without_video(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = fixture(root)
            code, first = execute(path, output_root=root / 'out')
            self.assertEqual(code, 2)
            before = json.loads((first / 'artifacts/time-projection.json').read_text())
            self.assertEqual(before['status'], 'provisional')
            self.assertEqual(before['timing_status'], 'pending')
            self.assertTrue(before['endpoint_status_counts'])
            self.assertTrue(all(e['calibration_status'] == 'unverified' for lane in before['lanes'] for e in lane['endpoints']))
            (root / 'video.mkv').unlink()
            with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no video decode')):
                code, again = execute(path, output_root=root / 'out', observation_cache=first / 'artifacts/observations')
            self.assertEqual(code, 2)
            self.assertEqual(before, json.loads((again / 'artifacts/time-projection.json').read_text()))
            events = json.loads((again / 'artifacts/events.json').read_text())
            self.assertEqual(events['time_projection_ref'], 'artifacts/time-projection.json')
            self.assertTrue(all(e['start_ms'] is None and e['end_ms'] is None for e in events['events']))
            chart = json.loads((again / 'chart.json').read_text())
            self.assertEqual(chart['notes'], [])
            self.assertEqual(chart['pending_stages'], ['timing'])
            run = json.loads((again / 'run.json').read_text())
            self.assertEqual(run['stages']['timing']['status'], 'pending')


if __name__ == '__main__':
    unittest.main()
