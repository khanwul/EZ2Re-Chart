"""Procedural temporal boundaries, provenance and scoped output contracts."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import cv2
from fractions import Fraction

import numpy as np

from ez2rc.v2.config import ConfigError, load_song, load_profile, load_algorithm
from ez2rc.v2.contracts import Event
from ez2rc.v2.observation import cache_keys, collect, open_observations
from ez2rc.v2.runner import execute
from ez2rc.v2.storage import CacheError, CacheWriter
from ez2rc.v2.window import analysis_scope, scope_events, song_window, window_settings
from v2_fixtures import fixture
from test_v2_timing_resolver import moving_taps, timing_input, resolve


def scene():
    profile = {'judge_y': 100, 'lanes': [{'roi': [0, 0, 20, 100], 'match_roi': [0, 0, 20, 100]}],
               'calibration': {k: {'status': 'unverified'} for k in ('tap', 'head', 'tail')}}
    clock = {'frame_id': np.arange(4), 'timestamp_ms': np.arange(4)*10.,
             'scroll_px': np.arange(4)*20., 'status': np.ones(4, int), 'error_px': np.ones(4)}
    chunk = {'frame_id': clock['frame_id'], 'timestamp_ms': clock['timestamp_ms'],
             'candidate_index': np.array([[i, 0, 0] for i in range(4)])}
    obs = SimpleNamespace(id='o', chunks=lambda: iter([chunk]), manifest={
        'key': {'parameters': {'profile': profile, 'template_shapes': [{k: [10, 10] for k in ('tap', 'head', 'tail')}]}},
        'metadata': {'profile': profile}})
    moved = SimpleNamespace(id='m', manifest={'key': {'parents': ['o']}})
    decoded = SimpleNamespace(id='d', manifest={'key': {'parents': ['o', 'm', 'f']}})
    lane = {'lane': 0, 'candidates': [], 'endpoints': [], 'events': []}
    # Direct core member, top fragment, fully later, bottom fragment, fully earlier.
    for i, (frame, q) in enumerate(((1, 70), (2, 125), (3, 160), (0, 25), (0, 15))):
        eid, cid = f'p{i}', f'c{i}'
        lane['endpoints'].append({'id': eid, 'kind': 'tap', 'observed_range_px': [q, q],
                                 'observation_ids': [f'f{frame}:l0:y0']})
        lane['candidates'].append({'id': cid, 'lane': 0, 'selected': True, 'competitive': True,
                                    'start': eid, 'end': None, 'body': None})
        lane['events'].append({'id': f'e{i}', 'candidate_ids': [cid], 'selected_candidate_ids': [cid]})
    return obs, moved, decoded, clock, lane


def project(data=None):
    obs, moved, decoded, clock, lane = data or scene()
    with patch('ez2rc.v2.motion.timeline', return_value=clock), patch('ez2rc.v2.decode.lanes', return_value=[lane]):
        return analysis_scope(obs, moved, decoded, window_settings(10., 20., 10., 20.))


class ObservationWindowScopeTest(unittest.TestCase):
    def test_direct_and_both_boundary_sides_remain_distinct_from_context_only(self):
        result = project()
        self.assertEqual([c['scope'] for c in result['candidates']],
                         ['requested', 'boundary_uncertain', 'auxiliary_only', 'boundary_uncertain', 'auxiliary_only'])
        self.assertEqual(result['requested_frame_count'], 1)
        self.assertEqual(result['auxiliary_frame_count'], 3)
        top = result['candidates'][1]
        self.assertEqual(top['endpoints']['start']['requested_observation_ids'], [])
        self.assertEqual(top['endpoints']['start']['auxiliary_observation_ids'], ['f2:l0:y0'])
        self.assertFalse(top['projected_overlap'][0]['verified_membership'])

    def test_touching_without_positive_overlap_does_not_enter_core(self):
        data = scene()
        data[4]['endpoints'][1]['observed_range_px'] = [130, 130]  # bottom == viewport top
        data[4]['endpoints'][3]['observed_range_px'] = [20, 20]  # top == viewport bottom
        result = project(data)
        self.assertEqual(result['candidates'][1]['scope'], 'auxiliary_only')
        self.assertEqual(result['candidates'][3]['scope'], 'auxiliary_only')

    def test_corrected_endpoint_coordinates_restore_template_origin(self):
        data = scene()
        data[0].manifest['metadata']['profile']['calibration']['tap'] = {'status': 'verified', 'offset_px': 8}
        for e in data[4]['endpoints']:
            e['observed_range_px'] = [v-8 for v in e['observed_range_px']]
        self.assertEqual(project(data), project())

    def test_body_envelope_and_missing_geometry_cannot_claim_direct_membership(self):
        data = scene()
        c = data[4]['candidates'][2]
        c.update(start=None, body={'frame_bounds': [0, 3]})
        result = project(data)['candidates'][2]
        self.assertEqual(result['scope'], 'boundary_uncertain')
        self.assertTrue(result['body_frame_envelope_intersects'])
        data = scene(); del data[0].manifest['key']['parameters']['template_shapes']
        self.assertEqual(project(data)['candidates'][2]['scope'], 'boundary_uncertain')

    def test_no_false_scope_from_unknown_raw_ids_or_mismatched_parents(self):
        for change in ('raw', 'parent', 'timeline'):
            data = scene()
            if change == 'raw': data[4]['endpoints'][0]['observation_ids'] = ['f1:l0:y999']
            elif change == 'parent': data[1].manifest['key']['parents'] = ['other']
            else: data[3]['frame_id'] = np.array([0, 1, 2, 9])
            with self.subTest(change=change), self.assertRaises(ValueError): project(data)

    def test_recorded_window_cannot_be_relabelled(self):
        data = scene()
        data[0].manifest['key']['parameters']['analysis_window'] = window_settings(0., 30., 0., 10.)
        with self.assertRaisesRegex(ValueError, 'recorded observation window'): project(data)

    def test_filter_preserves_selected_candidate_once_and_holds_boundary_resolution(self):
        scope = project()
        events = tuple(Event(f'e{i}', 0, 'incomplete', 'tap', candidate_ids=(f'c{i}',), reason='pending') for i in range(5))
        pixels = [{'id': e.id, 'status': 'resolved', 'reason': 'pixel', 'selected_intervals': [{'id': f'c{i}'}]}
                  for i, e in enumerate(events)]
        before = deepcopy(pixels)
        kept, output = scope_events(events, pixels, scope)
        self.assertEqual([e.id for e in kept], ['e0', 'e1', 'e3'])
        self.assertEqual([p['status'] for p in output], ['resolved', 'ambiguous', 'ambiguous'])
        self.assertEqual(pixels, before)
        self.assertEqual([p['selected_intervals'] for p in output], [pixels[i]['selected_intervals'] for i in (0, 1, 3)])

    def test_boundary_membership_cannot_be_promoted_even_with_verified_timing(self):
        events, pixels, projection = timing_input(('tap',))
        pixels[0]['reason'] = 'pixel'
        scope = {'events': [{'event_id': 'e0', 'scope': 'boundary_uncertain', 'included': True, 'hold_resolution': True}]}
        events, pixels = scope_events(events, pixels, scope)
        chart, events, _ = resolve((events, pixels, projection))
        self.assertEqual(chart.notes, ())
        self.assertEqual(events[0].status, 'ambiguous')

    def test_core_alternative_does_not_license_context_only_selected_note(self):
        data = scene(); lane = data[4]
        lane['candidates'][0]['selected'] = False
        lane['events'] = [{'id': 'mixed', 'candidate_ids': ['c0', 'c2'], 'selected_candidate_ids': ['c2']},
                          *[e for e in lane['events'] if e['id'] in ('e1', 'e3', 'e4')]]
        event = project(data)['events'][0]
        self.assertEqual(event['scope'], 'requested')
        self.assertTrue(event['hold_resolution'])


class ObservationWindowPipelineTest(unittest.TestCase):
    def test_vfr_range_context_cache_identity_and_no_duplicate_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = fixture(root)
            song = replace(load_song(path), start_ms=30., end_ms=101., context_before_ms=30., context_after_ms=40.)
            profile, algo = load_profile(song.profile), load_algorithm(song.algorithm)
            capture, obs, reused = collect(song, profile, algo, root/'cache')
            self.assertFalse(reused)
            ids = [int(f) for c in obs.chunks() for f in c['frame_id']]
            self.assertEqual(ids, list(range(6)))
            self.assertEqual(obs.manifest['metadata']['requested_frame_count'], 3)
            self.assertEqual(obs.manifest['metadata']['auxiliary_frame_count'], 3)
            self.assertEqual(capture.manifest['key']['parameters']['start_ms'], 30.)
            self.assertEqual(capture.manifest['key']['parameters']['end_ms'], 101.)
            with patch('ez2rc.v2.observation.VideoReader', side_effect=AssertionError('reuse')):
                self.assertTrue(collect(song, profile, algo, root/'cache')[2])
            different = replace(song, start_ms=20., context_before_ms=20.)
            self.assertEqual(song_window(song)['observation'], song_window(different)['observation'])
            self.assertNotEqual(cache_keys(song, profile, algo, 'v'), cache_keys(different, profile, algo, 'v'))

    def test_capture_rejects_empty_core_even_if_context_has_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = fixture(root)
            song = replace(load_song(path), start_ms=150., end_ms=180., context_before_ms=150.)
            with self.assertRaisesRegex(ValueError, 'no frames in the requested'):
                collect(song, load_profile(song.profile), load_algorithm(song.algorithm), root/'cache')
            self.assertFalse(list((root/'cache').glob('*/*/manifest.json')))

    def test_invalid_context_and_observation_manifest_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = fixture(root); song = load_song(path)
            for value in (-1, True, float('nan'), float('inf')):
                with self.subTest(value=value), self.assertRaises(ConfigError): replace(song, context_after_ms=value)
            song = replace(song, start_ms=30., end_ms=101., context_after_ms=40.)
            _, obs, _ = collect(song, load_profile(song.profile), load_algorithm(song.algorithm), root/'cache')
            for change in ('window', 'count'):
                meta = deepcopy(obs.manifest['metadata'])
                if change == 'window': meta['analysis_window']['requested']['end_ms'] = 90.
                else: meta['requested_frame_count'] += 1
                with CacheWriter(root/change, obs.manifest['key']) as writer:
                    for c in obs.chunks(): writer.append(**c)
                    bad = writer.commit(meta)
                with self.assertRaises(CacheError): open_observations(bad.path)

    def test_extract_reconstruct_scoped_report_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = moving_taps(root)
            path.write_text(path.read_text().replace('[input]', '[input]\nstart_ms = 200.0\nend_ms = 500.0\ncontext_before_ms = 100.0\ncontext_after_ms = 140.0'))
            code, run = execute(path, output_root=root/'out')
            self.assertEqual(code, 2, (run/'run.json').read_text())
            scope = json.loads((run/'artifacts/analysis-scope.json').read_text())
            self.assertEqual(scope['requested_frame_count'], 15)
            self.assertEqual(scope['auxiliary_frame_count'], 12)
            exported = json.loads((run/'artifacts/events.json').read_text())
            included = {e['event_id'] for e in scope['events'] if e['included']}
            self.assertEqual({e['id'] for e in exported['events']}, included)
            self.assertEqual(exported['analysis_scope_ref'], 'artifacts/analysis-scope.json')
            self.assertIn('Requested observation window', (run/'report.html').read_text())
            for name in ('video.mkv', 'profile.toml', 'note.png', 'head.png', 'tail.png'): (root/name).unlink()
            code, replay = execute(path, output_root=root/'out', observation_cache=run/'artifacts/observations')
            self.assertEqual(code, 2, (replay/'run.json').read_text())
            for name in ('analysis-scope.json', 'events.json', 'timing-resolution.json'):
                self.assertEqual((run/'artifacts'/name).read_bytes(), (replay/'artifacts'/name).read_bytes())
            path.write_text(path.read_text().replace('context_after_ms = 140.0', 'context_after_ms = 160.0'))
            code, mismatch = execute(path, output_root=root/'out', observation_cache=run/'artifacts/observations')
            self.assertEqual(code, 1)
            self.assertIn('interval', json.loads((mismatch/'run.json').read_text())['failure']['message'])

    def test_partial_sprite_recovers_with_context_without_duplicate_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path = moving_taps(root)
            template = cv2.imread(str(root/'note.png'))
            # Independently generated partial sprites, including two visible rows
            # at the requested end. Motion has ordinary full sprites in both lanes.
            with av.open(str(root/'clipped.mkv'), 'w') as container:
                stream = container.add_stream('ffv1', rate=50)
                stream.width = stream.height = 96
                stream.pix_fmt = 'bgr0'
                stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
                for i in range(25):
                    image = np.zeros((96, 96, 3), np.uint8)
                    image[70:76, 20:61] = (220, 120, 20)
                    for x, shift in ((22, 0), (42, 13)):
                        for gap in (0, 42, 93):
                            y = 10 + 3*i - shift - gap
                            lo, hi = max(0, y), min(70, y+7)
                            if hi > lo: image[lo:hi, x:x+16] = template[lo-y:hi-y]
                    frame = av.VideoFrame.from_ndarray(image, format='bgr24')
                    frame.pts, frame.time_base = 1000+i*20, Fraction(1, 1000)
                    for packet in stream.encode(frame): container.mux(packet)
                for packet in stream.encode(): container.mux(packet)
            path.write_text(path.read_text().replace('video.mkv', 'clipped.mkv').replace('[input]', '[input]\nend_ms = 200.0'))
            _, baseline = execute(path, output_root=root/'out')
            path.write_text(path.read_text().replace('[input]', '[input]\ncontext_after_ms = 120.0'))
            code, extended = execute(path, output_root=root/'out')
            self.assertEqual(code, 2, (extended/'run.json').read_text())
            from ez2rc.v2.decode import lanes, open_decode
            def selected(run):
                rows = []
                for lane in lanes(open_decode(run/'artifacts/decode')):
                    ends = {e['id']: e for e in lane['endpoints']}
                    rows.extend((c, ends[c['start']]) for c in lane['candidates']
                                if c['selected'] and c['lane'] == 0 and c['type'] == 'tap')
                return rows
            before, after = selected(baseline), selected(extended)
            recovered = [(c, e) for c, e in after if any(o.startswith('f13:') for o in e['observation_ids'])
                         and e['observed_q_px'] > 100]
            self.assertEqual(len(recovered), 1)
            self.assertFalse(any(e['observed_q_px'] > 100 for _, e in before))
            scope = json.loads((extended/'artifacts/analysis-scope.json').read_text())
            scoped = next(c for c in scope['candidates'] if c['candidate_id'] == recovered[0][0]['id'])
            self.assertIn(scoped['scope'], ('requested', 'boundary_uncertain'))
            self.assertTrue(scoped['endpoints']['start']['auxiliary_observation_ids'])

    def test_zero_context_retains_original_cache_keys_and_unbounded_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = fixture(root); song = load_song(path)
            keys = cache_keys(song, load_profile(song.profile), load_algorithm(song.algorithm), 'v')
            self.assertNotIn('analysis_window', keys[0]['parameters'])
            self.assertNotIn('analysis_window', keys[1]['parameters'])
            self.assertIsNone(window_settings(0., None, 50., 50.)['observation']['end_ms'])
            self.assertEqual(window_settings(10., 20., 50., 0.)['observation']['start_ms'], 0.)


if __name__ == '__main__': unittest.main()
