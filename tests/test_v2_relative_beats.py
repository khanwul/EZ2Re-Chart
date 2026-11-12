"""Procedural spatial references with independently varied capture clocks."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ez2rc.v2.config import ConfigError, load_song
from ez2rc.v2.decode import lanes, open_decode
from ez2rc.v2.export import read_chart, read_events
from ez2rc.v2.relative_beats import SpatialBeatMap, load_reference, project_relative_beats, unpack, validate_reference
from ez2rc.v2.runner import execute
from test_v2_timing_resolver import moving_taps


def reference():
    return {'format': 'ez2rc.spatial-beat-reference', 'schema_version': 1, 'version': 'procedural',
            'origin': 'first_barline_zero', 'measure_beats': [3, 1],
            'observation_cache_id': 'obs', 'motion_cache_id': 'motion', 'decode_cache_id': 'decode',
            'extrapolation': 'withhold',
            'anchors': [{'id': 'a', 'q_px': 0, 'q_range_px': [0, 0], 'beat': [0, 1]},
                        {'id': 'b', 'q_px': 120, 'q_range_px': [120, 120], 'beat': [3, 1]},
                        {'id': 'c', 'q_px': 360, 'q_range_px': [360, 360], 'beat': [6, 1]}],
            'endpoint_calibrations': [{'endpoint_id': eid, 'candidate_id': 'ln', 'version': 'procedural',
                                       'offset_px': [4, 4], 'position_error_px': 0} for eid in ('h', 't')]}


def scene():
    calibration = {k: {'status': 'unverified', 'version': 'unset'} for k in ('tap', 'head', 'tail')}
    obs = SimpleNamespace(id='obs', manifest={'metadata': {'profile': {'calibration': calibration}}})
    movement = SimpleNamespace(id='motion', manifest={'key': {'parents': ['obs']}})
    decoded = SimpleNamespace(id='decode', manifest={'key': {'parents': ['obs', 'motion', 'fusion']}})
    clock = {'frame_id': np.arange(5), 'timestamp_ms': np.array([100, 130, 180, 200, 280.]),
             'scroll_px': np.array([0, 60, 120, 240, 360.]), 'status': np.ones(5, np.uint8), 'error_px': np.zeros(5)}
    lane = {'lane': 0, 'events': [{'id': 'e', 'status': 'resolved', 'selected_candidate_ids': ['ln']}],
            'candidates': [{'id': 'ln', 'type': 'longnote', 'start': 'h', 'end': 't', 'selected': True}],
            'endpoints': [{'id': eid, 'kind': kind, 'q_px': q, 'observation_ids': [eid+'raw'],
                           'shared_motion_error_px': 2., 'fusion_error_px': 1., 'localization_error_px': None}
                          for eid, kind, q in (('h', 'head', 64.), ('t', 'tail', 244.))]}
    return obs, movement, decoded, clock, lane


def project(ref=None, data=None):
    obs, moved, decoded, clock, lane = data or scene()
    with patch('ez2rc.v2.relative_beats.lanes', return_value=[lane]), patch('ez2rc.v2.relative_beats.timeline', return_value=clock):
        return project_relative_beats(obs, moved, decoded, ref or reference(), measure_beats=F(3))


class SpatialMapTest(unittest.TestCase):
    def test_piecewise_spatial_scale_and_capture_clock_are_independent(self):
        first = project()['candidates'][0]
        self.assertEqual(unpack(first['endpoints']['start']['relative_beat']['beat_range']), (F(3, 2),)*2)
        self.assertEqual(unpack(first['endpoints']['end']['relative_beat']['beat_range']), (F(9, 2),)*2)
        self.assertEqual(unpack(first['duration_beat_range']), (F(3),)*2)
        data = scene(); data[3]['timestamp_ms'] = np.array([900., 915, 1000, 1400, 2000])
        second = project(data=data)['candidates'][0]
        for side in ('start', 'end'):
            self.assertEqual(first['endpoints'][side]['relative_beat'], second['endpoints'][side]['relative_beat'])
            self.assertNotEqual(first['endpoints'][side]['capture_time'], second['endpoints'][side]['capture_time'])
        self.assertNotEqual(first['capture_duration_ms'], second['capture_duration_ms'])
        self.assertEqual(first['duration_beat_range'], second['duration_beat_range'])

    def test_anchor_enclosure_covers_dense_independent_assignments_and_extrapolation(self):
        anchors = reference()['anchors']
        for a in anchors:
            a['q_range_px'] = [a['q_px']-2, a['q_px']+3]
        model = SpatialBeatMap(anchors, 'conditional_linear')
        for q in map(F, (-20, 0, 1, 60, 118, 120, 122, 240, 360, 400)):
            lo, hi = unpack(model.project(q, q)['beat_range'])
            for a in (-2, 0, 3):
                for b in (118, 120, 123):
                    for c in (358, 360, 363):
                        xs = [F(a), F(b), F(c)]
                        i = 0 if q <= b else 1
                        expected = 3*i + (q-xs[i])*3/(xs[i+1]-xs[i])
                        self.assertLessEqual(lo, expected)
                        self.assertGreaterEqual(hi, expected)

    def test_missing_support_and_extrapolation_stay_explicit(self):
        ref = reference()
        self.assertEqual(SpatialBeatMap([]).project(F(0), F(1))['reasons'], ['insufficient_barline_anchors'])
        self.assertEqual(SpatialBeatMap(ref['anchors']).project(F(-1), F(1))['status'], 'unavailable')
        value = SpatialBeatMap(ref['anchors'], 'conditional_linear').project(F(-1), F(1))
        self.assertEqual(value['coverage'], 'extrapolated')
        self.assertIn('barline_extrapolation', value['reasons'])

    def test_conflicting_anchors_and_note_labels_are_rejected(self):
        for change in ('overlap', 'reverse', 'fractional_bar', 'note_label', 'duplicate', 'nonfinite', 'schema_bool'):
            ref = reference()
            if change == 'overlap': ref['anchors'][1]['q_range_px'] = [0, 120]
            if change == 'reverse': ref['anchors'].reverse()
            if change == 'fractional_bar': ref['anchors'][1]['beat'] = [1, 2]
            if change == 'note_label': ref['endpoint_calibrations'][0]['beat'] = [1, 1]
            if change == 'duplicate': ref['endpoint_calibrations'].append(ref['endpoint_calibrations'][0])
            if change == 'nonfinite': ref['anchors'][0]['q_px'] = float('nan')
            if change == 'schema_bool': ref['schema_version'] = True
            with self.subTest(change=change), self.assertRaises(ValueError): validate_reference(ref)

    def test_offsets_replace_prior_profile_offset_without_double_application(self):
        before = project()
        data = scene()
        for kind in ('head', 'tail'):
            data[0].manifest['metadata']['profile']['calibration'][kind].update(status='verified', offset_px=3, error_px=1)
        for e in data[4]['endpoints']: e['q_px'] -= 3
        after = project(data=data)
        for side in ('start', 'end'):
            self.assertEqual(before['candidates'][0]['endpoints'][side]['coordinate_range_px'],
                             after['candidates'][0]['endpoints'][side]['coordinate_range_px'])

    def test_unknown_localization_does_not_become_zero(self):
        ref = reference(); ref['endpoint_calibrations'][0]['position_error_px'] = None
        row = project(ref)['candidates'][0]
        self.assertEqual(row['status'], 'unavailable')
        self.assertIsNone(row['duration_beat_range'])
        self.assertIn('localization_error_unknown', row['reasons'])

    def test_ambiguous_selection_retains_ranges_without_acceptance(self):
        data = scene(); data[4]['events'][0]['status'] = 'ambiguous'
        row = project(data=data)['candidates'][0]
        self.assertEqual(row['pixel_status'], 'ambiguous')
        self.assertEqual(row['status'], 'conditional')
        self.assertFalse(row['accepted'])
        self.assertIn('pixel_ambiguous', row['reasons'])

    def test_capture_gaps_do_not_erase_spatial_coordinates(self):
        data = scene(); data[3]['frame_id'] = np.array([0, 1, 2, 4, 5])
        data[4]['endpoints'][1]['q_px'] = 234.
        row = project(data=data)['candidates'][0]
        self.assertEqual(row['status'], 'conditional')
        self.assertIsNone(row['capture_duration_ms'])
        self.assertIn('missing_frames', row['endpoints']['end']['capture_time']['reasons'])

    def test_parent_and_selected_candidate_binding_fail_closed(self):
        for key in ('observation_cache_id', 'motion_cache_id', 'decode_cache_id'):
            ref = reference(); ref[key] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'lineage'): project(ref)
        ref = reference(); ref['endpoint_calibrations'][0]['candidate_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'candidate mismatch'): project(ref)
        ref = reference(); ref['endpoint_calibrations'][0]['endpoint_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'does not bind'): project(ref)

    def test_nonpositive_duration_is_preserved_not_clipped(self):
        ref = reference()
        for e in ref['endpoint_calibrations']: e['position_error_px'] = 110
        ref['extrapolation'] = 'conditional_linear'
        row = project(ref)['candidates'][0]
        self.assertLess(unpack(row['duration_beat_range'])[0], 0)
        self.assertIn('duration_may_be_nonpositive', row['reasons'])

    def test_hash_checked_and_missing_anchors_keeps_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'input.json'; p.write_text(json.dumps(reference()))
            request = {'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            self.assertEqual(load_reference(request), reference())
            p.write_text(p.read_text()+' ')
            with self.assertRaisesRegex(ValueError, 'checksum'): load_reference(request)
        ref = reference(); ref['anchors'] = []
        row = project(ref)['candidates'][0]
        self.assertIsNone(row['duration_beat_range'])
        self.assertIsNotNone(row['capture_duration_ms'])

    def test_common_bar_number_shift_cancels_in_duration(self):
        ref = reference()
        for a in ref['anchors']:
            a['q_range_px'] = [a['q_px']-1, a['q_px']+1]
        first = project(ref)['candidates'][0]
        for a in ref['anchors']: a['beat'][0] += 9
        second = project(ref)['candidates'][0]
        self.assertEqual(first['duration_beat_range'], second['duration_beat_range'])
        for side in ('start', 'end'):
            before = unpack(first['endpoints'][side]['relative_beat']['beat_range'])
            after = unpack(second['endpoints'][side]['relative_beat']['beat_range'])
            self.assertEqual(tuple(b-a for a,b in zip(before,after)), (F(9),F(9)))

    def test_profile_calibration_and_missing_longnote_endpoint(self):
        ref = reference(); ref['endpoint_calibrations'] = []
        data = scene()
        for kind in ('head', 'tail'):
            data[0].manifest['metadata']['profile']['calibration'][kind].update(status='verified', offset_px=4, error_px=1)
        for e in data[4]['endpoints']:
            e['q_px'] -= 4; e['localization_error_px'] = 2
        row = project(ref,data)['candidates'][0]
        self.assertEqual(unpack(row['endpoints']['start']['coordinate_range_px']), (F(57),F(63)))
        data[4]['candidates'][0]['end'] = None
        data[4]['events'][0]['status'] = 'incomplete'
        row = project(ref,data)['candidates'][0]
        self.assertEqual(row['status'], 'unavailable')
        self.assertIsNone(row['duration_beat_range'])
        self.assertIsNotNone(row['endpoints']['start']['relative_beat']['beat_range'])
        self.assertIn('missing_or_unlocalized_endpoint', row['reasons'])


class RelativePipelineTest(unittest.TestCase):
    def test_extract_reconstruct_exports_and_no_media_required(self):
        self.roundtrip(False)

    def test_motion_coordinate_dependence_reconstructs_from_observations(self):
        self.roundtrip(True)

    def test_redundant_spatial_constraints_reconstruct_from_observations(self):
        self.roundtrip('graph')

    def roundtrip(self, shared):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_taps(root)
            text = song.read_text().replace('bpm = "191.910"', 'bpm = "120"\nmeasure_beats = [3, 1]')
            song.write_text(text)
            code, first = execute(song, output_root=root/'out')
            self.assertEqual(code, 2)
            run = json.loads((first/'run.json').read_text())
            ref = reference(); ref['endpoint_calibrations'] = []
            for stage in ('observation', 'motion', 'decode'):
                ref[stage+'_cache_id'] = run['stages'][stage]['cache_id']
            for lane in lanes(open_decode(first/'artifacts/decode')):
                ep = {e['id']: e for e in lane['endpoints']}
                seen = set()
                for c in lane['candidates']:
                    if not c['selected']: continue
                    for eid in (c['start'], c['end']):
                        if eid and eid not in seen and ep[eid]['q_px'] is not None:
                            ref['endpoint_calibrations'].append({'endpoint_id': eid, 'candidate_id': c['id'],
                                'version': 'procedural', 'offset_px': [0, 0], 'position_error_px': 1})
                            seen.add(eid)
            ref['extrapolation'] = 'conditional_linear'
            if shared:
                from ez2rc.v2.motion import open_motion, timeline
                from ez2rc.v2.motion_dependence import VERSION
                clock = timeline(open_motion(first/'artifacts/motion'))
                supports = {}
                for anchor, index in zip(ref['anchors'], (5, 15, 25)):
                    fid = int(clock['frame_id'][index])
                    y = 20.
                    q = float(clock['scroll_px'][index]) + run['input']['profile']['judge_y'] - y
                    anchor.update(q_px=q, q_range_px=[q, q])
                    supports[anchor['id']] = [{'frame_id': fid, 'y_px': y, 'weight': [1, 1]}]
                ref['motion_dependence'] = {'version': VERSION, 'observation_error_px': 1.,
                                            'anchor_observations': supports,
                                            'endpoint_position_relation': 'same_raw_localization'}
                if shared == 'graph':
                    from ez2rc.v2.observed_geometry import VERSION as GEOMETRY_VERSION
                    ref.pop('motion_dependence')
                    ref['observed_geometry'] = {'version': GEOMETRY_VERSION, 'localization_error_px': 1.,
                        'scale': 'constant', 'endpoint_position_relation': 'same_raw_localization',
                        'anchor_observations': {k:[{'frame_id':p['frame_id'],'y_px':p['y_px']} for p in v]
                                                for k,v in supports.items()}}
            p = root/'spatial.json'; p.write_text(json.dumps(ref))
            checksum = hashlib.sha256(p.read_bytes()).hexdigest()
            song.write_text(text+f'\n[timing.relative_beats]\npath = "spatial.json"\nsha256 = "{checksum}"\n')
            self.assertEqual(load_song(song).snapshot()['relative_beats']['sha256'], checksum)
            code, extracted = execute(song, output_root=root/'out')
            self.assertEqual(code, 2)
            for name in ('video.mkv', 'profile.toml', 'note.png', 'head.png', 'tail.png'): (root/name).unlink()
            with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no decoding')):
                code, reconstructed = execute(song, output_root=root/'out', observation_cache=first/'artifacts/observations')
            self.assertEqual(code, 2)
            a = json.loads((extracted/'artifacts/relative-beats.json').read_text())
            b = json.loads((reconstructed/'artifacts/relative-beats.json').read_text())
            self.assertEqual(a, b)
            rhythm_a = json.loads((extracted/'artifacts/rhythm-candidates.json').read_text())
            rhythm_b = json.loads((reconstructed/'artifacts/rhythm-candidates.json').read_text())
            self.assertEqual(rhythm_a, rhythm_b)
            self.assertEqual(rhythm_a['relative_beats_sha256'], hashlib.sha256((extracted/'artifacts/relative-beats.json').read_bytes()).hexdigest())
            self.assertEqual(read_chart(reconstructed/'chart.json')['rhythm_candidates_ref'], 'artifacts/rhythm-candidates.json')
            song.write_text(song.read_text()+'\n[timing.rhythm_grid]\nsubdivisions = [2, 3]\n')
            code, changed = execute(song, output_root=root/'out', observation_cache=first/'artifacts/observations')
            self.assertEqual(code, 2)
            rhythm_changed = json.loads((changed/'artifacts/rhythm-candidates.json').read_text())
            self.assertNotEqual(rhythm_changed['identity'], rhythm_a['identity'])
            if shared == 'graph':
                self.assertNotEqual(rhythm_changed['relative_beats_sha256'], rhythm_a['relative_beats_sha256'])
                self.assertEqual(a['rhythm_refinement']['grid'], rhythm_a['grid'])
            else:
                self.assertEqual(rhythm_changed['relative_beats_sha256'], rhythm_a['relative_beats_sha256'])
            changed_run = json.loads((changed/'run.json').read_text())
            for stage in ('observation', 'motion', 'fusion', 'decode'):
                self.assertEqual(changed_run['stages'][stage]['cache_id'], run['stages'][stage]['cache_id'])
            self.assertGreater(a['conditional_candidate_count'], 0)
            if shared == 'graph':
                self.assertEqual(a['observed_geometry']['status'], 'conditional')
                self.assertFalse(a['observed_geometry']['verified'])
                lo, hi = unpack(a['observed_geometry']['scale_px_per_beat'])
                self.assertLessEqual(lo, 10)
                self.assertGreaterEqual(hi, 10)
                self.assertTrue(all(p['relative_beat']['dependency_model'] == GEOMETRY_VERSION
                                    for r in a['candidates'] for p in r['endpoints'].values()))
            elif shared:
                self.assertIsNotNone(a['motion_dependence'])
                self.assertTrue(a['coordinate_uncertainty']['sources'])
                self.assertTrue(a['motion_dependence']['shared_endpoint_position_ids'])
                from ez2rc.v2.observation import open_observations
                independent = deepcopy(ref)
                independent.pop('motion_dependence')
                prior = project_relative_beats(open_observations(first/'artifacts/observations'),
                    open_motion(first/'artifacts/motion'), open_decode(first/'artifacts/decode'),
                    independent, measure_beats=F(3))
                self.assertEqual([p['capture_time'] for r in a['candidates'] for p in r['endpoints'].values()],
                                 [p['capture_time'] for r in prior['candidates'] for p in r['endpoints'].values()])
            self.assertEqual(read_chart(reconstructed/'chart.json')['notes'], [])
            self.assertEqual(read_events(first/'artifacts/events.json')['events'], read_events(reconstructed/'artifacts/events.json')['events'])
            self.assertIn('relative-beats.json', (reconstructed/'report.html').read_text())
            self.assertIn('relative_beats_ref', read_chart(reconstructed/'chart.json'))
            p.write_text(p.read_text()+' ')
            code, failed = execute(song, output_root=root/'out', observation_cache=first/'artifacts/observations')
            self.assertEqual(code, 1)
            self.assertIn('checksum', json.loads((failed/'run.json').read_text())['failure']['message'])


if __name__ == '__main__':
    unittest.main()
