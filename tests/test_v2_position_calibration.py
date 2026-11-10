"""Explicit localization declarations through procedural video and cache paths."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import av
import cv2
import numpy as np

from ez2rc.v2.config import ConfigError, DecodeConfig, PositionCalibration, load_algorithm, load_profile, load_song
from ez2rc.v2.decode import infer, lanes, open_decode
from ez2rc.v2.fusion import open_fusion
from ez2rc.v2.observation import cache_keys, collect, open_observations
from ez2rc.v2.runner import execute
from ez2rc.v2.storage import Cache, CacheError, digest, write_json
from ez2rc.v2.timing import project_video_times
from test_v2_beat_ranges import INTERVAL_TOML
from test_v2_endpoint_beats import REQUEST
from test_v2_origin_adoption import ANCHOR_TOML
from test_v2_shared_judgment import UNKNOWN_TOML
from test_v2_timing_resolver import moving_taps
from v2_fixtures import fixture


def declaration(error=.75, *, status='verified', version='procedural-position-v1'):
    value = f'\n[position_calibration]\nstatus = "{status}"\nversion = "{version}"\nscope = "profile_endpoints"\n'
    return value + (f'error_px = {error}\n' if error is not None else '')


def load(run, name):
    return json.loads((run / name).read_bytes())


def manifest(run, stage):
    return load(run, f'artifacts/{stage}/manifest.json')


def rehash_manifest(path, value):
    value['manifest_sha256'] = digest({k: v for k, v in value.items() if k != 'manifest_sha256'})
    write_json(path, value)


def moving_notes(root, *, bridge_gap=140):
    """Two lanes, irregular taps and a visible head/body/tail, constant scroll."""
    song = moving_taps(root)
    templates = {kind: cv2.imread(str(root / name)) for kind, name in
                 (('tap', 'note.png'), ('head', 'head.png'), ('tail', 'tail.png'))}
    (root / 'video.mkv').unlink()
    with av.open(str(root / 'video.mkv'), 'w') as container:
        stream = container.add_stream('ffv1', rate=50)
        stream.width = stream.height = 96; stream.pix_fmt = 'bgr0'
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for i in range(65):
            frame_image = np.zeros((96, 96, 3), np.uint8)
            frame_image[70:76, 20:61] = (220, 120, 20)
            for x, gaps in ((22, (0, 42, 93, bridge_gap, 225)), (42, (0, 125, 196))):
                for gap in gaps:
                    y = 10 + 3 * i - gap
                    if 0 <= y and y + 7 <= 70:
                        frame_image[y:y + 7, x:x + 16] = templates['tap']
            head, tail = 10 + 3 * i - 42, 10 + 3 * i - 82
            lo, hi = max(0, tail + 7), min(70, head)
            if lo < hi: frame_image[lo:hi, 42:58] = 100
            for kind, y in (('head', head), ('tail', tail)):
                if 0 <= y and y + 7 <= 70:
                    frame_image[y:y + 7, 42:58] = templates[kind]
            frame = av.VideoFrame.from_ndarray(frame_image, format='bgr24')
            frame.pts, frame.time_base = 1000 + i * 20, Fraction(1, 1000)
            for packet in stream.encode(frame): container.mux(packet)
        for packet in stream.encode(): container.mux(packet)
    song.write_text(song.read_text().replace('bpm = "191.910"', 'bpm = "120"\nmeasure_beats = [4, 1]\n'
        '[timing.judgment_calibration]\nstatus = "verified"\nversion = "procedural"\noffset_ms = 7.5\nerror_ms = 2.0') +
        '\n' + ANCHOR_TOML.replace('error_ms = 0.0', 'error_ms = 2.0') + INTERVAL_TOML + UNKNOWN_TOML + REQUEST)
    return song


class PositionConfigTest(unittest.TestCase):
    def test_optional_verified_unverified_and_zero_preserve_exact_declarations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root); path = root / 'profile.toml'; base = path.read_text()
            self.assertIsNone(load_profile(path).position_calibration)
            self.assertNotIn('position_calibration', load_profile(path).snapshot)
            for value, expected in ((declaration(), PositionCalibration('verified', 'procedural-position-v1', 'profile_endpoints', .75)),
                                    (declaration(0), PositionCalibration('verified', 'procedural-position-v1', 'profile_endpoints', 0)),
                                    (declaration(None, status='unverified'), PositionCalibration('unverified', 'procedural-position-v1', 'profile_endpoints'))):
                path.write_text(base + value); profile = load_profile(path)
                self.assertEqual(profile.position_calibration, expected)
                self.assertEqual(profile.snapshot['position_calibration'], expected.snapshot())

    def test_invalid_status_scope_version_and_error_fail_before_analysis(self):
        bad = [declaration(-1), declaration('true'), declaration('nan'), declaration('inf'), declaration('"0.5"'),
               declaration(None), declaration(0, status='unverified'), declaration(status='unknown'),
               declaration(version=''), declaration(version='   '), declaration().replace('profile_endpoints', 'one_lane'),
               declaration().replace('scope = "profile_endpoints"', ''), declaration() + 'offset_px = 1\n',
               declaration().replace('version = "procedural-position-v1"', '')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); fixture(root); path = root / 'profile.toml'; base = path.read_text()
            for value in bad:
                path.write_text(base + value)
                with self.subTest(value=value), self.assertRaises(ConfigError): load_profile(path)
        for error in (True, -1, float('inf'), float('nan')):
            with self.assertRaises(ConfigError): PositionCalibration('verified', 'test', 'profile_endpoints', error)

    def test_radius_status_and_version_change_observation_identity_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = fixture(root); song = load_song(path); algorithm = load_algorithm(song.algorithm)
            profile_path = root / 'profile.toml'; base = profile_path.read_text(); keys = []
            for value in ('', declaration(), declaration(1.5), declaration(version='next-version'), declaration(None, status='unverified')):
                profile_path.write_text(base + value)
                keys.append(cache_keys(song, load_profile(profile_path), algorithm, 'procedural-video'))
            self.assertEqual(len({digest(k[0]) for k in keys}), 1)
            self.assertEqual(len({digest(k[1]) for k in keys}), len(keys))


class PositionCacheTest(unittest.TestCase):
    def test_rehashed_observation_metadata_cannot_override_the_key_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song_path, _ = fixture(root); path = root / 'profile.toml'
            path.write_text(path.read_text() + declaration(0))
            song = load_song(song_path)
            _, obs, _ = collect(song, load_profile(path), load_algorithm(song.algorithm), root / 'cache')
            self.assertEqual(open_observations(obs.path).manifest['metadata']['position_error_px'], 0)
            original = deepcopy(obs.manifest)
            for mode in ('radius', 'boolean_zero', 'version', 'scope', 'profile', 'undeclared'):
                destination = root / mode; shutil.copytree(obs.path, destination); changed = deepcopy(original)
                meta = changed['metadata']
                if mode == 'radius': meta['position_error_px'] = 9
                if mode == 'boolean_zero': meta['position_error_px'] = False
                if mode == 'version': meta['position_calibration']['version'] = 'other'
                if mode == 'scope': meta['position_calibration']['scope'] = 'one_lane'
                if mode == 'profile': meta['profile']['position_calibration']['error_px'] = 9
                if mode == 'undeclared': del meta['position_calibration']
                rehash_manifest(destination / 'manifest.json', changed)
                with self.subTest(mode=mode), self.assertRaises(CacheError): open_observations(destination)

    def test_missing_and_unverified_declarations_keep_unknown_radius(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = fixture(root); song = load_song(path); profile_path = root / 'profile.toml'; base = profile_path.read_text()
            for value in ('', declaration(None, status='unverified')):
                profile_path.write_text(base + value)
                _, obs, _ = collect(song, load_profile(profile_path), load_algorithm(song.algorithm), root / 'cache')
                self.assertIsNone(obs.manifest['metadata']['position_error_px'])
                self.assertEqual(obs.manifest['metadata']['position_error_reason'], 'not independently calibrated')
                self.assertEqual(open_observations(obs.path).id, obs.id)


class PositionPipelineTest(unittest.TestCase):
    def test_nonzero_tap_and_longnote_ranges_survive_extract_reuse_and_reconstruct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_notes(root); profile = root / 'profile.toml'
            profile.write_text(profile.read_text() + declaration())
            code, first = execute(song, output_root=root / 'out'); self.assertEqual(code, 2)
            declared = load_profile(profile).position_calibration.snapshot()
            for stage in ('observations', 'fusion', 'decode'):
                self.assertEqual(manifest(first, stage)['metadata']['position_calibration'], declared)
            self.assertEqual(manifest(first, 'observations')['metadata']['position_error_px'], .75)
            self.assertEqual(manifest(first, 'fusion')['metadata']['observation_position_error_px'], .75)
            decoded = open_decode(first / 'artifacts/decode')
            self.assertTrue(all(e['localization_error_px'] == .75 for lane in lanes(decoded) for e in lane['endpoints']))
            for name in ('time-projection', 'endpoint-time-ranges', 'beat-ranges'):
                self.assertEqual(load(first, f'artifacts/{name}.json')['position_calibration'], declared)
            ranges = load(first, 'artifacts/beat-ranges.json')
            bounded = [r for r in ranges['events'] if r['status'] == 'bounded']
            self.assertEqual({r['type'] for r in bounded}, {'tap', 'longnote'})
            for row in bounded:
                domain = row['start_capture_time_range_ms']
                self.assertLess(Fraction(*domain['lower']), Fraction(*domain['upper']))
                self.assertEqual(row['shared_judgment_id'], 'shared-judgment')
                if row['type'] == 'longnote':
                    width = row['duration_beat_range']
                    self.assertLess(Fraction(*width['lower']), Fraction(*width['upper']))
            self.assertEqual(load(first, 'chart.json')['notes'], [])
            self.assertEqual(load(first, 'run.json')['input']['profile']['position_calibration'], declared)
            report = (first / 'report.html').read_text()
            self.assertIn('bound: 0.75 px', report); self.assertIn('procedural-position-v1', report)
            artifacts = ('chart.json', 'artifacts/events.json', 'artifacts/time-projection.json', 'artifacts/endpoint-time-ranges.json',
                         'artifacts/timing-resolution.json', 'artifacts/beat-ranges.json')
            code, reused = execute(song, output_root=root / 'out'); self.assertEqual(code, 2)
            self.assertTrue(all(load(reused, 'run.json')['stages'][s]['reused'] for s in ('capture', 'observation', 'motion', 'fusion', 'decode')))
            code, fresh = execute(song, output_root=root / 'fresh'); self.assertEqual(code, 2)
            for name in artifacts:
                self.assertEqual(load(first, name), load(reused, name)); self.assertEqual(load(first, name), load(fresh, name))
            for path in (root / 'video.mkv', profile, root / 'note.png', root / 'head.png', root / 'tail.png'): path.unlink()
            with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no video decode')), patch('cv2.imread', side_effect=AssertionError('no template pixels')):
                code, again = execute(song, output_root=root / 'out', observation_cache=first / 'artifacts/observations')
            self.assertEqual(code, 2)
            for name in artifacts: self.assertEqual(load(first, name), load(again, name))
            for key in ('timing_resolution', 'origin_adoption', 'time_projection', 'endpoint_time_ranges'):
                self.assertEqual(ranges[key + '_sha256'], hashlib.sha256((again / ranges[key + '_ref']).read_bytes()).hexdigest())

    def test_radius_changes_widen_ranges_without_moving_nominal_notes_or_reusing_stale_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_notes(root); path = root / 'profile.toml'; base = path.read_text(); runs = []
            for error in (0., .75, 1.5):
                path.write_text(base + declaration(error))
                if runs:
                    code, failed = execute(song, output_root=root / 'out', observation_cache=runs[-1] / 'artifacts/observations')
                    self.assertEqual(code, 1); self.assertIn('profile changed', load(failed, 'run.json')['failure']['message'])
                code, run = execute(song, output_root=root / 'out'); self.assertEqual(code, 2); runs.append(run)
            stage_ids = [load(run, 'run.json')['stages'] for run in runs]
            self.assertEqual(len({s['capture']['cache_id'] for s in stage_ids}), 1)
            for stage in ('observation', 'motion', 'fusion', 'decode'):
                self.assertEqual(len({s[stage]['cache_id'] for s in stage_ids}), 3)
            charts = [load(run, 'chart.json') for run in runs]; self.assertTrue(all(c == charts[0] for c in charts))
            events = [load(run, 'artifacts/events.json')['events'] for run in runs]; self.assertTrue(all(e == events[0] for e in events))
            ranges = [{r['event_id']: r for r in load(run, 'artifacts/beat-ranges.json')['events']} for run in runs]
            checked = 0
            for eid, wide in ranges[-1].items():
                if wide['status'] != 'bounded': continue
                checked += 1
                for side in ('start', 'end'):
                    if wide[side + '_beat_range'] is None: continue
                    values = [r[eid][side + '_beat_range'] for r in ranges]
                    for a, b in zip(values, values[1:]):
                        self.assertGreater(Fraction(*a['lower']), Fraction(*b['lower']))
                        self.assertLess(Fraction(*a['upper']), Fraction(*b['upper']))
            self.assertGreater(checked, 0)

    def test_missing_unverified_and_outside_support_never_fall_back_to_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_notes(root); path = root / 'profile.toml'; base = path.read_text()
            for value, reason in (('', 'localization_error_unknown'), (declaration(None, status='unverified'), 'localization_error_unknown'),
                                  (declaration(10000.), 'outside_observed_scroll_range')):
                path.write_text(base + value)
                code, run = execute(song, output_root=root / 'out'); self.assertEqual(code, 2)
                ranges = load(run, 'artifacts/beat-ranges.json'); self.assertEqual(ranges['bounded_event_count'], 0)
                self.assertTrue(any('start_' + reason in r['reasons'] for r in ranges['events']))
                if value:
                    self.assertEqual(ranges['position_calibration'], load_profile(path).position_calibration.snapshot())
                else: self.assertNotIn('position_calibration', ranges)

    def test_location_declaration_does_not_verify_endpoint_timing_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_notes(root); path = root / 'profile.toml'
            path.write_text(path.read_text().replace('status = "verified"\noffset_px = 0.0\nerror_px = 0.0',
                                                    'status = "unverified"') + declaration())
            code, run = execute(song, output_root=root / 'out'); self.assertEqual(code, 2)
            self.assertEqual(load(run, 'artifacts/beat-ranges.json')['bounded_event_count'], 0)
            self.assertTrue(all(e['start_ms'] is None for e in load(run, 'artifacts/events.json')['events']))

    def test_declared_location_bound_does_not_bridge_uncertain_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_notes(root, bridge_gap=157); path = root / 'profile.toml'
            path.write_text(path.read_text() + declaration())
            code, run = execute(song, output_root=root / 'out'); self.assertEqual(code, 2)
            longs = [e for e in load(run, 'artifacts/beat-ranges.json')['events'] if e['type'] == 'longnote']
            self.assertTrue(longs)
            self.assertTrue(all(e['status'] == 'unavailable' and e['duration_beat_range'] is None for e in longs))
            self.assertTrue(any('end_uncertain_motion' in e['reasons'] for e in longs))

    def test_rehashed_downstream_declarations_must_match_used_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_notes(root); path = root / 'profile.toml'; path.write_text(path.read_text() + declaration())
            code, run = execute(song, output_root=root / 'out'); self.assertEqual(code, 2)
            for stage, reader in (('fusion', open_fusion), ('decode', open_decode)):
                target = root / ('bad-' + stage); shutil.copytree(run / 'artifacts' / stage, target)
                value = load(target, 'manifest.json'); value['metadata']['position_calibration']['error_px'] = 9.
                rehash_manifest(target / 'manifest.json', value)
                with self.subTest(stage=stage), self.assertRaises(CacheError): reader(target)

    def test_matching_radius_does_not_hide_a_changed_parent_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_notes(root); path = root / 'profile.toml'; path.write_text(path.read_text() + declaration())
            code, run = execute(song, output_root=root / 'out'); self.assertEqual(code, 2)
            obs = open_observations(run / 'artifacts/observations'); motion = Cache(run / 'artifacts/motion')
            for stage, reader in (('fusion', open_fusion), ('decode', open_decode)):
                target = root / ('changed-' + stage); shutil.copytree(run / 'artifacts' / stage, target)
                value = load(target, 'manifest.json'); value['metadata']['position_calibration']['version'] = 'other-basis'
                rehash_manifest(target / 'manifest.json', value); changed = reader(target)
                with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, 'declaration differs'):
                    if stage == 'fusion': infer(obs, motion, changed, DecodeConfig(), root / 'cache')
                    else: project_video_times(obs, motion, changed)


if __name__ == '__main__':
    unittest.main()
