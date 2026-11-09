"""Explicit synthetic timing references; no measured recording fixtures."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import cv2
import numpy as np

from ez2rc.v2.config import ConfigError, JudgmentCalibration, load_song
from ez2rc.v2.contracts import Event, Metadata
from ez2rc.v2.export import read_chart, read_events, write_outputs
from ez2rc.v2.runner import execute
from ez2rc.v2.timing import PROJECTION_VERSION, TimingResolver, beat_to_ms, resolve_pixel_timing
from v2_fixtures import fixture

META = Metadata('synthetic', '2k', 2, 'Timing', 'HD')
VERIFIED = JudgmentCalibration('verified', 'procedural-clock-1', 7.5, .25)


def timing_input(kinds=('tap', 'longnote')):
    events, pixels, endpoints = [], [], []
    for i, kind in enumerate(kinds):
        eid, cid, start, end = f'e{i}', f'c{i}', f'p{i}s', f'p{i}e' if kind == 'longnote' else None
        events.append(Event(eid, i % 2, 'incomplete', kind, observation_ids=(f'raw{i}',),
                            candidate_ids=(cid,), selected_cost=-2., alternative_cost=0., reason='timing pending'))
        pixels.append({'id': eid, 'status': 'resolved', 'selected_intervals': [
            {'id': cid, 'type': kind, 'start': start, 'end': end}]})
        for name, stamp in ((start, 100. + i * 200), (end, 180. + i * 200)):
            if name:
                endpoints.append({'id': name, 'status': 'interpolated', 'nominal_timestamp_ms': stamp,
                                  'calibration_status': 'verified'})
    projection = {'format': 'ez2rc.time-projection', 'algorithm_version': PROJECTION_VERSION,
                  'observation_cache_id': 'obs', 'motion_cache_id': 'motion', 'decode_cache_id': 'decode',
                  'lanes': [{'endpoints': endpoints}]}
    return tuple(events), pixels, projection


def resolve(data, **kwargs):
    options = {'beat_zero_ms': -50., 'measure_beats': Fraction(4), 'judgment_calibration': VERIFIED, **kwargs}
    return resolve_pixel_timing(*data, META, '123.456', **options)


def moving_taps(root):
    """Distinct procedural templates and two independently moving lanes."""
    song, _ = fixture(root)
    rng = np.random.default_rng(19)
    for kind in ('head', 'tail'):
        cv2.imwrite(str(root / f'{kind}.png'), rng.integers(50, 240, (7, 16, 3), dtype=np.uint8))
    profile = (root / 'profile.toml').read_text().replace('status = "unverified"',
        'status = "verified"\noffset_px = 0.0\nerror_px = 0.0')
    profile = profile.replace('head = "note.png"', 'head = "head.png"').replace('tail = "note.png"', 'tail = "tail.png"')
    (root / 'profile.toml').write_text(profile)
    template = cv2.imread(str(root / 'note.png'))
    (root / 'video.mkv').unlink()
    with av.open(str(root / 'video.mkv'), 'w') as container:
        stream = container.add_stream('ffv1', rate=50)
        stream.width = stream.height = 96
        stream.pix_fmt = 'bgr0'
        stream.time_base = stream.codec_context.time_base = Fraction(1, 1000)
        for i in range(42):
            image = np.zeros((96, 96, 3), np.uint8)
            image[70:76, 20:61] = (220, 120, 20)
            for x, shift in ((22, 0), (42, 13)):
                for gap in (0, 42, 93, 157):
                    y = 10 + 3 * i - shift - gap
                    if 0 <= y and y + 7 <= 70:
                        image[y:y + 7, x:x + 16] = template
            frame = av.VideoFrame.from_ndarray(image, format='bgr24')
            frame.pts, frame.time_base = 1000 + i * 20, Fraction(1, 1000)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return song


class TimingResolverTest(unittest.TestCase):
    def test_explicit_reference_converts_times_exactly_without_snapping(self):
        events = (Event('tap', 0, 'resolved', 'tap', 123.125),
                  Event('ln', 1, 'resolved', 'longnote', 250.5, 713.75))
        chart = TimingResolver(-50., Fraction(7, 2)).resolve(events, META, '123.456')
        self.assertEqual(chart.status, 'complete')
        self.assertEqual(chart.pending_stages, ())
        for note, event in zip(chart.notes, events):
            self.assertEqual(beat_to_ms(note.start_beat, chart.timing), Fraction(str(event.start_ms)))
            if note.end_beat is not None:
                self.assertEqual(beat_to_ms(note.end_beat, chart.timing), Fraction(str(event.end_ms)))
        self.assertEqual(chart.timing.measure_beats, Fraction(7, 2))

    def test_missing_origin_or_measure_keeps_known_times_but_no_notes(self):
        event = Event('tap', 0, 'resolved', 'tap', 12.)
        for origin, measure in ((None, Fraction(4)), (0., None)):
            chart = TimingResolver(origin, measure).resolve((event,), META, '120')
            self.assertEqual(chart.notes, ())
            self.assertEqual(chart.unresolved[0].start_ms, 12.)
            self.assertEqual(chart.unresolved[0].status, 'incomplete')
            self.assertEqual(chart.status, 'partial')

    def test_mixed_events_preserve_unknown_choices_and_visible_missing_ends(self):
        data = timing_input(('tap', 'tap', 'longnote'))
        data[1][1]['status'] = 'ambiguous'
        data[1][2]['status'] = 'incomplete'
        data[1][2]['selected_intervals'][0]['start'] = None
        chart, events, report = resolve(data)
        self.assertEqual([n.id for n in chart.notes], ['e0'])
        self.assertEqual(chart.status, 'partial')
        self.assertEqual(events[1].status, 'ambiguous')
        self.assertIsNone(events[1].start_ms)
        self.assertIsNone(events[2].start_ms)
        self.assertEqual(events[2].end_ms, 587.5)
        self.assertEqual(events[2].status, 'incomplete')
        self.assertEqual(events[2].observation_ids, ('raw2',))
        self.assertEqual(report['unresolved_event_count'], 2)

    def test_no_calibration_is_not_an_implicit_zero_offset(self):
        for calibration in (None, JudgmentCalibration('unverified', 'pending')):
            chart, events, report = resolve(timing_input(), judgment_calibration=calibration)
            self.assertEqual(chart.notes, ())
            self.assertTrue(all(e.start_ms is None and e.end_ms is None for e in events))
            self.assertIn('judgment_calibration_unverified', report['events'][0]['time_reasons'])
        chart, events, _ = resolve(timing_input(), judgment_calibration=JudgmentCalibration('verified', 'known-zero', 0., 0.))
        self.assertEqual(chart.status, 'complete')
        self.assertEqual(events[0].start_ms, 100.)

    def test_endpoint_calibration_and_time_support_are_checked_separately(self):
        for status in ('held', 'unavailable', 'outside_range', 'unlocalized'):
            data = timing_input(('tap',))
            data[2]['lanes'][0]['endpoints'][0].update(status=status, nominal_timestamp_ms=None)
            chart, events, _ = resolve(data)
            self.assertEqual(chart.notes, ())
            self.assertIsNone(events[0].start_ms)
        data = timing_input(('longnote',))
        data[2]['lanes'][0]['endpoints'][0]['calibration_status'] = 'unverified'
        chart, events, _ = resolve(data)
        self.assertIsNone(events[0].start_ms)
        self.assertEqual(events[0].end_ms, 187.5)
        self.assertEqual(chart.notes, ())

    def test_equal_endpoint_times_do_not_invent_a_longnote_duration(self):
        data = timing_input(('longnote',))
        data[2]['lanes'][0]['endpoints'][1]['nominal_timestamp_ms'] = 100.
        chart, events, report = resolve(data)
        self.assertEqual(chart.notes, ())
        self.assertIsNone(events[0].end_ms)
        self.assertEqual(report['events'][0]['proposed_corrected_times_ms'], {'start': 107.5, 'end': 107.5})
        self.assertIn('nonpositive_duration', report['events'][0]['time_reasons'])

    def test_explicit_correction_keeps_unknown_full_uncertainty_and_roundtrips_outputs(self):
        chart, events, report = resolve(timing_input())
        self.assertEqual(chart.status, 'complete')
        self.assertEqual(events[0].start_ms, 107.5)
        self.assertTrue(all(e.start_interval is None and e.end_interval is None for e in events))
        self.assertEqual(report['judgment_calibration']['error_ms'], .25)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_outputs(root, chart, events, observation_ref='observations', timing_resolution_ref='timing')
            self.assertEqual(read_chart(root / 'chart.json')['status'], 'complete')
            self.assertEqual(len(read_events(root / 'artifacts/events.json')['events']), 2)

    def test_projection_and_event_identity_mismatches_are_rejected(self):
        data = timing_input()
        data[1][0]['id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'identities'):
            resolve(data)
        data = timing_input()
        data[2]['algorithm_version'] = 'unsupported'
        with self.assertRaisesRegex(ValueError, 'projection'):
            resolve(data)


class TimingConfigurationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.song = Path(self.tmp.name) / 'song.toml'
        self.base = '''schema_version = 1
[input]
video = "unused.mkv"
profile = "unused.toml"
[metadata]
title = "Procedural"
difficulty = "HD"
[timing]
mode = "fixed"
bpm = "123.456"
'''

    def load(self, extra=''):
        self.song.write_text(self.base + extra)
        return load_song(self.song)

    def test_optional_reference_and_versioned_calibration_snapshot(self):
        self.assertIsNone(self.load().beat_zero_ms)
        config = self.load('beat_zero_ms = -17.5\nmeasure_beats = [7, 2]\n'
                           '[timing.judgment_calibration]\nstatus = "verified"\nversion = "procedural"\noffset_ms = 2.5\nerror_ms = 0.5\n')
        self.assertEqual(config.measure_beats, Fraction(7, 2))
        self.assertEqual(json.loads(json.dumps(config.snapshot()))['measure_beats'], [7, 2])
        self.assertEqual(config.judgment_calibration.offset_ms, 2.5)

    def test_invalid_reference_and_assumed_calibration_are_rejected(self):
        for extra in ('beat_zero_ms = true', 'beat_zero_ms = nan', 'beat_zero_ms = inf',
                      'measure_beats = [0, 1]', 'measure_beats = [4, 4]', 'measure_beats = [1, 0]',
                      'measure_beats = [true, 1]', 'measure_beats = 4',
                      '[timing.judgment_calibration]\nstatus = "verified"\nversion = "x"\noffset_ms = 0',
                      '[timing.judgment_calibration]\nstatus = "unverified"\nversion = "x"\noffset_ms = 0',
                      '[timing.judgment_calibration]\nstatus = "verified"\nversion = "x"\noffset_ms = 0\nerror_ms = -1'):
            with self.subTest(extra=extra), self.assertRaises(ConfigError):
                self.load(extra)


class TimingPipelineTest(unittest.TestCase):
    def test_verified_synthetic_notes_and_timing_only_reconstruction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_taps(root)
            text = song.read_text().replace('bpm = "191.910"', 'bpm = "120"\nbeat_zero_ms = 0.0\nmeasure_beats = [4, 1]\n'
                '[timing.judgment_calibration]\nstatus = "verified"\nversion = "procedural"\noffset_ms = 7.5\nerror_ms = 0.25')
            song.write_text(text)
            code, first = execute(song, output_root=root / 'out')
            self.assertIn(code, (0, 2))
            chart = read_chart(first / 'chart.json')
            self.assertGreater(len(chart['notes']), 0)
            before_events = read_events(first / 'artifacts/events.json')
            self.assertEqual(before_events['timing_resolution_ref'], 'artifacts/timing-resolution.json')
            projection = json.loads((first / 'artifacts/time-projection.json').read_text())
            resolution = json.loads((first / 'artifacts/timing-resolution.json').read_text())
            self.assertEqual(resolution['resolved_note_count'], len(chart['notes']))
            (root / 'video.mkv').unlink()
            song.write_text(text.replace('beat_zero_ms = 0.0', 'beat_zero_ms = 125.0'))
            with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no video decoding')):
                code, second = execute(song, output_root=root / 'out', observation_cache=first / 'artifacts/observations')
            self.assertIn(code, (0, 2))
            after = read_chart(second / 'chart.json')
            self.assertEqual(projection, json.loads((second / 'artifacts/time-projection.json').read_text()))
            self.assertEqual(before_events['events'], read_events(second / 'artifacts/events.json')['events'])
            for a, b in zip(chart['notes'], after['notes']):
                self.assertEqual(Fraction(*a['start_beat']) - Fraction(*b['start_beat']), Fraction(1, 4))
            stages = json.loads((second / 'run.json').read_text())['stages']
            self.assertTrue(all(stages[s]['reused'] for s in ('motion', 'fusion', 'decode')))


if __name__ == '__main__':
    unittest.main()
