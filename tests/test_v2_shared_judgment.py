"""Declared shared clock errors; no measurement fixtures or stochastic assumptions."""
from copy import deepcopy
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
from ez2rc.v2.contracts import decimal_bpm, read_ratio
from ez2rc.v2.runner import execute
from ez2rc.v2.storage import read_json, write_json
from test_v2_beat_ranges import INTERVAL_TOML, decision, resolved, window
from test_v2_origin_adoption import ANCHOR_TOML
from test_v2_reference_link import anchor, candidates
from test_v2_timing_resolver import moving_taps, timing_input

UNKNOWN = RangeUncertainty('procedural-errors', 'shared_all_events', 'unknown')
UNKNOWN_TOML = '''[timing.range_uncertainty]
version = "procedural-errors"
judgment_scope = "shared_all_events"
origin_relation = "unknown"
'''


def same_clock(reference=Fraction(100)):
    return RangeUncertainty('procedural-errors', 'shared_all_events', 'same_clock_shift', reference)


def inputs(error=3, *, bpm='120', origin=None, data=None, **options):
    chart, events, resolution = resolved(data, bpm=bpm,
        **{'judgment_calibration': JudgmentCalibration('verified', 'procedural-judgment', 7.5, error), **options})
    return chart, events, resolution, decision(candidates(bpm=bpm, measure=options.get('measure_beats', Fraction(4)))) if origin is None else origin


class SharedJudgmentTest(unittest.TestCase):
    def test_missing_declaration_keeps_origin_only_semantics(self):
        _, events, timing, origin = inputs(error=10)
        result = project_beat_ranges(events, timing, origin)
        self.assertEqual(result['schema_version'], 1)
        self.assertNotIn('shared_judgment', result)
        self.assertEqual(result['events'][0]['start_beat_range'], window(Fraction(-1, 200), Fraction(7, 200)))

    def test_unknown_dependence_retains_both_sources_and_conservative_bounds(self):
        chart, events, timing, origin = inputs()
        before = deepcopy((timing, origin))
        result = project_beat_ranges(events, timing, origin, uncertainty_model=UNKNOWN)
        self.assertEqual(result['schema_version'], 2)
        self.assertEqual(result['uncertainty_model']['bounds_kind'], 'outer_enclosure')
        self.assertEqual(result['combined_shared_shift_beats'], window(Fraction(-113, 500), Fraction(-87, 500)))
        self.assertEqual(result['events'][0]['start_beat_range'], window(Fraction(-11, 1000), Fraction(41, 1000)))
        self.assertEqual(result['events'][0]['start_origin_only_beat_range'], window(Fraction(-1, 200), Fraction(7, 200)))
        self.assertEqual(result['shared_judgment']['delta_ms'], window(-3, 3))
        self.assertEqual(result['events'][1]['duration_beats'], [4, 25])
        self.assertTrue(all(e['shared_judgment_id'] == 'shared-judgment' for e in result['events']))
        self.assertEqual((timing, origin), before)
        self.assertFalse(chart.notes)

    def test_same_clock_cancellation_requires_declared_reference_and_keeps_chart_partial(self):
        chart, events, timing, origin = inputs(error=10)
        result = project_beat_ranges(events, timing, origin, uncertainty_model=same_clock())
        self.assertEqual(result['uncertainty_model']['status'], 'applied')
        self.assertEqual(result['combined_shared_shift_beats'], window(Fraction(-1, 5), Fraction(-1, 5)))
        self.assertEqual(result['events'][0]['start_beat_range'], window(Fraction(3, 200), Fraction(3, 200)))
        self.assertEqual(result['shared_origin']['interval_ms'], window(90, 110))
        self.assertEqual(chart.status, 'partial'); self.assertFalse(chart.notes)
        self.assertTrue(all(e.status == 'incomplete' for e in events))
        for delta in (Fraction(-10), Fraction(7, 3), Fraction(10)):
            actual_time = Fraction('107.5') + delta
            actual_origin = 100 + delta
            self.assertEqual((actual_time - actual_origin) / 500, Fraction(3, 200))
        unknown = project_beat_ranges(events, timing, origin, uncertainty_model=UNKNOWN)
        self.assertNotEqual(unknown['events'][0]['start_beat_range'], result['events'][0]['start_beat_range'])

    def test_inconsistent_shared_source_withholds_combined_bounds_without_clipping(self):
        for error, reference in ((9, Fraction(100)), (10, Fraction(99)), (11, Fraction(100))):
            _, events, timing, origin = inputs(error=error)
            result = project_beat_ranges(events, timing, origin, uncertainty_model=same_clock(reference))
            self.assertEqual(result['uncertainty_model']['reasons'], ['shared_clock_bounds_mismatch'])
            self.assertEqual(result['bounded_event_count'], 0)
            self.assertIsNone(result['combined_shared_shift_beats'])
            self.assertIsNone(result['events'][0]['start_beat_range'])
            self.assertIsNotNone(result['events'][0]['start_origin_only_beat_range'])
            self.assertEqual(result['shared_origin']['interval_ms'], window(90, 110))

    def test_unknown_dependence_encloses_joint_domains_and_preserves_common_differences(self):
        rng = random.Random(904)
        for _ in range(50):
            center, width, error = rng.randrange(-200, 201), rng.randrange(1, 20), rng.randrange(0, 41) / 4
            bpm = rng.choice(('120', '123.456', '180'))
            origin = decision(candidates(bpm=bpm, measure=Fraction(4)), [anchor(time=center, error=width)])
            _, events, timing, origin = inputs(error, bpm=bpm, origin=origin)
            result = project_beat_ranges(events, timing, origin, uncertainty_model=UNKNOWN)
            scale = decimal_bpm(bpm) / 60000
            values = []
            for clock_origin in (center - width, center, center + width):
                for delta in (Fraction(str(-error)), Fraction(0), Fraction(str(error))):
                    beat = (Fraction('107.5') + delta - clock_origin) * scale
                    values.append(beat)
                    self.assertLessEqual(read_ratio(result['events'][0]['start_beat_range']['lower']), beat)
                    self.assertLessEqual(beat, read_ratio(result['events'][0]['start_beat_range']['upper']))
                    ln_start = (Fraction('307.5') + delta - clock_origin) * scale
                    ln_end = (Fraction('387.5') + delta - clock_origin) * scale
                    self.assertEqual(ln_end - ln_start, read_ratio(result['events'][1]['duration_beats']))
            self.assertEqual((min(values), max(values)),
                             (read_ratio(result['events'][0]['start_beat_range']['lower']),
                              read_ratio(result['events'][0]['start_beat_range']['upper'])))

    def test_fractional_reference_and_zero_judgment_error_are_exact(self):
        origin = decision(candidates(bpm='180', measure=Fraction(4)), [anchor(beat=Fraction(1), time=0, error=2)])
        _, events, timing, origin = inputs(2, bpm='180', origin=origin)
        result = project_beat_ranges(events, timing, origin, uncertainty_model=same_clock(Fraction(-1000, 3)))
        self.assertEqual(result['events'][0]['start_beat_range'], window(Fraction(529, 400), Fraction(529, 400)))
        _, events, timing, origin = inputs(0)
        base = project_beat_ranges(events, timing, origin)
        combined = project_beat_ranges(events, timing, origin, uncertainty_model=UNKNOWN)
        self.assertEqual(base['events'][0]['start_beat_range'], combined['events'][0]['start_beat_range'])

    def test_missing_calibration_origin_meter_or_event_evidence_never_acquires_ranges(self):
        for mode in ('judgment', 'origin', 'meter', 'endpoint', 'ambiguous', 'missing_end'):
            data = timing_input(('longnote',)); options = {}
            if mode == 'judgment': options['judgment_calibration'] = None
            if mode == 'meter': options['measure_beats'] = None
            if mode == 'endpoint': data[2]['lanes'][0]['endpoints'][0]['calibration_status'] = 'unverified'
            if mode == 'ambiguous': data[1][0]['status'] = 'ambiguous'
            if mode == 'missing_end': data[1][0]['selected_intervals'][0]['end'] = None
            origin = decision(anchors=[]) if mode == 'origin' else None
            _, events, timing, origin = inputs(data=data, origin=origin, **options)
            result = project_beat_ranges(events, timing, origin, uncertainty_model=UNKNOWN)
            self.assertEqual(result['bounded_event_count'], 0)
            self.assertIsNone(result['events'][0]['start_beat_range'])

    def test_invalid_direct_model_or_calibration_is_rejected_and_schema_is_explicit(self):
        _, events, timing, origin = inputs()
        with self.assertRaises(ValueError): project_beat_ranges(events, timing, origin, uncertainty_model={})
        for error in (-1, float('nan'), float('inf'), True, None):
            altered = deepcopy(timing); altered['judgment_calibration']['error_ms'] = error
            with self.subTest(error=error), self.assertRaises(ValueError):
                project_beat_ranges(events, altered, origin, uncertainty_model=UNKNOWN)
        result = project_beat_ranges(events, timing, origin, uncertainty_model=UNKNOWN)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ranges.json'; write_json(path, result)
            with self.assertRaises(ValueError): read_json(path, 'ez2rc.beat-ranges')
            self.assertEqual(read_json(path, 'ez2rc.beat-ranges', schema_versions=(2,)), result)


class SharedJudgmentConfigTest(unittest.TestCase):
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

    def test_optional_model_and_exact_reference_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'song.toml'; path.write_text(self.base)
            self.assertIsNone(load_song(path).range_uncertainty)
            path.write_text(self.base + UNKNOWN_TOML)
            self.assertEqual(load_song(path).range_uncertainty, UNKNOWN)
            path.write_text(self.base + UNKNOWN_TOML.replace('"unknown"', '"same_clock_shift"') +
                            'origin_reference_ms_exact = [-1000, 3]\n')
            loaded = load_song(path)
            self.assertEqual(loaded.range_uncertainty, same_clock(Fraction(-1000, 3)))
            self.assertEqual(json.loads(json.dumps(loaded.snapshot()))['range_uncertainty']['origin_reference_ms_exact'], [-1000, 3])

    def test_missing_scope_unknown_relations_and_invalid_references_fail(self):
        bad = [UNKNOWN_TOML.replace('shared_all_events', 'independent_per_event'),
               UNKNOWN_TOML.replace('"unknown"', '"independent"'),
               UNKNOWN_TOML.replace('"unknown"', '"same_clock_shift"'),
               UNKNOWN_TOML.replace('version = "procedural-errors"', ''),
               UNKNOWN_TOML.replace('procedural-errors', ''), UNKNOWN_TOML + 'origin_reference_ms_exact = [100, 1]\n',
               UNKNOWN_TOML + 'extra = 1\n']
        bad += [UNKNOWN_TOML.replace('"unknown"', '"same_clock_shift"') + f'origin_reference_ms_exact = {pair}\n'
                for pair in ('[2, 4]', '[1, 0]', '[1, -3]', '[true, 1]', '[1.0, 3]', '"1/3"')]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'song.toml'
            for value in bad:
                path.write_text(self.base + value)
                with self.subTest(value=value), self.assertRaises(ConfigError): load_song(path)


class SharedJudgmentPipelineTest(unittest.TestCase):
    def test_model_only_reconstruction_keeps_notes_timing_and_all_visual_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_taps(root)
            base = song.read_text().replace('bpm = "191.910"', 'bpm = "120"\nmeasure_beats = [4, 1]\n'
                '[timing.judgment_calibration]\nstatus = "verified"\nversion = "procedural"\noffset_ms = 7.5\nerror_ms = 2.0')
            base += '\n' + ANCHOR_TOML.replace('error_ms = 0.0', 'error_ms = 2.0') + INTERVAL_TOML
            song.write_text(base)
            code, first = execute(song, output_root=root / 'out'); self.assertEqual(code, 2)
            load = lambda run, name: json.loads((run / name).read_bytes())
            before = load(first, 'artifacts/beat-ranges.json'); self.assertGreater(before['bounded_event_count'], 0)
            (root / 'video.mkv').unlink()
            for mode, extra in (('unknown', UNKNOWN_TOML),
                                ('same', UNKNOWN_TOML.replace('"unknown"', '"same_clock_shift"') + 'origin_reference_ms_exact = [100, 1]\n'),
                                ('mismatch', UNKNOWN_TOML.replace('"unknown"', '"same_clock_shift"') + 'origin_reference_ms_exact = [101, 1]\n'),
                                ('removed', '')):
                song.write_text(base + '\n' + extra)
                with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no video decoding')):
                    code, run = execute(song, output_root=root / 'out', observation_cache=first / 'artifacts/observations')
                self.assertEqual(code, 2)
                for name in ('chart.json', 'artifacts/events.json', 'artifacts/timing-resolution.json',
                             'artifacts/reference-candidates.json', 'artifacts/reference-origin-link.json',
                             'artifacts/origin-adoption.json', 'artifacts/time-projection.json'):
                    self.assertEqual(load(first, name), load(run, name))
                ranges = load(run, 'artifacts/beat-ranges.json')
                if mode == 'removed': self.assertEqual(ranges, before)
                elif mode == 'mismatch': self.assertEqual(ranges['bounded_event_count'], 0)
                else:
                    self.assertEqual(ranges['bounded_event_count'], before['bounded_event_count'])
                    for a, b in zip(before['events'], ranges['events']):
                        if a['status'] != 'bounded': continue
                        if mode == 'same': self.assertEqual(b['start_beat_range']['lower'], b['start_beat_range']['upper'])
                        else:
                            self.assertEqual(read_ratio(a['start_beat_range']['lower']) - read_ratio(b['start_beat_range']['lower']), Fraction(1, 250))
                        self.assertEqual(b['start_origin_only_beat_range'], a['start_beat_range'])
                self.assertEqual(ranges['timing_resolution_sha256'], hashlib.sha256((run / ranges['timing_resolution_ref']).read_bytes()).hexdigest())
                stages = load(run, 'run.json')['stages']
                self.assertTrue(all(stages[s]['reused'] for s in ('capture', 'observation', 'motion', 'fusion', 'decode')))
                self.assertEqual(stages['timing']['beat_ranges_version'], ranges['algorithm_version'])


if __name__ == '__main__':
    unittest.main()
