"""Procedural interval propagation and shared-origin consistency checks."""
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
from ez2rc.v2.config import ConfigError, OriginAdoption, load_song
from ez2rc.v2.contracts import read_ratio
from ez2rc.v2.export import read_chart
from ez2rc.v2.origin import _covers_interval, adopt_reference_origin
from ez2rc.v2.reference_link import link_reference_origin
from ez2rc.v2.runner import execute
from ez2rc.v2.storage import read_json, write_json
from ez2rc.v2.timing import resolve_pixel_timing
from test_v2_origin_adoption import ANCHOR_TOML, REQUEST, REQUEST_TOML
from test_v2_reference_link import anchor, calibration, candidates, marker
from test_v2_timing_resolver import META, VERIFIED, moving_taps, timing_input

INTERVAL_REQUEST = OriginAdoption('anchor_interval', 'procedural-range-v1')
INTERVAL_TOML = REQUEST_TOML.replace('exact_anchor_intersection', 'anchor_interval')


def window(lo, hi, il=True, iu=True):
    return {'lower': [Fraction(lo).numerator, Fraction(lo).denominator],
            'upper': [Fraction(hi).numerator, Fraction(hi).denominator],
            'lower_inclusive': il, 'upper_inclusive': iu}


def decision(raw=None, anchors=None, calibrations=None, request=INTERVAL_REQUEST):
    linked = link_reference_origin(candidates(measure=Fraction(4)) if raw is None else raw,
                                   beat_anchors=[anchor(error=10)] if anchors is None else anchors,
                                   marker_calibrations=calibrations)
    return adopt_reference_origin(linked, request)


def resolved(data=None, *, bpm='120', **options):
    return resolve_pixel_timing(*(timing_input() if data is None else data), META, bpm,
                               **{'measure_beats': Fraction(4), 'judgment_calibration': VERIFIED, **options})


class PeriodicCoverageTest(unittest.TestCase):
    def test_closed_and_open_endpoints_wraparound_and_negative_lifts(self):
        p = Fraction(100)
        self.assertTrue(_covers_interval(window(0, 10), [window(0, 10)], p))
        self.assertFalse(_covers_interval(window(0, 10), [window(0, 10, True, False)], p))
        self.assertTrue(_covers_interval(window(-5, 5), [window(95, 100, True, False), window(0, 5)], p))
        self.assertFalse(_covers_interval(window(-5, 5), [window(95, 100, True, False), window(0, 5, False)], p))

    def test_wide_intervals_preserve_periodic_holes_without_enumeration(self):
        p = Fraction(100); span = window(-10**100, 10**100)
        self.assertFalse(_covers_interval(span, [window(0, 100, False, False)], p))
        self.assertFalse(_covers_interval(span, [window(0, 99)], p))
        self.assertTrue(_covers_interval(span, [window(0, 100, False, False), window(0, 0)], p))
        self.assertTrue(_covers_interval(span, [window(20, 121, False, False)], p))

    def test_periodic_coverage_matches_independent_finite_cell_oracle(self):
        rng = random.Random(812)
        for _ in range(120):
            lo = Fraction(rng.randrange(-200, 201), 2)
            hi = lo + rng.randrange(1, 401)
            windows = [window(a := Fraction(rng.randrange(-200, 201), 2), a + rng.randrange(0, 151),
                              bool(rng.randrange(2)), bool(rng.randrange(2))) for _ in range(rng.randrange(1, 5))]
            lifts = [(read_ratio(w['lower']) + k * 100, read_ratio(w['upper']) + k * 100,
                      w['lower_inclusive'], w['upper_inclusive']) for w in windows for k in range(-8, 9)]
            cuts = sorted({lo, hi, *(x for a, b, _, _ in lifts for x in (a, b) if lo <= x <= hi)})
            probes = cuts + [(a + b) / 2 for a, b in zip(cuts, cuts[1:])]
            expected = all(any((a < x < b) or (a <= x <= b and (x != a or ia) and (x != b or ib))
                               for a, b, ia, ib in lifts) for x in probes)
            self.assertEqual(_covers_interval(window(lo, hi), windows, Fraction(100)), expected)


class IntervalAdoptionTest(unittest.TestCase):
    def test_explicit_interval_policy_preserves_bounds_without_selecting_midpoint(self):
        self.assertEqual(decision(request=REQUEST)['reasons'], ['anchor_origin_is_interval'])
        self.assertEqual(decision(request=None)['status'], 'not_requested')
        result = decision()
        self.assertEqual(result['status'], 'bounded')
        self.assertEqual(result['schema_version'], 3)
        self.assertEqual(result['effective_origin_interval_ms'], window(90, 110))
        self.assertIsNone(result['effective_beat_zero_ms'])
        self.assertIsNone(result['candidate_origin_ms_exact'])
        self.assertNotIn('effective_beat_zero_ms_exact', result)
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_singleton_still_uses_exact_origin_and_conflicting_anchors_stay_blocked(self):
        self.assertEqual(decision(anchors=[anchor()])['status'], 'adopted')
        for anchors in ([], [anchor(status='unverified')], [anchor('a'), anchor('b', time=200)]):
            self.assertEqual(decision(anchors=anchors)['status'], 'blocked')
        raw = candidates(bpm='180', measure=Fraction(4))
        result = decision(raw, [anchor(beat=Fraction(1), time=0)])
        self.assertEqual(result['effective_beat_zero_ms_exact'], [-1000, 3])

    def test_marker_overlap_cannot_narrow_anchor_interval(self):
        raw = candidates([marker('p', 100, 105), marker('q', 600, 605)], measure=Fraction(4))
        result = decision(raw, calibrations={'beat_signal': calibration()})
        self.assertEqual(result['status'], 'blocked')
        self.assertIn('beat_signal:phase_does_not_cover_anchor_interval', result['reasons'])
        self.assertEqual(result['anchor_intersection_ms'], window(90, 110))
        self.assertFalse(result['channel_checks']['beat_signal']['hypotheses'][0]['whole_anchor_interval_covered'])

    def test_full_coverage_keeps_common_error_and_anchor_width_under_repetition(self):
        for count in (2, 20):
            raw = candidates([marker(str(i), 100 + 500 * i) for i in range(count)], measure=Fraction(4))
            result = decision(raw, calibrations={'beat_signal': calibration(error=12)})
            self.assertEqual(result['status'], 'bounded')
            self.assertEqual(result['effective_origin_interval_ms'], window(90, 110))
            self.assertTrue(result['channel_checks']['beat_signal']['hypotheses'][0]['whole_anchor_interval_covered'])

    def test_ambiguous_missing_and_full_cycle_error_channels_still_block(self):
        for markers, cal in (([marker('p', 100)], calibration()),
                             ([marker('p', 100), marker('q', 600), marker('x', 250), marker('y', 750)], calibration()),
                             ([marker('p', 100), marker('q', 600)], calibration(error=250))):
            self.assertEqual(decision(candidates(markers, measure=Fraction(4)),
                                      calibrations={'beat_signal': cal})['status'], 'blocked')

    def test_supplied_point_takes_precedence_but_requires_its_own_marker_check(self):
        for zero, status in ((102., 'confirmed_supplied'), (95., 'blocked'), (120., 'blocked')):
            raw = candidates([marker('p', 100, 105), marker('q', 600, 605)], zero=zero, measure=Fraction(4))
            result = decision(raw, calibrations={'beat_signal': calibration()})
            self.assertEqual(result['status'], status)
            self.assertEqual(result['effective_beat_zero_ms'], zero)
            self.assertNotIn('effective_origin_interval_ms', result)


class BeatRangeTest(unittest.TestCase):
    def test_common_origin_affine_model_and_duration_cancel_only_origin_error(self):
        chart, events, resolution = resolved()
        origin = decision(); before = deepcopy((resolution, origin))
        ranges = project_beat_ranges(events, resolution, origin)
        self.assertEqual(chart.status, 'partial'); self.assertFalse(chart.notes)
        self.assertTrue(all(e.status == 'incomplete' for e in events))
        self.assertEqual(ranges['bounded_event_count'], 2)
        tap, longnote = ranges['events']
        self.assertEqual(tap['start_beat_range'], window(Fraction(-1, 200), Fraction(7, 200)))
        self.assertEqual(longnote['duration_beats'], [4, 25])
        self.assertEqual(ranges['shared_origin']['shift_beats'], window(Fraction(-11, 50), Fraction(-9, 50)))
        for zero in (Fraction(90), Fraction(301, 3), Fraction(110)):
            shift = -zero / 500
            for row in ranges['events']:
                self.assertEqual(row['shared_origin_id'], ranges['shared_origin']['id'])
                start = read_ratio(row['start_time_term_beats']) + shift
                self.assertLessEqual(read_ratio(row['start_beat_range']['lower']), start)
                self.assertLessEqual(start, read_ratio(row['start_beat_range']['upper']))
                if row['type'] == 'longnote':
                    end = read_ratio(row['end_time_term_beats']) + shift
                    self.assertEqual(end - start, read_ratio(row['duration_beats']))
        independent_extremes = read_ratio(longnote['end_beat_range']['upper']) - read_ratio(longnote['start_beat_range']['lower'])
        self.assertNotEqual(independent_extremes, read_ratio(longnote['duration_beats']))
        self.assertEqual((resolution, origin), before)

    def test_fractional_negative_origin_bounds_and_decimal_bpm_are_exact(self):
        raw = candidates(bpm='123.456', measure=Fraction(4))
        origin = decision(raw, [anchor(beat=Fraction(-1, 3), time=-20, error=.5)])
        _, events, resolution = resolved(bpm='123.456')
        ranges = project_beat_ranges(events, resolution, origin)
        lo = read_ratio(origin['effective_origin_interval_ms']['lower'])
        hi = read_ratio(origin['effective_origin_interval_ms']['upper'])
        scale = Fraction(643, 312500)
        self.assertEqual(ranges['events'][0]['start_beat_range'],
                         window((Fraction('107.5') - hi) * scale, (Fraction('107.5') - lo) * scale))

    def test_calibration_pixel_timing_and_meter_gates_cannot_be_bypassed(self):
        for mode in ('judgment', 'endpoint', 'pixel', 'held', 'missing_end', 'meter'):
            data = timing_input(('longnote',)); options = {}
            if mode == 'judgment': options['judgment_calibration'] = None
            if mode == 'endpoint': data[2]['lanes'][0]['endpoints'][0]['calibration_status'] = 'unverified'
            if mode == 'pixel': data[1][0]['status'] = 'ambiguous'
            if mode == 'held': data[2]['lanes'][0]['endpoints'][0]['status'] = 'held'
            if mode == 'missing_end': data[1][0]['selected_intervals'][0]['end'] = None
            if mode == 'meter': options['measure_beats'] = None
            chart, events, resolution = resolved(data, **options)
            origin = decision(candidates(measure=None if mode == 'meter' else Fraction(4)))
            ranges = project_beat_ranges(events, resolution, origin)
            self.assertFalse(chart.notes)
            self.assertEqual(ranges['bounded_event_count'], 0)
            self.assertEqual(ranges['events'][0]['status'], 'unavailable')
            self.assertIsNone(ranges['events'][0]['start_beat_range'])

    def test_point_or_unrequested_origin_never_creates_interval_ranges(self):
        for origin in (decision(request=None), decision(request=REQUEST), decision(anchors=[anchor()])):
            _, events, resolution = resolved(beat_zero_ms=origin['effective_beat_zero_ms'])
            result = project_beat_ranges(events, resolution, origin)
            self.assertEqual(result['bounded_event_count'], 0)
            self.assertIn('bounded_origin_not_selected', result['reasons'])

    def test_lineage_identity_and_bounded_contract_mismatches_fail(self):
        _, events, resolution = resolved()
        with self.assertRaisesRegex(ValueError, 'effective origin'):
            project_beat_ranges(events, resolution, decision(anchors=[anchor()]))
        for field, value in (('observation_cache_id', 'other'), ('schema_version', 2),
                             ('effective_beat_zero_ms', 100.)):
            origin = decision(); origin[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                project_beat_ranges(events, resolution, origin)
        for change in ('reference', 'ids', 'times'):
            altered = deepcopy(resolution)
            if change == 'reference': altered['reference']['bpm'] = '121'
            if change == 'ids': altered['events'][0]['event_id'] = 'other'
            if change == 'times': altered['events'][0]['accepted_start_ms'] += 1
            with self.subTest(change=change), self.assertRaises(ValueError):
                project_beat_ranges(events, altered, decision())


class BeatRangePipelineTest(unittest.TestCase):
    def test_interval_extract_reconstruction_and_exact_fallback_preserve_visual_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_taps(root)
            base = song.read_text().replace('bpm = "191.910"', 'bpm = "120"\nmeasure_beats = [4, 1]\n'
                '[timing.judgment_calibration]\nstatus = "verified"\nversion = "procedural"\noffset_ms = 7.5\nerror_ms = 0.25')
            anchors = ANCHOR_TOML.replace('error_ms = 0.0', 'error_ms = 2.0')
            song.write_text(base + '\n' + anchors + INTERVAL_TOML)
            self.assertEqual(load_song(song).origin_adoption.policy, 'anchor_interval')
            code, first = execute(song, output_root=root / 'out')
            self.assertEqual(code, 2)
            load = lambda run, name: json.loads((run / name).read_bytes())
            chart = read_chart(first / 'chart.json'); self.assertFalse(chart['notes'])
            self.assertEqual(chart['beat_ranges_ref'], 'artifacts/beat-ranges.json')
            ranges = load(first, chart['beat_ranges_ref']); self.assertGreater(ranges['bounded_event_count'], 0)
            self.assertTrue(all(e['event_id'] in {u['id'] for u in chart['unresolved']} for e in ranges['events'] if e['status'] == 'bounded'))
            for key in ('timing_resolution', 'origin_adoption'):
                self.assertEqual(ranges[key + '_sha256'], hashlib.sha256((first / ranges[key + '_ref']).read_bytes()).hexdigest())
            with self.assertRaises(ValueError): read_json(first / 'artifacts/origin-adoption.json', 'ez2rc.origin-adoption')
            self.assertEqual(read_json(first / 'artifacts/origin-adoption.json', 'ez2rc.origin-adoption', schema_versions=(3,))['status'], 'bounded')
            self.assertIn('beat-ranges.json', (first / 'report.html').read_text())
            (root / 'video.mkv').unlink()
            variants = [('wider', base + '\n' + anchors.replace('error_ms = 2.0', 'error_ms = 3.0') + INTERVAL_TOML),
                        ('exact_policy', base + '\n' + anchors + REQUEST_TOML),
                        ('singleton', base + '\n' + ANCHOR_TOML + INTERVAL_TOML),
                        ('supplied_conflict', base.replace('bpm = "120"', 'bpm = "120"\nbeat_zero_ms = 0.0') + '\n' + anchors + INTERVAL_TOML)]
            for name, text in variants:
                song.write_text(text)
                with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no video decoding')):
                    code, run = execute(song, output_root=root / 'out', observation_cache=first / 'artifacts/observations')
                self.assertIn(code, (0, 2))
                after = read_chart(run / 'chart.json'); output = load(run, 'artifacts/beat-ranges.json')
                if name == 'wider':
                    self.assertEqual(after, chart)
                    self.assertEqual(output['bounded_event_count'], ranges['bounded_event_count'])
                    for a, b in zip(ranges['events'], output['events']):
                        if a['status'] == 'bounded':
                            self.assertEqual(read_ratio(a['start_beat_range']['lower']) - read_ratio(b['start_beat_range']['lower']), Fraction(1, 500))
                else:
                    self.assertEqual(output['bounded_event_count'], 0)
                    self.assertEqual(bool(after['notes']), name != 'exact_policy')
                self.assertEqual(load(first, 'artifacts/events.json')['pixel_intervals'], load(run, 'artifacts/events.json')['pixel_intervals'])
                stages = load(run, 'run.json')['stages']
                self.assertTrue(all(stages[s]['reused'] for s in ('capture', 'observation', 'motion', 'fusion', 'decode')))
                self.assertEqual(stages['timing']['bounded_events'], output['bounded_event_count'])
            song.write_text(base + '\n' + INTERVAL_TOML.replace('anchor_interval', 'midpoint'))
            with self.assertRaises(ConfigError): load_song(song)


if __name__ == '__main__':
    unittest.main()
