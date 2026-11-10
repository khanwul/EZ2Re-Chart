"""Independent procedural origins and explicit acceptance/rejection contracts."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ez2rc.v2.config import ConfigError, MarkerCalibration, OriginAdoption, load_song
from ez2rc.v2.contracts import decimal_bpm, read_origin
from ez2rc.v2.export import read_chart
from ez2rc.v2.origin import adopt_reference_origin
from ez2rc.v2.reference_link import link_reference_origin
from ez2rc.v2.runner import execute
from ez2rc.v2.timing import resolve_pixel_timing
from test_v2_reference_link import anchor, calibration, candidates, marker
from test_v2_timing_resolver import META, VERIFIED, moving_taps, timing_input

REQUEST = OriginAdoption('exact_anchor_intersection', 'procedural-adoption-v1')
REQUEST_TOML = '''[timing.origin_adoption]
policy = "exact_anchor_intersection"
version = "procedural-adoption-v1"
'''
ANCHOR_TOML = '''[[timing.beat_anchors]]
id = "known-beat"
beat = [2, 1]
timestamp_ms = 1100.0
error_ms = 0.0
status = "verified"
version = "procedural-anchor-v1"
'''


def decide(raw=None, anchors=None, calibrations=None, request=REQUEST):
    linked = link_reference_origin(candidates() if raw is None else raw,
                                   beat_anchors=[anchor()] if anchors is None else anchors,
                                   marker_calibrations=calibrations)
    return adopt_reference_origin(linked, request)


class OriginAdoptionTest(unittest.TestCase):
    def test_exact_anchor_needs_opt_in_and_does_not_mutate_diagnostics(self):
        linked = link_reference_origin(candidates(), beat_anchors=[anchor(beat=Fraction(-3, 2), time=-650)])
        before = deepcopy(linked)
        self.assertEqual(adopt_reference_origin(linked)['status'], 'not_requested')
        self.assertIsNone(adopt_reference_origin(linked)['effective_beat_zero_ms'])
        result = adopt_reference_origin(linked, REQUEST)
        self.assertEqual(result['status'], 'adopted')
        self.assertEqual(result['effective_beat_zero_ms'], 100.)
        self.assertEqual(result['candidate_origin_ms_exact'], [100, 1])
        self.assertEqual(linked, before)
        result['anchor_intersection_ms']['lower'][0] = 999
        self.assertEqual(linked, before)

    def test_no_anchor_or_unverified_anchor_cannot_supply_absolute_origin(self):
        raw = candidates([marker('p', 100), marker('q', 600)])
        for anchors in ([], [anchor(status='unverified')]):
            with self.subTest(anchors=anchors):
                result = decide(raw, anchors, {'beat_signal': calibration()})
                self.assertEqual(result['status'], 'blocked')
                self.assertIsNone(result['effective_beat_zero_ms'])
                self.assertIn('verified_anchor_unavailable', result['reasons'])

    def test_interval_is_not_replaced_by_midpoint_or_precise_periodic_marker(self):
        raw = candidates([marker('p', 100), marker('q', 600)])
        result = decide(raw, [anchor(error=5)], {'beat_signal': calibration()})
        self.assertEqual(result['reasons'], ['anchor_origin_is_interval'])
        self.assertEqual(result['anchor_intersection_ms']['lower'], [95, 1])
        self.assertEqual(result['anchor_intersection_ms']['upper'], [105, 1])
        self.assertIsNone(result['effective_beat_zero_ms'])
        self.assertIsNone(result['candidate_origin_ms_exact'])

    def test_all_verified_anchors_must_agree_and_unverified_records_remain_visible(self):
        anchors = [anchor('exact'), anchor('wide', time=101, error=3),
                   anchor('unverified', time=900, status='unverified')]
        result = decide(anchors=anchors)
        self.assertEqual(result['status'], 'adopted')
        self.assertEqual(result['anchor_ids'], ['exact', 'wide'])
        self.assertEqual(result['unverified_anchor_ids'], ['unverified'])
        result = decide(anchors=anchors + [anchor('contradiction', time=120, error=1)])
        self.assertIn('anchor_conflict', result['reasons'])
        self.assertIsNone(result['effective_beat_zero_ms'])

    def test_closed_intersection_can_be_singleton_without_averaging(self):
        result = decide(anchors=[anchor('left', time=90, error=10), anchor('right', time=110, error=10)])
        self.assertEqual(result['status'], 'adopted')
        self.assertEqual(result['effective_beat_zero_ms'], 100.)

    def test_nondecimal_or_overflowing_singleton_is_not_rounded(self):
        for raw, a in ((candidates(bpm='180'), anchor(beat=Fraction(1), time=0)),
                       (candidates(bpm='0.000001'), anchor(beat=Fraction(10**400), time=0))):
            with self.subTest(bpm=raw['supplied_reference']['bpm']):
                result = decide(raw, [a])
                self.assertEqual(result['status'], 'adopted')
                self.assertIsNone(result['effective_beat_zero_ms'])
                self.assertEqual(read_origin(result, result['schema_version'], 'effective_beat_zero_ms'),
                                 -a.beat * 60000 / decimal_bpm(raw['supplied_reference']['bpm']))
        result = decide(candidates(bpm='123.456'), [anchor(beat=Fraction(643), time=312499.9)])
        self.assertEqual(result['effective_beat_zero_ms'], -.1)

    def test_supplied_origin_is_preserved_on_acceptance_conflict_and_interval(self):
        for zero, expected in ((100., 'confirmed_supplied'), (0., 'blocked')):
            result = decide(candidates(zero=zero))
            self.assertEqual(result['status'], expected)
            self.assertEqual(result['effective_beat_zero_ms'], zero)
            self.assertEqual(result['origin_source'], 'supplied')
        self.assertIn('supplied_origin_conflict', result['reasons'])
        result = decide(candidates(zero=102.), [anchor(error=5)])
        self.assertEqual(result['reasons'], ['anchor_origin_is_interval'])
        self.assertEqual(result['effective_beat_zero_ms'], 102.)

    def test_unique_recurrent_calibrated_phase_can_support_exact_anchor(self):
        raw = candidates([marker('p', 109, 111), marker('q', 609, 611)],
                         [marker('b', 80), marker('c', 2080)], measure=Fraction(4))
        result = decide(raw, calibrations={'beat_signal': calibration(-10, 2),
                                          'barlines': calibration(20, 3, 'measure_boundary')})
        self.assertEqual(result['status'], 'adopted')
        self.assertTrue(all(c['status'] == 'compatible' for c in result['channel_checks'].values()))

    def test_multiple_phases_are_not_filtered_by_anchor_or_majority(self):
        raw = candidates([marker(str(i), 100 + 500 * i) for i in range(8)] +
                         [marker('x', 250), marker('y', 750)])
        result = decide(raw, calibrations={'beat_signal': calibration()})
        self.assertEqual(result['status'], 'blocked')
        hypotheses = result['channel_checks']['beat_signal']['hypotheses']
        self.assertEqual([h['anchor_comparison'] for h in hypotheses], ['compatible', 'conflict'])
        self.assertIn('beat_signal:unique_recurrent_phase_unavailable', result['reasons'])

    def test_any_configured_channel_can_block_adoption(self):
        raw = candidates([marker('p', 100), marker('q', 600)],
                         [marker('b', 250), marker('c', 2250)], measure=Fraction(4))
        result = decide(raw, calibrations={'beat_signal': calibration(),
                                          'barlines': calibration(meaning='measure_boundary')})
        self.assertEqual(result['channel_checks']['beat_signal']['status'], 'compatible')
        self.assertIn('barlines:phase_not_compatible_with_anchors', result['reasons'])
        self.assertIsNone(result['effective_beat_zero_ms'])

    def test_missing_recurrence_unverified_calibration_and_missing_measure_block(self):
        raw = candidates([marker('p', 100)])
        for channel, cal in (('beat_signal', calibration()),
                             ('beat_signal', MarkerCalibration('unverified', 'unknown', 'integer_beat')),
                             ('barlines', calibration(meaning='measure_boundary'))):
            result = decide(raw, calibrations={channel: cal})
            self.assertEqual(result['status'], 'blocked')
            self.assertIn(channel + ':unique_recurrent_phase_unavailable', result['reasons'])

    def test_common_clock_error_is_preserved_and_full_period_cannot_certify(self):
        for count in (2, 20):
            raw = candidates([marker(str(i), 100 + 500 * i) for i in range(count)])
            linked = link_reference_origin(raw, marker_calibrations={'beat_signal': calibration(error=7)},
                                           beat_anchors=[anchor(error=5)])
            before = deepcopy(linked)
            result = adopt_reference_origin(linked, REQUEST)
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(linked, before)
            self.assertEqual(linked['channels']['beat_signal']['hypotheses'][0]
                             ['corrected_periodic_windows_ms'][0]['lower'], [93, 1])
            result = decide(raw, calibrations={'beat_signal': calibration(error=250)})
            self.assertIn('beat_signal:clock_error_covers_full_period', result['reasons'])

    def test_wrapped_phase_and_negative_cycle_keep_exact_origin(self):
        raw = candidates([marker('p', 490, 510), marker('q', 990, 1010)])
        for zero in (0., -5., -500., 10.):
            result = decide(raw, [anchor(time=zero)], {'beat_signal': calibration()})
            self.assertEqual(result['status'], 'adopted')
            self.assertEqual(result['effective_beat_zero_ms'], zero)
        self.assertEqual(decide(raw, [anchor(time=11)], {'beat_signal': calibration()})['status'], 'blocked')

    def test_adoption_does_not_supply_meter_or_promote_uncalibrated_events(self):
        zero = decide()['effective_beat_zero_ms']
        for unverified in ('judgment', 'endpoint', 'pixel', 'measure'):
            events, pixels, projection = timing_input(('tap',))
            options = {'judgment_calibration': VERIFIED, 'measure_beats': Fraction(4)}
            if unverified == 'judgment':
                options['judgment_calibration'] = None
            if unverified == 'endpoint':
                projection['lanes'][0]['endpoints'][0]['calibration_status'] = 'unverified'
            if unverified == 'pixel':
                pixels[0]['status'] = 'ambiguous'
            if unverified == 'measure':
                options['measure_beats'] = None
            chart, _, _ = resolve_pixel_timing(events, pixels, projection, META, '120', beat_zero_ms=zero, **options)
            self.assertEqual(chart.status, 'partial')
            self.assertFalse(chart.notes)
        chart, _, _ = resolve_pixel_timing(*timing_input(('tap',)), META, '120', beat_zero_ms=zero,
                                          judgment_calibration=VERIFIED, measure_beats=Fraction(4))
        self.assertEqual(chart.status, 'complete')
        self.assertEqual(chart.notes[0].start_beat, Fraction(3, 200))


class OriginAdoptionConfigTest(unittest.TestCase):
    def test_versioned_opt_in_snapshot_and_invalid_requests(self):
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
        with tempfile.TemporaryDirectory() as tmp:
            song = Path(tmp) / 'song.toml'
            song.write_text(base)
            self.assertIsNone(load_song(song).origin_adoption)
            song.write_text(base + REQUEST_TOML)
            self.assertEqual(load_song(song).origin_adoption, REQUEST)
            self.assertEqual(json.loads(json.dumps(load_song(song).snapshot()))['origin_adoption']['policy'],
                             'exact_anchor_intersection')
            for extra in ('origin_adoption = true', '[timing.origin_adoption]',
                          REQUEST_TOML.replace('exact_anchor_intersection', 'midpoint'),
                          REQUEST_TOML.replace('procedural-adoption-v1', ''), REQUEST_TOML + 'threshold = 10\n'):
                song.write_text(base + extra)
                with self.subTest(extra=extra), self.assertRaises(ConfigError):
                    load_song(song)

    def test_wrong_artifact_version_or_request_type_fails(self):
        linked = link_reference_origin(candidates(), beat_anchors=[anchor()])
        with self.assertRaises(ValueError):
            adopt_reference_origin(linked, True)
        linked['algorithm_version'] = 'unknown'
        with self.assertRaises(ValueError):
            adopt_reference_origin(linked, REQUEST)


class OriginAdoptionPipelineTest(unittest.TestCase):
    def test_extract_and_cache_only_reconstruct_record_decision_and_exact_beat_shift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); song = moving_taps(root)
            base = song.read_text().replace('bpm = "191.910"', 'bpm = "120"\nmeasure_beats = [4, 1]\n'
                '[timing.judgment_calibration]\nstatus = "verified"\nversion = "procedural"\noffset_ms = 7.5\nerror_ms = 0.25')
            song.write_text(base + '\n' + ANCHOR_TOML + REQUEST_TOML)
            code, first = execute(song, output_root=root / 'out')
            self.assertIn(code, (0, 2))
            first_chart = read_chart(first / 'chart.json')
            self.assertTrue(first_chart['notes'])
            self.assertEqual(first_chart['timing']['beat_zero_ms'], 100.)
            load = lambda run, name: json.loads((run / name).read_text())
            decision = load(first, 'artifacts/origin-adoption.json')
            self.assertEqual(decision['status'], 'adopted')
            self.assertEqual(decision['reference_origin_link_sha256'],
                             hashlib.sha256((first / decision['reference_origin_link_ref']).read_bytes()).hexdigest())
            resolution = load(first, 'artifacts/timing-resolution.json')
            self.assertEqual(resolution['origin_adoption_sha256'],
                             hashlib.sha256((first / resolution['origin_adoption_ref']).read_bytes()).hexdigest())
            self.assertEqual(resolution['origin_source'], 'anchor_intersection')
            self.assertIn('origin-adoption.json', (first / 'report.html').read_text())
            self.assertEqual(load(first, 'artifacts/events.json')['origin_adoption_ref'], 'artifacts/origin-adoption.json')
            (root / 'video.mkv').unlink()
            variants = [
                ('shifted', base + '\n' + ANCHOR_TOML.replace('1100.0', '1225.0') + REQUEST_TOML),
                ('interval', base + '\n' + ANCHOR_TOML.replace('error_ms = 0.0', 'error_ms = 2.0') + REQUEST_TOML),
                ('no_request', base + '\n' + ANCHOR_TOML),
                ('supplied_conflict', base.replace('bpm = "120"', 'bpm = "120"\nbeat_zero_ms = 0.0') +
                 '\n' + ANCHOR_TOML + REQUEST_TOML)]
            for name, text in variants:
                with self.subTest(variant=name):
                    song.write_text(text)
                    with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no video decoding')):
                        code, run = execute(song, output_root=root / 'out', observation_cache=first / 'artifacts/observations')
                    self.assertIn(code, (0, 2))
                    chart = read_chart(run / 'chart.json')
                    decision = load(run, 'artifacts/origin-adoption.json')
                    if name in ('interval', 'no_request'):
                        self.assertEqual(code, 2)
                        self.assertFalse(chart['notes'])
                        self.assertIsNone(chart['timing']['beat_zero_ms'])
                        self.assertEqual(decision['status'], 'blocked' if name == 'interval' else 'not_requested')
                    else:
                        self.assertEqual(len(chart['notes']), len(first_chart['notes']))
                        expected_shift = Fraction(1, 4) if name == 'shifted' else Fraction(-1, 5)
                        for a, b in zip(first_chart['notes'], chart['notes']):
                            self.assertEqual(Fraction(*a['start_beat']) - Fraction(*b['start_beat']), expected_shift)
                        if name == 'supplied_conflict':
                            self.assertEqual(decision['status'], 'blocked')
                            self.assertIn('supplied_origin_conflict', decision['reasons'])
                            self.assertEqual(chart['timing']['beat_zero_ms'], 0.)
                    self.assertEqual(load(first, 'artifacts/time-projection.json'), load(run, 'artifacts/time-projection.json'))
                    self.assertEqual(load(first, 'artifacts/events.json')['pixel_intervals'],
                                     load(run, 'artifacts/events.json')['pixel_intervals'])
                    stages = load(run, 'run.json')['stages']
                    self.assertEqual(stages['timing']['origin_adoption_status'], decision['status'])
                    self.assertTrue(all(stages[s]['reused'] for s in ('capture', 'observation', 'motion', 'fusion', 'decode')))


if __name__ == '__main__':
    unittest.main()
