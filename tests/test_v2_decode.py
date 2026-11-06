from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch
import weakref

import numpy as np

from ez2rc.v2.config import AlgorithmConfig, DecodeConfig, FusionConfig, MotionConfig, ConfigError, load_algorithm
from ez2rc.v2 import decode
from ez2rc.v2.decode import infer, lanes, ordered_dp, open_decode, output_events
from ez2rc.v2.fusion import accumulate
from ez2rc.v2.motion import estimate
from ez2rc.v2.runner import execute
from ez2rc.v2.storage import Cache, CacheError, canonical, digest, file_hash, write_json
from v2_decode_fixtures import associate, scene, selected
from v2_motion_fixtures import observations
from v2_fixtures import fixture


class IntervalDPTest(unittest.TestCase):
    def test_exact_costs_and_all_marginals_against_exhaustive_enumeration(self):
        rng = np.random.default_rng(832)
        for _ in range(30):
            cs = [{'lo': int(a), 'hi': int(b), 'cost': float(rng.integers(-30, 20)) / 10}
                  for a, b in (sorted(rng.integers(0, 8, 2)) for _ in range(8))]
            paths = []
            for bits in itertools.product((0, 1), repeat=len(cs)):
                chosen = [i for i, bit in enumerate(bits) if bit]
                if any(not (cs[i]['hi'] < cs[j]['lo'] or cs[j]['hi'] < cs[i]['lo'])
                       for i, j in itertools.combinations(chosen, 2)):
                    continue
                paths.append((sum(cs[i]['cost'] for i in chosen), chosen))
            best = ordered_dp(8, cs, .35)
            self.assertEqual(ordered_dp(8, [MappingProxyType(c) for c in cs], .35, verify=True), best)
            self.assertAlmostEqual(best, min(p[0] for p in paths))
            for i, c in enumerate(cs):
                self.assertAlmostEqual(c['path_cost'], min(cost for cost, p in paths if i in p))
                if c['selected']:
                    self.assertAlmostEqual(c['exclusion_cost'], min(cost for cost, p in paths if i not in p))

    def test_dp_verification_rejects_each_derived_field_without_mutation(self):
        cs = [{'lo': 0, 'hi': 1, 'cost': -2.}, {'lo': 0, 'hi': 1, 'cost': -1.8},
              {'lo': 2, 'hi': 3, 'cost': 1.}]
        best = ordered_dp(4, cs, .35)
        self.assertEqual(ordered_dp(4, [MappingProxyType(c) for c in cs], .35, verify=True), best)
        for i in range(len(cs)):
            for field in ('path_cost', 'excess_cost', 'selected', 'competitive', 'exclusion_cost'):
                with self.subTest(candidate=i, field=field):
                    corrupted = [dict(c) for c in cs]
                    value = corrupted[i][field]
                    corrupted[i][field] = not value if isinstance(value, bool) else 999.
                    before = canonical(corrupted)
                    with self.assertRaisesRegex(ValueError, 'path or alternative cost'):
                        ordered_dp(4, [MappingProxyType(c) for c in corrupted], .35, verify=True)
                    self.assertEqual(canonical(corrupted), before)

    def test_decode_validation_reads_once_and_releases_each_lane(self):
        class Lane(dict):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); obs = observations(root, [0, 8, 16, 24])
            m, _ = estimate(obs, MotionConfig(), root)
            f, _ = accumulate(obs, m, FusionConfig(), root)
            d, _ = infer(obs, m, f, DecodeConfig(), root)
            refs, reads = [], []
            original_read, original_payload = Cache._read_chunk, decode._lane_payload

            def read(cache, chunk):
                self.assertTrue(all(ref() is None for ref in refs))
                arrays = original_read(cache, chunk)
                refs.append(weakref.ref(arrays['payload']))
                reads.append(chunk['file'])
                return arrays

            def payload(chunk):
                data = Lane(original_payload(chunk))
                refs.append(weakref.ref(data))
                return data

            with patch.object(Cache, '_read_chunk', read), patch.object(decode, '_lane_payload', payload):
                checked = open_decode(d.path)
            self.assertEqual(checked.manifest, d.manifest)
            self.assertEqual(reads, [c['file'] for c in d.manifest['chunks']])
            self.assertTrue(all(ref() is None for ref in refs))
            # Single-pass validation still rejects changes on subsequent reads.
            (d.path / 'chunk-000001.npz').write_bytes(b'changed after validation')
            with self.assertRaises(CacheError):
                list(lanes(checked))

    def test_inference_and_export_release_previous_lane_objects(self):
        class Lane(dict):
            pass

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); obs = observations(root, [0, 8, 16, 24])
            m, _ = estimate(obs, MotionConfig(), root)
            f, _ = accumulate(obs, m, FusionConfig(), root)
            refs = []
            original_infer, original_payload = decode._infer_lane, decode._lane_payload

            def watch(data):
                result = Lane(data)
                refs.append(weakref.ref(result))
                # Export deliberately retains competitive candidates. Watch a
                # discarded candidate too: a stale ID index can keep it alive
                # even after the enclosing lane dictionary has been released.
                for i, c in enumerate(result['candidates']):
                    if not c['competitive']:
                        candidate = Lane(c)
                        result['candidates'][i] = candidate
                        refs.append(weakref.ref(candidate))
                        break
                return result

            def infer_lane(*args):
                self.assertTrue(all(ref() is None for ref in refs))
                return watch(original_infer(*args))

            def payload(chunk):
                self.assertTrue(all(ref() is None for ref in refs))
                return watch(original_payload(chunk))

            with patch.object(decode, '_infer_lane', infer_lane), patch.object(decode, '_lane_payload', payload):
                d, _ = infer(obs, m, f, DecodeConfig(), root)
                exported = output_events(d)
            self.assertTrue(all(ref() is None for ref in refs))
            self.assertEqual(exported, output_events(d))

    def test_near_taps_and_short_longnote_keep_their_types(self):
        data = scene([(20, 0), (26, 0), (40, 1), (46, 2)], [(40, 46)])
        self.assertEqual(selected(data), [('tap', 20., None), ('tap', 26., None), ('longnote', 40., 46.)])
        self.assertTrue(all(e['status'] == 'resolved' for e in data['events']))

    def test_same_physical_endpoint_types_cannot_be_selected_together(self):
        data = scene([(20, [.99, .99, .99]), (80, 2)], [(20, 80)])
        chosen = [c for c in data['candidates'] if c['selected']]
        resources = [s for c in chosen for s in c['endpoint_site_ids']]
        self.assertEqual(len(resources), len(set(resources)))
        self.assertEqual(selected(data), [('longnote', 20., 80.)])
        duplicates = scene([(20, [.99, .3, .1]), (20, [.3, .99, .1]), (80, 2)], [(20, 80)])
        self.assertEqual(selected(duplicates), [('longnote', 20., 80.)])

    def test_tap_sprite_brightness_and_weak_head_do_not_force_short_longnote(self):
        data = scene([(20, [.14, .14, .14]), (26, [.99, .56, .72])], [(20, 26)])
        self.assertEqual(selected(data), [('tap', 26., None)])
        self.assertTrue(any(c['type'] == 'longnote' for c in data['candidates']))
        self.assertEqual(data['events'][0]['status'], 'resolved')
        old_reward = scene([(20, [.14, .14, .14]), (26, [.99, .56, .72])], [(20, 26)],
                           config=DecodeConfig(body_support_px=1))
        self.assertEqual(old_reward['events'][0]['status'], 'ambiguous')

    def test_weak_spatial_boundary_cannot_split_a_supported_endpoint(self):
        rows = [(0, 17., [.15, .1, .1]), (1, 20., [.96, .1, .1]), (2, 21.5, [.99, .1, .1])]
        sites, eps = associate(rows)
        strong = [e for e in eps if e['kind'] == 'tap' and e['score'] > .7]
        self.assertEqual(len(strong), 1)
        self.assertEqual(strong[0]['observed_range_px'], [20., 21.5])
        self.assertEqual(len({r for e in eps for r in e['observation_ids']}), 3)
        self.assertTrue(all(s['anchor_range_px'][1] - s['anchor_range_px'][0] <= 4 for s in sites))

    def test_coobserved_distinct_peaks_survive_inside_association_radius(self):
        data = scene([(20, 0), (22, 0), (20, 0)])
        self.assertEqual(selected(data), [('tap', 20., None), ('tap', 22., None)])
        self.assertEqual(len(data['sites']), 2)
        self.assertTrue(all(e['weight'] == 1.5 for e in data['endpoints']))

    def test_search_window_edge_response_is_not_a_distinct_peak(self):
        # A neighboring seed's search ends one pixel before the true maximum.
        # Both responses are strong, but only one is an interior local maximum.
        rows = [(0, 20, [.1, .1, .99]), (0, 19, [.1, .1, .95]), (1, 20.5, [.1, .1, .98])]
        sites, eps = associate(rows, interior=[[1, 1, 1], [0, 0, 0], [1, 1, 1]])
        self.assertEqual(len(sites), 1)
        tail = next(e for e in eps if e['kind'] == 'tail')
        self.assertEqual(len(tail['observation_ids']), 3)
        self.assertEqual(tail['search_boundary_observation_ids'], ['f0:l0:y1'])
        self.assertEqual(tail['weight'], 1.)
        # The same two coordinates with independent interior maxima stay split.
        self.assertEqual(len(associate(rows)[0]), 2)

    def test_association_does_not_chain_across_dense_observations(self):
        sites, _ = associate([(0, 20, [.99, .1, .1]), (1, 23, [.98, .1, .1]),
                              (2, 26, [.97, .1, .1]), (3, 29, [.96, .1, .1])])
        self.assertEqual(len(sites), 2)
        self.assertTrue(all(s['anchor_range_px'][1] - s['anchor_range_px'][0] <= 4 for s in sites))

    def test_duplicate_seeds_and_repeats_do_not_move_endpoint_center(self):
        rows = [(0, 20, [.99, .1, .1]), (1, 21, [.99, .1, .1]), (2, 22, [.99, .1, .1])]
        _, original = associate(rows)
        _, repeated = associate(rows + [(3, 20, [.99, .1, .1])] * 12, groups=[0, 1, 2] + [0] * 12)
        for a, b in zip(original, repeated):
            self.assertEqual(a['q_px'], b['q_px'])
            self.assertEqual(a['weight'], b['weight'])
            self.assertEqual(a['shared_motion_error_px'], b['shared_motion_error_px'])

    def test_tap_brightness_does_not_add_a_missing_head_fragment(self):
        points = [(30, [.1, .1, .344]), (43, [.954, .56, .736])]
        data = scene(points, [(21, 43)])
        self.assertEqual(selected(data), [('tap', 43., None)])
        self.assertTrue(any(c['missing_endpoints'] for c in data['candidates']))
        old = scene(points, [(21, 43)], config=DecodeConfig(body_support_px=1))
        self.assertIn(('longnote', None, 30.), selected(old))

    def test_short_body_only_noise_and_real_short_longnote_are_distinguished(self):
        self.assertEqual(selected(scene([], [(20, 22)])), [])
        data = scene([(50, 1), (51, 2)], [(50, 51)])
        self.assertEqual(selected(data), [('longnote', 50., 51.)])
        c = next(c for c in data['candidates'] if c['selected'])
        self.assertAlmostEqual(c['cost_terms']['body_support'], -1 / 24)
        self.assertTrue(all(c['cost'] == math.fsum(c['cost_terms'].values()) for c in data['candidates']))

    def test_review_causes_distinguish_hypotheses_and_timing(self):
        data = scene([(20, [.99, .99, .1]), (80, 2)], [(20, 80)], ambiguous_motion=True)
        self.assertTrue(all('motion_alternatives' in e['review_causes'] for e in data['events']))
        boundary = scene([], [(0, 200)])
        self.assertIn('coverage_boundary', boundary['events'][0]['review_causes'])
        self.assertIn('body_only_hypothesis', boundary['events'][0]['review_causes'])
        interior = scene([(80, 2)], [(20, 80)])
        self.assertIn('missing_endpoint', interior['events'][0]['review_causes'])
        self.assertNotIn('coverage_boundary', interior['events'][0]['review_causes'])

    def test_longnote_interior_is_exclusive_and_touching_ends_are_allowed(self):
        data = scene([(20, 1), (50, 0), (80, 2), (80, 1), (140, 2)], [(20, 140)])
        self.assertEqual(selected(data), [('longnote', 20., 80.), ('longnote', 80., 140.)])
        self.assertNotIn(('tap', 50., None), selected(data))
        data = scene([(20, 1), (80, 2), (80, 0)], [(20, 80)])
        self.assertEqual(selected(data), [('tap', 80., None), ('longnote', 20., 80.)])

    def test_similar_paths_and_noise_are_not_early_committed(self):
        data = scene([(20, [.42, .05, .05])])
        self.assertEqual(data['events'][0]['status'], 'ambiguous')
        self.assertAlmostEqual(data['events'][0]['alternative_cost'], 0)
        data = scene([(20, [.35, .05, .05])])
        self.assertEqual(selected(data), [])
        self.assertEqual(data['events'][0]['status'], 'ambiguous')
        self.assertGreater(len(data['endpoints']), 0)
        weak = scene([(20, [.01, .02, .03])])
        self.assertEqual(weak['events'], [])
        self.assertEqual(len(weak['endpoints']), 3)

    def test_missing_head_or_tail_stays_incomplete(self):
        for point, expected in (([(80, 2)], ('longnote', None, 80.)),
                                ([(20, 1)], ('longnote', 20., None))):
            data = scene(point, [(20, 80)])
            self.assertIn(expected, selected(data))
            self.assertEqual(data['events'][0]['status'], 'incomplete')

    def test_hidden_body_is_not_observed_absence(self):
        args = ([(20, 1), (120, 2)], [(20, 120)])
        for kwargs in ({'unavailable': [(45, 95)]}, {'occluded': [(45, 95)]}):
            data = scene(*args, **kwargs)
            self.assertIn(('longnote', 20., 120.), selected(data))
            c = next(c for c in data['candidates'] if c['selected'])
            self.assertEqual(c['body']['absent_px'], 0)
            self.assertEqual(c['body']['unavailable_px'], 50)
        data = scene(*args, absent=[(45, 95)])
        self.assertNotIn(('longnote', 20., 120.), selected(data))

    def test_video_boundaries_leave_null_missing_endpoints(self):
        data = scene([(40, 2), (120, 1)], [(0, 40), (120, 200)])
        self.assertEqual(selected(data), [('longnote', None, 40.), ('longnote', 120., None)])
        self.assertTrue(all(e['status'] == 'incomplete' for e in data['events']))

    def test_body_crossing_both_video_ends_is_incomplete(self):
        data = scene([], [(0, 200)])
        self.assertEqual(selected(data), [('longnote', None, None)])
        self.assertEqual(data['events'][0]['status'], 'incomplete')
        c = next(c for c in data['candidates'] if c['selected'])
        self.assertEqual(c['missing_start_reason'], 'coverage_boundary')
        self.assertEqual(c['missing_end_reason'], 'coverage_boundary')

    def test_longnote_has_no_maximum_duration_and_keeps_independent_kind_coordinates(self):
        data = scene([([18, 20, 23], 1), ([50010, 50015, 50020], 2)], [(20, 50020)],
                     bounds=(0, 50100), calibration_errors=(None, .2, .9))
        self.assertEqual(selected(data), [('longnote', 20., 50020.)])
        chosen = next(c for c in data['candidates'] if c['selected'])
        eps = {e['id']: e for e in data['endpoints']}
        self.assertEqual(eps[chosen['start']]['calibration_error_px'], .2)
        self.assertEqual(eps[chosen['end']]['calibration_error_px'], .9)

    def test_clipped_template_maximum_is_not_a_localized_endpoint(self):
        data = scene([(20, 1), (80, 2)], [(20, 80)], flags=[[1, 1, 1], [0, 0, 0]])
        for c in data['candidates']:
            if c['start'] and c['start'].startswith('l0:s0:'):
                self.assertIsNone(c['start_px'])
        self.assertTrue(any(e['observed_q_px'] == 20 and e['q_px'] is None for e in data['endpoints']))

    def test_independent_endpoint_errors_and_motion_alternatives(self):
        data = scene([(20, 1), (80, 2)], [(20, 80)], errors=(2, 4, 90), ambiguous_motion=True)
        self.assertTrue(all(e['status'] == 'ambiguous' for e in data['events']))
        self.assertTrue(all(e['shared_motion_error_px'] == 90 for e in data['endpoints']))
        self.assertTrue(all(e['calibration_error_px'] is None for e in data['endpoints']))
        self.assertTrue(all(e['localization_error_px'] is None for e in data['endpoints']))

    def test_empty_evidence_has_no_invented_events(self):
        self.assertEqual(scene([])['events'], [])

    def test_long_intervals_survive_tiles_chunks_repeats_and_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outcomes = []
            for chunk, tile, scroll in ((3, 32, list(range(0, 425, 8))),
                                       (7, 64, list(range(0, 425, 8))),
                                       (7, 64, [0, 8, 8, 8, *range(24, 425, 8)])):
                obs = observations(root, scroll, chunk_frames=chunk, duplicate_seeds=True)
                m, _ = estimate(obs, MotionConfig(), root)
                f, _ = accumulate(obs, m, FusionConfig(tile_bins=tile), root)
                d, _ = infer(obs, m, f, DecodeConfig(), root)
                data = list(lanes(d))[2]
                outcomes.append(selected(data))
                self.assertEqual(selected(data), [('longnote', 90., 490.)])
                self.assertTrue(all(e['status'] == 'resolved' for e in data['events']))
                if len(outcomes) == 1:
                    first = data
                elif len(outcomes) == 2:
                    self.assertEqual(first, data)
            self.assertEqual(outcomes[0], outcomes[1])

    def test_decode_cache_fresh_reuse_corruption_and_lineage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); obs = observations(root, list(range(0, 425, 16)))
            m, _ = estimate(obs, MotionConfig(), root)
            f, _ = accumulate(obs, m, FusionConfig(), root)
            d, hit = infer(obs, m, f, DecodeConfig(), root / 'a'); self.assertFalse(hit)
            reused, hit = infer(obs, m, f, DecodeConfig(), root / 'a'); self.assertTrue(hit)
            fresh, hit = infer(obs, m, f, DecodeConfig(), root / 'b'); self.assertFalse(hit)
            self.assertEqual(d.manifest, reused.manifest); self.assertEqual(d.manifest, fresh.manifest)
            self.assertEqual(output_events(d), output_events(fresh))
            (d.path / 'chunk-000000.npz').write_bytes(b'broken')
            with self.assertRaises(CacheError): open_decode(d.path)
            _, hit = infer(obs, m, f, DecodeConfig(), root / 'a'); self.assertFalse(hit)
            other = observations(root, [0, 8])
            with self.assertRaisesRegex(ValueError, 'lineage'): infer(other, m, f, DecodeConfig(), root)

    def test_rechecksummed_invalid_selection_is_rejected_and_regenerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); obs = observations(root, [0, 8, 16, 24])
            m, _ = estimate(obs, MotionConfig(), root)
            f, _ = accumulate(obs, m, FusionConfig(), root)
            d, _ = infer(obs, m, f, DecodeConfig(), root)
            for field, message in (('path', 'alternative cost'), ('terms', 'cost terms'),
                                   ('review', 'review diagnostics')):
                with self.subTest(field=field):
                    data = next(lanes(d))
                    selected_edge = next(c for c in data['candidates'] if c['selected'])
                    if field == 'path': selected_edge['path_cost'] += 20
                    elif field == 'terms': selected_edge['cost_terms']['note'] += 20
                    else: data['events'][0]['review_causes'] = ['invented cause']
                    chunk = d.path / 'chunk-000000.npz'
                    np.savez_compressed(chunk, payload=np.frombuffer(canonical(data), dtype=np.uint8))
                    manifest = d.manifest
                    manifest['chunks'][0]['sha256'] = file_hash(chunk)
                    with np.load(chunk) as a:
                        manifest['chunks'][0]['arrays']['payload']['shape'] = list(a['payload'].shape)
                    manifest['manifest_sha256'] = digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'})
                    write_json(d.path / 'manifest.json', manifest)
                    with self.assertRaisesRegex(CacheError, message): open_decode(d.path)
                    d, hit = infer(obs, m, f, DecodeConfig(), root)
                    self.assertFalse(hit)
                    self.assertTrue(list(lanes(d)))

    def test_endpoint_support_respects_fusion_group_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); obs = observations(root, [0, 8, 8, 8, 16], duplicate_seeds=True)
            m, _ = estimate(obs, MotionConfig(), root)
            f, _ = accumulate(obs, m, FusionConfig(repeat_group_cap=.1), root)
            d, _ = infer(obs, m, f, DecodeConfig(), root)
            for lane in lanes(d):
                for ep in lane['endpoints']:
                    self.assertLessEqual(ep['weight'], ep['group_count'] * .1 + 1e-10)

    def test_decode_only_parameters_and_pending_timing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); path, _ = fixture(root)
            _, run = execute(path, output_root=root / 'out')
            events = json.loads((run / 'artifacts/events.json').read_text())
            self.assertTrue((run / events['decode_ref']).is_file())
            self.assertEqual(events['pending_stages'], ['timing'])
            candidate_ids = {c['id'] for c in events['candidates']}
            self.assertTrue(all(set(e['candidate_ids']) <= candidate_ids for e in events['events']))
            self.assertTrue(all(e['start_ms'] is None and e['end_ms'] is None for e in events['events']))
            self.assertEqual(json.loads((run / 'chart.json').read_text())['notes'], [])
            (root / 'algo.toml').write_text('schema_version = 1\nchunk_frames = 2\n[decode]\nambiguity_cost = 0.7\n')
            with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no decoding')):
                code, second = execute(path, output_root=root / 'out', observation_cache=run / 'artifacts/observations')
            self.assertEqual(code, 2)
            stages = json.loads((second / 'run.json').read_text())['stages']
            self.assertTrue(stages['motion']['reused']); self.assertTrue(stages['fusion']['reused'])
            self.assertFalse(stages['decode']['reused'])
            path.write_text(path.read_text().replace('191.910', '200').replace('Synthetic', 'Renamed'))
            code, third = execute(path, output_root=root / 'out', observation_cache=run / 'artifacts/observations')
            self.assertEqual(code, 2)
            stages = json.loads((third / 'run.json').read_text())['stages']
            self.assertTrue(stages['decode']['reused'])
            self.assertEqual(AlgorithmConfig().observation_parameters(),
                             replace(AlgorithmConfig(), decode=DecodeConfig(note_cost=1)).observation_parameters())

    def test_decode_config_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'algorithm.toml'
            for text in ('note_cost = true', 'body_bright = 0.01', 'endpoint_radius_px = 0', 'unknown = 3',
                         'association_similarity = 1.1', 'body_support_px = 0'):
                path.write_text('schema_version = 1\n[decode]\n' + text)
                with self.assertRaises(ConfigError): load_algorithm(path)
