from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from ez2rc.v2.config import AlgorithmConfig, MotionConfig, FusionConfig, ConfigError, load_algorithm
from ez2rc.v2.motion import estimate, timeline, STATUS, cache_key as motion_key, open_motion
from ez2rc.v2.fusion import accumulate, cache_key as fusion_key, open_fusion
from ez2rc.v2.observation import cache_keys
from ez2rc.v2.storage import CacheError, digest, file_hash, write_json
from ez2rc.v2.runner import execute
from v2_motion_fixtures import observations, evidence_arrays
from v2_fixtures import fixture


class MotionFusionTest(unittest.TestCase):
    def test_common_motion_repeats_skips_and_fixed_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = [0, 8, 16, 16, 16, 40, 48, 64, 72, 80, 96]
            obs = observations(root, truth, static=True, chunk_frames=3)
            motion, _ = estimate(obs, MotionConfig(), root)
            a = timeline(motion)
            np.testing.assert_allclose(a['scroll_px'], truth, atol=.5)
            self.assertTrue(a['repeat_confirmed'][2:5].all())
            self.assertEqual(a['repeat_group'][2:5].tolist(), [102]*3)
            self.assertTrue((a['support_lanes'][[1, 2, 5, 6]] >= 2).all())
            self.assertTrue((a['status'][3:5] == STATUS['repeated']).all())
            links = np.concatenate([c['links'] for c in motion.chunks()])
            self.assertFalse(np.isin(links[:, 4], [12, 32, 145]).any())
            self.assertTrue((links[:, 1] < links[:, 0]).all())

    def test_sparse_gap_uses_pts_and_keeps_uncertainty(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = [0, 8, 16, 40, 48, 56, 64]
            times = [0, 16, 32, 80, 96, 112, 128]
            obs = observations(root, truth, blank=(3, 4), times=times)
            motion, _ = estimate(obs, MotionConfig(), root)
            a = timeline(motion)
            np.testing.assert_allclose(a['scroll_px'], truth, atol=.5)
            self.assertEqual(a['status'][3], STATUS['extrapolated'])
            self.assertGreater(a['error_px'][4], a['error_px'][2])
            self.assertGreater(a['step_error_px'][3], 200)
            self.assertEqual(a['status'][5], STATUS['observed'])

    def test_static_and_empty_frames_cannot_confirm_scroll(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for blank in ((), tuple(range(6))):
                obs = observations(root, [0]*6, blank=blank, static=True)
                motion, _ = estimate(obs, MotionConfig(), root)
                a = timeline(motion)
                self.assertFalse(a['repeat_confirmed'].any())
                self.assertTrue((a['status'][1:] == STATUS['unresolved']).all())
                self.assertGreater(a['error_px'][-1], 0)

    def test_single_lane_never_claims_multi_lane_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obs = observations(root, [0, 8, 16, 24], one_lane=range(4))
            motion, _ = estimate(obs, MotionConfig(), root)
            a = timeline(motion)
            self.assertTrue(np.isin(a['status'][1:], [STATUS['single_lane'], STATUS['ambiguous']]).all())
            self.assertTrue((a['support_lanes'][1:] == 1).all())
            np.testing.assert_allclose(a['scroll_px'], [0, 8, 16, 24])

    def test_displacement_aliases_remain_alternatives(self):
        from ez2rc.v2.motion import match
        a = np.array([[lane, 0, y, y, .99] for lane in (0, 1) for y in (10, 50, 90)], float)
        b = a.copy(); b[:, 2:4] += 8
        models = match(a, b, 80, MotionConfig())
        self.assertIn(8, [m['delta'] for m in models])
        self.assertIn(48, [m['delta'] for m in models])
        self.assertAlmostEqual(models[0]['quality'], models[1]['quality'])

    def test_chunks_do_not_split_long_body_or_duplicate_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            truth = list(range(0, 425, 8))
            results = []
            for chunk in (3, 7, 120):
                obs = observations(root, truth, chunk_frames=chunk)
                motion, _ = estimate(obs, MotionConfig(), root)
                np.testing.assert_allclose(timeline(motion)['scroll_px'], truth, atol=.5)
                fusion, _ = accumulate(obs, motion, FusionConfig(tile_bins=32), root)
                results.append(evidence_arrays(fusion))
                self.assertLess(fusion.manifest['metadata']['max_active_tiles'], 12)
            for other in results[1:]:
                for k in results[0]:
                    np.testing.assert_array_equal(results[0][k], other[k], err_msg=k)
            a = results[0]
            body = (a['index'][:, 0] == 2) & (a['index'][:, 1] >= 100) & (a['index'][:, 1] <= 480)
            self.assertEqual(body.sum(), 381)
            brightness = a['score_sum'][body, 3] / a['weight_sum'][body, 3]
            np.testing.assert_allclose(brightness, 100/255, atol=.025)
            for kind, q in ((1, 90), (2, 490)):
                at = (a['index'][:, 0] == 2) & (a['index'][:, 1] == q)
                self.assertGreater(a['score_sum'][at, kind].max()/a['weight_sum'][at, kind].max(), .98)
            self.assertEqual(fusion.manifest['metadata']['endpoint_timing_error_px'], [None]*3)

    def test_duplicate_seeds_and_repeat_groups_do_not_inflate_support(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = []
            for truth, duplicate in (([0,8,16,24], False), ([0,8,8,8,16,24], True)):
                obs = observations(root, truth, duplicate_seeds=duplicate, chunk_frames=2)
                motion, _ = estimate(obs, MotionConfig(), root)
                fusion, _ = accumulate(obs, motion, FusionConfig(tile_bins=32), root)
                results.append(evidence_arrays(fusion))
            for k in ('index','score_sum','weight_sum','group_count'):
                np.testing.assert_allclose(results[0][k], results[1][k], err_msg=k)
            self.assertGreater(len(results[1]['endpoint_index']), len(results[0]['endpoint_index']))

    def test_visibility_weights_and_observed_darkness_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            totals = []
            for flag in (0, 4, 1, 8, 2):
                obs = observations(root, [0, 8], flags=flag)
                motion, _ = estimate(obs, MotionConfig(), root)
                fusion, _ = accumulate(obs, motion, FusionConfig(), root)
                a = evidence_arrays(fusion)
                first = a['endpoint_index'][:, 0] == 100
                # Fixed effect candidates always have weight zero.
                moving = first & (a['endpoint_index'][:, 2] != 145)
                totals.append(a['endpoint_weight'][moving, 0].max())
                body = a['weight_sum'][:, 3] > 0
                self.assertTrue((a['score_sum'][body, 3] == 0).any())
                self.assertTrue((a['visibility_weight'][:, 3, 1] > 0).any())
            np.testing.assert_allclose(totals, [1,.5,.25,.1,0])

    def test_cache_keys_separate_motion_fusion_and_observation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            path,_=fixture(root)
            from ez2rc.v2.config import load_song, load_profile
            song=load_song(path); profile=load_profile(song.profile); algo=AlgorithmConfig()
            self.assertEqual(cache_keys(song,profile,algo,'hash'),cache_keys(song,profile,replace(algo,motion=MotionConfig(min_similarity=.8),fusion=FusionConfig(unknown_weight=.3)),'hash'))
            obs=observations(root,[0,8,16]); motion,_=estimate(obs,algo.motion,root)
            self.assertNotEqual(motion_key(obs,algo.motion),motion_key(obs,MotionConfig(min_similarity=.8)))
            self.assertNotEqual(fusion_key(obs,motion,algo.fusion),fusion_key(obs,motion,FusionConfig(unknown_weight=.3)))

    def test_new_calculation_and_cache_reuse_are_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); obs=observations(root,[0,8,16,16,32,48])
            m,_=estimate(obs,MotionConfig(),root/'a'); f,_=accumulate(obs,m,FusionConfig(),root/'a')
            m2,hit=estimate(obs,MotionConfig(),root/'a'); self.assertTrue(hit)
            f2,hit=accumulate(obs,m2,FusionConfig(),root/'a'); self.assertTrue(hit)
            m3,hit=estimate(obs,MotionConfig(),root/'b'); self.assertFalse(hit)
            f3,hit=accumulate(obs,m3,FusionConfig(),root/'b'); self.assertFalse(hit)
            self.assertEqual(m.manifest,m2.manifest); self.assertEqual(m.manifest,m3.manifest)
            self.assertEqual(f.manifest,f2.manifest); self.assertEqual(f.manifest,f3.manifest)

    def test_corrupt_and_semantically_invalid_caches_are_regenerated(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); obs=observations(root,[0,8,16])
            m,_=estimate(obs,MotionConfig(),root)
            chunk=m.path/'chunk-000000.npz'; data=next(m.chunks()); data['scroll_px'][1]=-1
            np.savez_compressed(chunk,**data)
            manifest=m.manifest; manifest['chunks'][0]['sha256']=file_hash(chunk)
            manifest['manifest_sha256']=digest({k:v for k,v in manifest.items() if k!='manifest_sha256'})
            write_json(m.path/'manifest.json',manifest)
            with self.assertRaises(CacheError): open_motion(m.path)
            repaired,hit=estimate(obs,MotionConfig(),root); self.assertFalse(hit)
            np.testing.assert_allclose(timeline(repaired)['scroll_px'],[0,8,16])
            fusion,_=accumulate(obs,repaired,FusionConfig(),root)
            (fusion.path/'chunk-000000.npz').write_bytes(b'broken')
            with self.assertRaises(CacheError): open_fusion(fusion.path)
            _,hit=accumulate(obs,repaired,FusionConfig(),root); self.assertFalse(hit)

    def test_pipeline_reconstruct_needs_no_decoder_and_recomputes_from_changed_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,_=fixture(root)
            code,run=execute(path,output_root=root/'out'); self.assertEqual(code,2)
            status=json.loads((run/'run.json').read_text())
            self.assertEqual(status['stages']['motion']['status'],'complete')
            self.assertEqual(status['stages']['fusion']['status'],'complete')
            chart=json.loads((run/'chart.json').read_text()); self.assertEqual(chart['pending_stages'],['decode','timing'])
            events=json.loads((run/'artifacts/events.json').read_text())
            self.assertTrue((run/events['fusion_ref']).is_file())
            (root/'algo.toml').write_text('schema_version = 1\nchunk_frames = 2\n[fusion]\nunknown_weight = 0.3\n')
            with patch('ez2rc.v2.capture.av.open',side_effect=AssertionError('no decoding')):
                code,again=execute(path,output_root=root/'out',observation_cache=run/'artifacts/observations')
            self.assertEqual(code,2)
            status=json.loads((again/'run.json').read_text())
            self.assertTrue(status['stages']['motion']['reused'])
            self.assertFalse(status['stages']['fusion']['reused'])

    def test_algorithm_tables_validate_types_and_ranges(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'algo.toml'
            for bad in ('[motion]\nmin_lanes = 1', '[motion]\nmax_speed_px_per_ms = 0', '[fusion]\nunknown_weight = 1.5', '[fusion]\ntile_bins = true','[motion]\nunknown = 1'):
                path.write_text('schema_version = 1\n'+bad)
                with self.assertRaises(ConfigError): load_algorithm(path)

    def test_close_weak_candidates_remain_at_distinct_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); obs=observations(root,[0,8,16,24],close=True)
            motion,_=estimate(obs,MotionConfig(),root)
            fusion,_=accumulate(obs,motion,FusionConfig(),root)
            a=evidence_arrays(fusion)
            for q,score in ((101,.99),(107,.45)):
                cell=(a['index'][:,0]==0)&(a['index'][:,1]==q)
                self.assertAlmostEqual(float((a['score_sum'][cell,0]/a['weight_sum'][cell,0])[0]),score,places=5)
            self.assertTrue(np.isclose(a['endpoint_q'][:,0],107).any())

    def test_repeat_cap_survives_output_and_observation_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            results=[]
            for truth in ([0,8,16], [0]+[8]*125+[16]):
                obs=observations(root,truth,chunk_frames=7,near_repeat=True)
                motion,_=estimate(obs,MotionConfig(),root)
                fusion,_=accumulate(obs,motion,FusionConfig(tile_bins=32),root)
                results.append(evidence_arrays(fusion))
            for k in ('index','score_sum','weight_sum','group_count'):
                # The procedural near-repeat changes only row zero, which is
                # clipped; moving endpoint and interior body support is capped.
                keep=(results[0]['index'][:,1] < 150)
                np.testing.assert_allclose(results[0][k][keep],results[1][k][keep],err_msg=k)

    def test_independent_verified_offsets_do_not_borrow_from_other_kinds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            c={k:{'status':'verified','detection_reference':'template_top','version':'synthetic',
                  'offset_px':offset,'error_px':.2} for k,offset in zip(('tap','head','tail'),(2,4,-3))}
            obs=observations(root,[0,8,16],calibration=c)
            motion,_=estimate(obs,MotionConfig(),root)
            fusion,_=accumulate(obs,motion,FusionConfig(),root)
            a=evidence_arrays(fusion)
            endpoint=(a['endpoint_index'][:,0]==100)&(a['endpoint_index'][:,1]==0)&(a['endpoint_index'][:,2]==49)
            np.testing.assert_allclose(a['endpoint_q'][endpoint],[ [99,97,104] ])
            self.assertEqual(fusion.manifest['metadata']['endpoint_timing_error_px'],[.2]*3)

    def test_foreign_motion_lineage_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); a=observations(root,[0,8]); b=observations(root,[0,16])
            motion,_=estimate(a,MotionConfig(),root)
            with self.assertRaisesRegex(ValueError,'lineage'):
                accumulate(b,motion,FusionConfig(),root)

    def test_gap_larger_than_viewport_does_not_link_unrelated_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            obs=observations(root,[0,8,16,256,264,272],times=[0,16,32,512,528,544])
            motion,_=estimate(obs,MotionConfig(),root)
            a=timeline(motion)
            self.assertEqual(a['status'][3],STATUS['extrapolated'])
            self.assertEqual(a['reference_frame'][3],-1)
            self.assertGreater(a['step_error_px'][3],2000)
            np.testing.assert_allclose(a['scroll_px'],[0,8,16,256,264,272],atol=.5)


if __name__ == '__main__':
    unittest.main()
