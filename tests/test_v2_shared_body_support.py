"""Procedural evidence accounting and asset-free cache reconstruction."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from ez2rc.v2 import decode
from ez2rc.v2.config import DecodeConfig,load_song,load_profile,load_algorithm
from ez2rc.v2.decode import BodyEvidence,endpoint_footprints,shared_body_footprints,union_ranges
from ez2rc.v2.observation import cache_keys,template_shapes
from ez2rc.v2.storage import CacheError,CacheWriter,canonical
from ez2rc.v2.runner import execute
from v2_fixtures import fixture
import v2_decode_fixtures as fixtures


def accounted_scene(*args,height=28.,**kwargs):
    original_sites,original_summary=fixtures.endpoint_sites,fixtures.summarize_lane
    def sites(lane,ledger,scores,flags,clock,meta,config):
        sites,eps=original_sites(lane,ledger,scores,flags,clock,meta,config)
        endpoint_footprints(eps,ledger,flags,[height]*3,[0.]*3)
        return sites,eps
    def summary(lane,sites,endpoints,body,config):
        # Reconstruct an equivalent body from its original primitive cells.
        masked=BodyEvidence(body.source_cells,body.grid,config,
                            endpoint_footprints=shared_body_footprints(sites,endpoints,config))
        return original_summary(lane,sites,endpoints,masked,config)
    original_body=fixtures.BodyEvidence
    def body(cells,*args,**kwargs):
        result=original_body(cells,*args,**kwargs);result.source_cells=cells
        return result
    with patch.object(fixtures,'endpoint_sites',side_effect=sites),patch.object(fixtures,'BodyEvidence',side_effect=body),patch.object(fixtures,'summarize_lane',side_effect=summary):
        return fixtures.scene(*args,**kwargs)


class SharedBodySupportTest(unittest.TestCase):
    def test_short_visible_boundary_endpoint_survives_body_only_competition(self):
        for q in (12,28,44):
            for kind in (1,2):
                at=q if kind==2 else 200-q
                body=(0,at) if kind==2 else (at,200)
                expected=('longnote',None,float(at)) if kind==2 else ('longnote',float(at),None)
                result=accounted_scene([(at,kind)],[body],height=q)
                self.assertEqual(fixtures.selected(result),[expected])
                self.assertEqual(result['events'][0]['status'],'incomplete')
                self.assertTrue(any(c['start'] is None and c['end'] is None for c in result['candidates']))

    def test_tap_sprite_is_not_independent_longnote_body_support(self):
        args=([(48,[.96,.12,.82])],[(20,48)])
        before=fixtures.scene(*args);after=accounted_scene(*args)
        self.assertEqual(fixtures.selected(before),fixtures.selected(after))
        self.assertEqual(before['events'][0]['status'],'ambiguous')
        self.assertEqual(after['events'][0]['status'],'resolved')
        self.assertEqual(len(before['candidates']),len(after['candidates']))
        for c in after['candidates']:
            if c['body']:
                self.assertLessEqual(c['body']['endpoint_overlap_supported_px'],c['body']['supported_bright_px'])

    def test_motion_and_remaining_near_cost_alternatives_stay_ambiguous(self):
        args=([(48,[.96,.12,.82])],[(20,48)])
        motion=accounted_scene(*args,ambiguous_motion=True)
        self.assertEqual(motion['events'][0]['status'],'ambiguous')
        self.assertIn('motion_alternatives',motion['events'][0]['review_causes'])
        competing=accounted_scene(*args,config=DecodeConfig(missing_endpoint_cost=0.))
        self.assertEqual(competing['events'][0]['status'],'ambiguous')
        self.assertTrue(competing['events'][0]['has_alternatives'])

    def test_short_complete_notes_survive_shared_footprints(self):
        for length in (1,2,9,40):
            data=accounted_scene([(60,1),(60+length,2)],[(60,60+length)])
            self.assertEqual(fixtures.selected(data),[('longnote',60.,60.+length)])

    def test_body_only_without_supported_endpoint_keeps_its_reward(self):
        for points in ([],[(50,[.4,.5,.6])]):
            args=(points,[(20,100)])
            a=fixtures.scene(*args);b=accounted_scene(*args)
            self.assertEqual(fixtures.selected(a),fixtures.selected(b))
            self.assertEqual(b['shared_body_footprints_px'],[])
            self.assertTrue(all(c['body']['endpoint_overlap_supported_px']==0 for c in b['candidates'] if c['body']))

    def test_unknown_geometry_and_clipped_evidence_cannot_mask_body(self):
        args=([(48,[.96,.12,.82])],[(20,48)])
        before=fixtures.scene(*args)
        after=accounted_scene(*args,height=None)
        self.assertEqual(before['events'],after['events'])
        marked=accounted_scene(*args,flags=[[1,1,1]])
        self.assertEqual(marked['shared_body_footprints_px'],[])

    def test_missing_geometry_in_one_supported_interpretation_is_not_ignored(self):
        data=accounted_scene([(48,[.96,.12,.82])],[(20,48)])
        endpoint=next(e for e in data['endpoints'] if e['kind']=='tail')
        endpoint['template_footprint_px']=None
        self.assertEqual(shared_body_footprints(data['sites'],data['endpoints'],DecodeConfig()),[])

    def test_darkness_unavailable_and_sparse_support_keep_original_meaning(self):
        for weight in (.05,.5,2.):
            args=([(20,1),(120,2)],[(20,120)])
            before=fixtures.scene(*args,body_weight=weight,absent=[(50,60)],unavailable=[(80,90)])
            after=accounted_scene(*args,body_weight=weight,absent=[(50,60)],unavailable=[(80,90)])
            for a,b in zip(before['candidates'],after['candidates']):
                if a['body']:
                    for k,v in a['body'].items():self.assertEqual(v,b['body'][k])
                    self.assertLessEqual(b['body']['endpoint_overlap_supported_px'],b['body']['supported_bright_px'])

    def test_fractional_mask_union_matches_scalar_pixel_integrals(self):
        q=np.array([-2,0,4,6.]);support=np.array([.2,.6,.4,.8])
        cells={'index':np.column_stack([np.zeros(4,int),q/2]),'weight_sum':np.zeros((4,4)),
               'score_sum':np.zeros((4,4)),'visibility_weight':np.zeros((4,4,5))}
        cells['weight_sum'][:,3]=support;cells['visibility_weight'][:,3,1]=support;cells['score_sum'][:,3]=support*.6
        masks=[[-1.5,.3],[0,1.5],[4.25,7.3],[5.,6.]]
        merged=union_ranges(masks)
        body=BodyEvidence(cells,2.,DecodeConfig(),endpoint_footprints=masks)
        for a,b in ((-3,9),(-1,5),(0,0),(2,4),(4.5,6.5)):
            expected=sum(max(0.,min(b,y,x+2)-max(a,z,x))*v for x,v in zip(q,support) for z,y in merged)
            self.assertAlmostEqual(body.summary(a,b)['endpoint_overlap_supported_px'],expected)

    def test_total_offsets_are_removed_before_screen_rectangle(self):
        endpoint={'kind':'tail','observation_ids':['f0:l0:y0','f1:l0:y0','f2:l0:y0']}
        ledger={'endpoint_index':np.array([[0,0,0],[1,0,0],[2,0,0]]),
                'endpoint_q':np.repeat(np.array([70.,72.,150.])[:,None],3,axis=1),'endpoint_weight':np.ones((3,3))}
        flags=np.zeros((3,3),np.uint8);flags[2,2]=1
        endpoint_footprints([endpoint],ledger,flags,[12.]*3,[0.,0.,4.])
        self.assertEqual(endpoint['template_footprint_px'],[64.,74.])
        flags[:]=0
        endpoint_footprints([endpoint],ledger,flags,[12.]*3,[0.]*3)
        self.assertIsNone(endpoint['template_footprint_px'])

    def test_cache_validation_rejects_mask_and_overlap_tampering(self):
        data=accounted_scene([(48,[.96,.12,.82])],[(20,48)])
        decode._validate_lane(0,data,DecodeConfig(),1.)
        for field in ('mask','negative','too_large','missing'):
            bad=deepcopy(data);body=next(c['body'] for c in bad['candidates'] if c['body'])
            if field=='mask':bad['shared_body_footprints_px']=[[0.,100.]]
            if field=='negative':body['endpoint_overlap_supported_px']=-1
            if field=='too_large':body['endpoint_overlap_supported_px']=body['supported_bright_px']+1
            if field=='missing':del body['endpoint_overlap_supported_px']
            with self.subTest(field=field),self.assertRaises(CacheError):decode._validate_lane(0,bad,DecodeConfig(),1.)


class TemplateGeometryTest(unittest.TestCase):
    def test_geometry_is_keyed_and_validated_with_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);song,_=fixture(root)
            song=load_song(song);profile=load_profile(song.profile);algo=load_algorithm(None)
            before=cache_keys(song,profile,algo,'video')
            self.assertEqual(template_shapes(before[1]['parameters'])[0]['tap'],[7,16])
            profile.lanes[0].templates['tap']=profile.lanes[0].templates['tap'][:-1]
            after=cache_keys(song,profile,algo,'video')
            self.assertNotEqual(before[1],after[1]);self.assertEqual(before[0],after[0])
            self.assertIsNone(template_shapes({'profile':profile.snapshot}))
            for value in ([],[{}],{'tap':[7,16]}):
                bad=deepcopy(before[1]['parameters']);bad['template_shapes']=value
                with self.assertRaises(CacheError):template_shapes(bad)
            for size in ([0,16],[True,16],[7.5,16],[1000,16],[7,1000]):
                bad=deepcopy(before[1]['parameters']);bad['template_shapes'][0]['tap']=size
                with self.assertRaises(CacheError):template_shapes(bad)

    def test_extract_and_asset_free_reconstruct_use_identical_geometry(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);song,_=fixture(root)
            code,first=execute(song,output_root=root/'out');self.assertEqual(code,2)
            manifest=json.loads((first/'artifacts/observations/manifest.json').read_text())
            self.assertIn('template_shapes',manifest['key']['parameters'])
            old=[canonical(x) for x in decode.lanes(decode.open_decode(first/'artifacts/decode'))]
            for f in ('video.mkv','profile.toml','note.png'):(root/f).unlink()
            with patch('ez2rc.v2.capture.av.open',side_effect=AssertionError('no video needed')):
                code,second=execute(song,output_root=root/'out',observation_cache=first/'artifacts/observations')
            self.assertEqual(code,2)
            cache=decode.open_decode(second/'artifacts/decode')
            self.assertEqual(cache.manifest['metadata']['body_overlap_policy'],'shared_supported_template_footprints')
            self.assertEqual(old,[canonical(x) for x in decode.lanes(cache)])

    def test_cache_cannot_claim_overlap_accounting_without_its_lane_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);song,_=fixture(root)
            code,run=execute(song,output_root=root/'out');self.assertEqual(code,2)
            cache=decode.open_decode(run/'artifacts/decode')
            metadata=deepcopy(cache.manifest['metadata'])
            metadata['body_overlap_policy']='unavailable_template_geometry'
            with CacheWriter(root/'corrupt',cache.manifest['key']) as writer:
                for chunk in cache.chunks():writer.append(**chunk)
                corrupted=writer.commit(metadata)
            with self.assertRaisesRegex(CacheError,'overlap policy'):decode.open_decode(corrupted.path)


if __name__=='__main__':unittest.main()
