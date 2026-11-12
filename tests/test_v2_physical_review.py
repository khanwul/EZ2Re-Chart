"""Generated identity reviews keep physical decisions separate from timing."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ez2rc.v2.decode import lanes, open_decode
from ez2rc.v2.export import read_chart, read_events
from ez2rc.v2.motion import open_motion
from ez2rc.v2.observation import open_observations
from ez2rc.v2.physical_review import VERSION, LOCALIZATION_VERSION, load_physical_review
from ez2rc.v2.runner import execute
from ez2rc.v2.relative_beats import project_relative_beats
from ez2rc.v2.rhythm import RhythmGrid, rhythm_candidates
from ez2rc.v2.spatial_resolution import resolve_spatial_chart
from test_v2_barline_pipeline import marked_video
from test_v2_relative_beats import reference, scene


def review_for(observations, movement, decoded, lane_data):
    value = {'format':'ez2rc.physical-review','schema_version':1,'version':VERSION,
             'video_sha256':observations.manifest['metadata']['video_sha256'],
             'observation_cache_id':observations.id,'motion_cache_id':movement.id,'decode_cache_id':decoded.id,
             'reviewer':{'id':'procedural-scene-author','kind':'human'},
             'review_context':'Procedurally specified identity witnesses; no independent accuracy claim.',
             'evidence':[{'id':'scene-specification','sha256':hashlib.sha256(b'procedural specification').hexdigest()}],
             'decisions':[]}
    for lane in lane_data:
        endpoints={e['id']:e for e in lane['endpoints']}
        candidates={c['id']:c for c in lane['candidates']}
        for event in lane['events']:
            for cid in event['selected_candidate_ids']:
                candidate=candidates[cid]
                value['decisions'].append({'event_id':event['id'],'candidate_id':cid,'lane':lane['lane'],
                    'type':candidate['type'], 'evidence_ids':['scene-specification'],
                    **{side:{'endpoint_id':candidate[side],
                             'witness_observation_ids':endpoints[candidate[side]]['observation_ids'][:1]}
                       if candidate[side] else None for side in ('start','end')}})
    return value


def save_review(root, value):
    data=json.dumps(value).encode();path=root/'review.json';path.write_bytes(data)
    return {'path':str(path),'sha256':hashlib.sha256(data).hexdigest()}


class PhysicalReviewTest(unittest.TestCase):
    def test_reviewed_pixels_require_bound_evidence_and_scoped_verification(self):
        from ez2rc.v2.observation import template_shapes
        from ez2rc.v2.observed_geometry import search_censored
        from ez2rc.v2.calibration_verification import VERSION as VERIFY_VERSION,code_fingerprint,measurement_scope
        from ez2rc.v2.storage import digest
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);song=marked_video(root)
            song.write_text(song.read_text()+'\n[timing.rhythm_grid]\nsubdivisions = [4]\n')
            _,first=execute(song,output_root=root/'out')
            obs=open_observations(first/'artifacts/observations')
            movement=open_motion(first/'artifacts/motion');decoded=open_decode(first/'artifacts/decode')
            lane_data=list(lanes(decoded));value=review_for(obs,movement,decoded,lane_data)
            decision=value['decisions'][0];point=decision['start'];eid=point['endpoint_id']
            endpoint=next(e for lane in lane_data for e in lane['endpoints'] if e['id']==eid)
            parameters=obs.manifest['key']['parameters'];profile=obs.manifest['metadata']['profile']
            shapes=template_shapes(parameters);lane=decision['lane'];roi=profile['lanes'][lane]['roi']
            chosen=None
            for chunk in obs.chunks():
                for index,(frame,lid,seed) in enumerate(chunk['candidate_index']):
                    oid=f'f{frame}:l{lid}:y{seed}';y=float(chunk['positions'][index,0,1])
                    if (oid not in endpoint['observation_ids'] or int(chunk['flags'][index,0])&3
                            or float(chunk['scores'][index,0])<.9 or not roi[1]+3<y<roi[3]-shapes[lane]['tap'][0]-3
                            or search_censored(int(seed),y,roi,shapes[lane]['tap'][0],parameters['algorithm']['search_radius_px'])):continue
                    chosen=(oid,int(frame),y);break
                if chosen:break
            self.assertIsNotNone(chosen);oid,frame,y=chosen
            point['witness_observation_ids']=[oid]
            value.update(schema_version=2,version=LOCALIZATION_VERSION,localizations=[{
                'endpoint_id':eid,'observation_id':oid,'frame_id':frame,'template_top_interval_px':[y-1,y+1],
                'visibility':'fully_visible_unoccluded','evidence_ids':['scene-specification']}])
            review=load_physical_review(save_review(root,value),obs,movement,decoded)
            self.assertEqual(review.receipt()['localization_count'],1)
            for change in ('frame','witness','interval','evidence','visibility','duplicate','beat','version'):
                altered=deepcopy(value);row=altered['localizations'][0]
                if change=='frame':row['frame_id']+=1
                if change=='witness':row['observation_id']='f999:l0:y20'
                if change=='interval':row['template_top_interval_px']=[y+1,y+2]
                if change=='evidence':row['evidence_ids']=['absent']
                if change=='visibility':row['visibility']='unknown'
                if change=='duplicate':altered['localizations']*=2
                if change=='beat':row['beat']=[1,2]
                if change=='version':altered.update(schema_version=1,version=VERSION)
                with self.subTest(change=change),self.assertRaises(ValueError):
                    load_physical_review(save_review(root,altered),obs,movement,decoded)
            binding=save_review(root,value)
            calibration={'format':'ez2rc.spatial-calibration','schema_version':1,'version':'procedural-pixels',
                'observation_cache_id':obs.id,'motion_cache_id':movement.id,'decode_cache_id':decoded.id,
                'localization_error_px':1,'endpoint_calibrations':[]}
            for lane_record in lane_data:
                for candidate in lane_record['candidates']:
                    if candidate['selected']:
                        calibration['endpoint_calibrations'].append({'endpoint_id':candidate['start'],
                            'candidate_id':candidate['id'],'version':'procedural-template-top',
                            'offset_px':[0,0],'position_error_px':1})
            data=json.dumps(calibration).encode();(root/'calibration.json').write_bytes(data)
            song.write_text(song.read_text()+f'\n[timing.spatial_calibration]\npath = "calibration.json"\nsha256 = "{hashlib.sha256(data).hexdigest()}"\n'
                            +f'\n[timing.physical_review]\npath = "review.json"\nsha256 = "{binding["sha256"]}"\n')
            code,conditional=execute(song,output_root=root/'out');self.assertEqual(code,2)
            relative=json.loads((conditional/'artifacts/relative-beats.json').read_text())
            audit=json.loads((conditional/'artifacts/spatial-resolution.json').read_text())
            self.assertEqual(relative['observed_geometry']['reviewed_localizations'],value['localizations'])
            self.assertIn('reviewed_localization_requires_scoped_verification',audit['reasons'])
            review.check_relative(relative)
            tampered=deepcopy(relative);tampered['observed_geometry']['reviewed_localizations']=[]
            with self.assertRaises(ValueError):review.check_relative(tampered)
            verification={'format':'ez2rc.calibration-verification','schema_version':1,'version':VERIFY_VERSION,
                'status':'verified','code_sha256':code_fingerprint(),'scope':measurement_scope(obs,relative),
                'method':'procedural_measurement','assessor':'procedural renderer',
                'criteria_sha256':digest({'known_template_and_reviewed_pixel_bounds':True}),
                'evidence':[{'id':'procedural-renderer','sha256':digest({'known_scene':'marked_video'})}],
                'limitations':'This generated renderer and its explicit pixel review only.'}
            data=json.dumps(verification).encode();(root/'verification.json').write_bytes(data)
            song.write_text(song.read_text()+f'\n[timing.calibration_verification]\npath = "verification.json"\nsha256 = "{hashlib.sha256(data).hexdigest()}"\n')
            code,confirmed=execute(song,output_root=root/'out');self.assertEqual(code,0,(confirmed/'run.json').read_text())
            chart=read_chart(confirmed/'chart.json')
            self.assertEqual([(n['lane'],F(*n['start_beat'])) for n in chart['notes']],
                             [(0,F(1,2)),(1,F(3,4)),(0,F(3,2)),(1,F(11,4)),(0,F(7,2)),(1,F(19,4))])
            for name in ('video.mkv','profile.toml','note.png','head.png','tail.png'):(root/name).unlink()
            with patch('ez2rc.v2.capture.av.open',side_effect=AssertionError('no source decoding')):
                code,reconstructed=execute(song,output_root=root/'out',observation_cache=first/'artifacts/observations')
            self.assertEqual(code,0)
            for artifact in ('chart.json','artifacts/events.json','artifacts/physical-review.json',
                             'artifacts/relative-beats.json','artifacts/spatial-resolution.json'):
                self.assertEqual((confirmed/artifact).read_bytes(),(reconstructed/artifact).read_bytes())

    def test_review_with_independent_spatial_reference_preserves_conditional_output(self):
        from ez2rc.v2.contracts import Chart, Event, Metadata, Timing
        obs,movement,decoded,clock,lane=scene()
        obs.id,movement.id,decoded.id='a'*64,'b'*64,'c'*64
        obs.manifest['metadata']['video_sha256']='d'*64
        movement.manifest['key']['parents']=[obs.id]
        decoded.manifest['key']['parents']=[obs.id,movement.id,'f'*64]
        ref=reference()
        for key,cache in [('observation_cache_id',obs),('motion_cache_id',movement),('decode_cache_id',decoded)]:
            ref[key]=cache.id
        with tempfile.TemporaryDirectory() as directory, patch('ez2rc.v2.physical_review.lanes',return_value=[lane]):
            review=load_physical_review(save_review(Path(directory),review_for(obs,movement,decoded,[lane])),obs,movement,decoded)
            with patch('ez2rc.v2.relative_beats.lanes',return_value=[lane]),patch('ez2rc.v2.relative_beats.timeline',return_value=clock):
                relative=project_relative_beats(obs,movement,decoded,ref,measure_beats=F(3),
                                                rhythm_grid=RhythmGrid([2]),physical_review=review)
            self.assertEqual(relative['candidates'][0]['status'],'conditional')
            events=(Event('e',0,'incomplete','longnote',reason='timing pending'),)
            chart=Chart(Metadata('game','mode',1,'generated','test'),Timing('120',None,F(3)),unresolved=events)
            result,_,audit=resolve_spatial_chart(chart,events,relative,rhythm_candidates(relative,RhythmGrid([2])),
                {'status':'unavailable','hypotheses':[]},obs.manifest['metadata']['profile'],physical_review=review)
            self.assertEqual(result.notes,())
            self.assertIn('joint_grid_feasibility_unavailable',audit['reasons'])
            self.assertEqual(audit['physical_review'],review.receipt())

    def test_stale_reviews_wrong_connections_and_beat_labels_are_rejected(self):
        obs,movement,decoded,_,lane=scene()
        obs.id,movement.id,decoded.id='a'*64,'b'*64,'c'*64
        obs.manifest['metadata']['video_sha256']='d'*64
        movement.manifest['key']['parents']=[obs.id]
        decoded.manifest['key']['parents']=[obs.id,movement.id,'f'*64]
        # One intact procedural LN has two distinct endpoint witnesses.
        value=review_for(obs,movement,decoded,[lane])
        with tempfile.TemporaryDirectory() as directory, patch('ez2rc.v2.physical_review.lanes',return_value=[lane]):
            root=Path(directory)
            valid=load_physical_review(save_review(root,value),obs,movement,decoded)
            snapshot=valid.snapshot();snapshot['decisions'][0]['type']='tap'
            self.assertEqual(valid.snapshot(),value)
            for change in ('video','observation','motion','decode','lane','kind','connection','witness',
                           'duplicate','unselected','beat','coordinate','empty_witness','evidence','digest'):
                changed=deepcopy(value);decision=changed['decisions'][0]
                if change=='video':changed['video_sha256']='e'*64
                if change in ('observation','motion','decode'):changed[change+'_cache_id']='e'*64
                if change=='lane':decision['lane']=1
                if change=='kind':decision['type']='tap';decision['end']=None
                if change=='connection':decision['end']['endpoint_id']='h'
                if change=='witness':decision['end']['witness_observation_ids']=['hraw']
                if change=='duplicate':changed['decisions'].append(deepcopy(decision))
                if change=='unselected':decision['candidate_id']='other'
                if change=='beat':decision['start_beat']=[2,1]
                if change=='coordinate':decision['start']['q_px']=64
                if change=='empty_witness':decision['start']['witness_observation_ids']=[]
                if change=='evidence':decision['evidence_ids']=['absent']
                request=save_review(root,changed)
                if change=='digest':request['sha256']='0'*64
                with self.subTest(change=change),self.assertRaises(ValueError):
                    load_physical_review(request,obs,movement,decoded)

    def test_reviewed_ambiguous_selection_confirms_and_reconstructs_with_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);song=marked_video(root)
            profile=root/'profile.toml'
            profile.write_text(profile.read_text()+'''\n[position_calibration]
status = "verified"
version = "procedural-template-top"
scope = "profile_endpoints"
error_px = 1.0
''')
            algorithm=root/'algo.toml'
            # A marginal procedural score retains the exclusion alternative.
            # The generated image identities supply a separate review decision.
            algorithm.write_text(algorithm.read_text()+'\n[decode]\nnote_cost = 1.8\n')
            song.write_text(song.read_text()+'\n[timing.rhythm_grid]\nsubdivisions = [4]\n')
            _,first=execute(song,output_root=root/'out')
            obs=open_observations(first/'artifacts/observations')
            movement=open_motion(first/'artifacts/motion')
            decoded=open_decode(first/'artifacts/decode')
            lane_data=list(lanes(decoded))
            self.assertTrue(all(e['status']=='ambiguous' for l in lane_data for e in l['events']))
            calibration={'format':'ez2rc.spatial-calibration','schema_version':1,'version':'procedural-reference',
                         'localization_error_px':1,'endpoint_calibrations':[],
                         'observation_cache_id':obs.id,'motion_cache_id':movement.id,'decode_cache_id':decoded.id}
            data=json.dumps(calibration).encode();(root/'calibration.json').write_bytes(data)
            song.write_text(song.read_text()+f'\n[timing.spatial_calibration]\npath = "calibration.json"\nsha256 = "{hashlib.sha256(data).hexdigest()}"\n')
            _,unreviewed=execute(song,output_root=root/'out')
            self.assertEqual(read_chart(unreviewed/'chart.json')['notes'],[])
            review=review_for(obs,movement,decoded,lane_data)
            request=save_review(root,review)
            song.write_text(song.read_text()+f'\n[timing.physical_review]\npath = "review.json"\nsha256 = "{request["sha256"]}"\n')
            code,reviewed=execute(song,output_root=root/'out')
            self.assertEqual(code,0,(reviewed/'run.json').read_text())
            chart=read_chart(reviewed/'chart.json')
            expected=[(0,F(1,2)),(1,F(3,4)),(0,F(3,2)),(1,F(11,4)),(0,F(7,2)),(1,F(19,4))]
            self.assertEqual([(n['lane'],F(*n['start_beat'])) for n in chart['notes']],expected)
            self.assertTrue(all(n['type']=='tap' and n['end_beat'] is None for n in chart['notes']))
            self.assertEqual(chart['unresolved'],[])
            self.assertEqual(chart['physical_review_ref'],'artifacts/physical-review.json')
            relative=json.loads((reviewed/'artifacts/relative-beats.json').read_text())
            self.assertTrue(all(r['pixel_status']=='ambiguous' for r in relative['candidates']))
            proof=relative['rhythm_refinement']
            self.assertTrue(all(p['physical_basis']=='manual_review' and p['automatic_pixel_status']=='ambiguous'
                                for step in proof['iterations'] for p in step['added_unique_labels']))
            audit=json.loads((reviewed/'artifacts/spatial-resolution.json').read_text())
            self.assertTrue(all(d['accepted'] and d['automatic_pixel_status']=='ambiguous'
                                and d['physical_basis']['method']=='manual_review' for d in audit['decisions']))
            events=read_events(reviewed/'artifacts/events.json')
            self.assertEqual(events['physical_review_ref'],chart['physical_review_ref'])
            self.assertTrue(all(p['status']=='ambiguous' for p in events['pixel_intervals']))
            for name in ('video.mkv','profile.toml','note.png','head.png','tail.png'):
                (root/name).unlink()
            with patch('ez2rc.v2.capture.av.open',side_effect=AssertionError('must reconstruct stored observations')):
                code,reconstructed=execute(song,output_root=root/'out',observation_cache=first/'artifacts/observations')
            self.assertEqual(code,0)
            for name in ('chart.json','artifacts/events.json','artifacts/relative-beats.json',
                         'artifacts/rhythm-candidates.json','artifacts/spatial-resolution.json','artifacts/physical-review.json'):
                self.assertEqual((reviewed/name).read_bytes(),(reconstructed/name).read_bytes(),name)

            # Reusing a result flag without the bound input, changing its
            # identity, or leaving calibration unverified cannot confirm notes.
            from ez2rc.v2.contracts import Chart, Metadata, Timing
            from ez2rc.v2.decode import output_events
            decoded_events,_=output_events(decoded)
            base=Chart(Metadata('game','mode',2,'generated','test'),Timing('150',None,F(1)),
                       unresolved=decoded_events,pending_stages=('timing',))
            rhythm=json.loads((reviewed/'artifacts/rhythm-candidates.json').read_text())
            bars=json.loads((reviewed/'artifacts/barline-grid.json').read_text())
            checked=load_physical_review(request,obs,movement,decoded)
            profile_data=obs.manifest['metadata']['profile']
            with self.assertRaises(ValueError):
                resolve_spatial_chart(base,decoded_events,relative,rhythm,bars,profile_data)
            for change in ('candidate','proof','parent'):
                altered=deepcopy(relative)
                if change=='candidate':altered['candidates'][0]['endpoints']['start']['endpoint_id']='wrong'
                if change=='proof':altered['rhythm_refinement'].pop('physical_review')
                if change=='parent':altered['reference']['decode_cache_id']='0'*64
                with self.subTest(change=change),self.assertRaises(ValueError):
                    resolve_spatial_chart(base,decoded_events,altered,rhythm,bars,profile_data,physical_review=checked)
            profile_data=deepcopy(profile_data);profile_data['position_calibration']['status']='unverified'
            result,_,audit=resolve_spatial_chart(base,decoded_events,relative,rhythm,bars,profile_data,physical_review=checked)
            self.assertEqual(result.notes,())
            self.assertIn('profile_localization_unverified',audit['reasons'])
