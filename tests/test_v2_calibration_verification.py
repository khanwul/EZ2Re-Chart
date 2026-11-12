"""Independent procedural rendering declarations and scoped confirmation gates."""
from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ez2rc.v2.calibration_verification import (
    VERSION,code_fingerprint,load_calibration_verification,measurement_scope)
from ez2rc.v2.contracts import Chart,Metadata,Timing
from ez2rc.v2.decode import lanes,open_decode,output_events
from ez2rc.v2.export import read_chart,read_events
from ez2rc.v2.observation import open_observations
from ez2rc.v2.projection_edges import VERSION as EDGE_VERSION,REGISTRATION_VERSION
from ez2rc.v2.runner import execute
from ez2rc.v2.spatial_resolution import resolve_spatial_chart
from ez2rc.v2.storage import digest
from test_v2_barline_pipeline import marked_video


class CalibrationVerificationTest(unittest.TestCase):
    def test_scoped_optical_verification_confirms_and_reconstructs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);song=marked_video(root)
            song.write_text(song.read_text()+'\n[timing.rhythm_grid]\nsubdivisions = [4]\n')
            _,first=execute(song,output_root=root/'out')
            obs=open_observations(first/'artifacts/observations')
            decoded=open_decode(first/'artifacts/decode')
            run=json.loads((first/'run.json').read_text())
            calibration={'format':'ez2rc.spatial-calibration','schema_version':1,'version':'generated-optical',
                         'localization_error_px':1,'endpoint_calibrations':[]}
            for stage in ('observation','motion','decode'):calibration[stage+'_cache_id']=run['stages'][stage]['cache_id']
            for lane in lanes(decoded):
                for candidate in lane['candidates']:
                    if not candidate['selected']:continue
                    calibration['endpoint_calibrations'].append({'endpoint_id':candidate['start'],
                        'candidate_id':candidate['id'],'version':'generated-top','offset_px':[0,0],
                        'position_error_px':1,
                        'optical_reference':{'version':EDGE_VERSION,'direction':'falling','thresholds':[10,20],
                            'window_offset_px':[6,10],'reference_offset_px':[-7,-7],'error_px':0},
                        'template_registration':{'version':REGISTRATION_VERSION,'direction':'falling',
                            'window_offset_px':[6,10],'error_px':0,
                            'source_crossings':[{'threshold':t,'row_interval_px':[6,7]} for t in (10,20)],
                            'template_sha256':obs.manifest['key']['parameters']['asset_sha256'][f'lane{lane["lane"]}.tap']}})
            data=json.dumps(calibration).encode();(root/'calibration.json').write_bytes(data)
            song.write_text(song.read_text()+f'\n[timing.spatial_calibration]\npath = "calibration.json"\nsha256 = "{hashlib.sha256(data).hexdigest()}"\n')
            code,conditional=execute(song,output_root=root/'out')
            self.assertEqual(code,2)
            self.assertEqual(read_chart(conditional/'chart.json')['notes'],[])
            relative=json.loads((conditional/'artifacts/relative-beats.json').read_text())
            rhythm=json.loads((conditional/'artifacts/rhythm-candidates.json').read_text())
            bars=json.loads((conditional/'artifacts/barline-grid.json').read_text())
            expected=[(0,F(1,2)),(1,F(3,4)),(0,F(3,2)),(1,F(11,4)),(0,F(7,2)),(1,F(19,4))]
            value={'format':'ez2rc.calibration-verification','schema_version':1,'version':VERSION,
                'status':'verified','code_sha256':code_fingerprint(),'scope':measurement_scope(obs,relative),
                'method':'procedural_measurement','assessor':'procedural renderer',
                'criteria_sha256':digest({'measurements':'lossless template top and bracketed lower edge'}),
                'evidence':[{'id':'renderer','sha256':digest({'known_scene':[(l,str(b)) for l,b in expected]})}],
                'limitations':'Only this renderer, observation settings and declared measurement definitions.'}
            def request(document):
                data=json.dumps(document).encode();(root/'verified.json').write_bytes(data)
                return {'path':str(root/'verified.json'),'sha256':hashlib.sha256(data).hexdigest()}
            binding=request(value);verification=load_calibration_verification(binding,obs)
            self.assertEqual(verification.check_relative(relative),[])
            edited=verification.snapshot();edited['scope']['localization_error_px']=[100,1]
            self.assertEqual(verification.snapshot(),value)
            song.write_text(song.read_text()+f'\n[timing.calibration_verification]\npath = "verified.json"\nsha256 = "{binding["sha256"]}"\n')
            code,confirmed=execute(song,output_root=root/'out')
            self.assertEqual(code,0,(confirmed/'run.json').read_text())
            chart=read_chart(confirmed/'chart.json');events=read_events(confirmed/'artifacts/events.json')
            self.assertEqual([(n['lane'],F(*n['start_beat'])) for n in chart['notes']],expected)
            self.assertEqual(chart['unresolved'],[])
            self.assertEqual(chart['calibration_verification_ref'],'artifacts/calibration-verification.json')
            resolution=json.loads((confirmed/'artifacts/spatial-resolution.json').read_text())
            self.assertTrue(all(d['calibration_basis']['method']=='scoped_verification' for d in resolution['decisions']))
            # Neither the original profile nor raw optical measurements acquire
            # a verified flag; the explicit assessment is a separate basis.
            after=json.loads((confirmed/'artifacts/relative-beats.json').read_text())
            self.assertEqual(relative,after)
            self.assertTrue(all(not p['optical_reference']['verified'] for r in after['candidates'] for p in r['endpoints'].values()))
            for name in ('video.mkv','profile.toml','note.png','head.png','tail.png'):(root/name).unlink()
            with patch('ez2rc.v2.capture.av.open',side_effect=AssertionError('no source decoding')):
                code,reconstructed=execute(song,output_root=root/'out',observation_cache=first/'artifacts/observations')
            self.assertEqual(code,0)
            self.assertEqual(chart,read_chart(reconstructed/'chart.json'))
            self.assertEqual(events,read_events(reconstructed/'artifacts/events.json'))
            for artifact in ('relative-beats','rhythm-candidates','spatial-resolution','calibration-verification'):
                self.assertEqual((confirmed/f'artifacts/{artifact}.json').read_bytes(),
                                 (reconstructed/f'artifacts/{artifact}.json').read_bytes())

            original_events,_=output_events(decoded)
            original=Chart(Metadata('game','mode',2,'generated','test'),Timing('150',None,F(1)),
                           unresolved=original_events,pending_stages=('timing',))
            for kind in ('geometry','offset','shift','optical','registration','physical','joint','origin'):
                with self.subTest(changed=kind):
                    r,g,b=deepcopy(relative),deepcopy(rhythm),deepcopy(bars)
                    point=r['candidates'][0]['endpoints']['start']
                    if kind=='geometry':r['reference']['observed_geometry']['localization_error_px']=2
                    if kind=='offset':point['calibration']['offset_px']=[-1,1]
                    if kind=='shift':point['calibration']['beat_shift']=[1,4]
                    if kind=='optical':point['calibration']['optical_reference']['thresholds']=[11,20]
                    if kind=='registration':point['calibration']['template_registration']['error_px']=1
                    if kind=='physical':r['candidates'][0]['pixel_status']='ambiguous'
                    if kind=='joint':r['rhythm_refinement']['status']='conflict'
                    if kind=='origin':b['hypotheses'][0]['assignments'][0]['beat']=[1,1]
                    result,_,audit=resolve_spatial_chart(original,original_events,r,g,b,obs.manifest['metadata']['profile'],
                                                       calibration_verification=verification)
                    self.assertEqual(result.notes,())
                    self.assertEqual(audit['accepted_note_count'],0)
            for kind in ('code','observation','alignment','status','evidence','duplicate','labels','definition','policy','missing_policy','review_scope'):
                changed=deepcopy(value)
                if kind=='code':changed['code_sha256']='0'*64
                if kind=='observation':changed['scope']['observation_parameters_sha256']='0'*64
                if kind=='alignment':changed['scope']['alignment_offset_px']=[1,0]
                if kind=='status':changed['status']='conditional'
                if kind=='evidence':changed['evidence']=[]
                if kind=='duplicate':changed['scope']['endpoint_definitions']*=2
                if kind=='labels':changed['scope']['note_beats']=[[1,2]]
                if kind=='definition':changed['scope']['endpoint_definitions'][0]['definition']['candidate_id']='c1'
                if kind=='policy':changed['scope']['reviewed_localization_policy']='all_observations'
                if kind=='missing_policy':changed['scope'].pop('reviewed_localization_policy')
                if kind=='review_scope':changed['scope']['reviewed_localization_scope']=[{'lane':0,'kind':'tap'}]
                with self.subTest(invalid=kind),self.assertRaises(ValueError):load_calibration_verification(request(changed),obs)
            changed=deepcopy(value);changed['scope']['endpoint_definitions'].pop()
            partial=load_calibration_verification(request(changed),obs)
            self.assertEqual(partial.check_relative(relative),['calibration_verification_definition_scope_mismatch'])
            binding=request(value);binding['sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'checksum'):load_calibration_verification(binding,obs)


if __name__=='__main__':unittest.main()
