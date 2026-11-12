from copy import deepcopy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ez2rc.v2.contracts import Chart, Event, Metadata, Note, Timing
from ez2rc.v2.export import chart_payload, events_payload, read_chart, read_events, write_outputs
from ez2rc.v2.runner import execute
from ez2rc.v2.spatial_resolution import resolve_spatial_chart
from ez2rc.v2.timing import beat_to_ms
from test_v2_barline_pipeline import marked_video


class RelativeContractTest(unittest.TestCase):
    def test_exact_relative_note_has_no_invented_recording_origin(self):
        timing=Timing('123.45',None,F(3),'resolved',None,'first_barline_zero')
        note=Note('ln',0,'longnote',F(2,3),F(5,4))
        event=Event('ln',0,'resolved','longnote',start_beat=note.start_beat,end_beat=note.end_beat,
                    beat_coordinate='first_barline_zero')
        chart=Chart(Metadata('game','mode',1,'generated','test'),timing,(note,))
        with self.assertRaises(ValueError):beat_to_ms(note.start_beat,timing)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            write_outputs(root,chart,(event,),observation_ref='observations')
            self.assertEqual(read_chart(root/'chart.json')['schema_version'],3)
            self.assertEqual(read_events(root/'artifacts/events.json')['schema_version'],2)
            value=chart_payload(chart);value['schema_version']=1
            (root/'chart.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError):read_chart(root/'chart.json')
            value=events_payload((event,));value['schema_version']=1
            (root/'events.json').write_text(json.dumps(value))
            with self.assertRaises(ValueError):read_events(root/'events.json')
            with self.assertRaises(ValueError):
                write_outputs(root,chart,(Event('ln',0,'resolved','longnote',start_beat=F(1),end_beat=F(2),
                    beat_coordinate='first_barline_zero'),),observation_ref='observations')


class SpatialResolutionTest(unittest.TestCase):
    def test_generated_longnote_ends_confirm_and_reconstruct_from_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);song=marked_video(root,longnotes=True)
            profile=root/'profile.toml'
            profile.write_text(profile.read_text()+'''\n[position_calibration]
status = "verified"
version = "procedural-template-top"
scope = "profile_endpoints"
error_px = 1.0
''')
            song.write_text(song.read_text()+'\n[timing.rhythm_grid]\nsubdivisions = [4]\n')
            _,first=execute(song,output_root=root/'out')
            run=json.loads((first/'run.json').read_text())
            calibration={'format':'ez2rc.spatial-calibration','schema_version':1,
                         'version':'generated-reference','localization_error_px':1,
                         'endpoint_calibrations':[]}
            for stage in ('observation','motion','decode'):
                calibration[stage+'_cache_id']=run['stages'][stage]['cache_id']
            data=json.dumps(calibration).encode();(root/'calibration.json').write_bytes(data)
            song.write_text(song.read_text()+f'\n[timing.spatial_calibration]\npath = "calibration.json"\nsha256 = "{hashlib.sha256(data).hexdigest()}"\n')
            code,automatic=execute(song,output_root=root/'out')
            self.assertEqual(code,0,(automatic/'run.json').read_text())
            chart=read_chart(automatic/'chart.json')
            expected=[(0,'tap',F(1,2),None),(1,'tap',F(3,4),None),
                      (0,'longnote',F(3,2),F(5,2)),(1,'longnote',F(11,4),F(17,4)),
                      (0,'tap',F(7,2),None),(1,'tap',F(19,4),None)]
            self.assertEqual([(n['lane'],n['type'],F(*n['start_beat']),
                               F(*n['end_beat']) if n['end_beat'] else None)
                              for n in chart['notes']],expected)
            self.assertEqual(chart['status'],'complete')
            self.assertEqual(chart['unresolved'],[])
            self.assertEqual(chart['timing']['coordinate'],'first_barline_zero')
            self.assertIsNone(chart['timing']['beat_zero_ms'])
            for name in ('video.mkv','profile.toml','note.png','head.png','tail.png'):
                (root/name).unlink()
            with patch('ez2rc.v2.capture.av.open',side_effect=AssertionError('no source decoding')):
                code,reconstructed=execute(song,output_root=root/'out',
                                           observation_cache=first/'artifacts/observations')
            self.assertEqual(code,0)
            for artifact in ('chart.json','artifacts/events.json','artifacts/barline-grid.json',
                             'artifacts/relative-beats.json','artifacts/rhythm-candidates.json',
                             'artifacts/spatial-resolution.json'):
                self.assertEqual((automatic/artifact).read_bytes(),(reconstructed/artifact).read_bytes())

    def test_automatic_confirmed_notes_reconstruct_without_video_or_clock_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);song=marked_video(root)
            profile=root/'profile.toml'
            profile.write_text(profile.read_text()+'''\n[position_calibration]
status = "verified"
version = "procedural-template-top"
scope = "profile_endpoints"
error_px = 1.0
''')
            # The independently constructed scene uses quarters. Its input
            # supplies that grid, never the expected note labels below.
            song.write_text(song.read_text()+'\n[timing.rhythm_grid]\nsubdivisions = [4]\n')
            _,first=execute(song,output_root=root/'out')
            run=json.loads((first/'run.json').read_text())
            calibration={'format':'ez2rc.spatial-calibration','schema_version':1,'version':'generated-reference',
                         'localization_error_px':1,'endpoint_calibrations':[]}
            for stage in ('observation','motion','decode'):calibration[stage+'_cache_id']=run['stages'][stage]['cache_id']
            data=json.dumps(calibration).encode();(root/'calibration.json').write_bytes(data)
            song.write_text(song.read_text()+f'\n[timing.spatial_calibration]\npath = "calibration.json"\nsha256 = "{hashlib.sha256(data).hexdigest()}"\n')
            code,automatic=execute(song,output_root=root/'out')
            self.assertIn(code,(0,2), (automatic/'run.json').read_text())
            chart=read_chart(automatic/'chart.json')
            expected=[(0,F(1,2)),(1,F(3,4)),(0,F(3,2)),(1,F(11,4)),(0,F(7,2)),(1,F(19,4))]
            self.assertEqual([(r['lane'],F(*r['start_beat'])) for r in chart['notes']],expected)
            self.assertEqual(chart['timing']['coordinate'],'first_barline_zero')
            self.assertIsNone(chart['timing']['beat_zero_ms'])
            evidence=read_events(automatic/'artifacts/events.json')
            self.assertTrue(all(e['start_ms'] is None for e in evidence['events'] if e['status']=='resolved'))
            resolution=json.loads((automatic/'artifacts/spatial-resolution.json').read_text())
            self.assertEqual(resolution['accepted_note_count'],6)
            for name in ('video.mkv','profile.toml','note.png','head.png','tail.png'):(root/name).unlink()
            with patch('ez2rc.v2.capture.av.open',side_effect=AssertionError('no media decoding')):
                _,reconstructed=execute(song,output_root=root/'out',observation_cache=first/'artifacts/observations')
            self.assertEqual(read_chart(reconstructed/'chart.json'),chart)
            self.assertEqual(read_events(reconstructed/'artifacts/events.json'),evidence)
            for artifact in ('relative-beats','rhythm-candidates','spatial-resolution'):
                self.assertEqual(json.loads((automatic/f'artifacts/{artifact}.json').read_text()),
                                 json.loads((reconstructed/f'artifacts/{artifact}.json').read_text()))

            # Neither an unverified profile, a physical alternative, nor an
            # algebraic conflict may be turned into an accepted musical note.
            from ez2rc.v2.decode import open_decode,output_events
            from ez2rc.v2.observation import open_observations
            obs=open_observations(first/'artifacts/observations')
            decoded=open_decode(first/'artifacts/decode')
            events,_=output_events(decoded)
            original=Chart(Metadata('game','mode',2,'generated','test'),Timing('150',None,F(1)),
                           unresolved=events,pending_stages=('timing',))
            relative=json.loads((automatic/'artifacts/relative-beats.json').read_text())
            rhythm=json.loads((automatic/'artifacts/rhythm-candidates.json').read_text())
            bars=json.loads((automatic/'artifacts/barline-grid.json').read_text())
            profile=obs.manifest['metadata']['profile']
            # Older references retain their original reconstruction semantics,
            # but a profile endpoint bound does not certify every template alias.
            from ez2rc.v2.motion import open_motion
            from ez2rc.v2.relative_beats import project_relative_beats
            from ez2rc.v2.rhythm import RhythmGrid,rhythm_candidates
            legacy=deepcopy(relative['reference'])
            legacy['observed_geometry']['version']='observed-spatial-constraints-4'
            reconstructed_legacy=project_relative_beats(obs,open_motion(first/'artifacts/motion'),decoded,
                legacy,measure_beats=F(1),rhythm_grid=RhythmGrid((4,)))
            legacy_grid=rhythm_candidates(reconstructed_legacy,RhythmGrid((4,)))
            self.assertEqual([r['note_candidates'] for r in legacy_grid['candidates']],
                             [r['note_candidates'] for r in rhythm['candidates']])
            candidate,_,audit=resolve_spatial_chart(original,events,reconstructed_legacy,legacy_grid,bars,profile)
            self.assertEqual(candidate.notes,())
            self.assertIn('raw_localization_scope_unverified',audit['reasons'])
            for kind in ('profile','physical','dependency','joint','origin','shift','optical','registration'):
                r,g,p,b=deepcopy(relative),deepcopy(rhythm),deepcopy(profile),deepcopy(bars)
                if kind=='profile':p['position_calibration']['status']='unverified'
                if kind=='physical':
                    for row in g['candidates']:row['pixel_status']='ambiguous'
                if kind=='dependency':r['candidates'][0]['pixel_status']='ambiguous'
                if kind=='joint':r['rhythm_refinement']['status']='conflict'
                if kind=='origin':b['hypotheses'][0]['assignments'][0]['beat']=[1,1]
                if kind=='shift':
                    for row in r['candidates']:
                        for point in row['endpoints'].values():point['calibration']={'offset_px':[0,0],'beat_shift':[1,8]}
                if kind in ('optical','registration'):
                    for row in r['candidates']:
                        for point in row['endpoints'].values():
                            point['optical_reference' if kind=='optical' else 'template_registration']={'status':'measured'}
                candidate,_,audit=resolve_spatial_chart(original,events,r,g,b,p)
                self.assertEqual(candidate.notes,(),kind)
                self.assertEqual(audit['accepted_note_count'],0,kind)
