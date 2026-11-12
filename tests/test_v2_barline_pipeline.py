"""Native first-line inference and cache reconstruction from a generated video."""
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import cv2
import numpy as np

from ez2rc.v2.decode import lanes, open_decode
from ez2rc.v2.export import read_chart
from ez2rc.v2.runner import execute
from test_v2_timing_resolver import moving_taps


def marked_video(root, *, longnotes=False):
    song=moving_taps(root)
    text=song.read_text().replace('bpm = "191.910"','bpm = "150"\nmeasure_beats = [1, 1]')
    song.write_text(text)
    template=cv2.imread(str(root/'note.png'))
    heads=cv2.imread(str(root/'head.png'))
    tails=cv2.imread(str(root/'tail.png'))
    (root/'video.mkv').unlink()
    with av.open(str(root/'video.mkv'),'w') as container:
        stream=container.add_stream('ffv1',rate=50)
        stream.width=stream.height=96;stream.pix_fmt='bgr0'
        stream.time_base=stream.codec_context.time_base=F(1,1000)
        for frame_id in range(126):
            image=np.zeros((96,96,3),np.uint8);image[70:76,20:61]=(220,120,20)
            for entry in (6,26,46,66,86,106):
                y=3*(frame_id-entry)
                if 0<=y<65:image[y,20:60]=70
            for lane,q in ((0,120),(1,135),(0,180),(1,255),(0,300),(1,375)):
                y=3*frame_id+72-q
                duration={180:60,255:90}.get(q) if longnotes else None
                left,right=22+lane*20,38+lane*20
                if duration is None:
                    if 0<=y and y+7<=65:image[y:y+7,left:right]=template
                else:
                    tail=y-duration
                    lower,upper=max(0,tail+7),min(65,y)
                    if lower<upper:image[lower:upper,left:right]=100
                    for top,patch in ((y,heads),(tail,tails)):
                        if 0<=top and top+7<=65:image[top:top+7,left:right]=patch
            frame=av.VideoFrame.from_ndarray(image,format='bgr24')
            frame.pts,frame.time_base=frame_id*20,F(1,1000)
            for packet in stream.encode(frame):container.mux(packet)
        for packet in stream.encode():container.mux(packet)
    return song


class BarlinePipelineTest(unittest.TestCase):
    def test_optical_landmarks_use_saved_projections_and_preserve_capture_ranges(self):
        from ez2rc.v2.projection_edges import VERSION as EDGE_VERSION, REGISTRATION_VERSION
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);song=marked_video(root)
            _,first=execute(song,output_root=root/'out')
            run=json.loads((first/'run.json').read_text())
            calibration={'format':'ez2rc.spatial-calibration','schema_version':1,'version':'generated-optical',
                         'localization_error_px':1,'endpoint_calibrations':[]}
            endpoint_lanes={}
            for stage in ('observation','motion','decode'):calibration[stage+'_cache_id']=run['stages'][stage]['cache_id']
            for lane in lanes(open_decode(first/'artifacts/decode')):
                for candidate in lane['candidates']:
                    if candidate['selected']:
                        endpoint_lanes[candidate['start']]=lane['lane']
                        calibration['endpoint_calibrations'].append({'endpoint_id':candidate['start'],
                            'candidate_id':candidate['id'],'version':'generated-top','offset_px':[0,0],
                            'position_error_px':1})
            original=song.read_text()
            def configure():
                data=json.dumps(calibration).encode();(root/'calibration.json').write_bytes(data)
                song.write_text(original+f'\n[timing.spatial_calibration]\npath = "calibration.json"\nsha256 = "{hashlib.sha256(data).hexdigest()}"\n')
            configure();_,baseline=execute(song,output_root=root/'out')
            observation=json.loads((first/'artifacts/observations/manifest.json').read_text())
            for item in calibration['endpoint_calibrations']:
                item['optical_reference']={'version':EDGE_VERSION,'direction':'falling','thresholds':[10,20],
                    'window_offset_px':[6,10],'reference_offset_px':[-7,-7],'error_px':0}
                item['template_registration']={'version':REGISTRATION_VERSION,'direction':'falling',
                    'window_offset_px':[6,10],'error_px':0,
                    'source_crossings':[{'threshold':t,'row_interval_px':[6,7]} for t in (10,20)],
                    'template_sha256':observation['key']['parameters']['asset_sha256'][
                        f'lane{endpoint_lanes[item["endpoint_id"]]}.tap']}
            configure();code,optical=execute(song,output_root=root/'out')
            self.assertEqual(code,2,(optical/'run.json').read_text())
            before=json.loads((baseline/'artifacts/relative-beats.json').read_text())
            after=json.loads((optical/'artifacts/relative-beats.json').read_text())
            measured=after['observed_geometry']['optical_endpoints']
            self.assertGreater(sum(e['measured_frames'] for e in measured.values()),0)
            self.assertTrue(all(not e['verified'] for e in measured.values()))
            registered=after['observed_geometry']['registered_endpoints']
            self.assertGreater(sum(e['measured_frames'] for e in registered.values()),0)
            self.assertTrue(all(not e['verified'] for e in registered.values()))
            for a,b in zip(before['candidates'],after['candidates']):
                self.assertEqual(a['candidate_id'],b['candidate_id'])
                for side in a['endpoints']:
                    self.assertEqual(a['endpoints'][side]['capture_time'],b['endpoints'][side]['capture_time'])
            for name in ('video.mkv','profile.toml','note.png','head.png','tail.png'):(root/name).unlink()
            with patch('ez2rc.v2.capture.av.open',side_effect=AssertionError('no source decoding')):
                code,reconstructed=execute(song,output_root=root/'out',observation_cache=first/'artifacts/observations')
            self.assertEqual(code,2)
            self.assertEqual(after,json.loads((reconstructed/'artifacts/relative-beats.json').read_text()))
            self.assertEqual(read_chart(reconstructed/'chart.json')['notes'],[])
            calibration['endpoint_calibrations'][0]['template_registration']['template_sha256']='0'*64
            configure();code,failed=execute(song,output_root=root/'out',observation_cache=first/'artifacts/observations')
            self.assertEqual(code,1)
            self.assertIn('registration template differs',json.loads((failed/'run.json').read_text())['failure']['message'])

    def test_automatic_reference_has_no_reference_note_beats_and_reconstructs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);song=marked_video(root)
            code,first=execute(song,output_root=root/'out')
            self.assertEqual(code,2)
            run=json.loads((first/'run.json').read_text())
            lines=json.loads((first/'artifacts/barline-grid.json').read_text())
            self.assertEqual(lines['status'],'supported')
            self.assertEqual([a['measure_index'] for a in lines['hypotheses'][0]['assignments']],list(range(6)))
            calibration={'format':'ez2rc.spatial-calibration','schema_version':1,'version':'procedural-geometry',
                         'localization_error_px':1,'endpoint_calibrations':[]}
            for stage in ('observation','motion','decode'):calibration[stage+'_cache_id']=run['stages'][stage]['cache_id']
            for lane in lanes(open_decode(first/'artifacts/decode')):
                for c in lane['candidates']:
                    if c['selected']:
                        for eid in (c['start'],c['end']):
                            if eid:calibration['endpoint_calibrations'].append({'endpoint_id':eid,'candidate_id':c['id'],
                                'version':'procedural-top','offset_px':[0,0],'position_error_px':1})
            p=root/'calibration.json';p.write_text(json.dumps(calibration));checksum=hashlib.sha256(p.read_bytes()).hexdigest()
            song.write_text(song.read_text()+f'\n[timing.spatial_calibration]\npath = "calibration.json"\nsha256 = "{checksum}"\n')
            code,automatic=execute(song,output_root=root/'out')
            self.assertEqual(code,2)
            for name in ('video.mkv','profile.toml','note.png','head.png','tail.png'):(root/name).unlink()
            with patch('ez2rc.v2.capture.av.open',side_effect=AssertionError('no media decode')):
                code,reconstructed=execute(song,output_root=root/'out',observation_cache=first/'artifacts/observations')
            self.assertEqual(code,2)
            for artifact in ('barline-grid','relative-beats','rhythm-candidates'):
                a=json.loads((automatic/f'artifacts/{artifact}.json').read_text())
                b=json.loads((reconstructed/f'artifacts/{artifact}.json').read_text())
                self.assertEqual(a,b)
            spatial=json.loads((automatic/'artifacts/relative-beats.json').read_text())
            self.assertGreater(spatial['conditional_candidate_count'],0)
            self.assertEqual(spatial['barline_grid_ref'],'artifacts/barline-grid.json')
            self.assertEqual(read_chart(automatic/'chart.json')['notes'],[])
            self.assertIn('barline_grid_ref',read_chart(automatic/'chart.json'))
            # Calibration accepts no reference-note or externally numbered-line input.
            calibration['anchors']=[{'beat':[0,1]}];p.write_text(json.dumps(calibration))
            song.write_text(song.read_text().replace(checksum,hashlib.sha256(p.read_bytes()).hexdigest()))
            code,failed=execute(song,output_root=root/'out',observation_cache=first/'artifacts/observations')
            self.assertEqual(code,1)
            self.assertIn('unknown settings',json.loads((failed/'run.json').read_text())['failure']['message'])
