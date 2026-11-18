"""Procedural moving lines with occlusion, decoys, gaps and phase alternatives."""
from fractions import Fraction as F
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ez2rc.v2.barline_grid import infer_barline_grid, number_barlines, reconcile_cycles, thin_rows
from ez2rc.v2.config import AlgorithmConfig, MotionConfig
from dataclasses import asdict


def recording(omit=(), start=0, chunk_size=17):
    ids=np.arange(start,132,dtype=np.int64);timestamps=ids*10.
    frames=[]
    for fid in ids:
        image=np.full((3,84),11.,dtype=np.float16)
        # A broad three-lane body is not a thin moving marker.
        image[:,64:80]=100
        for number,entry in enumerate((5,55,105)):
            row=4*(int(fid)-entry)
            if number not in omit and 0<=row<64:image[:,row]=72
        frames.append(image)
    chunks=[{'frame_id':ids[i:i+chunk_size],'timestamp_ms':timestamps[i:i+chunk_size],
             'projection':np.array(frames[i:i+chunk_size])} for i in range(0,len(ids),chunk_size)]
    obs=SimpleNamespace(id='observations',chunks=lambda:iter(chunks),manifest={
        'metadata':{'profile':{'lanes':[{'roi':[0,0,10,84]}]*3,'effect_top':64,'judge_y':70}},
        'key':{'parameters':{'algorithm':AlgorithmConfig().observation_parameters()}}})
    mov=SimpleNamespace(id='motion',manifest={'key':{'parents':[obs.id],'parameters':asdict(MotionConfig())}})
    clock={'frame_id':ids,'timestamp_ms':timestamps,'scroll_px':ids*4.,'status':np.ones(len(ids),np.uint8)}
    return obs,mov,clock


def infer(**kwargs):
    obs,mov,clock=recording(**kwargs)
    with patch('ez2rc.v2.barline_grid.timeline',return_value=clock):
        return infer_barline_grid(obs,mov,'120',F(1))


class BarlineGridTest(unittest.TestCase):
    def test_observed_meter_includes_a_grid_beat_on_an_exact_pts_boundary(self):
        obs, mov, clock = recording(chunk_size=132)
        chunk, = obs.chunks()
        frames = clock['frame_id']
        times = np.array([float(F(int(f) * 32, 3)) for f in frames])
        chunk['timestamp_ms'] = clock['timestamp_ms'] = times
        capture = SimpleNamespace(id='capture', manifest={'metadata': {
            'first_pts': 17, 'first_time_base': [1, 3000]}},
            chunks=lambda: iter([{'frame_id': frames, 'pts': frames*32 + 17,
                                  'time_base': np.tile([1, 3000], (len(frames), 1)),
                                  'fps_fallback': np.zeros(len(frames), dtype=bool),
                                  'timestamp_ms': times}]))
        obs.manifest['key']['parents'] = [capture.id]
        with patch('ez2rc.v2.barline_grid.timeline', return_value=clock):
            grid = infer_barline_grid(obs, mov, '117.1875', None, observed_meter=True, capture=capture)
        # 48 frame steps between the end of the first crossing and the start
        # of the next are exactly 512 ms, one beat at the supplied tempo.
        self.assertEqual(grid['barline_beats'][:2], [[0, 1], [1, 1]])

    def test_first_line_and_numbering_without_judgment_clock_or_note_labels(self):
        result=infer()
        self.assertEqual(result['status'],'supported')
        hypothesis=result['hypotheses'][0]
        self.assertTrue(hypothesis['leading_coverage'])
        self.assertEqual([a['beat'] for a in hypothesis['assignments']],[[0,1],[1,1],[2,1]])
        self.assertEqual(hypothesis['missing_measure_indices'],[])
        self.assertFalse(result['accepted'])
        self.assertNotIn('beat_zero_ms',result)
        self.assertEqual(result,infer(chunk_size=31))

    def test_missing_line_preserves_cycle_gap_instead_of_renumbering(self):
        result=infer(omit=(1,))
        self.assertEqual(result['status'],'supported')
        self.assertEqual([a['measure_index'] for a in result['hypotheses'][0]['assignments']],[0,2])
        self.assertEqual(result['hypotheses'][0]['missing_measure_indices'],[1])

    def test_split_outside_crossing_does_not_erase_the_clean_segment(self):
        for frame, expected in ((58, [0, 1, 2]), (63, [0, 2]), (68, [0, 1, 2])):
            with self.subTest(frame=frame):
                obs, mov, clock = recording(chunk_size=132)
                chunk, = obs.chunks()
                row = 4 * (frame - 55)
                chunk['projection'][frame, :, row + 3] = 72
                with patch('ez2rc.v2.barline_grid.timeline', return_value=clock):
                    grid = infer_barline_grid(obs, mov, '120', F(1))
                self.assertEqual(grid['status'], 'supported')
                assignments = grid['hypotheses'][0]['assignments']
                self.assertEqual([a['measure_index'] for a in assignments], expected)
                if 1 in expected:
                    track = next(t for t in grid['tracks'] if t['id'] == assignments[1]['marker_id'])
                    self.assertTrue(track['association_conflict'])
                    self.assertNotIn(frame, [r['frame_id'] for r in track['observations']])

    def test_clipped_leading_coverage_does_not_establish_first_line(self):
        result=infer(start=9)
        self.assertEqual(result['status'],'ambiguous')
        self.assertTrue(all('first_line_leading_coverage_unestablished' in h['reasons'] for h in result['hypotheses']))

    def test_two_recurring_phases_remain_competing_hypotheses(self):
        markers=[{'id':str(i),'timestamp_interval_ms':[a,a+2]} for i,a in enumerate((10,110,210,40,140,240))]
        phase,hypotheses=number_barlines(markers,F(100),F(3))
        self.assertEqual(phase['status'],'ambiguous')
        self.assertEqual(len(hypotheses),2)
        self.assertTrue(all(len(h['conflicting_marker_ids'])==3 for h in hypotheses))

    def test_overlapping_clock_supports_can_agree_on_relative_cycles(self):
        # No common clock phase fits all three intervals. The middle crossing
        # connects both local supports and fixes the same integer cycle counts.
        markers=[{'id':str(i),'timestamp_interval_ms':span} for i,span in
                 enumerate(([0,4],[102,108],[206,110+100]))]
        phase,hypotheses=number_barlines(markers,F(100),F(3))
        self.assertEqual(phase['status'],'ambiguous')
        self.assertEqual(len(hypotheses),1)
        self.assertEqual([a['measure_index'] for a in hypotheses[0]['assignments']],[0,1,2])
        self.assertIsNone(hypotheses[0]['phase_witness_ms'])
        self.assertEqual(phase['cycle_reconciliation']['components'],1)
        self.assertTrue(phase['numbering_alternatives'])

    def test_conflicting_cycle_differences_are_not_merged(self):
        def h(a,b,n):return {'assignments':[{'marker_id':a,'measure_index':0},{'marker_id':b,'measure_index':n}]}
        result,diagnostic=reconcile_cycles([h('a','b',1),h('b','c',1),h('a','c',3)],['a','b','c'],F(4))
        self.assertIsNone(result)
        self.assertTrue(diagnostic['conflicts'])

    def test_broad_body_and_single_lane_peak_are_not_common_thin_lines(self):
        data=np.full((3,90),10.,dtype=np.float16)
        data[:,15:35]=80;data[0,48]=80;data[:,64]=60
        rows=thin_rows(data,0,90,AlgorithmConfig().observation_parameters())
        self.assertEqual([r['y_px'] for r in rows],[64.])

    def test_missing_meter_has_explicit_unavailable_result(self):
        obs,mov,_=recording()
        result=infer_barline_grid(obs,mov,'120',None)
        self.assertEqual(result['reasons'],['measure_length_not_supplied'])
