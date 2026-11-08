"""Dense pixels are adjacent samples; they are not duplicate endpoint seeds."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import tempfile
import unittest
import numpy as np

from ez2rc.v2.config import FusionConfig
from ez2rc.v2.fusion import accumulate
from ez2rc.v2.storage import digest
from v2_motion_fixtures import evidence_arrays


def dense_fixture(root, rows, scrolls, *, groups=None, tile=32, cap=1., status=1):
    rows=np.asarray(rows,float);n,height=rows.shape
    groups=list(range(n)) if groups is None else groups
    profile={'judge_y':32.,'effect_top':height-4.,'lanes':[{'roi':[0,0,10,height]}],
             'calibration':{k:{'status':'unverified'} for k in ('tap','head','tail')}}
    obs=SimpleNamespace(id=digest({'rows':rows.tolist(),'groups':groups}),manifest={'metadata':{'profile':profile}})
    motion=SimpleNamespace(id=digest({'scrolls':list(scrolls),'status':status}),manifest={'key':{'parents':[obs.id]}})
    clock={'frame_id':np.arange(n),'timestamp_ms':np.arange(n)*16.,'scroll_px':np.array(scrolls,float),
           'status':np.full(n,status),'error_px':np.zeros(n),'repeat_group':np.array(groups),'repeat_confirmed':np.ones(n,bool)}
    frames=[{'id':i,'ms':i*16.,'group':groups[i],'projection':rows[i,None],
             'candidate_index':np.empty((0,3),np.int64),'positions':np.empty((0,3,2)),
             'scores':np.empty((0,4)),'flags':np.empty((0,3),np.uint8)} for i in range(n)]
    with patch('ez2rc.v2.fusion.frames',return_value=iter(frames)),patch('ez2rc.v2.fusion.timeline',return_value=clock):
        result,_=accumulate(obs,motion,FusionConfig(tile_bins=tile,repeat_group_cap=cap),root)
    return evidence_arrays(result)


class DenseBodySplatTest(unittest.TestCase):
    def test_fractional_translation_preserves_total_dense_sample_mass(self):
        with tempfile.TemporaryDirectory() as tmp:
            for height in (16,33,64):
                for shift in (0.,.125,.5,.875,1.,32.5):
                    with self.subTest(height=height,shift=shift):
                        a=dense_fixture(Path(tmp),np.full((1,height),153.),[shift])
                        expected=.25+.5*(height-5)
                        self.assertAlmostEqual(a['weight_sum'][:,3].sum(),expected)
                        self.assertAlmostEqual(a['score_sum'][:,3].sum(),expected*.6)

    def test_uniform_interior_support_is_independent_of_grid_phase(self):
        with tempfile.TemporaryDirectory() as tmp:
            for shift in (0.,.25,.5,.75):
                with self.subTest(shift=shift):
                    a=dense_fixture(Path(tmp),np.full((1,32),153.),[shift])
                    at=(a['index'][:,1]>=10)&(a['index'][:,1]<25)
                    np.testing.assert_allclose(a['weight_sum'][at,3],.5)
                    np.testing.assert_allclose(a['score_sum'][at,3],.3)

    def test_adjacent_distinct_brightness_samples_are_interpolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            row=np.arange(32)*4.
            a=dense_fixture(Path(tmp),row[None],[.25])
            for q in range(10,25):
                i=np.flatnonzero(a['index'][:,1]==q)[0]
                expected=(.75*row[32-q]+.25*row[33-q])/255
                self.assertAlmostEqual(a['score_sum'][i,3]/a['weight_sum'][i,3],expected)

    def test_visibility_mass_survives_clipped_unknown_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=dense_fixture(Path(tmp),np.full((1,32),153.),[.5])
            i=np.flatnonzero(a['index'][:,1]==32)[0]
            self.assertAlmostEqual(a['visibility_weight'][i,3,2],.125)
            self.assertAlmostEqual(a['visibility_weight'][i,3,1],.25)
            np.testing.assert_allclose(a['visibility_weight'].sum(axis=2),a['weight_sum'])

    def test_repeated_display_group_does_not_multiply_dense_mass(self):
        with tempfile.TemporaryDirectory() as tmp:
            one=dense_fixture(Path(tmp)/'one',np.full((1,32),153.),[.5])
            repeated=dense_fixture(Path(tmp)/'repeat',np.full((4,32),153.),[.5]*4,groups=[0]*4)
            for k in ('index','score_sum','weight_sum','visibility_weight','group_count'):
                np.testing.assert_allclose(one[k],repeated[k],err_msg=k)

    def test_group_cap_scales_mixed_visibility_mass(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=dense_fixture(Path(tmp),np.full((1,32),153.),[.5],cap=.2)
            self.assertLessEqual(a['weight_sum'][:,3].max(),.2)
            np.testing.assert_allclose(a['visibility_weight'].sum(axis=2),a['weight_sum'])

    def test_single_lane_weight_preserves_relative_mass(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=dense_fixture(Path(tmp)/'full',np.full((1,32),153.),[.5])
            b=dense_fixture(Path(tmp)/'single',np.full((1,32),153.),[.5],status=2)
            for k in ('score_sum','weight_sum','visibility_weight'):np.testing.assert_allclose(b[k],a[k]*.25)

    def test_integer_origin_and_tile_layout_preserve_interpolated_cells(self):
        with tempfile.TemporaryDirectory() as tmp:
            baseline=None
            for shift,tile in [(0,16),(32,32),(2048,2048)]:
                a=dense_fixture(Path(tmp)/str(shift),np.full((130,32),153.),np.arange(130)*.25+shift,tile=tile)
                a['index'][:,1]-=shift
                if baseline is None:baseline=a
                else:
                    for k in ('index','score_sum','weight_sum','visibility_weight','group_count'):
                        np.testing.assert_allclose(a[k],baseline[k],err_msg=k)

    def test_dark_rows_are_observed_and_effect_rows_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=dense_fixture(Path(tmp),np.zeros((1,32)),[.5])
            self.assertTrue((a['weight_sum'][:,3]>0).all())
            self.assertEqual(a['score_sum'][:,3].sum(),0)
            self.assertEqual(a['visibility_weight'][:,:,3].sum(),0)

if __name__=='__main__':unittest.main()
