"""Frame-coordinate lineage from procedural common landmark motion."""
from fractions import Fraction as F
from types import SimpleNamespace
import unittest

import numpy as np

from ez2rc.v2.coordinate_errors import Coordinate
from ez2rc.v2.motion import STATUS
from ez2rc.v2.motion_dependence import MotionDependence, outward


def movement(conflict=False, gap=True):
    states = [STATUS['origin'], STATUS['observed'], STATUS['extrapolated'] if gap else STATUS['observed'],
              STATUS['observed'], STATUS['observed'], STATUS['repeated']]
    clock = {'frame_id': np.arange(6), 'timestamp_ms': np.arange(6)*20.,
             'scroll_px': np.array([0., 10., 20., 31., 40., 40.]), 'status': np.array(states),
             'reference_frame': np.array([-1, 0, -1 if gap else 1, 2, 3, 4]),
             'repeat_group': np.array([0, 1, 2, 3, 4, 4]),
             'repeat_confirmed': np.array([False]*4+[True]*2)}
    links = []
    for fid, ref in ((1,0), (2,1), (3,2), (4,3)):
        for lane in (0,1):
            links.append([fid,ref,lane,0,0,0,10+(8 if conflict and fid==3 and lane==1 else 0),1,0])
    clock['links'] = np.array(links)
    return SimpleNamespace(id='procedural-motion', chunks=lambda: iter([clock]))


def at(fid):
    return [{'frame_id': fid, 'weight': [1, 1]}]


class MotionDependenceTest(unittest.TestCase):
    def test_shared_gap_cancels_locally_but_not_across_gap(self):
        model = MotionDependence(movement(), .25)
        sources = {k: None if v['range_px'] is None else tuple(F(str(x)) for x in v['range_px'])
                   for k,v in model.sources.items()}
        a = Coordinate(F(0), model.terms(at(3)))
        b = Coordinate(F(0), model.terms(at(4)))
        c = Coordinate(F(0), model.terms(at(1)))
        # The measured 10 px displacement also corrects the nominal 9 px step.
        self.assertEqual((b-a).domain(sources), (F(1, 2), F(3, 2)))
        self.assertIsNone((b-c).domain(sources))
        self.assertIn('motion:2', model.terms(at(4)))
        self.assertIsNone(model.sources['motion:2']['range_px'])

    def test_confirmed_repeat_reuses_same_coordinate(self):
        model = MotionDependence(movement(), .5)
        difference = Coordinate(F(0), model.terms(at(5))) - Coordinate(F(0), model.terms(at(4)))
        self.assertEqual(difference.terms, {'motion:5': 1.})
        self.assertEqual(model.sources['motion:5']['range_px'], [0., 0.])

    def test_unknown_or_conflicting_localization_never_becomes_zero(self):
        unknown = MotionDependence(movement(), None)
        self.assertIsNone(unknown.sources['position:f1:l0:y0:k0']['range_px'])
        conflict = MotionDependence(movement(conflict=True), .5)
        self.assertEqual(conflict.issues, {3: 'landmark_bounds_conflict'})
        self.assertIsNone(conflict.sources['motion:3']['range_px'])

    def test_reused_landmark_errors_cancel_along_an_observed_path(self):
        model = MotionDependence(movement(gap=False), .25)
        sources = {k: None if v['range_px'] is None else tuple(F(str(x)) for x in v['range_px'])
                   for k,v in model.sources.items()}
        difference = Coordinate(F(40), model.terms(at(4))) - Coordinate(F(0), model.terms(at(0)))
        self.assertEqual(difference.domain(sources), (F(79,2), F(81,2)))
        self.assertFalse(any(f'position:f{i}:' in key for i in (1,2,3) for key in difference.terms))

    def test_endpoint_localization_cancels_only_with_the_same_raw_observation(self):
        model = MotionDependence(movement(gap=False), None)
        a, b = at(0), at(4)
        a[0]['position_source'] = model.position_source(0,0,0,0)
        b[0]['position_source'] = model.position_source(4,0,0,0)
        declaration = model.model({}, {'a':a, 'b':b})
        sources = {s['id']:None if s['range_px'] is None else tuple(F(str(x)) for x in s['range_px']) for s in declaration['sources']}
        x, y = [Coordinate(F(0),declaration['endpoints'][k]) for k in ('a','b')]
        self.assertEqual((y-x).domain(sources), (0,0))
        b[0]['position_source'] = model.position_source(4,0,0,1)
        declaration = model.model({}, {'a':a, 'b':b})
        sources = {s['id']:None if s['range_px'] is None else tuple(F(str(x)) for x in s['range_px']) for s in declaration['sources']}
        x, y = [Coordinate(F(0),declaration['endpoints'][k]) for k in ('a','b')]
        self.assertIsNone((y-x).domain(sources))

    def test_two_median_observations_share_ancestor_once(self):
        model = MotionDependence(movement(), .5)
        weights = [{'frame_id': 3, 'weight': [1, 2]}, {'frame_id': 4, 'weight': [1, 2]}]
        terms = model.terms(weights)
        self.assertEqual(terms['motion:0'], 1)
        self.assertEqual(terms['motion:2'], 1)
        self.assertEqual(terms['motion:4'], .5)
        for invalid in ([{'frame_id': 9, 'weight': [1,1]}], [{'frame_id': 3, 'weight': [1,2]}]):
            with self.assertRaises(ValueError): model.terms(invalid)

    def test_outward_serialization_preserves_exact_bounds(self):
        for value in (F(1,3), F(-1,3), F(199999999,7), F(0)):
            self.assertLessEqual(F(str(outward(value, True))), value)
            self.assertGreaterEqual(F(str(outward(value, False))), value)


if __name__ == '__main__':
    unittest.main()
