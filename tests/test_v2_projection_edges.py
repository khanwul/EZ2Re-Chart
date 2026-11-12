from fractions import Fraction as F
import unittest

import numpy as np

from ez2rc.v2.contracts import read_ratio
from ez2rc.v2.difference_constraints import ConstraintConflict, DifferenceGraph
from ez2rc.v2.projection_edges import (REGISTRATION_VERSION, VERSION, measure_edge, measure_registration,
                                      reference_interval, validate_registration, validate_spec)


def specification(**changes):
    return dict(version=VERSION, direction='falling', window_offset_px=[3, 12],
                thresholds=[64, 128, 192], reference_offset_px=[0, 0], error_px=0, **changes)


class ProjectionEdgeTest(unittest.TestCase):
    def test_generated_subpixel_edges_and_translated_viewports(self):
        spec = specification()
        for fraction in (F(0), F(1, 4), F(1, 2), F(3, 4)):
            boundary = 17 + fraction
            signal = 255 * np.clip((float(boundary) + 1 - np.arange(40))/2, 0, 1)
            for origin in (-20, 0, 130):
                measured = measure_edge(signal, 10+origin, origin, spec)
                self.assertEqual(measured['status'], 'measured')
                lower, upper = reference_interval(measured, spec)
                self.assertLessEqual(lower, boundary+origin)
                self.assertGreaterEqual(upper, boundary+origin)
                self.assertLessEqual(upper-lower, 3)
                spec_rising = {**spec, 'direction':'rising'}
                rising = measure_edge(255-signal, 10+origin, origin, spec_rising)
                # Complementing 8-bit intensity does not complement the
                # thresholds (nor their inclusive side). Check the independent
                # known boundary rather than assuming identical brackets.
                a,b = reference_interval(rising, spec_rising)
                self.assertLessEqual(a, boundary+origin)
                self.assertGreaterEqual(b, boundary+origin)
                self.assertLessEqual(b-a, 3)

    def test_missing_or_multiple_crossings_are_not_chosen_or_clamped(self):
        spec = specification()
        for signal in (np.zeros(40), np.full(40, 255), np.tile([255, 0], 20)):
            result = measure_edge(signal, 10, 0, spec)
            self.assertEqual(result['status'], 'unavailable')
            self.assertIsNone(reference_interval(result, spec))
            self.assertEqual(result['reason'], 'missing_or_multiple_threshold_crossings')
            self.assertEqual(len(result['threshold_crossings']), 3)
        result = measure_edge(np.arange(40), 35, 0, spec)
        self.assertEqual(result['reason'], 'edge_window_outside_observed_rows')
        self.assertEqual(result['threshold_crossings'], [])

    def test_error_and_reference_offset_are_explicit_and_exact(self):
        spec = {**specification(), 'reference_offset_px': [[1, 3], [2, 3]], 'error_px': .5}
        measured = measure_edge(np.r_[np.full(18, 255), np.zeros(22)], 10, 0, spec)
        self.assertEqual(measured['row_interval_px'], [17, 18])
        self.assertEqual(reference_interval(measured, spec), (F(101, 6), F(115, 6)))
        self.assertNotIn('verified', measured)

    def test_invalid_definitions_and_signals_fail(self):
        for change in ({'thresholds':[]}, {'thresholds':[0]}, {'thresholds':[255]},
                       {'thresholds':[128,64]}, {'thresholds':[64,64]}, {'thresholds':[True]},
                       {'error_px':-1}, {'reference_offset_px':[2,1]},
                       {'window_offset_px':[0,False]}, {'window_offset_px':[1,1]},
                       {'direction':'nearest'}, {'version':'unknown'}, {'verified':True}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_spec({**specification(), **change})
        for signal in (np.array([[1,2]]), np.array([float('nan')])):
            with self.assertRaises(ValueError): measure_edge(signal, 10, 0, specification())


class TemplateRegistrationTest(unittest.TestCase):
    def definition(self):
        return {'version':REGISTRATION_VERSION, 'direction':'rising', 'window_offset_px':[0,9],
                'template_sha256':'1'*64, 'error_px':0,
                'source_crossings':[{'threshold':t,'row_interval_px':[a,a+1]}
                                    for t,a in ((64,2),(128,3),(192,4))]}

    def signal(self, translation):
        return np.r_[np.zeros(translation), [0,0,30,100,180,240,255,255,255,255], np.zeros(12)]

    def test_paired_thresholds_retain_known_translation_dependency(self):
        for shift in (9,14,31):
            for origin in (0,150):
                spec=self.definition()
                measured=measure_registration(self.signal(shift),shift+origin,origin,spec)
                self.assertEqual(measured['status'],'measured')
                span=tuple(map(read_ratio,measured['template_top_interval_px']))
                self.assertEqual(span,(F(shift+origin-1),F(shift+origin+1)))
                raw=measured['row_interval_px']
                unpaired=(raw[0]-5,raw[1]-2)
                self.assertLess(span[1]-span[0],unpaired[1]-unpaired[0])
                spec['error_px']=.25
                wider=measure_registration(self.signal(shift),shift+origin,origin,spec)
                self.assertEqual(tuple(map(read_ratio,wider['template_top_interval_px'])),
                                 (span[0]-F(1,4),span[1]+F(1,4)))

    def test_inconsistent_threshold_translation_preserves_conflicting_constraints(self):
        spec=self.definition()
        for row,start in zip(spec['source_crossings'],(0,5,7)):row['row_interval_px']=[start,start+1]
        measured=measure_registration(self.signal(14),14,0,spec)
        self.assertEqual(measured['status'],'conflict')
        self.assertEqual(len(measured['paired_top_intervals']),3)
        graph=DifferenceGraph()
        with self.assertRaises(ConstraintConflict):
            for row in measured['paired_top_intervals']:
                a,b=map(read_ratio,row['top_interval_px'])
                graph.bound('frame','point',100-b,100-a,str(row['threshold']))

    def test_missing_crossings_and_invalid_source_do_not_localize(self):
        spec=self.definition()
        measured=measure_registration(np.zeros(40),14,0,spec)
        self.assertEqual(measured['status'],'unavailable')
        self.assertEqual(measured['paired_top_intervals'],[])
        for change in ({'source_crossings':[]}, {'template_sha256':'unknown'}, {'error_px':-1},
                       {'source_crossings':[{'threshold':64,'row_interval_px':[1,3]}]}):
            with self.subTest(change=change),self.assertRaises(ValueError):
                validate_registration({**spec,**change})
