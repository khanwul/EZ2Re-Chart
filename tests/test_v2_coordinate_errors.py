"""Procedural shared shifts, drift and interval fractional optimization."""
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import random
import unittest

from ez2rc.v2.coordinate_errors import Coordinate, quotient_range
from ez2rc.v2.relative_beats import SharedSpatialBeatMap, unpack, validate_reference
from test_v2_relative_beats import project, reference, scene


def uncertainty(ref):
    return {'version': 'procedural-common-translation',
            'sources': [{'id': 'translation', 'range_px': None, 'reason': 'unobserved origin'}],
            'anchors': {a['id']: {'translation': 1} for a in ref['anchors']},
            'endpoints': {e['endpoint_id']: {'translation': 1} for e in ref['endpoint_calibrations']}}


class CoordinateErrorsTest(unittest.TestCase):
    def test_exact_pixel_offset_bounds_avoid_serialization_rounding(self):
        ref = reference()
        ref['endpoint_calibrations'][0]['offset_px'] = [[1, 3], [2, 3]]
        result = project(ref)['candidates'][0]['endpoints']['start']
        self.assertEqual(unpack(result['coordinate_range_px']), (F(190,3),F(191,3)))

    def test_fractional_extrema_equal_exhaustive_vertices(self):
        rng = random.Random(731)
        for _ in range(80):
            sources = {str(k): (F(-rng.randrange(1, 4)), F(rng.randrange(1, 4))) for k in range(4)}
            n = Coordinate(F(rng.randrange(-8, 9)), {k: F(rng.randrange(-3, 4)) for k in sources})
            d = Coordinate(F(80), {k: F(rng.randrange(-3, 4)) for k in sources})
            expected = []
            for values in product(*sources.values()):
                numerator = n.nominal + sum(n.terms.get(k, F(0))*x for k, x in zip(sources, values))
                denominator = d.nominal + sum(d.terms.get(k, F(0))*x for k, x in zip(sources, values))
                expected.append(numerator / denominator)
            actual, reason = quotient_range(n, d, sources)
            self.assertIsNone(reason)
            self.assertEqual(actual, (min(expected), max(expected)))

    def test_common_unknown_translation_cancels_before_interval_arithmetic(self):
        ref = reference(); before = project(ref)
        ref['coordinate_uncertainty'] = uncertainty(ref)
        after = project(ref)
        for side in ('start', 'end'):
            a = before['candidates'][0]['endpoints'][side]['relative_beat']
            b = after['candidates'][0]['endpoints'][side]['relative_beat']
            self.assertEqual(a['beat_range'], b['beat_range'])
            self.assertEqual(b['cancelled_sources'], ['translation'])
        self.assertFalse(after['candidates'][0]['accepted'])

    def test_gap_shift_shared_only_by_later_anchor_and_endpoint_does_not_cancel(self):
        ref = reference(); model = uncertainty(ref)
        model['sources'].append({'id': 'gap', 'range_px': None, 'reason': 'no displacement observation'})
        model['anchors']['b']['gap'] = 1
        model['anchors']['c']['gap'] = 1
        model['endpoints']['t']['gap'] = 1
        ref['coordinate_uncertainty'] = model
        row = project(ref)['candidates'][0]
        self.assertIn('unbounded_relative_coordinate_error', row['endpoints']['start']['reasons'])
        self.assertEqual(row['endpoints']['end']['status'], 'conditional')
        self.assertIn('gap', row['endpoints']['end']['relative_beat']['cancelled_sources'])
        self.assertIsNone(row['duration_beat_range'])

    def test_bounded_shared_drift_encloses_correlated_assignments(self):
        ref = reference(); model = uncertainty(ref)
        model['sources'].append({'id': 'drift', 'range_px': [-6, 8], 'reason': 'bounded displacement'})
        for i, a in enumerate(ref['anchors']): model['anchors'][a['id']]['drift'] = i
        model['endpoints']['h']['drift'] = .25
        ref['coordinate_uncertainty'] = model
        p = project(ref)['candidates'][0]['endpoints']['start']['relative_beat']
        lo, hi = unpack(p['beat_range'])
        for shift in range(-6, 9):
            truth = (F(60) + F(shift, 4))*3/(120+shift)
            self.assertLessEqual(lo, truth); self.assertGreaterEqual(hi, truth)
        self.assertGreater(hi, lo)
        self.assertEqual(p['cancelled_sources'], ['translation'])

    def test_uncertain_anchor_crossing_uses_both_pieces(self):
        ref = reference(); model = uncertainty(ref)
        line = SharedSpatialBeatMap(ref['anchors'], 'withhold', model)
        p = line.project(F(117), F(123), 'h')
        self.assertEqual(unpack(p['beat_range']), (F(117, 40), F(243, 80)))
        self.assertEqual(p['support_anchor_ids'], ['a', 'b', 'c'])

    def test_missing_binding_and_unknown_nonshared_sources_withhold(self):
        ref = reference(); model = uncertainty(ref); del model['endpoints']['h']
        ref['coordinate_uncertainty'] = model
        p = project(ref)['candidates'][0]['endpoints']['start']
        self.assertEqual(p['reasons'], ['endpoint_coordinate_dependence_unknown'])
        model['endpoints']['h'] = {}
        self.assertIn('unbounded_relative_coordinate_error', project(ref)['candidates'][0]['reasons'])

    def test_unknown_absolute_origin_is_not_required_for_local_bracket(self):
        ref = reference(); model = uncertainty(ref)
        model['sources'].append({'id': 'earlier-gap', 'range_px': None, 'reason': 'unobserved earlier motion'})
        model['anchors']['a']['earlier-gap'] = 1
        ref['coordinate_uncertainty'] = model
        row = project(ref)['candidates'][0]
        self.assertEqual(row['endpoints']['end']['status'], 'conditional')
        self.assertEqual(unpack(row['endpoints']['end']['relative_beat']['beat_range']), (F(9, 2),)*2)

    def test_invalid_dependency_inputs_and_nonpositive_scale_fail_closed(self):
        for invalid in ('missing_anchor', 'duplicate', 'unknown', 'nonfinite', 'unused_endpoint', 'reserved', 'unordered_range'):
            ref = reference(); model = uncertainty(ref); ref['coordinate_uncertainty'] = model
            if invalid == 'missing_anchor': del model['anchors']['b']
            if invalid == 'duplicate': model['sources'].append(deepcopy(model['sources'][0]))
            if invalid == 'unknown': model['endpoints']['h']['absent'] = 1
            if invalid == 'nonfinite': model['endpoints']['h']['translation'] = float('nan')
            if invalid == 'unused_endpoint': model['endpoints']['other'] = {}
            if invalid == 'reserved': model['sources'][0]['id'] = 'residual:anchor:a'
            if invalid == 'unordered_range': model['sources'][0]['range_px'] = [2, 1]
            with self.subTest(invalid=invalid), self.assertRaises(ValueError): project(ref)
        ref = reference(); model = uncertainty(ref)
        model['sources'].append({'id': 'scale', 'range_px': [-120, 120], 'reason': 'scale not established'})
        model['anchors']['b']['scale'] = 1
        model['anchors']['c']['scale'] = 2
        ref['coordinate_uncertainty'] = model
        self.assertIn('anchor_order_not_established', project(ref)['candidates'][0]['reasons'])

    def test_capture_gap_does_not_invalidate_independent_spatial_reference(self):
        data = scene(); data[3]['frame_id'][3:] += 1
        data[4]['endpoints'][1]['q_px'] = 234.
        result = project(data=data)['candidates'][0]
        self.assertEqual(result['status'], 'conditional')
        self.assertIn('missing_frames', result['endpoints']['end']['capture_time']['reasons'])

    def test_exact_internal_anchor_can_have_two_compatible_brackets(self):
        ref = reference(); line = SharedSpatialBeatMap(ref['anchors'], 'withhold', uncertainty(ref))
        p = line.project(F(120), F(120), 'h')
        self.assertEqual(unpack(p['beat_range']), (F(3), F(3)))


if __name__ == '__main__':
    unittest.main()
