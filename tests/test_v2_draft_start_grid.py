"""Shared moving coordinates for independently drawn note starts."""
from fractions import Fraction as F
import unittest

from ez2rc.v2.difference_constraints import DifferenceGraph
from ez2rc.v2.draft_start_grid import refine_draft_starts
from ez2rc.v2.endpoint_ranges import closed
from ez2rc.v2.spatial_constraints import ConstantSpacingMap


class SharedDraftStartsTest(unittest.TestCase):
    def scene(self, *, shape='resolved', gap=0, second_pin=None, variable_offset=0):
        graph = DifferenceGraph()
        graph.bound('bar', 'bar2', 192, 192, 'numbered lines')
        graph.bound('bar', 'known', 40, 56, 'frame path')
        graph.bound('known', 'variable', gap+variable_offset, gap+variable_offset, 'same frame')
        model = ConstantSpacingMap(graph, {'bar': F(0), 'bar2': F(1)}, {'known', 'variable'})
        def point(name, lower, upper, status, offset):
            return dict(id=name, node=name, side='start', pixel_status=status, beat_shift=F(0),
                        offset=(F(offset),F(offset)), result={'relative_beat':{'beat_range':closed(F(lower,192),F(upper,192))}})
        points = [point('known', 48, 48, shape, 0), point('variable', 40, 56, 'resolved', variable_offset)]
        if second_pin is not None:
            graph.bound('known', 'other', 0, 0, 'paired')
            points.append(point('other', second_pin, second_pin, 'resolved', 0))
        return refine_draft_starts(model, points), points

    def test_shared_position_selects_label_and_applies_pixel_offset(self):
        proof, points = self.scene(gap=4, variable_offset=3)
        self.assertEqual(proof['status'], 'conditional')
        self.assertEqual(points[1]['result']['draft_start_ticks'], [52])

    def test_ambiguous_shape_does_not_anchor_another_note(self):
        proof, points = self.scene(shape='ambiguous')
        self.assertEqual(proof['status'], 'conditional')
        self.assertGreater(len(points[1]['result']['draft_start_ticks']), 1)

    def test_incompatible_pins_withhold_reductions(self):
        proof, points = self.scene(second_pin=52)
        self.assertEqual(proof['status'], 'conflict')
        self.assertNotIn('draft_start_ticks', points[1]['result'])
