"""Independent moving caps with known pixel displacements."""
from fractions import Fraction as F
from types import SimpleNamespace
import unittest

import numpy as np

from ez2rc.v2.difference_constraints import DifferenceGraph
from ez2rc.v2.draft_tap_anchors import extend_draft_tap_anchors
from ez2rc.v2.spatial_constraints import ConstantSpacingMap


class DraftTapAnchorTest(unittest.TestCase):
    def scene(self, *, kind='tap', flag=2, dark_gap=False, reverse_pattern=False,
              seed_offset=0, current_y=65, previous_y=40, contradictory=False, subpixel=False):
        graph = DifferenceGraph()
        graph.bound('frame:1', 'line0', 40, 40, 'screen-line')
        graph.bound('frame:1', 'line1', 90, 90, 'screen-line')
        graph.bound('frame:0', 'cap', 100-previous_y, 100-previous_y, 'earlier-cap')
        graph.bound('frame:0', 'neighbor', 70, 70, 'earlier-neighbor')
        anchors = {'line0': F(0), 'line1': F(1)}
        model = ConstantSpacingMap(graph, anchors, {'cap', 'neighbor'})
        self.assertIsNone(model.project('neighbor'))
        values = [54., 170., 115., 225.]
        projection = np.zeros((2, 1, 100))
        projection[0, 0, previous_y:previous_y+4] = values
        projection[1, 0, current_y:current_y+4] = values[::-1] if reverse_pattern else values
        if dark_gap:
            projection[1, 0, current_y + 4] = 200
        seed = current_y + seed_offset
        rows = [[0, 0, previous_y], [1, 0, seed]]
        positions = [previous_y, current_y + .5 if subpixel else current_y]
        flags = [0, flag]
        if contradictory:
            rows.append([1, 0, seed+8])
            positions.append(current_y+8)
            flags.append(flag)
            projection[1, 0, current_y+8:current_y+12] = values
        chunk = {'frame_id': np.arange(2), 'candidate_index': np.array(rows),
                 'positions': np.array([[[0,y]] for y in positions]), 'flags': np.array(flags)[:,None],
                 'scores': np.ones((len(rows),1)), 'projection': projection}
        obs = SimpleNamespace(chunks=lambda: iter([chunk]), manifest={'metadata': {'profile': {
            'judge_y': 100, 'effect_top': 62, 'lanes': [{'roi': [0,0,10,100]}]}},
            'key': {'parameters': {'algorithm': {'search_radius_px': 3, 'brightness_threshold': 35}}}})
        endpoints = {'tap': {'kind': kind, 'observation_ids': [f'f{f}:l{l}:y{s}' for f,l,s in rows]}}
        return extend_draft_tap_anchors(model, {'tap': 'cap'}, endpoints, obs,
                                        [{'tap': (4,10)}], .85, F(0))

    def test_shared_frame_recovers_neighbor_without_snapping(self):
        model, evidence = self.scene()
        self.assertEqual(model.project('cap'), (F(-1,10), F(-1,10)))
        self.assertEqual(model.project('neighbor'), (F(1,10), F(1,10)))
        self.assertEqual(evidence[0]['preceding_observation_id'], 'f0:l0:y40')

    def test_rejects_body_clipping_stationary_and_search_boundary(self):
        for changes in ({'kind':'head'}, {'dark_gap':True}, {'reverse_pattern':True},
                        {'flag':3}, {'previous_y':65}, {'seed_offset':3}, {'current_y':96},
                        {'subpixel':True}):
            with self.subTest(changes=changes):
                model, evidence = self.scene(**changes)
                self.assertIsNone(model.project('neighbor'))
                self.assertEqual(evidence, [])

    def test_conflicting_measurements_preserve_original_geometry(self):
        model, evidence = self.scene(contradictory=True)
        self.assertIsNone(model.project('cap'))
        self.assertEqual(len(evidence), 2)
        self.assertTrue(all('rejected' in row for row in evidence))
