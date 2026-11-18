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


class DraftEntryTest(unittest.TestCase):
    def reconstruct(self, *, covered=False, flat=False, reverse=False, jump=False, short=False):
        # The exiting cap and the entering cap are both seen at frame 1.
        # Frame 1 originally belongs to neither spatial component.
        graph = DifferenceGraph()
        graph.bound('frame:0', 'zero', 0, 0, 'origin')
        graph.bound('frame:0', 'one', 60, 60, 'measure')
        graph.bound('frame:0', 'exit', 38, 38, 'full')
        graph.bound('frame:2', 'entry', 83, 83, 'full')
        graph.bound('frame:3', 'entry', 63 if jump else 59, 63 if jump else 59, 'full')
        model = ConstantSpacingMap(graph, {'zero':F(0), 'one':F(1)}, {'entry','exit'})
        values = np.array([120, 190, 80, 165, 230, 90, 130, 200, 70, 170, 100, 150])
        image = np.zeros((4, 2, 110))
        image[0,0,62:74] = values
        image[1,0,86:98] = values
        if covered:
            image[1,0,98] = 180
        image[2,1,17:29] = values
        suffix = values[-5:]
        image[1,1,:5] = suffix[::-1] if reverse else suffix
        if flat:
            image[1,1,:5] = 100
        if short:
            image[1,1,2:] = 0
        rows = [(0,0,62),(1,0,86),(2,1,17),(3,1,41)]
        chunk = dict(frame_id=np.arange(4), candidate_index=np.array(rows),
                     positions=np.array([[[0,y]] for _,_,y in rows]),
                     flags=np.array([[0],[2],[0],[0]]), scores=np.ones((4,1)), projection=image)
        profile = dict(judge_y=100, effect_top=85, lanes=[dict(roi=[0,0,1,110])]*2)
        obs = SimpleNamespace(chunks=lambda: iter([chunk]), manifest={'metadata':{'profile':profile},
                              'key':{'parameters':{'algorithm':{'search_radius_px':4,'brightness_threshold':40}}}})
        ends = {key:{'kind':'tap','observation_ids':[f'f{f}:l{l}:y{y}' for f,l,y in group]}
                for key, group in [('exit',rows[:2]), ('entry',rows[2:])]}
        return extend_draft_tap_anchors(model, {k:k for k in ends}, ends, obs, [{'tap':(12,1)}]*2, .8, F(0))

    def test_shared_partial_and_full_caps_create_a_measured_frame(self):
        model, evidence = self.reconstruct()
        self.assertEqual(model.project('entry'), (F(131,60), F(131,60)))
        entry, = (r for r in evidence if r['endpoint_id']=='entry')
        self.assertEqual((entry['visible_bottom_px'], entry['tap_y_px']), (4,-7))

    def test_requires_the_suffix_motion_and_a_separate_coordinate(self):
        for key in ('covered','flat','reverse','jump','short'):
            with self.subTest(key=key):
                model, evidence = self.reconstruct(**{key:True})
                self.assertIsNone(model.project('entry'))
                self.assertFalse(evidence)
