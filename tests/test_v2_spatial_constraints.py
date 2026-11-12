from fractions import Fraction as F
from itertools import combinations
import random
import unittest

from ez2rc.v2.difference_constraints import DifferenceGraph
from ez2rc.v2.spatial_constraints import ConstantSpacingMap, essential_constraints, minimum_envelope


class SpatialConstraintsTest(unittest.TestCase):
    def test_leaf_anchor_extension_matches_full_reconstruction(self):
        graph = DifferenceGraph()
        for i in range(8):
            graph.bound('origin', f'p{i}', 17*i-1, 17*i+1, 'generated observation')
        model = ConstantSpacingMap(graph, {'p0':F(0), 'p7':F(7)}, [f'p{i}' for i in range(8)])
        for batch in ([('v2','p2',F(-1,3),F(2,3),F(2),'label')],
                      [('v4','p4',F(-1,2),F(1,2),F(4),'label'),
                       ('v6','p6',F(-1),F(1),F(6),'label')]):
            extended = model.with_leaf_anchors(batch)
            rebuilt = ConstantSpacingMap(graph, extended.anchors, extended.destinations)
            self.assertEqual(extended.status, rebuilt.status)
            self.assertEqual(extended.scale, rebuilt.scale)
            self.assertEqual(extended.anchor_pair_count, rebuilt.anchor_pair_count)
            for node in extended.destinations:
                for offset in ((F(0),F(0)),(F(-1),F(2))):
                    self.assertEqual(extended.project(node,offset,F(1,3)), rebuilt.project(node,offset,F(1,3)))
                    for beat in (F(i,3) for i in range(-3,25)):
                        self.assertEqual(extended.scales_for_beat(node,beat,offset),
                                         rebuilt.scales_for_beat(node,beat,offset))
            model = extended
        conflicting = model.with_leaf_anchors([('wrong','p4',F(0),F(0),F(0),'contradictory label')])
        rebuilt = ConstantSpacingMap(graph, conflicting.anchors, conflicting.destinations)
        self.assertEqual(conflicting.status, 'conflict')
        self.assertEqual(conflicting.status, rebuilt.status)

    def test_leaf_extension_rejects_other_graph_changes_and_invalid_batches(self):
        graph = DifferenceGraph()
        graph.bound('a','b',10,12,'observation')
        model = ConstantSpacingMap(graph,{'a':F(0),'b':F(1)},['a','b'])
        revision = graph.revision
        with self.assertRaises(ValueError):
            model.with_leaf_anchors([('v','a',0,0,F(0),'valid'),('v','b',0,0,F(1),'duplicate')])
        self.assertEqual(graph.revision, revision)
        graph.bound('a','b',11,12,'new evidence')
        with self.assertRaises(ValueError):
            model.with_leaf_anchors([('v','a',0,0,F(0),'label')])

    def test_redundant_constraints_preserve_both_exact_envelopes(self):
        rng = random.Random(941)
        rows = [(str(i), F(rng.randrange(-30, 31), 3), F(a), F(a+w))
                for i, (a, w) in enumerate((rng.randrange(-80, 81), rng.randrange(1, 9))
                                           for _ in range(120))]
        reduced = essential_constraints(rows)
        self.assertLess(len(reduced), len(rows))
        self.assertEqual((min(r[1] for r in reduced), max(r[1] for r in reduced)),
                         (min(r[1] for r in rows), max(r[1] for r in rows)))
        # Every change of active constraint occurs at a line intersection.
        # Check all such points and the unbounded outer regions, including
        # common endpoint offsets and musical shifts.
        for offset, shift in ((F(0), F(0)), (F(7, 3), F(1, 6))):
            for sign, bound in ((1, 2), (-1, 3)):
                lines = [(sign*(r[bound]-offset), sign*(r[1]+shift)) for r in rows]
                kept = [(sign*(r[bound]-offset), sign*(r[1]+shift)) for r in reduced]
                points = {(c-b)/(m-n) for (m, b), (n, c) in combinations(lines, 2) if m != n}
                points.update((min(points)-1, max(points)+1))
                for x in points:
                    self.assertEqual(max(m*x+b for m, b in kept), max(m*x+b for m, b in lines))

    def test_many_anchor_storage_preserves_projection_and_grid_feasibility(self):
        graph = DifferenceGraph()
        anchors = {f'a{i}': F(i, 4) for i in range(81)}
        for i, node in enumerate(anchors):
            graph.bound('origin', node, 13*i-1, 13*i+1, 'generated landmark')
        graph.bound('origin', 'note', 517, 520, 'generated endpoint')
        model = ConstantSpacingMap(graph, anchors, ['note'])
        rows = [(node, beat, -graph.distances(node, reverse=True)['note'],
                 graph.distances(node)['note']) for node, beat in anchors.items()]
        self.assertLess(len(model.constraints['note']), 10)
        self.assertEqual(model.anchor_pair_count, 81*80//2)
        self.assertEqual(model.scale, (F(1038, 20), F(1042, 20)))
        self.assertEqual(model.scale_witnesses['lower'][2], model.scale[0])
        self.assertEqual(model.scale_witnesses['upper'][3], model.scale[1])
        lo, hi = 1/model.scale[1], 1/model.scale[0]
        for offset, shift in (((F(0), F(0)), F(0)), ((F(2), F(3)), F(1, 6))):
            expected = (minimum_envelope([(a-offset[1], beat+shift) for _, beat, a, _ in rows], lo, hi),
                        -minimum_envelope([(-(b-offset[0]), -beat-shift) for _, beat, _, b in rows], lo, hi))
            self.assertEqual(model.project('note', offset, shift), expected)
            for beat in (F(i, 12) for i in range(108, 133)):
                lower, upper = model.scale
                for _, anchor, a, b in rows:
                    delta = beat-anchor-shift
                    if delta == 0:
                        if not a-offset[1] <= 0 <= b-offset[0]:
                            lower, upper = F(1), F(0)
                    else:
                        left, right = sorted(((a-offset[1])/delta, (b-offset[0])/delta))
                        lower, upper = max(lower, left), min(upper, right)
                self.assertEqual(model.scales_for_beat('note', beat, offset, shift),
                                 (lower, upper) if lower <= upper else None)

    def test_insufficient_anchors_retain_an_unavailable_model(self):
        graph=DifferenceGraph();graph.bound('a','p',10,11,'observation')
        for anchors in ({},{'a':F(0)}):
            model=ConstantSpacingMap(graph,anchors,['p'])
            self.assertEqual(model.status,'unavailable')
            self.assertIsNone(model.project('p'))

    def test_envelope_matches_exhaustive_line_intersections(self):
        rng=random.Random(152)
        for _ in range(100):
            lines=[(F(rng.randrange(-30,30)),F(rng.randrange(-30,30))) for _ in range(10)]
            lower,upper=F(-2,3),F(5,4);points={lower,upper}
            for (m,b),(n,c) in combinations(lines,2):
                if m!=n and lower <= (c-b)/(m-n) <= upper:points.add((c-b)/(m-n))
            expected=min(max(m*x+b for m,b in lines) for x in points)
            self.assertEqual(minimum_envelope(lines,lower,upper),expected)

    def test_numbered_marker_restores_a_gap_without_fabricating_motion(self):
        graph=DifferenceGraph()
        graph.bound('bar0','bar1',118,122,'observed marker displacement')
        graph.bound('bar8','note',14,16,'co-observed line and endpoint')
        model=ConstantSpacingMap(graph,{'bar0':F(0),'bar1':F(3),'bar8':F(24)},['note'])
        self.assertEqual(model.scale,(F(118,3),F(122,3)))
        self.assertEqual(model.project('note',(F(2),F(3))), (F(24)+F(33,122),F(24)+F(21,59)))
        self.assertIsNone(graph.interval('bar0','bar8'))
        self.assertEqual(model.scales_for_beat('note',F(73,3),(F(2),F(3))), model.scale)
        self.assertIsNone(model.scales_for_beat('note',F(24),(F(2),F(3))))

    def test_scale_conflict_and_unobserved_endpoint_stay_unavailable(self):
        graph=DifferenceGraph()
        graph.bound('a','b',10,11,'first gap');graph.bound('c','d',20,21,'second gap')
        model=ConstantSpacingMap(graph,{'a':F(0),'b':F(1),'c':F(2),'d':F(3)},['z'])
        self.assertEqual(model.status,'conflict');self.assertIsNone(model.project('z'))
        model=ConstantSpacingMap(graph,{'a':F(0),'b':F(1)},['z'])
        self.assertEqual(model.status,'conditional');self.assertIsNone(model.project('z'))

    def test_common_scale_gives_tight_projection_and_retains_beat_shift(self):
        graph=DifferenceGraph()
        graph.bound('a','b',100,120,'scale')
        graph.bound('a','p',50,60,'first')
        graph.bound('p','b',50,60,'second')
        model=ConstantSpacingMap(graph,{'a':F(0),'b':F(1)},['p'])
        self.assertEqual(model.project('p'),(F(5,11),F(6,11)))
        self.assertEqual(model.project('p',beat_shift=F(1,3)),(F(26,33),F(29,33)))
