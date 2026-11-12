"""Generated geometric constraints, checked against exhaustive closure."""
from fractions import Fraction as F
import random
import unittest

from ez2rc.v2.difference_constraints import ConstraintConflict, DifferenceGraph


class DifferenceConstraintsTest(unittest.TestCase):
    def test_redundant_observations_tighten_a_shared_translation(self):
        graph = DifferenceGraph()
        graph.bound('frame-a', 'note', 20, 22, 'note observation a')
        graph.bound('frame-b', 'note', 8, 10, 'note observation b')
        graph.bound('frame-a', 'line', 50, 51, 'line observation a')
        graph.bound('frame-b', 'line', 39, 40, 'line observation b')
        self.assertEqual(graph.interval('frame-a','frame-b'), (F(10),F(12)))
        self.assertEqual(graph.interval('note','line'), (F(29),F(31)))
        graph.bound('gap-frame', 'isolated', 0, 1, 'unconnected observation')
        self.assertIsNone(graph.interval('note','isolated'))

    def test_inconsistent_cycle_has_observation_certificate(self):
        graph = DifferenceGraph()
        graph.bound('a','b',1,2,'one'); graph.bound('b','c',1,2,'two')
        graph.bound('a','c',5,6,'three')
        with self.assertRaises(ConstraintConflict) as caught: graph.prepare()
        self.assertLess(sum(e['upper'] for e in caught.exception.edges),0)
        self.assertEqual({e['evidence'] for e in caught.exception.edges},{'one','two','three'})

    def test_duplicate_observation_conflict_keeps_both_sources(self):
        graph=DifferenceGraph();graph.bound('frame','point',10,12,'first patch')
        with self.assertRaises(ConstraintConflict) as caught:
            graph.bound('frame','point',14,16,'second patch')
        self.assertEqual({r['evidence'] for r in caught.exception.edges},{'first patch','second patch'})
        self.assertLess(sum(r['upper'] for r in caught.exception.edges),0)

    def test_sparse_exact_solver_matches_independent_floyd_warshall(self):
        rng=random.Random(104)
        for _ in range(30):
            size=8;truth=[F(rng.randrange(-100,100),3) for _ in range(size)]
            graph=DifferenceGraph();closed={}
            for u in range(size):
                closed[u,u]=F(0)
                graph.bound(str(u),str(u),0,0,'identity')
                for v in range(u):
                    if rng.random()<.4:
                        delta=truth[v]-truth[u];lo,hi=delta-F(rng.randrange(8),5),delta+F(rng.randrange(8),7)
                        graph.bound(str(u),str(v),lo,hi,f'{u}-{v}')
                        closed[u,v],closed[v,u]=hi,-lo
            for k in range(size):
                for i in range(size):
                    for j in range(size):
                        if (i,k) in closed and (k,j) in closed:
                            d=closed[i,k]+closed[k,j]
                            closed[i,j]=min(closed.get((i,j),d),d)
            graph.prepare({str(i):truth[i]+100 for i in range(size)})
            for i in range(size):
                self.assertEqual(graph.distances(str(i)),{str(j):closed[i,j] for j in range(size) if (i,j) in closed})
                self.assertEqual(graph.distances(str(i),reverse=True),{str(j):closed[j,i] for j in range(size) if (j,i) in closed})
