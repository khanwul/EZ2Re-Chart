"""Exact bounds for shared translations with redundant observed constraints.

An edge u -> v with weight c means x[v] - x[u] <= c. Unobserved
translations have no edge. All constraints must coexist; contradictory cycles
are reported rather than silently dropping observations or widening bounds.
"""
from collections import deque
from fractions import Fraction
from heapq import heappop, heappush
from math import lcm

from .contracts import exact_ms


class ConstraintConflict(ValueError):
    def __init__(self, edges):
        self.edges = edges
        super().__init__('inconsistent coordinate constraints: negative cycle')


class DifferenceGraph:
    def __init__(self):
        self.edges = {}
        self.potentials = None
        self.compiled = None
        self.revision = 0

    def bound(self, left, right, lower, upper, evidence):
        """Constrain x[right] - x[left] to one closed observed interval."""
        lower, upper = exact_ms(lower), exact_ms(upper)
        if lower > upper or not evidence:
            raise ValueError('invalid coordinate constraint or missing evidence')
        self.revision += 1
        self.edges.setdefault(left, {})
        self.edges.setdefault(right, {})
        for u, v, weight in ((left, right, upper), (right, left, -lower)):
            previous = self.edges[u].get(v)
            if previous is None or weight < previous[0]:
                self.edges[u][v] = weight, evidence
        self.potentials = None
        self.compiled = None
        if self.edges[left][right][0]+self.edges[right][left][0] < 0:
            raise ConstraintConflict([{'left':u,'right':v,'upper':self.edges[u][v][0],
                                       'evidence':self.edges[u][v][1]} for u,v in ((left,right),(right,left))])

    def prepare(self, nominal=None):
        """Find feasible potentials, retaining an exact conflict certificate."""
        distance = {v: exact_ms((nominal or {}).get(v, 0)) for v in self.edges}
        queue = deque(self.edges)
        queued = set(queue)
        length, previous = {v: 0 for v in queue}, {}
        updates = 0

        def negative_cycle(at):
            order, seen = [],{}
            while at in previous and at not in seen:
                seen[at] = len(order); order.append(at); at = previous[at]
            if at not in seen:
                return None
            cycle = []
            for target in order[seen[at]:]:
                parent = previous[target]
                weight, proof = self.edges[parent][target]
                cycle.append({'left':parent,'right':target,'upper':weight,'evidence':proof})
            return cycle if sum(e['upper'] for e in cycle)<0 else None
        while queue:
            u = queue.popleft(); queued.remove(u)
            for v, (weight, evidence) in self.edges[u].items():
                value = distance[u] + weight
                if value >= distance[v]:
                    continue
                distance[v] = value
                previous[v] = u
                length[v] = length[u] + 1
                updates += 1
                # A short contradictory observation loop need not circulate
                # once per unrelated node before its certificate is reported.
                if updates%1024 == 0:
                    cycle = negative_cycle(v)
                    if cycle:
                        raise ConstraintConflict(cycle)
                if length[v] >= len(self.edges):
                    # Walk through the predecessor graph to reach its cycle.
                    at = v
                    for _ in self.edges:
                        at = previous[at]
                    start, cycle = at, []
                    while True:
                        parent = previous[at]
                        w, proof = self.edges[parent][at]
                        cycle.append({'left': parent, 'right': at, 'upper': w, 'evidence': proof})
                        at = parent
                        if at == start:
                            break
                    if sum(e['upper'] for e in cycle) >= 0:
                        raise RuntimeError('invalid negative cycle certificate')
                    raise ConstraintConflict(cycle)
                if v not in queued:
                    queue.append(v); queued.add(v)
        self.potentials = distance
        self.compiled = None

    def _compile(self):
        # Reweight once. Integral shortest-path arithmetic avoids repeatedly
        # normalizing the same rational denominators for every anchor query.
        scale = 1
        forward, reverse = {u:{} for u in self.edges},{u:{} for u in self.edges}
        for u, edges in self.edges.items():
            for v, (weight, _) in edges.items():
                step = weight+self.potentials[u]-self.potentials[v]
                if step < 0:
                    raise RuntimeError('invalid coordinate potentials')
                forward[u][v] = step
                scale = lcm(scale,step.denominator)
        for u, edges in forward.items():
            for v, step in edges.items():
                value = step.numerator*(scale//step.denominator)
                forward[u][v] = value
                reverse[v][u] = value
        self.compiled = forward,reverse,scale

    def distances(self, start, *, reverse=False):
        """Tight upper differences to/from a node, with absent paths omitted."""
        if self.potentials is None:
            self.prepare()
        if start not in self.edges:
            raise ValueError('unknown coordinate node')
        if self.compiled is None:
            self._compile()
        forward,backward,scale = self.compiled
        p = self.potentials
        graph = backward if reverse else forward
        if reverse:
            p = {u: -v for u, v in p.items()}
        costs, heap = {start: 0}, [(0, start)]
        while heap:
            cost, u = heappop(heap)
            if cost != costs[u]:
                continue
            for v, step in graph[u].items():
                new = cost + step
                if v not in costs or new < costs[v]:
                    costs[v] = new
                    heappush(heap, (new, v))
        return {v: Fraction(cost,scale) - p[start] + p[v] for v, cost in costs.items()}

    def interval(self, left, right):
        forward = self.distances(left)
        backward = self.distances(left, reverse=True)
        if right not in forward or right not in backward:
            return None
        return -backward[right], forward[right]
