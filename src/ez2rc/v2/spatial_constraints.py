"""Constant spatial-spacing hypotheses constrained by shared observations.

This model assumes that numbered measure boundaries share one positive spatial
scale. Capture timestamps do not determine that scale. Calibration and physical
associations remain explicit inputs, separate from algebraic feasibility.
"""
from fractions import Fraction

from .contracts import exact_ms


def essential_constraints(rows):
    """Retain exact envelope supports, not a sampled approximation.

    Each lower or negated upper bound is affine in inverse scale. A line
    below their upper hull for every real inverse scale cannot affect a
    projection or a grid-label feasibility check. Common offset/beat shifts
    translate all lines equally and preserve this dominance.
    """
    keep = set()
    for upper in (False, True):
        lines = {}
        for index, (_, beat, lower, higher) in enumerate(rows):
            slope, intercept = (-higher, -beat) if upper else (lower, beat)
            if slope not in lines or intercept > lines[slope][0]:
                lines[slope] = intercept, index
        hull = []
        for slope, (intercept, index) in sorted(lines.items()):
            start = None
            while hull:
                pm, pb, _, previous_start = hull[-1]
                start = (pb-intercept)/(slope-pm)
                if previous_start is None or start > previous_start:
                    break
                hull.pop()
            hull.append((slope, intercept, index, start if hull else None))
        keep.update(row[2] for row in hull)
    if rows:
        # Coverage is a property of all observed anchors, separate from which
        # constraints are active on the numerical envelope.
        keep.add(min(range(len(rows)), key=lambda i: rows[i][1]))
        keep.add(max(range(len(rows)), key=lambda i: rows[i][1]))
    return [row for i, row in enumerate(rows) if i in keep]


def minimum_envelope(lines, lower, upper):
    """Exact minimum of a convex maximum of affine lines on a closed interval."""
    slopes = {}
    for slope, intercept in lines:
        slopes[slope] = max(intercept, slopes.get(slope, intercept))
    hull = []
    for slope, intercept in sorted(slopes.items()):
        start = None
        while hull:
            pm, pb, previous_start = hull[-1]
            start = (pb-intercept)/(slope-pm)
            if previous_start is None or start > previous_start:
                break
            hull.pop()
        if not hull:
            start = None
        hull.append((slope, intercept, start))
    values = []
    for i, (slope, intercept, start) in enumerate(hull):
        a = lower if start is None else max(lower, start)
        stop = hull[i+1][2] if i+1 < len(hull) else upper
        b = min(upper, stop)
        if a <= b:
            values.extend((slope*a+intercept, slope*b+intercept))
    if not values:
        raise ValueError('empty line envelope')
    return min(values)


class ConstantSpacingMap:
    def __init__(self, graph, anchors, destinations):
        if any(not isinstance(v, Fraction) for v in anchors.values()):
            raise ValueError('numbered spatial anchors require exact beat values')
        self.graph, self.anchors, self.destinations = graph,dict(anchors),set(destinations)
        self.constraints = {node: [] for node in destinations}
        self.scale = None
        self.equality_conflicts = []
        self.anchor_pair_count = 0
        self.scale_witnesses = {}
        self._include(anchors, set())

    def _include(self, anchors, previous):
        for node, beat in sorted(anchors.items(), key=lambda v: v[1]):
            graph = self.graph
            forward, backward = graph.distances(node), graph.distances(node, reverse=True)
            for target in self.destinations:
                if target in forward and target in backward:
                    self.constraints[target].append((node, beat, -backward[target], forward[target]))
                    if len(self.constraints[target]) >= 64:
                        self.constraints[target] = essential_constraints(self.constraints[target])
            for other, other_beat in self.anchors.items():
                if other_beat == beat and other in forward and other in backward:
                    if not -backward[other] <= 0 <= forward[other]:
                        self.equality_conflicts.append((node,other,-backward[other],forward[other]))
                if (other_beat > beat or other in previous and other_beat < beat) and other in forward and other in backward:
                    delta = other_beat - beat
                    lower, upper = sorted((-backward[other]/delta, forward[other]/delta))
                    self.anchor_pair_count += 1
                    witness = (node, other, lower, upper) if delta > 0 else (other, node, lower, upper)
                    if self.scale is None or lower > self.scale[0]:
                        self.scale_witnesses['lower'] = witness
                    if self.scale is None or upper < self.scale[1]:
                        self.scale_witnesses['upper'] = witness
                    self.scale = (max(lower,self.scale[0]),min(upper,self.scale[1])) if self.scale else (lower,upper)
        self.constraints = {node:essential_constraints(rows) for node,rows in self.constraints.items()}
        self.status = ('conflict' if self.equality_conflicts else 'unavailable' if self.scale is None else 'conflict' if self.scale[0] > self.scale[1]
                       else 'unavailable' if self.scale[0] <= 0 else 'conditional')
        self.graph_revision = self.graph.revision

    def with_leaf_anchors(self, additions):
        """Add calibrated leaf anchors without repeating unchanged queries.

        Each tuple is (new node, existing node, lower, upper, beat, evidence).
        A leaf's only round trip has nonnegative width, so it cannot shorten
        paths between old nodes. Existing projection and scale constraints
        remain exact; only pairs involving a new anchor require new queries.
        Arbitrary graph changes do not have this property and use a fresh map.
        """
        graph = self.graph
        if graph.revision != self.graph_revision:
            raise ValueError('coordinate graph changed since the original map')
        additions = tuple(additions)
        new = {}
        for virtual, node, lower, upper, beat, evidence in additions:
            if (virtual in graph.edges or virtual in new or node not in graph.edges
                    or not isinstance(beat, Fraction) or exact_ms(lower) > exact_ms(upper) or not evidence):
                raise ValueError('invalid leaf anchor extension')
            new[virtual] = beat
        nominal = dict(graph.potentials)
        for virtual, node, lower, upper, beat, evidence in additions:
            graph.bound(virtual, node, lower, upper, evidence)
            nominal[virtual] = nominal[node]-(exact_ms(lower)+exact_ms(upper))/2
        graph.prepare(nominal)
        model = object.__new__(ConstantSpacingMap)
        model.graph, model.anchors, model.destinations = graph, {**self.anchors, **new}, self.destinations
        model.constraints = {node:list(rows) for node,rows in self.constraints.items()}
        model.scale = self.scale
        model.equality_conflicts = list(self.equality_conflicts)
        model.anchor_pair_count = self.anchor_pair_count
        model.scale_witnesses = dict(self.scale_witnesses)
        model._include(new, set(self.anchors))
        return model

    def project(self, node, offset=(Fraction(0), Fraction(0)), beat_shift=Fraction(0)):
        offset = tuple(map(exact_ms, offset))
        if len(offset) != 2 or offset[0] > offset[1] or not isinstance(beat_shift, Fraction):
            raise ValueError('invalid spatial endpoint calibration')
        if self.status != 'conditional' or not self.constraints.get(node):
            return None
        lo, hi = (1/self.scale[1], 1/self.scale[0])
        rows = self.constraints[node]
        lower = minimum_envelope([(a-offset[1], beat+beat_shift) for _, beat, a, _ in rows], lo, hi)
        upper = -minimum_envelope([(-(b-offset[0]), -beat-beat_shift) for _, beat, _, b in rows], lo, hi)
        return lower, upper

    def scales_for_beat(self, node, beat, offset=(Fraction(0), Fraction(0)), beat_shift=Fraction(0)):
        """Necessary scale domain for one exact grid label, without rounding."""
        if self.status != 'conditional' or not self.constraints.get(node):
            return None
        lower, upper = self.scale
        for _, anchor, a, b in self.constraints[node]:
            a, b, delta = a-offset[1], b-offset[0], beat-anchor-beat_shift
            if delta == 0:
                if not a <= 0 <= b:
                    return None
            else:
                lo, hi = sorted((a/delta, b/delta))
                lower, upper = max(lower, lo), min(upper, hi)
                if lower > upper:
                    return None
        return lower, upper
