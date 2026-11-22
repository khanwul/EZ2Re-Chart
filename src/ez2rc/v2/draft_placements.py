"""Find two translated tap placements supported by co-observed screen edges."""
from collections import defaultdict

import numpy as np

from .contracts import exact_ms
from .draft_start_grid import intersect_scale


def pixel_separation(track, other, error):
    """Closed bounds on y(other) - y(track) across their common frames."""
    common, i, j = np.intersect1d(track[:, 0], other[:, 0], assume_unique=True, return_indices=True)
    if len(common) < 3:
        return None
    deltas = other[j, 1] - track[i, 1]
    return exact_ms(float(max(deltas))) - 2*error, exact_ms(float(min(deltas))) + 2*error


def translated_placements(domains, tracks, error, scale):
    """Each candidate beat already has a scale interval from its fixed anchors."""
    edges = defaultdict(dict)
    ordered = sorted(domains, key=lambda eid: tracks[eid][0, 0])
    for index, eid in enumerate(ordered):
        for j in range(index + 1, len(ordered)):
            other = ordered[j]
            if tracks[other][0, 0] > tracks[eid][-1, 0]:
                break
            bounds = pixel_separation(tracks[eid], tracks[other], error)
            if bounds is not None:
                low, high = bounds
                edges[eid][other] = (low, high)
                edges[other][eid] = (-high, -low)
    if not edges:
        return None

    # Visit a connected set in frontier order so each new beat is constrained
    # immediately. Isolated taps keep their own review items.
    order = []
    frontier = {min(edges, key=lambda eid: len(domains[eid]))}
    visited = set()
    while frontier:
        eid = min(frontier, key=lambda e: (len(domains[e]), e))
        frontier.remove(eid)
        order.append(eid)
        visited.add(eid)
        frontier.update(edges[eid].keys() - visited)
    if len(order) != len(edges):
        return None  # Independent groups need independent timing decisions.

    solutions = []
    stack = [({}, scale)]
    while stack:
        placed, interval = stack.pop()
        if len(placed) == len(order):
            solutions.append(placed)
            if len(solutions) == 3:
                return None
            continue
        eid = order[len(placed)]
        for beat, (lo, hi) in domains[eid].items():
            fit = max(interval[0], lo), min(interval[1], hi)
            if fit[0] > fit[1]:
                continue
            for other in edges[eid].keys() & placed.keys():
                fit = intersect_scale(fit, *edges[eid][other], beat - placed[other])
                if fit is None:
                    break
            if fit is not None:
                stack.append(({**placed, eid: beat}, fit))
    if len(solutions) != 2:
        return None
    early, late = sorted(solutions, key=lambda s: s[order[0]])
    shifts = {late[e] - early[e] for e in order}
    if len(shifts) == 1:
        return {e: (early[e], late[e]) for e in order}
    return None
