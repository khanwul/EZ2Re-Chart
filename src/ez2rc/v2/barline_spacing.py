"""Place observed measure lines using numbered crossings and measured screen displacements."""
from collections import defaultdict
from fractions import Fraction
from math import ceil, floor

from .contracts import ratio, read_ratio
from .difference_constraints import ConstraintConflict, DifferenceGraph
from .spatial_constraints import ConstantSpacingMap
from .motion import STATUS


def measured_steps(movement):
    """Use displacements of matched features, including their search tolerance."""
    tolerance = movement.manifest['key']['parameters']['match_tolerance_px']
    steps = {}
    for chunk in movement.chunks():
        links = defaultdict(list)
        for frame, parent, lane, kind, previous, seed, delta, weight, residual in chunk['links']:
            links[int(frame), int(parent)].append(delta)
        for frame, parent, state, spread in zip(chunk['frame_id'], chunk['reference_frame'],
                                               chunk['status'], chunk['step_error_px']):
            frame, parent = int(frame), int(parent)
            if state == STATUS['repeated']:
                steps[frame] = (frame - 1, 0, 0)
            elif state == STATUS['observed'] and (frame, parent) in links:
                values = links[frame, parent]
                error = tolerance + float(spread)
                steps[frame] = (parent, float(min(values)) - error, float(max(values)) + error)
    return steps


def spatial_crossings(tracks, assignments, unplaced, steps, scroll=None):
    """Fit crossing positions on one spatial scale using observed motion only."""
    known = {r['marker_id']: read_ratio(r['beat']) for r in assignments}
    wanted = set(unplaced)
    graph = DifferenceGraph()
    nominal = {str(f): q for f, q in (scroll or {}).items()}
    try:
        for frame, (parent, low, high) in steps.items():
            graph.bound(str(parent), str(frame), low, high, 'matched displacement')
        for track in tracks:
            if track['id'] in known or track['id'] in wanted:
                for row in track['observations']:
                    if row.get('association_conflict'):
                        continue
                    low, high = map(float, row['visible_range_px'])
                    graph.bound(str(row['frame_id']), track['id'], -high, -low, 'visible line')
                    nominal.setdefault(track['id'], nominal.get(str(row['frame_id']), 0) - float(row['y_px']))
        graph.prepare(nominal)
        model = ConstantSpacingMap(graph, known, wanted)
    except ConstraintConflict:
        return {}, {'status': 'inconsistent_line_motion'}
    candidates, placements = {}, {}
    if model.status != 'conditional':
        return {}, {'status': model.status}
    for identity in unplaced:
        bounds = model.project(identity)
        if bounds is None:
            candidates[identity] = {'reason': 'disconnected_line_motion'}
            continue
        domain = [Fraction(t, 4) for t in range(ceil(4*bounds[0]), floor(4*bounds[1]) + 1)]
        domain = [beat for beat in domain if model.scales_for_beat(identity, beat)]
        candidates[identity] = {'beat_range': list(map(ratio, bounds)), 'candidate_beats': list(map(ratio, domain)),
                                'anchors': [r[0] for r in model.constraints[identity]]}
        if len(domain) == 1:
            placements[identity] = domain[0]
    # Marginal singletons still have to fit together at the same scale.
    if placements and ConstantSpacingMap(graph, known | placements, ()).status != 'conditional':
        return {}, {'status': 'joint_position_conflict', 'crossings': candidates}
    return placements, {'status': 'conditional', 'crossings': candidates,
                        'scale_px_per_beat': list(map(ratio, model.scale))}
