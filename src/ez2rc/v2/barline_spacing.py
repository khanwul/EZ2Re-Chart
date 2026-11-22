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
        return {}, {'status': model.status, 'no_measured_scale': model.scale is None}
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


def row_crossings(rows, times):
    """Bracket passage of each screen row between two observed frames."""
    crossings = {}
    for a, b in zip(rows, rows[1:]):
        f, g = a['frame_id'], b['frame_id']
        if (g != f+1 or a.get('association_conflict') or b.get('association_conflict')
                or b['y_px'] < a['y_px'] or times[g] <= times[f]):
            return {}
        low = ceil(a['visible_range_px'][1])
        high = floor(b['visible_range_px'][0])
        for y in range(low, high + 1):
            crossings[y] = times[f], times[g]
    return crossings


def clock_bracket(target, left, right, beats, period, center):
    """Intersect local crossing clocks over the full observed sweep.

    The surrounding labels must themselves admit the same crossing clock
    at every comparable height. Row samples within one video frame share
    its exposure interval and do not count as separate motion observations.
    """
    common = target.keys() & left.keys() & right.keys()
    if (not common or not min(common) < center < max(common)
            or any(len({track[y] for y in common}) < 3 for track in (left, target, right))):
        return None
    start, stop = beats
    elapsed = (stop-start)*period
    if any(max(left[y][0]+elapsed, right[y][0]) > min(left[y][1]+elapsed, right[y][1])
           for y in left.keys() & right.keys()):
        return None
    low, high = start, stop
    for track, beat in ((left, start), (right, stop)):
        for y in track.keys() & target.keys():
            low = max(low, beat+(target[y][0]-track[y][1])/period)
            high = min(high, beat+(target[y][1]-track[y][0])/period)
    first, last = ceil(4*low), floor(4*high)
    if first == last and start < Fraction(first, 4) < stop:
        return Fraction(first, 4), {'beat_range': [ratio(low), ratio(high)],
                                    'shared_screen_rows': len(common),
                                    'screen_row_range': [min(common), max(common)]}
    return None


def recover_disconnected_crossings(markers, tracks, assignments, evidence, times, period, center):
    labels = {a['marker_id']: read_ratio(a['beat']) for a in assignments}
    ordered = [m['id'] for m in sorted(markers, key=lambda m: m['timestamp_interval_ms'])]
    tracks = {t['id']: t['observations'] for t in tracks}
    recovered = {}
    for left, target, right in zip(ordered, ordered[1:], ordered[2:]):
        data = evidence.get('crossings', {}).get(target, {})
        disconnected = (evidence['status'] == 'conditional' and data.get('reason') == 'disconnected_line_motion'
                        or evidence['status'] == 'unavailable' and evidence.get('no_measured_scale'))
        if not disconnected or target in labels or left not in labels or right not in labels:
            continue
        result = clock_bracket(*(row_crossings(tracks[k], times) for k in (target, left, right)),
                               (labels[left], labels[right]), period, center)
        if result:
            beat, support = result
            recovered[target] = beat
            data = evidence.setdefault('crossings', {}).setdefault(target, {})
            data['local_crossing_clock'] = dict(support, beat=ratio(beat), anchors=[left, right])
    return recovered
