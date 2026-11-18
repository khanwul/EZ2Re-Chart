"""Constrain draft starts by a common scale and shared frame coordinates."""
from fractions import Fraction

from .contracts import ratio, read_ratio
from .rhythm import DRAFT_GRID, grid_candidate_ticks


def _intersection(scale, lower, upper, delta):
    if delta == 0:
        return scale if lower <= 0 <= upper else None
    a, b = sorted((lower / delta, upper / delta))
    lo, hi = max(scale[0], a), min(scale[1], b)
    return (lo, hi) if lo <= hi else None


def _relations(graph, node, fixed):
    # Every path through another fixed coordinate splits into two inequalities
    # there. Distances through the remaining free coordinates retain their
    # exact bounds even when several notes use the same frame.
    forward = graph.distances(node, stop_at=fixed)
    backward = graph.distances(node, reverse=True, stop_at=fixed)
    return [(other, -backward[other], limit) for other, limit in forward.items()
            if other != node]


def _fit(scale, beat, offset, fixed, relations):
    for other, lower, upper in relations:
        other_beat, other_offset = fixed[other]
        shift = offset - other_offset
        scale = _intersection(scale, lower + shift, upper + shift, other_beat - beat)
        if scale is None:
            break
    return scale


def refine_draft_starts(model, points):
    """Only a unique label of a resolved shape may constrain another start.

    A candidate must coexist with every pinned start and barline for the same
    scale. Recheck new pins together before returning a reduced candidate set.
    Optical ranges and continuous tail coordinates stay available separately.
    """
    fixed = {node: (beat, Fraction(0)) for node, beat in model.anchors.items()}
    ends = {p['candidate_id']: p['result'].get('relative_beat') for p in points if p['side'] == 'end'}
    candidates = []
    for point in points:
        if point['side'] != 'start' or point['offset'][0] != point['offset'][1] or point['beat_shift']:
            continue
        spatial = point['result'].get('relative_beat')
        ticks = grid_candidate_ticks(spatial['beat_range'], DRAFT_GRID) if spatial else None
        if ticks:
            eligible = point['pixel_status'] == 'resolved'
            if point.get('candidate_id') in ends:
                end = ends[point['candidate_id']]
                eligible = bool(eligible and end and end['beat_range'] and
                                Fraction(max(ticks), DRAFT_GRID.ticks_per_beat) < read_ratio(end['beat_range']['lower']))
            candidates.append((point, sorted(ticks), eligible))
    if not any(len(ticks) > 1 for _, ticks, _ in candidates):
        return {'status': 'no_ambiguous_starts'}
    scale = model.scale
    resolution = DRAFT_GRID.ticks_per_beat
    choices = {}
    while True:
        for point, ticks, eligible in candidates:
            ticks = choices.get(point['id'], ticks)
            if len(ticks) == 1 and eligible:
                value = (Fraction(ticks[0], resolution), point['offset'][0])
                node = point['node']
                if node in fixed:
                    beat, offset = fixed[node]
                    shift = offset - value[1]
                    scale = _intersection(scale, shift, shift, value[0] - beat)
                    if scale is None:
                        return {'status': 'conflict', 'endpoint_id': point['id']}
                else:
                    fixed[node] = value
        for node, (beat, offset) in fixed.items():
            scale = _fit(scale, beat, offset, fixed, _relations(model.graph, node, fixed))
            if scale is None:
                return {'status': 'conflict', 'coordinate': node}
        new_pin = False
        for point, original, eligible in candidates:
            if len(original) == 1:
                continue
            node, offset = point['node'], point['offset'][0]
            relations = _relations(model.graph, node, fixed)
            supported = []
            for tick in choices.get(point['id'], original):
                beat = Fraction(tick, resolution)
                candidate_scale = scale
                if node in fixed:
                    other_beat, other_offset = fixed[node]
                    candidate_scale = _intersection(scale, other_offset-offset, other_offset-offset, beat-other_beat)
                    if candidate_scale is None:
                        continue
                if _fit(candidate_scale, beat, offset, fixed, relations) is not None:
                    supported.append(tick)
            if not supported and eligible:
                return {'status': 'conflict', 'endpoint_id': point['id']}
            choices[point['id']] = supported
            new_pin |= len(supported) == 1 and eligible and node not in fixed
        if not new_pin:
            break
    for point, original, eligible in candidates:
        if point['id'] in choices:
            point['result']['draft_start_ticks'] = choices[point['id']]
    return {'status': 'conditional', 'fixed_coordinates': len(fixed),
            'scale_px_per_beat': [ratio(v) for v in scale]}
