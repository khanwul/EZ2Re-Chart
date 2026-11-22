"""Check an isolated tap against numbered lines through co-visible tap edges."""
from collections import defaultdict
from fractions import Fraction
from itertools import groupby

from .barline_grid import _unbranched_rows, independent_assignments
from .contracts import exact_ms, ratio, read_ratio
from .difference_constraints import ConstraintConflict, DifferenceGraph
from .draft_shape import isolated_bar
from .endpoint_ranges import closed
from .rhythm import DRAFT_GRID, grid_candidate_ticks
from .spatial_constraints import ConstantSpacingMap


def _segments(rows):
    """Keep a bridge on the same continuously observed, moving tap."""
    segment = {}
    for frame, ys in sorted(rows.items()):
        if len(ys) != 1:
            if segment:
                yield segment
            segment = {}
            continue
        y, = ys
        if segment and (frame-1 not in segment or y <= segment[frame-1]):
            yield segment
            segment = {}
        segment[frame] = y
    if segment:
        yield segment


def _longest(frames):
    return max(([f for _, f in group] for _, group in
                groupby(enumerate(sorted(frames)), key=lambda item: item[1]-item[0])), key=len, default=[])


def neighbor_position(target, neighbors, markers, labels, scale, error):
    """Test one target without assigning a musical position to any neighbor."""
    graph = DifferenceGraph()
    anchors = {}
    bridges = []

    def tap_rows(name, rows):
        for f, y in rows.items():
            graph.bound(str(f), name, -y-error, -y+error, 'tap bottom')
    try:
        tap_rows('target', target)
        frames = set(target)
        for cid, segments in neighbors.items():
            for segment in segments:
                shared = target.keys() & segment.keys()
                if not shared:
                    continue
                by_marker = defaultdict(list)
                for f in segment:
                    for marker in markers.get(f, {}):
                        by_marker[marker].append(f)
                tap_rows(f'neighbor:{cid}:{min(segment)}', segment)
                frames.update(segment)
                common = _longest(shared)
                for marker, seen in by_marker.items():
                    support = _longest(seen)
                    if len(common) >= 8 and len(support) >= 8:
                        bridges.append({'neighbor_id': cid, 'marker_id': marker,
                                        'shared_frames': common, 'numbered_frames': support})
        if not bridges:
            return None
        for f in frames:
            for marker, (low, high) in markers.get(f, {}).items():
                node = 'marker:' + marker
                graph.bound(str(f), node, -high, -low, 'numbered line')
                anchors[node] = labels[marker]
        model = ConstantSpacingMap(graph, anchors, ['target'], scale_bounds=scale)
    except ConstraintConflict:
        return None
    bounds = model.project('target')
    if bounds is None:
        return None
    ticks = grid_candidate_ticks(closed(*bounds), DRAFT_GRID)
    values = [Fraction(t, DRAFT_GRID.ticks_per_beat) for t in ticks]
    feasible = [v for v in values if model.scales_for_beat('target', v)]
    if len(feasible) == 1:
        beat, = feasible
        return beat, {'bridges': bridges, 'beat_range': list(map(ratio, bounds)),
                      'scale_px_per_beat': list(map(ratio, model.scales_for_beat('target', beat)))}
    return None


def confirm_neighbors(notes, targets, neighbors, observations, shapes, barlines, scale, error):
    remaining = {n['id']: n for n in notes if n['issues'] == ['pixel_ambiguous']
                 and n['type'] == 'tap' and n['start_beat'] is not None}
    targets = {k: cid for k, cid in targets.items() if cid in remaining}
    if not targets or not scale or len(barlines['hypotheses']) != 1:
        return
    target_frames = {f for f, _, _ in targets}
    near = {cid for (f, _, _), cid in neighbors.items() if f in target_frames}
    wanted = {k: cid for k, cid in neighbors.items() if cid in near}
    wanted.update(targets)
    frames = {f for f, _, _ in wanted}
    profile = observations.manifest['metadata']['profile']
    threshold = observations.manifest['key']['parameters']['algorithm']['brightness_threshold']
    points = defaultdict(lambda: defaultdict(set))
    for chunk in observations.chunks():
        local = {int(f): j for j, f in enumerate(chunk['frame_id']) if f in frames}
        if not local:
            continue
        for j, (f, lane, seed) in enumerate(chunk['candidate_index']):
            cid = wanted.get((f, lane, seed))
            if cid is None or chunk['flags'][j, 0] & 3 or chunk['scores'][j, 0] < .7:
                continue
            height = shapes[lane]['tap'][0]
            y = int(chunk['positions'][j, 0, 1])
            if isolated_bar(chunk['projection'][local[f], lane], y-profile['lanes'][lane]['roi'][1], height, threshold):
                points[cid][int(f)].add(y+height-1)
    markers = defaultdict(dict)
    labels = {a['marker_id']: read_ratio(a['beat']) for a in independent_assignments(barlines['hypotheses'][0])}
    for track in barlines['tracks']:
        if track['id'] in labels:
            for row in _unbranched_rows(track):
                if row['frame_id'] in frames:
                    markers[row['frame_id']][track['id']] = tuple(map(float, row['visible_range_px']))
    lanes = {cid: lane for (_, lane, _), cid in wanted.items()}
    segments = {cid: list(_segments(points[cid])) for cid in near}
    for cid in set(targets.values()):
        note = remaining[cid]
        if read_ratio(note['evidence']['start']['offset_px']) != shapes[note['lane']]['tap'][0]-1:
            continue
        runs = list(_segments(points[cid]))
        if len(runs) != 1 or len(runs[0]) != len(points[cid]):
            continue
        target = runs[0]
        # Two IDs on the same lane must denote distinct visible bars. A
        # second template response on the target cannot corroborate it.
        lane = note['lane']
        height = shapes[lane]['tap'][0]
        other_tracks = {other: [s for s in tracks if lanes[other] != lane or
                                all(abs(s[f]-target[f]) >= height for f in s.keys() & target.keys())]
                        for other, tracks in segments.items() if other != cid}
        result = neighbor_position(target, other_tracks, markers, labels, tuple(map(read_ratio, scale)), exact_ms(error))
        if result and result[0] == read_ratio(note['start_beat']):
            note['evidence']['neighbor_tap'] = result[1]
            note.update(status='draft', issues=[])
