"""Corroborate an isolated draft tap directly against numbered screen lines."""
from collections import defaultdict
from itertools import groupby, product

from .barline_grid import _unbranched_rows
from .contracts import exact_ms, read_ratio
from .endpoint_ranges import closed
from .rhythm import DRAFT_GRID, grid_candidate_ticks


def tap_requests(candidate, event, endpoint):
    if (candidate['type'] != 'tap' or not candidate['selected'] or event['status'] != 'ambiguous'
            or event['candidate_ids'] != [candidate['id']]
            or 'motion_alternatives' not in event['review_causes']
            or set(event['review_causes']) - {'motion_alternatives', 'search_window_boundary'}):
        return ()
    return set(endpoint['observation_ids']) - set(endpoint['search_boundary_observation_ids'])


def isolated_bar(field, top, height, threshold):
    bottom = top + height
    return (height >= 6 and top >= 4 and bottom + 4 <= len(field)
            and max(field[top-4:top-1]) < threshold and max(field[bottom+1:bottom+4]) < threshold
            and min(field[top+1:bottom-1]) >= max(threshold, 128))


def confirm_taps(notes, requests, observations, shapes, barlines, scale, error):
    targets = {n['id']: n for n in notes if n['issues'] == ['pixel_ambiguous']
               and n['start_beat'] is not None and n['type'] == 'tap'}
    wanted = {key: cid for key, cid in requests.items() if cid in targets}
    if not wanted or not scale or len(barlines['hypotheses']) != 1:
        return
    frames = {f for f, _, _ in wanted}
    labels = {a['marker_id']: read_ratio(a['beat']) for a in barlines['hypotheses'][0]['assignments']}
    markers = defaultdict(list)
    for t in barlines['tracks']:
        if t['id'] in labels:
            for row in _unbranched_rows(t):
                if row['frame_id'] in frames:
                    markers[row['frame_id']].append((t['id'], row))
    if not markers:
        return
    scale = tuple(map(read_ratio, scale))
    profile = observations.manifest['metadata']['profile']
    threshold = observations.manifest['key']['parameters']['algorithm']['brightness_threshold']
    evidence = defaultdict(dict)
    conflicts = set()
    for chunk in observations.chunks():
        local = {int(f): j for j, f in enumerate(chunk['frame_id']) if f in markers}
        if not local:
            continue
        for j, (frame, lane, seed) in enumerate(chunk['candidate_index']):
            cid = wanted.get((frame, lane, seed))
            if cid is None or frame not in local or chunk['flags'][j, 0] & 3 or chunk['scores'][j, 0] < .7:
                continue
            height = shapes[lane]['tap'][0]
            offset = targets[cid]['evidence']['start']['offset_px']
            if read_ratio(offset) != height - 1:
                continue
            top = int(chunk['positions'][j, 0, 1])
            field = chunk['projection'][local[frame], lane]
            if not isolated_bar(field, top-profile['lanes'][lane]['roi'][1], height, threshold):
                continue
            edge = top + height - 1
            tick = read_ratio(targets[cid]['start_beat']) * DRAFT_GRID.ticks_per_beat
            for marker, row in markers[frame]:
                lo, hi = map(exact_ms, row['visible_range_px'])
                shifts = [(y-edge+d)/s for (y, d), s in product(((lo, -error), (hi, error)), scale)]
                domain = closed(labels[marker]+min(shifts), labels[marker]+max(shifts))
                ticks = grid_candidate_ticks(domain, DRAFT_GRID)
                if tick not in ticks:
                    conflicts.add(cid)
                elif len(ticks) == 1:
                    evidence[cid, marker][int(frame)] = edge
    for (cid, marker), rows in evidence.items():
        if cid in conflicts or not targets[cid]['issues']:
            continue
        for _, group in groupby(enumerate(sorted(rows)), key=lambda item: item[1]-item[0]):
            run = [f for _, f in group]
            if len(run) >= 8 and all(rows[a] < rows[b] for a, b in zip(run, run[1:])):
                note = targets[cid]
                note['evidence']['tap_shape'] = {'marker_id': marker, 'frames': run,
                                                 'bottom_rows': [rows[f] for f in run]}
                note.update(status='draft', issues=[])
                break
