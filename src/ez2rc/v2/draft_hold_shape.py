"""Trace a hold's two terminal edges and the visible body that joins them."""
from collections import defaultdict
from itertools import groupby

import numpy as np

from .contracts import read_ratio
from .observation import runs


def hold_request(candidate, event, endpoints):
    if (candidate['type'] != 'longnote' or not candidate['selected'] or event['status'] != 'ambiguous'
            or not candidate['start'] or not candidate['end']):
        return None
    edges = {}
    for kind, side in ((1, 'start'), (2, 'end')):
        endpoint = endpoints[candidate[side]]
        for oid in set(endpoint['observation_ids']) - set(endpoint['search_boundary_observation_ids']):
            frame, lane, seed = (int(v[1:]) for v in oid.split(':'))
            edges[frame, lane, seed, kind] = side
    if len(set(edges.values())) != 2:
        return None
    single = (event['candidate_ids'] == [candidate['id']] and 'motion_alternatives' in event['review_causes']
              and not set(event['review_causes']) - {'motion_alternatives', 'search_window_boundary'})
    return edges, single


def _crossing(samples, center):
    crossings = []
    for _, group in groupby(enumerate(sorted(samples)), key=lambda pair: pair[1]-pair[0]):
        frames = [f for _, f in group]
        if len(frames) < 8 or any(len(samples[f]) != 1 for f in frames):
            continue
        ys = [next(iter(samples[f])) for f in frames]
        if any(a >= b for a, b in zip(ys, ys[1:])):
            continue
        for i, (a, b) in enumerate(zip(ys, ys[1:])):
            if a < center <= b:
                crossings.append((frames[i], frames[i+1], frames))
    return crossings[0] if len(crossings) == 1 else None


def joined_body(head, tail, spans, center, speed, slack):
    """Adjacent bright components must overlap at every allowed translation."""
    first, last = _crossing(head, center), _crossing(tail, center)
    if not first or not last or first[1] > last[0]:
        return None
    start, stop = first[1], last[0]
    previous = None
    for frame in range(start, stop + 1):
        if frame not in spans:
            return None
        low, high, time = spans[frame]
        if frame == start and high != next(iter(head[frame])):
            return None
        if frame == stop and low != next(iter(tail[frame])):
            return None
        if previous:
            a, b, t = previous
            if time <= t or low > b or a + speed * (time-t) + slack > high:
                return None
        previous = spans[frame]
    return {'head_frames': first[2], 'tail_frames': last[2], 'body_frames': [start, stop],
            'screen_row': center}


def confirm_holds(notes, requests, observations, shapes, error, motion):
    targets = {n['id']: n for n in notes if 'pixel_ambiguous' in n['issues'] and n['id'] in requests}
    if not targets:
        return
    profile = observations.manifest['metadata']['profile']
    threshold = observations.manifest['key']['parameters']['algorithm']['brightness_threshold']
    wanted, spans, edges = defaultdict(list), defaultdict(dict), defaultdict(lambda: defaultdict(set))
    windows = defaultdict(set)
    for cid, note in targets.items():
        keys, _ = requests[cid]
        for key, side in keys.items():
            wanted[key].append((cid, side))
        frames = [k[0] for k in keys]
        for frame in range(min(frames), max(frames) + 1):
            windows[frame].add(note['lane'])
    for chunk in observations.chunks():
        fields = {}
        for j, frame in enumerate(chunk['frame_id']):
            for lane in windows.get(frame, ()):
                top = profile['lanes'][lane]['roi'][1]
                center = (top + int(profile['effect_top'])) // 2
                field = chunk['projection'][j, lane, :int(profile['effect_top'])-top]
                bands = [(int(a)+top, int(b)-1+top) for a,b in runs(field >= threshold)]
                fields[frame, lane] = field, bands
                for a, b in bands:
                    if a <= center <= b:
                        spans[lane][int(frame)] = a, b, float(chunk['timestamp_ms'][j])
        if not fields:
            continue
        for j, (frame, lane, seed) in enumerate(chunk['candidate_index']):
            if (frame, lane) not in fields:
                continue
            field, bands = fields[frame, lane]
            top = profile['lanes'][lane]['roi'][1]
            for kind, name in ((1, 'head'), (2, 'tail')):
                items = wanted.get((frame, lane, seed, kind))
                if not items or chunk['flags'][j, kind] & 3 or chunk['scores'][j, kind] < .7:
                    continue
                height = shapes[lane][name][0]
                y = int(chunk['positions'][j, kind, 1]) + (height-1 if kind == 1 else 0)
                for a, b in bands:
                    edge = b if kind == 1 else a
                    if b-a < height or abs(edge-y) > error:
                        continue
                    pos = edge-top
                    if pos < 4 or pos + 4 >= len(field):
                        continue
                    inside, outside = ((field[pos-3:pos], field[pos+1:pos+4]) if kind == 1 else
                                       (field[pos+1:pos+4], field[pos-3:pos]))
                    if np.min(inside) >= max(128, threshold) and np.max(outside) < threshold:
                        for cid, side in items:
                            edges[cid, side][int(frame)].add(edge)
    for cid, note in targets.items():
        lane = note['lane']
        center = (profile['lanes'][lane]['roi'][1] + int(profile['effect_top'])) // 2
        proof = joined_body(edges[cid, 'start'], edges[cid, 'end'], spans[lane], center,
                            motion['max_speed_px_per_ms'], motion['displacement_slack_px'])
        if not proof:
            continue
        note['evidence']['hold_shape'] = proof
        if not requests[cid][1]:
            continue
        start, end = note['evidence']['start'], note['evidence']['end']
        if (note['start_beat'] is None or note['end_beat'] is None
                or len(start.get('candidate_ticks', ())) != 1 or not end.get('beat_range')
                or read_ratio(end['beat_range']['lower']) <= read_ratio(note['start_beat'])):
            continue
        note['issues'].remove('pixel_ambiguous')
        if not note['issues']:
            note['status'] = 'draft'
