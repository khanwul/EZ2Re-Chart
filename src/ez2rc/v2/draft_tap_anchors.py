"""Conditional draft coordinates from taps that remain visible near judgment."""
from collections import defaultdict

import numpy as np

from .contracts import exact_ms
from .difference_constraints import ConstraintConflict, DifferenceGraph
from .spatial_constraints import ConstantSpacingMap


def _tap_profile(rows, y, height, threshold):
    """A full bright cap with a dark row on each side of the matched patch."""
    if not 0 < y < len(rows) - height:
        return None
    patch = rows[y:y + height]
    if rows[y - 1] >= threshold or rows[y + height] >= threshold or np.any(patch < threshold):
        return None
    # Compare the pattern *inside* the cap. A flat bright effect has no
    # internal structure even if its two exterior edges match a tap.
    centered = patch.astype(float) - patch.mean(dtype=float)
    norm = np.linalg.norm(centered)
    return centered / norm if norm else None


def _components(graph):
    components = {}
    for start in graph.edges:
        if start in components:
            continue
        components[start] = start
        stack = [start]
        while stack:
            for neighbor in graph.edges[stack.pop()]:
                if neighbor not in components:
                    components[neighbor] = start
                    stack.append(neighbor)
    return components


def _entry_bottom(rows, full, top, height, threshold, similarity):
    """Locate a moving cap's visible suffix against the viewport's top edge."""
    dark = np.flatnonzero(rows < threshold)
    if not len(dark):
        return None
    count = int(dark[0])
    if not max(4, (height + 2) // 3) <= count < height:
        return None
    if _tap_profile(full, top, height, threshold) is None:
        return None
    suffix = full[top + height - count:top + height].astype(float)
    prefix = rows[:count].astype(float)
    suffix -= suffix.mean()
    prefix -= prefix.mean()
    norm = np.linalg.norm(suffix) * np.linalg.norm(prefix)
    if not norm or prefix @ suffix < similarity * norm:
        return None
    return count - 1


def _entry_anchors(model, destinations, endpoints, observations, shapes, components,
                   anchored, similarity):
    profile = observations.manifest['metadata']['profile']
    threshold = observations.manifest['key']['parameters']['algorithm']['brightness_threshold']
    requests = defaultdict(list)
    for eid, node in destinations.items():
        if components[node] in anchored or endpoints[eid]['kind'] != 'tap':
            continue
        frames = {int(oid.split(':', 1)[0][1:]) for oid in endpoints[eid]['observation_ids']}
        lane = int(endpoints[eid]['observation_ids'][0].split(':')[1][1:])
        height = shapes[lane]['tap'][0]
        # Use two consecutive localized full patches of the selected track.
        observed = model.graph.edges[node]
        usable = sorted(f for f in frames if f'frame:{f}' in observed)
        if len(usable) < 2 or usable[1] != usable[0] + 1:
            continue
        first, second = usable[:2]
        if components.get(f'frame:{first - 1}') == components[node]:
            continue
        roi_top = profile['lanes'][lane]['roi'][1]
        def top_interval(frame):
            at = f'frame:{frame}'
            return (profile['judge_y'] - model.graph.edges[at][node][0] - roi_top,
                    profile['judge_y'] + observed[at][0] - roi_top)
        lo, hi = top_interval(first)
        top = (lo + hi) / 2
        if not float(top).is_integer() or top <= 0:
            continue
        later_lo, later_hi = top_interval(second)
        # The measured entry must agree with the next observed displacement.
        # This is an identity check. Only the visible bottom supplies a bound.
        requests[first].append((eid, lane, int(top), height, later_lo - hi, later_hi - lo))
    evidence = []
    if requests:
        previous = None
        for chunk in observations.chunks():
            for i, fid in enumerate(chunk['frame_id']):
                frame = int(fid)
                if frame in requests and previous is not None and previous[0] == frame - 1:
                    for eid, lane, top, height, lo, hi in requests[frame]:
                        bottom = _entry_bottom(previous[1][lane], chunk['projection'][i, lane],
                                               top, height, threshold, similarity)
                        if bottom is None or not lo <= top + height - 1 - bottom <= hi:
                            continue
                        y = profile['lanes'][lane]['roi'][1] + bottom - height + 1
                        evidence.append({'frame_id': frame - 1, 'endpoint_id': eid, 'tap_y_px': y,
                                         'observation_id': f'entry-bottom:f{frame-1}:l{lane}',
                                         'visible_bottom_px': y + height - 1,
                                         'full_frames': [frame, frame + 1]})
                previous = (frame, chunk['projection'][i].copy()) if frame + 1 in requests else None
    return evidence


def extend_draft_tap_anchors(model, destinations, endpoints, observations, shapes,
                             min_similarity, error):
    """Bridge disconnected coordinates only with a measured frame-to-cap relation."""
    from .observed_geometry import search_censored

    if all(model.constraints.values()):
        return model, []
    components = _components(model.graph)
    anchored = {components[node] for node in model.anchors}
    profile = observations.manifest['metadata']['profile']
    params = observations.manifest['key']['parameters']['algorithm']
    entries = _entry_anchors(model, destinations, endpoints, observations, shapes, components,
                             anchored, min_similarity)
    entry_nodes = defaultdict(set)
    for row in entries:
        entry_nodes[row['frame_id']].add(components[destinations[row['endpoint_id']]])
    requests = defaultdict(dict)
    targets = set()
    for eid, node in destinations.items():
        endpoint = endpoints[eid]
        if endpoint['kind'] != 'tap':
            continue
        rows = defaultdict(list)
        for oid in endpoint['observation_ids']:
            frame, lane, seed = (int(v[1:]) for v in oid.split(':'))
            rows[frame].append((lane, seed))
        for frame, sources in rows.items():
            frame_component = components.get(f'frame:{frame}')
            if ((frame_component is None and not entry_nodes.get(frame, set()) - {components[node]})
                    or frame_component == components[node]
                    or frame_component in anchored and components[node] in anchored
                    or frame - 1 not in rows):
                continue
            for lane, seed in sources:
                # Seeds outside this band cannot produce an effect-zone patch.
                roi = profile['lanes'][lane]['roi']
                if roi[1] + seed + params['search_radius_px'] + shapes[lane]['tap'][0] < profile['effect_top']:
                    continue
                requests[frame][lane, seed] = eid
                targets.add((frame, lane, seed))
                for source in rows[frame - 1]:
                    requests[frame - 1][source] = eid
    if not requests and not entries:
        return model, []

    clean_profiles, evidence = defaultdict(dict), []
    for chunk in observations.chunks():
        indices = chunk['candidate_index']
        for j, fid in enumerate(chunk['frame_id']):
            frame = int(fid)
            if frame not in requests:
                continue
            for i in np.flatnonzero(indices[:, 0] == frame):
                lane, seed = map(int, indices[i, 1:])
                eid = requests[frame].get((lane, seed))
                if eid is None:
                    continue
                flags = int(chunk['flags'][i, 0])
                measured_y = float(chunk['positions'][i, 0, 1])
                if not measured_y.is_integer():
                    continue
                y = int(measured_y)
                roi = profile['lanes'][lane]['roi']
                height = shapes[lane]['tap'][0]
                if (flags & 1 or chunk['scores'][i, 0] < min_similarity
                        or y + height >= profile['judge_y']
                        or search_censored(seed, y, roi, height, params['search_radius_px'])):
                    continue
                patch = _tap_profile(chunk['projection'][j, lane], y - roi[1], height,
                                     params['brightness_threshold'])
                if patch is None:
                    continue
                oid = f'f{frame}:l{lane}:y{seed}'
                if not flags & 2:
                    clean_profiles[eid, frame][y] = patch, oid
                    continue
                if (frame, lane, seed) not in targets:
                    continue
                previous = clean_profiles.get((eid, frame - 1), {})
                if len(previous) != 1:
                    continue
                ((previous_y, (previous_patch, previous_id)),) = previous.items()
                if previous_y >= y or previous_patch @ patch < min_similarity:
                    continue
                evidence.append({'frame_id': frame, 'endpoint_id': eid, 'tap_y_px': y,
                                 'observation_id': oid, 'preceding_observation_id': previous_id})
    # A new frame has no coordinate until at least two different components
    # observe it. Do not keep a dangling entry (or an unpaired effect patch).
    peers = defaultdict(set)
    for row in [*entries, *evidence]:
        peers[row['frame_id']].add(components[destinations[row['endpoint_id']]])
    evidence = [r for r in [*evidence, *entries] if f"frame:{r['frame_id']}" in components
                or len(peers[r['frame_id']]) > 1]
    if not evidence:
        return model, []
    # Test all admitted links together. Keep the original graph intact for
    # cases where independent paths disagree on displacement or line spacing.
    graph = DifferenceGraph()
    graph.edges = {node: dict(edges) for node, edges in model.graph.edges.items()}
    try:
        for row in evidence:
            delta = exact_ms(profile['judge_y']) - row['tap_y_px']
            graph.bound(f"frame:{row['frame_id']}", destinations[row['endpoint_id']],
                        delta - error, delta + error, row['observation_id'])
        graph.prepare(model.graph.potentials)
        extended = ConstantSpacingMap(graph, model.anchors, model.destinations)
        if extended.status == 'conditional':
            return extended, evidence
        reason = extended.status
    except ConstraintConflict:
        reason = 'inconsistent_coordinate_constraints'
    for row in evidence:
        row['rejected'] = reason
    return model, evidence
