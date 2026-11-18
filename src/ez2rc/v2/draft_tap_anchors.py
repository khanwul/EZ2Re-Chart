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
            if (frame_component is None or frame_component == components[node]
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
    if not requests:
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
