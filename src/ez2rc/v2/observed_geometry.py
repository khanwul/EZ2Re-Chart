"""Shared frame/landmark geometry reconstructed from stored observations."""
from collections import Counter
from fractions import Fraction

from .contracts import exact_ms, read_ratio
from .decode import KINDS, lanes
from .difference_constraints import DifferenceGraph
from .motion import STATUS, timeline
from .observation import template_shapes
from .projection_edges import measure_edge, measure_registration, reference_interval, validate_registration
from .spatial_constraints import ConstantSpacingMap

VERSION = 'observed-spatial-constraints-6'
LOCALIZATION_SCOPE = 'selected_endpoint_kind'
LEGACY_LOCALIZATION_SCOPE = 'retained_template_responses'


def search_censored(seed, y, roi, height, radius):
    """A maximum on the finite search edge has no two-sided location bound.

    This is distinct from viewport clipping: a completely visible patch can
    still have its actual maximum outside the searched rows.
    """
    available = roi[3] - roi[1] - height
    lower = max(0, min(seed - radius, available))
    upper = min(available, seed + radius)
    return y <= roi[1] + lower or y >= roi[1] + upper


class _Union:
    def __init__(self):
        self.parents = {}

    def find(self, value):
        self.parents.setdefault(value, value)
        root = value
        while self.parents[root] != root:
            root = self.parents[root]
        while self.parents[value] != value:
            value, self.parents[value] = self.parents[value], root
        return root

    def join(self, left, right):
        left, right = self.find(left), self.find(right)
        if left != right:
            self.parents[max(left, right)] = min(left, right)


def _raw(frame, lane, kind, seed):
    return f'f{frame}:l{lane}:k{kind}:y{seed}'


def observed_geometry(observations, movement, decoded, anchors, anchor_observations, error_px,
                      optical_calibrations=(), *, localization_scope=LOCALIZATION_SCOPE,
                      reviewed_localizations=()):
    """Condition on retained links and selected physical endpoint hypotheses.

    A raw template response receives a location bound only under a selected
    endpoint hypothesis of the same kind. Correlation alone does not identify
    the physical feature to which a template's position error applies. Other
    responses remain in the observations and decode alternatives.

    Shared frames and object identities are variables, not independent error
    terms. No forecast displacement is inserted across a missing link. The
    selected identities and the declared error bound remain assumptions;
    restricting their scope does not verify either one.
    """
    error = exact_ms(error_px)
    if error < 0:
        raise ValueError('localization bound must be nonnegative')
    if localization_scope not in (LOCALIZATION_SCOPE, LEGACY_LOCALIZATION_SCOPE):
        raise ValueError('unsupported raw localization scope')
    clock = timeline(movement)
    states = dict(zip(map(int, clock['frame_id']), map(int, clock['status'])))
    parents = dict(zip(map(int, clock['frame_id']), map(int, clock['reference_frame'])))
    scroll = {int(f): exact_ms(float(q)) for f, q in zip(clock['frame_id'], clock['scroll_px'])}
    profile = observations.manifest['metadata']['profile']
    parameters = observations.manifest['key']['parameters']
    shapes = template_shapes(parameters)
    if shapes is None:
        raise ValueError('observed geometry requires stored template shapes to bound search localization')
    radius = parameters['algorithm']['search_radius_px']
    judge = exact_ms(profile['judge_y'])
    threshold = movement.manifest['key']['parameters']['min_similarity']
    union, linked_pairs = _Union(), 0
    for chunk in movement.chunks():
        for row in chunk['links']:
            frame, parent, lane, kind, previous_seed, seed = map(int, row[:6])
            if states[frame] == STATUS['observed'] and parents[frame] == parent:
                union.join(_raw(parent,lane,kind,previous_seed), _raw(frame,lane,kind,seed))
                linked_pairs += 1
    endpoints, endpoint_lanes = {}, {}
    for lane in lanes(decoded):
        eps = {e['id']: e for e in lane['endpoints']}
        for candidate in lane['candidates']:
            if candidate['selected']:
                for eid in (candidate['start'], candidate['end']):
                    if eid is not None:
                        endpoints[eid] = eps[eid]
                        endpoint_lanes[eid] = lane['lane']
    members = {}
    for eid, endpoint in endpoints.items():
        kind = KINDS.index(endpoint['kind'])
        values = []
        for identity in endpoint['observation_ids']:
            frame, lane, seed = [int(v[1:]) for v in identity.split(':')]
            key = _raw(frame,lane,kind,seed)
            union.find(key); values.append(key)
        members[eid] = values
    reviewed = {}
    for row in reviewed_localizations:
        eid = row['endpoint_id']
        if eid not in members:
            raise ValueError('reviewed localization does not bind a selected endpoint')
        frame,lane,seed = [int(v[1:]) for v in row['observation_id'].split(':')]
        key = _raw(frame,lane,KINDS.index(endpoints[eid]['kind']),seed)
        if key not in members[eid] or key in reviewed:
            raise ValueError('reviewed localization witness mismatch or duplicate')
        reviewed[key] = row
    if reviewed and localization_scope != LOCALIZATION_SCOPE:
        raise ValueError('reviewed localizations require selected endpoint scope')
    optical = {e['endpoint_id']: e for e in optical_calibrations if 'optical_reference' in e}
    registrations = {e['endpoint_id']: e for e in optical_calibrations if 'template_registration' in e}
    requests = {'optical_reference':optical, 'template_registration':registrations}
    requested_raw = {}
    measurements = {kind:{eid:{} for eid in definitions} for kind,definitions in requests.items()}
    for kind, definitions in requests.items():
        for eid, definition in definitions.items():
            if eid not in members:
                raise ValueError('optical calibration does not bind a selected endpoint')
            if kind == 'template_registration':
                spec = definition[kind]
                validate_registration(spec)
                lane, endpoint_kind = endpoint_lanes[eid], endpoints[eid]['kind']
                if parameters.get('asset_sha256',{}).get(f'lane{lane}.{endpoint_kind}') != spec['template_sha256']:
                    raise ValueError('registration template differs from the observation template')
                height = shapes[lane][endpoint_kind][0]
                if any(not -1 <= r['row_interval_px'][0] < r['row_interval_px'][1] <= height
                       for r in spec['source_crossings']):
                    raise ValueError('source crossing lies outside the template boundary')
            for key in members[eid]:
                requested_raw.setdefault(key, []).append((kind,eid))
    needed, positions, excluded = set(union.parents), {}, Counter()
    selected_raw = {key for values in members.values() for key in values}
    for chunk in observations.chunks():
        frame_rows = {int(f): i for i, f in enumerate(chunk['frame_id'])}
        for index, (frame,lane,seed) in enumerate(chunk['candidate_index']):
            for kind in range(len(KINDS)):
                key = _raw(int(frame),int(lane),kind,int(seed))
                if key not in needed:
                    continue
                if localization_scope == LOCALIZATION_SCOPE and key not in selected_raw:
                    excluded['outside_selected_endpoint_kind'] += 1
                    continue
                flag, score = int(chunk['flags'][index,kind]), float(chunk['scores'][index,kind])
                if flag & 1 or flag & 2 and key not in reviewed or score < threshold:
                    excluded['clipped_effect_or_low_score'] += 1
                    continue
                y = exact_ms(float(chunk['positions'][index,kind,1]))
                if search_censored(int(seed), y, profile['lanes'][int(lane)]['roi'],
                                   shapes[int(lane)][KINDS[kind]][0], radius):
                    excluded['finite_search_boundary'] += 1
                    continue
                positions[key] = int(frame), y
                if key in reviewed:
                    lower,upper = map(exact_ms,reviewed[key]['template_top_interval_px'])
                    if not lower <= y-error <= y+error <= upper:
                        raise ValueError('reviewed interval understates the declared localization bound')
                for method, eid in requested_raw.get(key, ()):
                    previous = measurements[method][eid].get(int(frame))
                    if previous is not None and previous['score'] >= score:
                        continue
                    spec = requests[method][eid][method]
                    if y.denominator != 1:
                        raise ValueError('projection edges require integer matched image rows')
                    measure = measure_edge if method == 'optical_reference' else measure_registration
                    measurement = measure(chunk['projection'][frame_rows[int(frame)], int(lane)],
                                          int(y), profile['lanes'][int(lane)]['roi'][1], spec)
                    measurements[method][eid][int(frame)] = {
                        'frame_id': int(frame), 'observation_id': f'f{frame}:l{lane}:y{seed}',
                        'score': score, **measurement}
    if reviewed.keys() - positions.keys():
        raise ValueError('reviewed localization remains unusable under the observation contract')
    endpoint_roots = {}
    for eid, values in members.items():
        usable = [v for v in values if v in positions]
        if not usable:
            continue
        for value in usable[1:]:
            union.join(usable[0], value)
        endpoint_roots[eid] = usable[0]
    graph, nominal = DifferenceGraph(), {}
    for key, (frame,y) in positions.items():
        at, point = 'frame:'+str(frame), 'landmark:'+union.find(key)
        lower,upper = (tuple(map(exact_ms,reviewed[key]['template_top_interval_px']))
                       if key in reviewed else (y-error,y+error))
        graph.bound(at,point,judge-upper,judge-lower,key)
        nominal[at] = scroll[frame]
        nominal.setdefault(point,scroll[frame]+judge-y)
    for i, frame in enumerate(clock['frame_id']):
        if int(clock['status'][i]) == STATUS['repeated'] and clock['repeat_confirmed'][i]:
            graph.bound('frame:'+str(int(clock['repeat_group'][i])), 'frame:'+str(int(frame)),
                        0, 0, 'confirmed-repeat:'+str(int(frame)))
    numbered = {}
    for identity, beat in anchors.items():
        node = 'anchor:'+identity
        numbered[node] = beat
        for i, row in enumerate(anchor_observations[identity]):
            frame, y = row['frame_id'], exact_ms(row['y_px'])
            if frame not in scroll:
                raise ValueError('spatial anchor refers to an unobserved frame')
            graph.bound('frame:'+str(frame),node,judge-y-error,judge-y+error,
                        f'anchor:{identity}:observation:{i}')
            nominal.setdefault(node,scroll[frame]+judge-y)
    destinations = {eid:'landmark:'+union.find(value) for eid,value in endpoint_roots.items()}
    registration_diagnostics = {}
    for eid, rows in measurements['template_registration'].items():
        available = [r for r in rows.values() if r['status'] != 'unavailable']
        registration_diagnostics[eid] = {'status':'measured' if available else 'unavailable',
            'definition':registrations[eid]['template_registration'],
            'observations':sorted(rows.values(), key=lambda r:r['frame_id']),
            'measured_frames':len(available), 'verified':False}
        for row in available:
            for pair in row['paired_top_intervals']:
                lower,upper = map(read_ratio, pair['top_interval_px'])
                graph.bound('frame:'+str(row['frame_id']), destinations[eid], judge-upper, judge-lower,
                            f'template-registration:{row["observation_id"]}:threshold:{pair["threshold"]}')
    optical_diagnostics = {}
    for eid, rows in measurements['optical_reference'].items():
        spec = optical[eid]['optical_reference']
        available = [r for r in rows.values() if r['status'] == 'measured']
        optical_diagnostics[eid] = {'status': 'measured' if available else 'unavailable',
            'definition': spec, 'observations': sorted(rows.values(), key=lambda r:r['frame_id']),
            'measured_frames': len(available), 'verified': False}
        if not available:
            continue
        node, raw = 'optical:'+eid, destinations[eid]
        for row in available:
            lower, upper = reference_interval(row, spec)
            graph.bound('frame:'+str(row['frame_id']), node, judge-upper, judge-lower,
                        'optical:'+row['observation_id'])
        lower, upper = [Fraction(*v) if isinstance(v, list) else exact_ms(v)
                        for v in optical[eid]['offset_px']]
        graph.bound(node, raw, lower, upper, 'optical-reference-offset:'+eid)
        nominal[node] = nominal[raw] - (lower+upper)/2
        destinations[eid] = node
        optical_diagnostics[eid]['node'] = node
    graph.prepare(nominal)
    model = ConstantSpacingMap(graph,numbered,set(destinations.values()))
    diagnostics = {'version': VERSION, 'observation_cache_id': observations.id, 'motion_cache_id': movement.id,
                   'decode_cache_id': decoded.id, 'status': model.status,
                   'nodes': len(graph.edges), 'edges': sum(map(len,graph.edges.values())),
                   'raw_observations': len(positions), 'linked_pairs': linked_pairs,
                   'localization_scope': localization_scope,
                   'excluded_observations': dict(excluded),
                   'unbound_endpoint_ids': sorted(set(endpoints)-destinations.keys()),
                   'assumptions': ['retained_landmark_associations', 'selected_endpoint_identity',
                                   'declared_raw_localization_bound', 'numbered_barline_identity',
                                   'constant_positive_spatial_scale'],
                   'verified': False}
    if reviewed:
        diagnostics['reviewed_localizations'] = list(reviewed.values())
        diagnostics['reviewed_localization_semantics'] = (
            'Explicit visibility and template-top intervals bound to raw witnesses; '
            'no narrowing of the declared localization error and no calibration promotion.')
    if optical:
        diagnostics['optical_endpoints'] = optical_diagnostics
        diagnostics['assumptions'].extend(['declared_optical_landmark_relation',
                                          'declared_optical_measurement_error'])
    if registrations:
        diagnostics['registered_endpoints'] = registration_diagnostics
        diagnostics['assumptions'].extend(['declared_template_photometric_registration',
                                          'declared_registration_error'])
    return model, destinations, diagnostics
