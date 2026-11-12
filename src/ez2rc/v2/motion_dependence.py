"""Coordinate-error lineage conditional on observed landmark associations.

Forecasts never supply a measured displacement bound. Each missing relation
has its own unknown additive source. Later observations can share that source
without measuring it, so local relative coordinates can still be supported.
"""
from collections import defaultdict
from fractions import Fraction
import math

from .config import number
from .contracts import exact_ms, read_ratio
from .motion import STATUS, timeline

VERSION = 'landmark-coordinate-dependence-2'


def outward(value, lower):
    """Serialize an exact bound without rounding the enclosure inward."""
    result = float(value)
    converted = exact_ms(result)
    if converted > value if lower else converted < value:
        result = math.nextafter(result, -math.inf if lower else math.inf)
    return result


class MotionDependence:
    def __init__(self, movement, observation_error_px):
        if observation_error_px is not None:
            number(observation_error_px, 'motion observation error', 0)
        self.error = exact_ms(observation_error_px) if observation_error_px is not None else None
        clock = timeline(movement)
        self.rows = {int(fid): i for i, fid in enumerate(clock['frame_id'])}
        links = defaultdict(list)
        for chunk in movement.chunks():
            for row in chunk['links']:
                links[int(row[0]), int(row[1])].append(row)
        self.parents, self.sources, self.issues, self.edge_terms = {}, {}, {}, {}
        self.motion_cache_id = movement.id
        previous = None
        for fid, i in self.rows.items():
            state = int(clock['status'][i])
            ref = int(clock['reference_frame'][i])
            domain, reason = None, 'unobserved displacement'
            self.edge_terms[fid] = {f'motion:{fid}': Fraction(1)}
            if previous is None:
                parent = None
                reason = 'arbitrary common coordinate origin'
            elif state == STATUS['repeated'] and bool(clock['repeat_confirmed'][i]):
                parent = int(clock['repeat_group'][i])
                if parent not in self.parents:
                    raise ValueError('repeat dependence has no preceding origin')
                domain = (Fraction(0), Fraction(0))
                reason = 'confirmed displayed-frame equality'
            elif state == STATUS['observed'] and ref in self.parents and links[fid, ref]:
                parent = ref
                rows = links[fid, ref]
                shifts = [exact_ms(float(row[6])) for row in rows]
                if self.error is not None and max(shifts)-min(shifts) > 4*self.error:
                    reason = 'matched landmark displacement bounds conflict'
                    self.issues[fid] = 'landmark_bounds_conflict'
                else:
                    # A deterministic retained match supplies one equality.
                    # It does not acquire a smaller error from the number of
                    # links. Reused raw positions share the SAME error variable.
                    row = min(rows, key=lambda r: (-r[7], *r[2:6]))
                    lane, kind, previous_seed, seed = map(int, row[2:6])
                    a = self.position_source(ref, lane, kind, previous_seed)
                    b = self.position_source(fid, lane, kind, seed)
                    self.edge_terms[fid].update({a: Fraction(1), b: Fraction(-1)})
                    delta = exact_ms(float(row[6]))
                    domain = (delta, delta)
                    reason = 'retained landmark equality with shared raw-position errors'
            else:
                # This identity is valid even if no physical relation was
                # observed: its unrestricted source covers the whole difference.
                parent = previous
                reason = 'uncertain or missing motion relation'
            self.parents[fid] = parent
            if domain is not None:
                nominal = exact_ms(float(clock['scroll_px'][i])) - exact_ms(float(clock['scroll_px'][self.rows[parent]]))
                domain = domain[0] - nominal, domain[1] - nominal
            self.sources[f'motion:{fid}'] = {'id': f'motion:{fid}',
                'range_px': [outward(domain[0], True), outward(domain[1], False)] if domain is not None else None,
                'reason': reason}
            previous = fid

    def position_source(self, fid, lane, kind, seed):
        key = f'position:f{fid}:l{lane}:y{seed}:k{kind}'
        self.sources.setdefault(key, {'id': key,
            'range_px': [float(-self.error), float(self.error)] if self.error is not None else None,
            'reason': 'raw detection-reference localization in q direction; shared whenever this observation is reused'})
        return key

    def terms(self, frame_weights):
        result = defaultdict(Fraction)
        total = Fraction(0)
        for point in frame_weights:
            frame, weight = point['frame_id'], read_ratio(point['weight'])
            if type(frame) is not int or frame not in self.parents or weight <= 0:
                raise ValueError('invalid coordinate observation frame or weight')
            total += weight
            while frame is not None:
                for key, coefficient in self.edge_terms[frame].items():
                    result[key] += coefficient * weight
                frame = self.parents[frame]
        if total != 1:
            raise ValueError('coordinate observation weights must sum to one')
        # Median representatives use halves, which have exact JSON decimals.
        if any(v.denominator not in (1, 2) for v in result.values()):
            raise ValueError('coordinate observation weights must represent a median')
        return {k: float(v) for k, v in sorted(result.items()) if v}

    def model(self, anchor_weights, endpoint_weights):
        anchors = {k: self.terms(v) for k, v in anchor_weights.items()}
        endpoints = {k: self.terms(v) for k, v in endpoint_weights.items()}
        for eid, points in endpoint_weights.items():
            for point in points:
                if 'position_source' in point:
                    key = point['position_source']
                    endpoints[eid][key] = endpoints[eid].get(key, 0.) + float(read_ratio(point['weight']))
            endpoints[eid] = {k:v for k,v in endpoints[eid].items() if v}
        used = {k for terms in (*anchors.values(), *endpoints.values()) for k in terms}
        return {'version': VERSION, 'sources': [self.sources[k] for k in sorted(used)],
                'anchors': anchors, 'endpoints': endpoints}


def derive_coordinate_model(observations, movement, decoded, reference):
    """Bind a declaration to the actual median observations, never note beats."""
    from .decode import KINDS, lanes

    request = reference['motion_dependence']
    dependence = MotionDependence(movement, request['observation_error_px'])
    clock = timeline(movement)
    scroll = {int(f): float(s) for f, s in zip(clock['frame_id'], clock['scroll_px'])}
    profile = observations.manifest['metadata']['profile']
    anchors = {}
    for anchor in reference['anchors']:
        support = request['anchor_observations'][anchor['id']]
        dependence.terms(support)  # Validate frame identities and median weights.
        q = sum((scroll[p['frame_id']] + profile['judge_y'] - p['y_px'])
                * float(read_ratio(p['weight'])) for p in support)
        if q != anchor['q_px']:
            raise ValueError('anchor coordinate differs from its declared motion observations')
        anchors[anchor['id']] = support
    selected = {}
    for lane in lanes(decoded):
        eps = {e['id']: e for e in lane['endpoints']}
        for c in lane['candidates']:
            if c['selected']:
                for eid in (c['start'], c['end']):
                    if eid is not None and eps[eid]['q_px'] is not None:
                        selected[eid] = eps[eid]
    needed = {p['observation_id'] for e in selected.values() for p in e['coordinate_observations']}
    raw = {}
    for chunk in observations.chunks():
        for i, (f, lane, seed) in enumerate(chunk['candidate_index']):
            identity = f'f{f}:l{lane}:y{seed}'
            if identity in needed:
                raw[identity] = (int(f), chunk['positions'][i])
    endpoints, shared = {}, []
    overrides = {e['endpoint_id']: e for e in reference['endpoint_calibrations']}
    for eid, endpoint in selected.items():
        support = [dict(p) for p in endpoint['coordinate_observations']]
        calibration = profile['calibration'][endpoint['kind']]
        offset = calibration['offset_px'] if calibration['status'] == 'verified' else 0.
        error = overrides[eid]['position_error_px'] if eid in overrides else endpoint['localization_error_px']
        same = request.get('endpoint_position_relation') == 'same_raw_localization' and error is not None
        if same and (dependence.error is None or exact_ms(error) != dependence.error):
            raise ValueError('shared endpoint position bound differs from the raw motion position bound')
        for point in support:
            fid, positions = raw[point['observation_id']]
            q = scroll[fid] + profile['judge_y'] - float(positions[KINDS.index(endpoint['kind']), 1]) - offset
            if fid != point['frame_id'] or q != point['q_px']:
                raise ValueError('endpoint coordinate differs from its raw motion observation')
            if same:
                _, lane, seed = point['observation_id'].split(':')
                point['position_source'] = dependence.position_source(fid, int(lane[1:]), KINDS.index(endpoint['kind']), int(seed[1:]))
        if same:
            shared.append(eid)
        endpoints[eid] = support
    model = dependence.model(anchors, endpoints)
    diagnostics = {'motion_cache_id': movement.id, 'observation_error_px': request['observation_error_px'],
                   'landmark_conflicts': sorted(dependence.issues),
                   'shared_endpoint_position_ids': sorted(shared),
                   'semantics': 'Conditional on retained landmark and endpoint associations and the declared raw-position bound. Search tolerances and forecast rates supply no physical error bounds. Different source dependence is unspecified, not independent.'}
    return model, diagnostics
