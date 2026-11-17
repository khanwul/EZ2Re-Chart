"""Conditional musical coordinates from explicitly numbered spatial barlines.

Capture timestamps never define this map. Inputs are bound to visual caches;
neither note labels nor a fitted capture-clock offset enter the conversion.
"""
from bisect import bisect_left
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from .config import number, string, table
from .contracts import exact_ms, ratio, read_ratio
from .coordinate_errors import coordinate, quotient_range
from .decode import lanes
from .endpoint_ranges import EndpointTimeMap, closed
from .motion import timeline
from .motion_dependence import VERSION as DEPENDENCE_VERSION, derive_coordinate_model
from .storage import reject_constant

VERSION = 'spatial-relative-beats-10'


def screen_end_reference(reference, decoded):
    """Remove musical tail extensions before geometry and grid inference.

    Preserve pixel landmark definitions and the original input document. This
    is the draft release convention, not a claim of verified release timing.
    """
    from copy import deepcopy
    result = deepcopy(reference)
    tails = {e['id'] for lane in lanes(decoded) for e in lane['endpoints'] if e['kind'] == 'tail'}
    removed = []
    for calibration in result['endpoint_calibrations']:
        if calibration['endpoint_id'] in tails and 'beat_shift' in calibration:
            shift = calibration.pop('beat_shift')
            removed.append({'endpoint_id': calibration['endpoint_id'], 'beat_shift': shift})
    return result, {'convention': 'screen_end_without_musical_extension', 'removed_tail_shifts': removed}


def load_spatial_calibration(request):
    data = Path(request['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != request['sha256']:
        raise ValueError('spatial calibration checksum mismatch')
    value = json.loads(data, parse_constant=reject_constant)
    keys = {'format','schema_version','version','observation_cache_id','motion_cache_id','decode_cache_id',
            'endpoint_calibrations','localization_error_px'}
    table(value,keys,keys)
    if (value['format'] != 'ez2rc.spatial-calibration' or type(value['schema_version']) is not int
            or value['schema_version'] != 1):
        raise ValueError('unsupported spatial calibration')
    for key in ('version','observation_cache_id','motion_cache_id','decode_cache_id'):
        string(value[key],key)
    number(value['localization_error_px'],'raw localization bound',0)
    if not isinstance(value['endpoint_calibrations'],list):
        raise ValueError('spatial endpoint calibrations must be an array')
    return value


def bounds(values):
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError('expected two ordered pixel bounds')
    result = tuple(read_ratio(v) if isinstance(v, list) else exact_ms(number(v, 'pixel bound')) for v in values)
    if result[0] > result[1]:
        raise ValueError('unordered pixel bounds')
    return result


def unpack(value):
    return read_ratio(value['lower']), read_ratio(value['upper'])


def difference(end, start):
    lo, hi = unpack(end)
    a, b = unpack(start)
    return closed(lo - b, hi - a)


def load_reference(request):
    """Hash and parse the same immutable bytes, including during reconstruct."""
    data = Path(request['path']).read_bytes()
    if hashlib.sha256(data).hexdigest() != request['sha256']:
        raise ValueError('relative beat reference checksum mismatch')
    value = json.loads(data, parse_constant=reject_constant)
    validate_reference(value)
    return value


def validate_reference(value):
    required = {'format', 'schema_version', 'version', 'origin', 'measure_beats',
                'observation_cache_id', 'motion_cache_id', 'decode_cache_id',
                'anchors', 'endpoint_calibrations', 'extrapolation'}
    table(value, required | {'coordinate_uncertainty', 'motion_dependence', 'observed_geometry'}, required)
    if (value['format'] != 'ez2rc.spatial-beat-reference'
            or type(value['schema_version']) is not int or value['schema_version'] != 1):
        raise ValueError('unsupported spatial beat reference')
    for name in ('version', 'observation_cache_id', 'motion_cache_id', 'decode_cache_id'):
        string(value[name], name)
    if value['origin'] != 'first_barline_zero':
        raise ValueError('spatial beats require the first_barline_zero convention')
    if value['extrapolation'] not in ('withhold', 'conditional_linear'):
        raise ValueError('unsupported spatial extrapolation policy')
    measure = read_ratio(value['measure_beats'])
    if measure <= 0:
        raise ValueError('measure length must be positive')
    if not isinstance(value['anchors'], list):
        raise ValueError('anchors must be an array')
    seen = set()
    for anchor in value['anchors']:
        table(anchor, {'id', 'beat', 'q_px', 'q_range_px'}, {'id', 'beat', 'q_px', 'q_range_px'})
        string(anchor['id'], 'anchor ID')
        if anchor['id'] in seen:
            raise ValueError('duplicate barline anchor')
        seen.add(anchor['id'])
        beat = read_ratio(anchor['beat'])
        if beat < 0 or (beat / measure).denominator != 1:
            raise ValueError('barline beat must be a nonnegative measure boundary')
        q = exact_ms(number(anchor['q_px'], 'anchor coordinate'))
        lo, hi = bounds(anchor['q_range_px'])
        if not lo <= q <= hi:
            raise ValueError('anchor nominal coordinate outside its range')
    if not isinstance(value['endpoint_calibrations'], list):
        raise ValueError('endpoint calibrations must be an array')
    seen = set()
    for item in value['endpoint_calibrations']:
        required = {'endpoint_id', 'candidate_id', 'version', 'offset_px', 'position_error_px'}
        table(item, required | {'beat_shift', 'optical_reference', 'template_registration'}, required)
        for key in ('endpoint_id', 'candidate_id', 'version'):
            string(item[key], key)
        if item['endpoint_id'] in seen:
            raise ValueError('duplicate endpoint calibration')
        seen.add(item['endpoint_id'])
        bounds(item['offset_px'])
        if item['position_error_px'] is not None:
            number(item['position_error_px'], 'position error', 0)
        if 'beat_shift' in item:
            read_ratio(item['beat_shift'])
            if 'observed_geometry' not in value:
                raise ValueError('a spatial beat shift requires observed geometry')
        if 'optical_reference' in item:
            from .projection_edges import validate_spec
            validate_spec(item['optical_reference'])
            if 'observed_geometry' not in value:
                raise ValueError('an optical reference requires observed geometry')
        if 'template_registration' in item:
            from .projection_edges import validate_registration
            validate_registration(item['template_registration'])
            if 'observed_geometry' not in value:
                raise ValueError('template registration requires observed geometry')
    if 'observed_geometry' in value:
        from .observed_geometry import VERSION as GEOMETRY_VERSION
        if 'coordinate_uncertainty' in value or 'motion_dependence' in value:
            raise ValueError('supply one observed coordinate model')
        request = value['observed_geometry']
        keys = {'version', 'localization_error_px', 'anchor_observations', 'scale', 'endpoint_position_relation'}
        table(request, keys, keys)
        if request['version'] not in (GEOMETRY_VERSION, 'observed-spatial-constraints-2', 'observed-spatial-constraints-3', 'observed-spatial-constraints-4', 'observed-spatial-constraints-5') or request['scale'] != 'constant':
            raise ValueError('unsupported observed geometry model')
        if request['version'] == 'observed-spatial-constraints-2' and any('optical_reference' in e for e in value['endpoint_calibrations']):
            raise ValueError('optical references require the current observed geometry version')
        if request['version'] not in (GEOMETRY_VERSION, 'observed-spatial-constraints-4', 'observed-spatial-constraints-5') and any('template_registration' in e for e in value['endpoint_calibrations']):
            raise ValueError('template registration requires the current observed geometry version')
        if request['endpoint_position_relation'] != 'same_raw_localization':
            raise ValueError('observed geometry requires shared raw endpoint localization')
        number(request['localization_error_px'], 'geometry localization error', 0)
        support = request['anchor_observations']
        if not isinstance(support, dict) or set(support) != {a['id'] for a in value['anchors']}:
            raise ValueError('observed geometry must bind every numbered anchor')
        if len({tuple(a['beat']) for a in value['anchors']}) != len(value['anchors']):
            raise ValueError('duplicate numbered anchor')
        for rows in support.values():
            if not isinstance(rows, list) or not rows:
                raise ValueError('observed geometry anchor requires raw observations')
            frames = set()
            for row in rows:
                table(row, {'frame_id', 'y_px'}, {'frame_id', 'y_px'})
                if type(row['frame_id']) is not int or row['frame_id'] < 0 or row['frame_id'] in frames:
                    raise ValueError('invalid or duplicate anchor observation frame')
                frames.add(row['frame_id'])
                number(row['y_px'], 'anchor observed row')
    else:
        SpatialBeatMap(value['anchors'], value['extrapolation'])
    if 'coordinate_uncertainty' in value:
        validate_coordinate_uncertainty(value['coordinate_uncertainty'], value['anchors'])
    if 'motion_dependence' in value:
        if 'coordinate_uncertainty' in value:
            raise ValueError('supply one coordinate dependence model')
        request = value['motion_dependence']
        keys = {'version', 'observation_error_px', 'anchor_observations'}
        table(request, keys | {'endpoint_position_relation'}, keys)
        if request['version'] != DEPENDENCE_VERSION:
            raise ValueError('unsupported motion dependence model')
        if request['observation_error_px'] is not None:
            number(request['observation_error_px'], 'motion observation error', 0)
        if request.get('endpoint_position_relation', 'unspecified') not in ('unspecified', 'same_raw_localization'):
            raise ValueError('unsupported endpoint position dependence')
        support = request['anchor_observations']
        if not isinstance(support, dict) or set(support) != {a['id'] for a in value['anchors']}:
            raise ValueError('motion dependence must bind every anchor')
        for rows in support.values():
            if not isinstance(rows, list) or len(rows) not in (1, 2):
                raise ValueError('anchor requires median observation representatives')
            for row in rows:
                table(row, {'frame_id', 'y_px', 'weight'}, {'frame_id', 'y_px', 'weight'})
                if type(row['frame_id']) is not int or row['frame_id'] < 0:
                    raise ValueError('invalid anchor observation frame')
                number(row['y_px'], 'anchor observed row')
                if read_ratio(row['weight']) != 1 / len(rows):
                    raise ValueError('invalid anchor median weight')


def validate_coordinate_uncertainty(model, anchors):
    required = {'version', 'sources', 'anchors', 'endpoints'}
    table(model, required, required)
    string(model['version'], 'coordinate uncertainty version')
    if not isinstance(model['sources'], list):
        raise ValueError('coordinate sources must be an array')
    sources = set()
    for source in model['sources']:
        table(source, {'id', 'range_px', 'reason'}, {'id', 'range_px', 'reason'})
        string(source['id'], 'coordinate source ID')
        string(source['reason'], 'coordinate source reason')
        if source['id'].startswith('residual:') or source['id'] in sources:
            raise ValueError('reserved or duplicate coordinate source ID')
        sources.add(source['id'])
        if source['range_px'] is not None:
            bounds(source['range_px'])
    for name in ('anchors', 'endpoints'):
        if not isinstance(model[name], dict):
            raise ValueError('coordinate bindings must be objects')
        for identity, terms in model[name].items():
            string(identity, 'coordinate binding ID')
            if not isinstance(terms, dict) or set(terms) - sources:
                raise ValueError('unknown coordinate error source')
            for coefficient in terms.values():
                number(coefficient, 'coordinate error coefficient')
    if set(model['anchors']) != {a['id'] for a in anchors}:
        raise ValueError('coordinate model must bind every anchor')


class SharedSpatialBeatMap:
    """Numbered spatial interpolation with common additive errors cancelled.

    Only a proven local bracket, or complete bounded support over the whole
    map, permits projection. An unbounded common origin can cancel locally;
    an unknown displacement between the bracketing anchors cannot.
    """
    def __init__(self, anchors, extrapolation, model):
        self.nominal = SpatialBeatMap(anchors, extrapolation)
        self.model = model
        self.sources = {s['id']: bounds(s['range_px']) if s['range_px'] is not None else None
                        for s in model['sources']}
        self.anchors = [coordinate(a['q_px'], a['q_range_px'], model['anchors'][a['id']],
                                   self.sources, 'residual:anchor:' + a['id']) for a in anchors]
        self.ids = [a['id'] for a in anchors]

    def project(self, lower, upper, endpoint_id):
        result = {'status': 'unavailable', 'coverage': None, 'beat_range': None,
                  'nominal_anchor_beat_range': None, 'reasons': [],
                  'dependency_model': self.model['version'], 'support_anchor_ids': [],
                  'cancelled_sources': [], 'propagated_sources': []}
        if endpoint_id not in self.model['endpoints']:
            result['reasons'] = ['endpoint_coordinate_dependence_unknown']
            return result
        if len(self.anchors) < 2:
            result['reasons'] = ['insufficient_barline_anchors']
            return result
        sources = dict(self.sources)
        point = coordinate((lower + upper) / 2, (lower, upper), self.model['endpoints'][endpoint_id],
                           sources, 'residual:endpoint:' + endpoint_id)
        start = max(1, min(bisect_left(self.nominal.q, point.nominal), len(self.anchors)-1)) - 1
        stop = start + 1
        # Expand only when finite bounds can cross a neighboring anchor. An
        # unrelated unknown origin elsewhere in the recording is immaterial.
        while True:
            left = (point - self.anchors[start]).domain(sources)
            right = (self.anchors[stop] - point).domain(sources)
            if left is None or right is None:
                result['reasons'] = ['unbounded_relative_coordinate_error']
                return result
            expand_left, expand_right = left[0] < 0 and start > 0, right[0] < 0 and stop < len(self.anchors)-1
            if not expand_left and not expand_right:
                break
            start -= int(expand_left)
            stop += int(expand_right)
        inside = left[0] >= 0 and right[0] >= 0
        result['coverage'] = 'interpolated' if inside else 'extrapolated'
        if not inside and self.nominal.extrapolation == 'withhold':
            result['reasons'] = ['outside_barline_support']
            return result
        values, propagated, cancelled, ids = [], set(), set(), set()
        for i in range(start, stop):
            numerator, denominator = point - self.anchors[i], self.anchors[i+1] - self.anchors[i]
            domain, issue = quotient_range(numerator, denominator, sources)
            if issue:
                result['reasons'] = [issue]
                return result
            lo, hi = domain
            if i != 0 or self.nominal.extrapolation == 'withhold':
                lo = max(lo, 0)
            if i != len(self.anchors)-2 or self.nominal.extrapolation == 'withhold':
                hi = min(hi, 1)
            if lo > hi:
                continue
            a, b = self.nominal.beats[i:i+2]
            values.extend((a + lo * (b-a), a + hi * (b-a)))
            used = numerator.terms.keys() | denominator.terms.keys()
            original = point.terms.keys() | self.anchors[i].terms.keys() | self.anchors[i+1].terms.keys()
            propagated.update(used)
            cancelled.update(original - used)
            ids.update(self.ids[i:i+2])
        if not values:
            result['reasons'] = ['outside_barline_support']
            return result
        result.update(status='conditional', beat_range=closed(min(values), max(values)),
                      nominal_anchor_beat_range=closed(self.nominal.nominal(lower), self.nominal.nominal(upper)),
                      support_anchor_ids=sorted(ids), cancelled_sources=sorted(cancelled - propagated),
                      propagated_sources=sorted(propagated),
                      reasons=[] if result['coverage'] == 'interpolated' else ['barline_extrapolation'])
        return result


class SpatialBeatMap:
    """Piecewise linear, positive spatial scale with shared anchor variables.

The envelope includes every supplied anchor position interval, with no
independence assumption. Extrapolation is opt-in and always labelled.
"""
    def __init__(self, anchors, extrapolation='withhold'):
        self.anchors = anchors
        self.q = [exact_ms(a['q_px']) for a in anchors]
        self.beats = [read_ratio(a['beat']) for a in anchors]
        self.domains = [bounds(a['q_range_px']) for a in anchors]
        self.extrapolation = extrapolation
        if any(b <= a for a, b in zip(self.beats, self.beats[1:])):
            raise ValueError('barline beats must be strictly increasing')
        if any(a[1] >= b[0] for a, b in zip(self.domains, self.domains[1:])):
            raise ValueError('barline coordinate ranges conflict or overlap')

    def nominal(self, q):
        i = max(1, min(bisect_left(self.q, q), len(self.q) - 1))
        return self.beats[i-1] + (q-self.q[i-1]) * (self.beats[i]-self.beats[i-1]) / (self.q[i]-self.q[i-1])

    def extrema(self, q):
        values = []
        for i, ((al, au), (bl, bu)) in enumerate(zip(self.domains, self.domains[1:])):
            # Interior pieces require a <= q <= b. The first/last piece
            # additionally supports explicitly requested linear extrapolation.
            if i != 0:
                au = min(au, q)
            if i != len(self.domains) - 2:
                bl = max(bl, q)
            if al > au or bl > bu:
                continue
            values.extend(self.beats[i] + (q-a)*(self.beats[i+1]-self.beats[i])/(b-a)
                          for a in (al, au) for b in (bl, bu))
        return min(values), max(values)

    def project(self, lower, upper):
        if lower > upper:
            raise ValueError('unordered endpoint coordinate domain')
        result = {'status': 'unavailable', 'coverage': None, 'beat_range': None,
                  'nominal_anchor_beat_range': None, 'reasons': []}
        if len(self.q) < 2:
            result['reasons'] = ['insufficient_barline_anchors']
            return result
        inside = self.domains[0][1] <= lower <= upper <= self.domains[-1][0]
        result['coverage'] = 'interpolated' if inside else 'extrapolated'
        if not inside and self.extrapolation == 'withhold':
            result['reasons'] = ['outside_barline_support']
            return result
        result.update(status='conditional', beat_range=closed(self.extrema(lower)[0], self.extrema(upper)[1]),
                      nominal_anchor_beat_range=closed(self.nominal(lower), self.nominal(upper)),
                      reasons=[] if inside else ['barline_extrapolation'])
        return result


def project_relative_beats(observations, movement, decoded, reference, *, measure_beats, rhythm_grid=None,
                           physical_review=None):
    """Produce relative beat ranges and separate capture ranges for selections.

Scoped offsets are total downward offsets from the detection reference, not
increments on an already calibrated endpoint. Unknown errors stay unknown.
All results are conditional; this path does not turn bounded or ambiguous
coordinates into exact chart notes or accepted judgment times.
"""
    validate_reference(reference)
    if physical_review is not None:
        physical_review.check_parents(observations, movement, decoded)
    if read_ratio(reference['measure_beats']) != measure_beats:
        raise ValueError('spatial reference and song measure lengths differ')
    if (movement.manifest['key']['parents'] != [observations.id]
            or decoded.manifest['key']['parents'][:2] != [observations.id, movement.id]
            or any(reference[key] != cache.id for key, cache in (
                ('observation_cache_id', observations), ('motion_cache_id', movement), ('decode_cache_id', decoded)))):
        raise ValueError('spatial reference parent lineage mismatch')
    geometry, geometry_nodes, geometry_diagnostics = None, {}, None
    if 'observed_geometry' in reference:
        from .observed_geometry import (VERSION as GEOMETRY_VERSION, LOCALIZATION_SCOPE,
                                        LEGACY_LOCALIZATION_SCOPE, observed_geometry)
        request = reference['observed_geometry']
        localizations = physical_review.localizations() if physical_review is not None else ()
        if localizations and request['version'] != GEOMETRY_VERSION:
            raise ValueError('reviewed localizations require the current observed geometry version')
        geometry, geometry_nodes, geometry_diagnostics = observed_geometry(
            observations, movement, decoded, {a['id']:read_ratio(a['beat']) for a in reference['anchors']},
            request['anchor_observations'], request['localization_error_px'], reference['endpoint_calibrations'],
            localization_scope=(LOCALIZATION_SCOPE if request['version'] in (GEOMETRY_VERSION,'observed-spatial-constraints-5')
                                else LEGACY_LOCALIZATION_SCOPE), reviewed_localizations=localizations)
        geometry_diagnostics['scale_bounds_px_per_beat'] = [ratio(v) for v in geometry.scale] if geometry.scale else None
        geometry_diagnostics['scale_px_per_beat'] = closed(*geometry.scale) if geometry.status == 'conditional' else None
        geometry_diagnostics['anchor_pair_count'] = geometry.anchor_pair_count
    line = SpatialBeatMap(reference['anchors'], reference['extrapolation']) if geometry is None else None
    if geometry is None and physical_review is not None and physical_review.localizations():
        raise ValueError('reviewed localizations require an observed spatial geometry model')
    model = reference.get('coordinate_uncertainty')
    diagnostics = None
    if 'motion_dependence' in reference:
        model, diagnostics = derive_coordinate_model(observations, movement, decoded, reference)
        validate_coordinate_uncertainty(model, reference['anchors'])
    shared_line = SharedSpatialBeatMap(reference['anchors'], reference['extrapolation'], model) if model else None
    clock = EndpointTimeMap(timeline(movement))
    offsets = {e['endpoint_id']: e for e in reference['endpoint_calibrations']}
    used_offsets = set()
    selected_endpoints = set()
    output, geometry_points = [], []
    profile = observations.manifest['metadata']['profile']['calibration']
    for lane in lanes(decoded):
        endpoints = {e['id']: e for e in lane['endpoints']}
        candidates = {c['id']: (i, c) for i, c in enumerate(lane['candidates'])}
        for event in lane['events']:
            selected = [c for _, c in sorted(candidates[cid] for cid in set(event['selected_candidate_ids']))]
            for candidate in selected:
                row = {'event_id': event['id'], 'candidate_id': candidate['id'], 'lane': lane['lane'],
                       'type': candidate['type'], 'pixel_status': event['status'], 'status': 'unavailable',
                       'endpoints': {}, 'duration_beat_range': None, 'capture_duration_ms': None,
                       'reasons': [], 'accepted': False}
                for side in ('start', 'end'):
                    if side == 'end' and candidate['type'] == 'tap':
                        continue
                    eid = candidate[side]
                    if eid is not None:
                        selected_endpoints.add(eid)
                    point = endpoints.get(eid)
                    result = {'endpoint_id': eid, 'status': 'unavailable', 'coordinate_range_px': None,
                              'relative_beat': None, 'capture_time': None, 'reasons': []}
                    row['endpoints'][side] = result
                    if point is None or point['q_px'] is None:
                        result['reasons'] = ['missing_or_unlocalized_endpoint']
                        continue
                    expected_kind = 'tap' if candidate['type'] == 'tap' else 'head' if side == 'start' else 'tail'
                    if point['kind'] != expected_kind:
                        raise ValueError('selected endpoint kind mismatch')
                    result.update(kind=point['kind'], observation_ids=point['observation_ids'],
                                  motion_error_px=point['shared_motion_error_px'], fusion_error_px=point['fusion_error_px'])
                    q = exact_ms(point['q_px'])
                    calibration = profile[point['kind']]
                    override = offsets.get(eid)
                    if override is not None:
                        if override['candidate_id'] != candidate['id']:
                            raise ValueError('scoped endpoint calibration candidate mismatch')
                        used_offsets.add(eid)
                        lo, hi = bounds(override['offset_px'])
                        error = override['position_error_px']
                        # Decode coordinates already contain the profile offset.
                        if calibration['status'] == 'verified':
                            q += exact_ms(calibration['offset_px'])
                        result['calibration'] = override
                        raw_offset = (lo, hi)
                    elif calibration['status'] == 'verified':
                        lo, hi = -exact_ms(calibration['error_px']), exact_ms(calibration['error_px'])
                        error = point['localization_error_px']
                        result['calibration'] = {'source': 'profile', 'version': calibration['version']}
                        raw_offset = (exact_ms(calibration['offset_px'])+lo, exact_ms(calibration['offset_px'])+hi)
                    else:
                        result['reasons'] = ['endpoint_calibration_unavailable']
                        continue
                    if error is None:
                        result['reasons'] = ['localization_error_unknown']
                        continue
                    error = exact_ms(error)
                    if geometry is not None and error != exact_ms(reference['observed_geometry']['localization_error_px']):
                        raise ValueError('endpoint and geometry localization bounds differ')
                    lower, upper = q-hi-error, q-lo+error
                    shift = read_ratio(override['beat_shift']) if override and 'beat_shift' in override else Fraction(0)
                    if shift and geometry is not None and geometry.status == 'conditional':
                        displacements = sorted(s*shift for s in geometry.scale)
                        lower, upper = lower+displacements[0], upper+displacements[1]
                        result['capture_spatial_beat_shift'] = ratio(shift)
                    spatial_lower, spatial_upper = lower, upper
                    if diagnostics and eid in diagnostics['shared_endpoint_position_ids']:
                        result['localization_component'] = 'shared_raw_observation_sources'
                        spatial_lower, spatial_upper = q-hi, q-lo
                        result['shared_coordinate_residual_px'] = closed(spatial_lower, spatial_upper)
                    if geometry is not None:
                        node = geometry_nodes.get(eid)
                        registration = geometry_diagnostics.get('registered_endpoints', {}).get(eid)
                        if registration:
                            result['template_registration'] = {'status':registration['status'],
                                'measured_frames':registration['measured_frames'], 'verified':False}
                        optical = geometry_diagnostics.get('optical_endpoints', {}).get(eid)
                        geometric_offset = (Fraction(0), Fraction(0)) if optical and optical['status']=='measured' else raw_offset
                        if optical:
                            result['optical_reference'] = {'status':optical['status'],
                                'measured_frames':optical['measured_frames'], 'verified':False}
                        geometry_points.append({'id':eid,'node':node,'candidate_id':candidate['id'],
                                                'pixel_status':event['status'],'side':side,
                                                'offset':geometric_offset,'beat_shift':shift,'result':result})
                        domain = geometry.project(node, geometric_offset, shift)
                        relations = geometry.constraints.get(node, [])
                        coverage = 'interpolated' if domain and relations and min(r[1] for r in relations) <= domain[0] <= domain[1] <= max(r[1] for r in relations) else 'extrapolated'
                        reason = ('constant_spatial_scale_conflict' if geometry.status == 'conflict' else
                                  'constant_spatial_scale_unavailable' if geometry.status != 'conditional' else
                                  'no_observed_path_to_numbered_barline') if domain is None else None
                        if domain and coverage == 'extrapolated' and reference['extrapolation'] == 'withhold':
                            domain, reason = None, 'outside_barline_support'
                        spatial = {'status': 'conditional' if domain else 'unavailable', 'coverage': coverage,
                                   'beat_range': closed(*domain) if domain else None, 'nominal_anchor_beat_range': None,
                                   'reasons': [reason] if reason else ['constant_spatial_scale_extension'] if coverage == 'extrapolated' else [],
                                   'support_anchor_ids': [r[0].removeprefix('anchor:') for r in relations],
                                   'dependency_model': geometry_diagnostics['version']}
                    else:
                        spatial = shared_line.project(spatial_lower, spatial_upper, eid) if shared_line else line.project(lower, upper)
                    result.update(coordinate_range_px=closed(lower, upper), relative_beat=spatial,
                                  capture_time=clock.project(lower, upper))
                    if shift and geometry is not None and geometry.status != 'conditional':
                        result['visual_capture_time'], result['capture_time'] = result['capture_time'], None
                        result['capture_reasons'] = ['spatial_beat_shift_unavailable']
                    result['status'] = result['relative_beat']['status']
                    result['reasons'] = result['relative_beat']['reasons'][:]
                if physical_review is not None:
                    basis = physical_review.basis(row)
                    if basis is not None:
                        row['physical_review'] = basis
                points = list(row['endpoints'].values())
                row['reasons'] = sorted({reason for point in points for reason in point['reasons']})
                if event['status'] != 'resolved':
                    row['reasons'].append('pixel_' + event['status'])
                if all(p['status'] == 'conditional' for p in points):
                    row['status'] = 'conditional'
                    if candidate['type'] == 'longnote':
                        head, tail = row['endpoints']['start'], row['endpoints']['end']
                        row['duration_beat_range'] = difference(tail['relative_beat']['beat_range'], head['relative_beat']['beat_range'])
                        if unpack(row['duration_beat_range'])[0] <= 0:
                            row['reasons'].append('duration_may_be_nonpositive')
                if candidate['type'] == 'longnote':
                    head, tail = row['endpoints']['start'], row['endpoints']['end']
                    if all(p['capture_time'] and p['capture_time']['status'] == 'bounded' for p in (head, tail)):
                        row['capture_duration_ms'] = difference(tail['capture_time']['time_range_ms'], head['capture_time']['time_range_ms'])
                output.append(row)
    if set(offsets) != used_offsets:
        raise ValueError('scoped calibration does not bind a localized selected endpoint')
    if shared_line and set(model['endpoints']) - selected_endpoints:
        raise ValueError('coordinate dependence does not bind a selected endpoint')
    refinement = None
    if geometry is not None and rhythm_grid is not None:
        from .rhythm_refinement import refine_grid
        original_scale = geometry.scale
        reviewed = {r['candidate_id'] for r in output if 'physical_review' in r} & {
            p['candidate_id'] for p in geometry_points}
        geometry, refinement = refine_grid(geometry,geometry_points,rhythm_grid,
                                           reviewed_candidate_ids=reviewed)
        if physical_review is not None:
            refinement['physical_review'] = physical_review.receipt()
        refinement['grid'] = rhythm_grid.snapshot()
        # Capture time is projected separately using the original, wider scale.
        # A musical-grid constraint cannot manufacture new clock observations.
        refinement['capture_scale_bounds_px_per_beat'] = [ratio(v) for v in original_scale] if original_scale else None
        geometry_diagnostics.update(status=geometry.status,
            scale_bounds_px_per_beat=[ratio(v) for v in geometry.scale] if geometry.scale else None,
            scale_px_per_beat=closed(*geometry.scale) if geometry.status == 'conditional' else None)
        for point in geometry_points:
            result = point['result']
            spatial = result['relative_beat']
            domain = geometry.project(point['node'],point['offset'],point['beat_shift'])
            # An explicit coverage restriction remains in force after refinement.
            if 'outside_barline_support' in spatial['reasons']:
                continue
            spatial.update(status='conditional' if domain else 'unavailable',
                           beat_range=closed(*domain) if domain else None)
            if geometry.status=='conflict':
                spatial['reasons']=['shared_grid_scale_conflict']
            result.update(status=spatial['status'],reasons=spatial['reasons'][:])
        for row in output:
            points=list(row['endpoints'].values())
            row['status']='conditional' if all(p['status']=='conditional' for p in points) else 'unavailable'
            row['reasons']=sorted({r for p in points for r in p['reasons']})
            if row['pixel_status']!='resolved':row['reasons'].append('pixel_'+row['pixel_status'])
            if row['type']=='longnote':
                row['duration_beat_range']=(difference(row['endpoints']['end']['relative_beat']['beat_range'],
                    row['endpoints']['start']['relative_beat']['beat_range']) if row['status']=='conditional' else None)
                if row['duration_beat_range'] and unpack(row['duration_beat_range'])[0]<=0:
                    row['reasons'].append('duration_may_be_nonpositive')
    return {'format': 'ez2rc.relative-beats', 'schema_version': 1, 'algorithm_version': VERSION,
            'status': 'conditional', 'reference': reference,
            **({'physical_review': physical_review.receipt()} if physical_review is not None else {}),
            **({'coordinate_uncertainty': model, 'motion_dependence': diagnostics} if shared_line else {}),
            **({'observed_geometry': geometry_diagnostics} if geometry is not None else {}),
            **({'rhythm_refinement': refinement} if refinement is not None else {}),
            'coordinate': 'first_barline_zero', 'capture_coordinate': 'recording_timestamp_ms',
            'rule': ('relative beat = numbered observation constraints with constant spatial scale; capture time = inverse observed scroll' if geometry is not None else
                     'relative beat = numbered spatial barline interpolation; capture time = inverse observed scroll'),
            'uncertainty': ('Raw localization intervals constrain shared frame and physical landmark variables jointly. Constant spacing and retained identities are conditional hypotheses. No empirical calibration status is promoted.' if geometry is not None else
                            'Named coordinate error sources are propagated jointly; identical additive terms cancel before division. Unknown noncommon errors withhold the projection. Unspecified motion/fusion errors remain unpropagated. No independence or physical confidence claim.' if shared_line else
                            'Endpoint and anchor ranges are conditional outer enclosures. Motion/fusion error and their dependence remain unpropagated. No independence or physical confidence claim.'),
            'duration_rule': 'tail.lower - head.upper through tail.upper - head.lower; beat numbering origin cancels, no endpoint or anchor errors are assumed to cancel',
            'conditional_candidate_count': sum(r['status'] == 'conditional' for r in output),
            'accepted_note_count': 0, 'candidates': output}
