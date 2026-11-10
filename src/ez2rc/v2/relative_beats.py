"""Conditional musical coordinates from explicitly numbered spatial barlines.

Capture timestamps never define this map. Inputs are bound to visual caches;
neither note labels nor a fitted capture-clock offset enter the conversion.
"""
from bisect import bisect_left
import hashlib
import json
from pathlib import Path

from .config import number, string, table
from .contracts import exact_ms, ratio, read_ratio
from .decode import lanes
from .endpoint_ranges import EndpointTimeMap, closed
from .motion import timeline
from .storage import reject_constant

VERSION = 'spatial-relative-beats-1'


def bounds(values):
    if not isinstance(values, list) or len(values) != 2:
        raise ValueError('expected two ordered pixel bounds')
    result = tuple(exact_ms(number(v, 'pixel bound')) for v in values)
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
    table(value, required, required)
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
        table(item, required, required)
        for key in ('endpoint_id', 'candidate_id', 'version'):
            string(item[key], key)
        if item['endpoint_id'] in seen:
            raise ValueError('duplicate endpoint calibration')
        seen.add(item['endpoint_id'])
        bounds(item['offset_px'])
        if item['position_error_px'] is not None:
            number(item['position_error_px'], 'position error', 0)
    SpatialBeatMap(value['anchors'], value['extrapolation'])


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


def project_relative_beats(observations, movement, decoded, reference, *, measure_beats):
    """Produce relative beat ranges and separate capture ranges for selections.

Scoped offsets are total downward offsets from the detection reference, not
increments on an already calibrated endpoint. Unknown errors stay unknown.
All results are conditional; this path does not turn bounded or ambiguous
coordinates into exact chart notes or accepted judgment times.
"""
    validate_reference(reference)
    if read_ratio(reference['measure_beats']) != measure_beats:
        raise ValueError('spatial reference and song measure lengths differ')
    if (movement.manifest['key']['parents'] != [observations.id]
            or decoded.manifest['key']['parents'][:2] != [observations.id, movement.id]
            or any(reference[key] != cache.id for key, cache in (
                ('observation_cache_id', observations), ('motion_cache_id', movement), ('decode_cache_id', decoded)))):
        raise ValueError('spatial reference parent lineage mismatch')
    line = SpatialBeatMap(reference['anchors'], reference['extrapolation'])
    clock = EndpointTimeMap(timeline(movement))
    offsets = {e['endpoint_id']: e for e in reference['endpoint_calibrations']}
    used_offsets = set()
    output = []
    profile = observations.manifest['metadata']['profile']['calibration']
    for lane in lanes(decoded):
        endpoints = {e['id']: e for e in lane['endpoints']}
        for event in lane['events']:
            selected = [c for c in lane['candidates'] if c['id'] in event['selected_candidate_ids']]
            for candidate in selected:
                row = {'event_id': event['id'], 'candidate_id': candidate['id'], 'lane': lane['lane'],
                       'type': candidate['type'], 'pixel_status': event['status'], 'status': 'unavailable',
                       'endpoints': {}, 'duration_beat_range': None, 'capture_duration_ms': None,
                       'reasons': [], 'accepted': False}
                for side in ('start', 'end'):
                    if side == 'end' and candidate['type'] == 'tap':
                        continue
                    eid = candidate[side]
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
                    elif calibration['status'] == 'verified':
                        lo, hi = -exact_ms(calibration['error_px']), exact_ms(calibration['error_px'])
                        error = point['localization_error_px']
                        result['calibration'] = {'source': 'profile', 'version': calibration['version']}
                    else:
                        result['reasons'] = ['endpoint_calibration_unavailable']
                        continue
                    if error is None:
                        result['reasons'] = ['localization_error_unknown']
                        continue
                    error = exact_ms(error)
                    lower, upper = q-hi-error, q-lo+error
                    result.update(coordinate_range_px=closed(lower, upper),
                                  relative_beat=line.project(lower, upper), capture_time=clock.project(lower, upper))
                    result['status'] = result['relative_beat']['status']
                    result['reasons'] = result['relative_beat']['reasons'][:]
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
    return {'format': 'ez2rc.relative-beats', 'schema_version': 1, 'algorithm_version': VERSION,
            'status': 'conditional', 'reference': reference,
            'coordinate': 'first_barline_zero', 'capture_coordinate': 'recording_timestamp_ms',
            'rule': 'relative beat = numbered spatial barline interpolation; capture time = inverse observed scroll',
            'uncertainty': 'Endpoint and anchor ranges are conditional outer enclosures. Motion/fusion error and their dependence remain unpropagated. No independence or physical confidence claim.',
            'duration_rule': 'tail.lower - head.upper through tail.upper - head.lower; beat numbering origin cancels, no endpoint or anchor errors are assumed to cancel',
            'conditional_candidate_count': sum(r['status'] == 'conditional' for r in output),
            'accepted_note_count': 0, 'candidates': output}
