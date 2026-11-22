"""Join endpoint time domains to an explicitly shared musical error model."""
from copy import deepcopy

from .contracts import decimal_bpm, exact_ms, read_ratio
from .endpoint_ranges import VERSION as ENDPOINT_VERSION, closed
from .timing import PROJECTION_VERSION

VERSION = 'endpoint-aware-beat-ranges-4'


def _bounds(value):
    lo, hi = read_ratio(value['lower']), read_ratio(value['upper'])
    if lo > hi or value['lower_inclusive'] is not True or value['upper_inclusive'] is not True:
        raise ValueError('endpoint propagation requires ordered closed bounds')
    return lo, hi


def _by_id(values, key='id'):
    result = {value[key]: value for value in values}
    if len(result) != len(values):
        raise ValueError('duplicate endpoint propagation identity')
    return result


def _join_parents(resolution, projection, endpoint_ranges):
    if (projection is None or endpoint_ranges is None
            or projection['format'] != 'ez2rc.time-projection'
            or projection['algorithm_version'] != PROJECTION_VERSION
            or type(projection['schema_version']) is not int or projection['schema_version'] != 1
            or endpoint_ranges['format'] != 'ez2rc.endpoint-time-ranges'
            or endpoint_ranges['algorithm_version'] != ENDPOINT_VERSION
            or type(endpoint_ranges['schema_version']) is not int or endpoint_ranges['schema_version'] != 1):
        raise ValueError('unsupported endpoint propagation parents')
    for key in ('observation_cache_id', 'motion_cache_id', 'decode_cache_id'):
        if projection[key] != resolution[key] or endpoint_ranges[key] != resolution[key]:
            raise ValueError('endpoint propagation parent lineage mismatch')
    if projection.get('position_calibration') != endpoint_ranges.get('position_calibration'):
        raise ValueError('endpoint propagation position declaration mismatch')
    lanes = _by_id(projection['lanes'], 'lane')
    range_lanes = _by_id(endpoint_ranges['lanes'], 'lane')
    if lanes.keys() != range_lanes.keys():
        raise ValueError('endpoint propagation lanes differ')
    points, ranges, candidates = {}, {}, {}
    for lane_id, lane in lanes.items():
        other = range_lanes[lane_id]
        source_points, range_points = _by_id(lane['endpoints']), _by_id(other['endpoints'])
        source_candidates, range_candidates = _by_id(lane['candidates']), _by_id(other['candidates'])
        if source_points.keys() != range_points.keys() or source_candidates != range_candidates:
            raise ValueError('endpoint propagation evidence identities differ')
        for eid, point in source_points.items():
            match = range_points[eid]
            if (eid in points or any(match[key] != point[key] for key in ('id', 'kind', 'q_px'))
                    or match['nominal_projection_status'] != point['status']):
                raise ValueError('endpoint propagation point mismatch')
            points[eid], ranges[eid] = (lane_id, point), match
        for cid, candidate in source_candidates.items():
            if cid in candidates:
                raise ValueError('duplicate endpoint propagation candidate')
            candidates[cid] = (lane_id, candidate)
    return points, ranges, candidates


def _component(point, evidence):
    component = evidence['components']['calibration_and_localization']
    if evidence['status'] != component['status']:
        raise ValueError('endpoint component status mismatch')
    if component['status'] == 'unavailable':
        if component['time_range_ms'] is not None or not component['reasons']:
            raise ValueError('unavailable endpoint component must retain reasons and no bounds')
        return None, component['reasons']
    if (component['status'] != 'bounded' or component['reasons']
            or point['calibration_status'] != 'verified' or point['q_px'] is None
            or point['calibration_error_px'] is None or point['localization_error_px'] is None):
        raise ValueError('bounded endpoint component requires known calibrated errors')
    errors = [exact_ms(point[key]) for key in ('calibration_error_px', 'localization_error_px')]
    error = sum(errors)
    if min(errors) < 0 or read_ratio(component['error_px_exact']) != error:
        raise ValueError('endpoint component error mismatch')
    q = exact_ms(point['q_px'])
    if _bounds(component['pixel_range_px']) != (q - error, q + error):
        raise ValueError('endpoint component pixel domain mismatch')
    domain, support = _bounds(component['time_range_ms']), _bounds(component['source_bracket_ms'])
    if domain[0] < support[0] or domain[1] > support[1]:
        raise ValueError('endpoint component time domain exceeds its source support')
    return domain, []


def apply_endpoint_ranges(result, events, resolution, pixels, projection, endpoint_ranges):
    """Replace nominal-time marginals with conditional endpoint outer bounds.

    Dependencies of endpoint errors on each other and on shared terms are
    unspecified. Only the explicitly common musical terms cancel in durations.
    """
    points, ranges, candidates = _join_parents(resolution, projection, endpoint_ranges)
    if pixels is None:
        raise ValueError('endpoint propagation requires selected pixel candidates')
    pixel_rows = _by_id(pixels)
    timing_rows = _by_id(resolution['events'], 'event_id')
    if pixel_rows.keys() != timing_rows.keys():
        raise ValueError('endpoint propagation event identities differ')
    scale = decimal_bpm(result['bpm']) / 60000
    calibration = resolution['judgment_calibration']
    offset = exact_ms(calibration['offset_ms']) if calibration is not None and calibration['status'] == 'verified' else None
    shared = _bounds(result['combined_shared_shift_beats']) if result['combined_shared_shift_beats'] is not None else None
    for row, event in zip(result['events'], events):
        pixel, timing = pixel_rows[event.id], timing_rows[event.id]
        if pixel['status'] != timing['pixel_status'] or pixel['lane'] != event.lane:
            raise ValueError('endpoint propagation pixel event mismatch')
        decode_status = pixel.get('decode_status', pixel['status'])
        expected_status = 'ambiguous' if pixel.get('hold_resolution') else decode_status
        if pixel['status'] != expected_status:
            raise ValueError('endpoint propagation scoped status mismatch')
        issues = list(row['reasons'])
        row.update(nominal_duration_beats=row['duration_beats'], duration_beats=None,
                   duration_beat_range=None, diagnostics=[], selected_candidate_id=None)
        for side in ('start', 'end'):
            row[side + '_nominal_time_beat_range'] = deepcopy(row[side + '_beat_range'])
            row[side + '_beat_range'] = None
            row[side + '_endpoint_id'] = None
            row[side + '_capture_time_range_ms'] = None
            row[side + '_judgment_time_range_ms'] = None
        selected = pixel['selected_intervals']
        if len(selected) != 1:
            issues.append('no_unique_selected_candidate')
        else:
            candidate = selected[0]
            if candidate['id'] not in candidates:
                raise ValueError('selected endpoint propagation candidate absent')
            lane_id, source = candidates[candidate['id']]
            if (lane_id != event.lane or candidate['id'] not in event.candidate_ids
                    or source['selected'] is not True or source['pixel_status'] != decode_status
                    or any(source[key] != candidate[key] for key in ('id', 'type', 'start', 'end'))
                    or candidate['type'] != event.type):
                raise ValueError('selected endpoint propagation candidate mismatch')
            row['selected_candidate_id'] = candidate['id']
            for side in ('start', 'end'):
                if side == 'end' and event.type == 'tap':
                    continue
                eid = candidate[side]
                row[side + '_endpoint_id'] = eid
                if eid is None:
                    issues.append(side + '_missing_endpoint')
                    continue
                if eid not in points or points[eid][0] != event.lane:
                    raise ValueError('selected endpoint propagation endpoint absent or in another lane')
                point = points[eid][1]
                expected_kind = 'tap' if event.type == 'tap' else 'head' if side == 'start' else 'tail'
                if point['kind'] != expected_kind:
                    raise ValueError('selected endpoint propagation endpoint kind mismatch')
                domain, reasons = _component(point, ranges[eid])
                issues.extend(side + '_' + reason for reason in reasons)
                accepted = getattr(event, side + '_ms')
                if accepted is not None and (offset is None or point['status'] not in ('sample', 'interpolated')
                        or accepted != point['nominal_timestamp_ms'] + calibration['offset_ms']):
                    raise ValueError('accepted event time differs from its selected endpoint')
                if domain is not None:
                    row[side + '_capture_time_range_ms'] = closed(*domain)
                    if offset is not None:
                        row[side + '_judgment_time_range_ms'] = closed(domain[0] + offset, domain[1] + offset)
        row['reasons'] = list(dict.fromkeys(issues))
        row['status'] = 'unavailable' if issues else 'bounded'
        if not issues:
            for side in ('start', 'end'):
                value = row[side + '_judgment_time_range_ms']
                if value is not None:
                    lo, hi = _bounds(value)
                    row[side + '_beat_range'] = closed(lo * scale + shared[0], hi * scale + shared[1])
            if event.type == 'longnote':
                start, end = (_bounds(row[side + '_capture_time_range_ms']) for side in ('start', 'end'))
                lo, hi = (end[0] - start[1]) * scale, (end[1] - start[0]) * scale
                row['duration_beat_range'] = closed(lo, hi)
                if lo <= 0:
                    row['diagnostics'].append('duration_may_be_nonpositive')
    result.update(schema_version=3, algorithm_version=VERSION,
                  **({'position_calibration': deepcopy(endpoint_ranges['position_calibration'])}
                     if 'position_calibration' in endpoint_ranges else {}),
                  endpoint_component='calibration_and_localization',
                  equation='beat = (endpoint_capture_time + nominal_judgment_offset) * BPM / 60000 + combined_shared_shift; common sources obey uncertainty_model.relation.',
                  uncertainty='Endpoint calibration/localization and shared origin/judgment domains are combined as a conservative outer enclosure with unspecified remaining dependence. Unknown endpoint errors are not zero. Motion/fusion error remains unpropagated; these are not physical confidence intervals.',
                  duration_rule='Only common origin/judgment terms cancel. Endpoint times use an outer difference interval with no independence or shared endpoint-error assumption. Nonpositive bounds are retained; duration_beats is unavailable and nominal_duration_beats is historical nominal-time arithmetic.')
    result['bounded_event_count'] = sum(row['status'] == 'bounded' for row in result['events'])
    result['unavailable_event_count'] = len(result['events']) - result['bounded_event_count']
    result['status'] = 'available' if result['bounded_event_count'] else 'unavailable'
