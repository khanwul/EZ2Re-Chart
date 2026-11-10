"""Conditional pixel-domain error propagation through observed scroll support."""
from bisect import bisect_left, bisect_right
from copy import deepcopy

from .contracts import exact_ms, finite_number, ratio
from .motion import STATUS, timeline
from .timing import PROJECTION_VERSION, ScrollTimeMap

VERSION = 'conditional-endpoint-time-ranges-1'


def closed(lower, upper):
    return {'lower': ratio(lower), 'upper': ratio(upper),
            'lower_inclusive': True, 'upper_inclusive': True}


class EndpointTimeMap:
    """Exact inverse image bounds, conditional on the representative curve.

    Every sample/link intersecting the pixel domain must be usable. Holds have
    multiple inverse times and retain their full extent. Motion error is not
    treated as independent of the endpoint coordinate or added a second time.
    """
    def __init__(self, clock):
        self.nominal = ScrollTimeMap(clock)
        curve = self.nominal
        self.scroll = tuple(exact_ms(v.item()) for v in curve.scroll)
        self.times = tuple(exact_ms(v.item()) for v in curve.ms)
        ids, states = curve.ids.tolist(), curve.status.tolist()
        self.bad = {}
        for reason, values in (
            ('uncertain_motion', [s not in (STATUS['origin'], STATUS['observed'], STATUS['repeated']) for s in states]),
            ('missing_frames', [False, *[b - a != 1 for a, b in zip(ids, ids[1:])]]),
            ('nonincreasing_sample_times', [False, *[b <= a for a, b in zip(self.times, self.times[1:])]])):
            prefix = [0]
            for value in values:
                prefix.append(prefix[-1] + value)
            self.bad[reason] = prefix

    def project(self, lower, upper):
        lower, upper = exact_ms(lower), exact_ms(upper)
        if lower > upper:
            raise ValueError('pixel interval must be ordered')
        result = {'pixel_range_px': closed(lower, upper), 'status': 'unavailable',
                  'time_range_ms': None, 'source_frame_id_range': None,
                  'source_sample_count': 0, 'source_bracket_ms': None, 'reasons': []}
        if lower < self.scroll[0] or upper > self.scroll[-1]:
            result['reasons'] = ['outside_observed_scroll_range']
            return result
        left = bisect_left(self.scroll, lower)
        if self.scroll[left] != lower:
            left -= 1
        right = bisect_right(self.scroll, upper) - 1
        if self.scroll[right] != upper:
            right += 1
        result.update(source_frame_id_range=[int(self.nominal.ids[left]), int(self.nominal.ids[right])],
                      source_sample_count=right - left + 1,
                      source_bracket_ms=closed(self.times[left], self.times[right]))
        for reason, prefix in self.bad.items():
            # Link faults belong to the sample on the right; exclude the link
            # into the first support sample. Sample faults include both ends.
            start = left if reason == 'uncertain_motion' else left + 1
            if prefix[right + 1] != prefix[start]:
                result['reasons'].append(reason)
        if result['reasons']:
            return result

        def inverse(q, earliest):
            index = bisect_left(self.scroll, q) if earliest else bisect_right(self.scroll, q) - 1
            if self.scroll[index] == q:
                return self.times[index]
            lo = index - 1 if earliest else index
            hi = lo + 1
            return self.times[lo] + (q - self.scroll[lo]) * (self.times[hi] - self.times[lo]) / (self.scroll[hi] - self.scroll[lo])

        result.update(status='bounded', time_range_ms=closed(inverse(lower, True), inverse(upper, False)))
        return result


def _error(value):
    if value is not None and (not finite_number(value) or value < 0):
        raise ValueError('endpoint error must be finite and nonnegative or unknown')
    return exact_ms(value) if value is not None else None


def project_endpoint_time_ranges(observations, movement, projection):
    """Keep component ranges separate from accepted times and musical notes.

    Observed spread is diagnostic, never substituted for a missing location
    error. Summing known error magnitudes gives an outer enclosure with unknown
    dependence; it does not declare independent errors or shared cancellation.
    """
    if (projection['format'] != 'ez2rc.time-projection'
            or projection['schema_version'] != 1
            or projection['algorithm_version'] != PROJECTION_VERSION):
        raise ValueError('unsupported endpoint range projection')
    if (movement.manifest['key']['parents'] != [observations.id]
            or projection['observation_cache_id'] != observations.id
            or projection['motion_cache_id'] != movement.id):
        raise ValueError('endpoint range parent lineage mismatch')
    curve = EndpointTimeMap(timeline(movement))
    metadata = observations.manifest['metadata']
    if projection.get('position_calibration') != metadata.get('position_calibration'):
        raise ValueError('endpoint range position declaration differs from its observation parent')
    calibrations = metadata['profile']['calibration']
    output, seen, counts = [], set(), {'bounded': 0, 'unavailable': 0}
    calibration_count = 0
    for lane in projection['lanes']:
        endpoints = []
        for point in lane['endpoints']:
            if point['id'] in seen:
                raise ValueError('duplicate endpoint range identity')
            seen.add(point['id'])
            calibration = calibrations[point['kind']]
            verified = calibration['status'] == 'verified'
            expected_error = calibration.get('error_px') if verified else None
            if (point['calibration_status'] != calibration['status']
                    or point['calibration_error_px'] != expected_error
                    or point['localization_error_px'] != metadata.get('position_error_px')):
                raise ValueError('endpoint range calibration differs from its observation parent')
            cal_error, loc_error = _error(point['calibration_error_px']), _error(point['localization_error_px'])
            q = point['q_px']
            if q is not None:
                nominal = curve.nominal.project(q)
                if any(point[key] != value for key, value in nominal.items()):
                    raise ValueError('endpoint range projection differs from its motion parent')
                q = exact_ms(q)
            issues = ([] if verified else ['endpoint_calibration_unverified']) + ([] if q is not None else ['endpoint_unlocalized'])
            components = {}
            for name, error, missing in (
                ('calibration', cal_error, ['calibration_error_unknown'] if cal_error is None else []),
                ('localization', loc_error, ['localization_error_unknown'] if loc_error is None else []),
                ('calibration_and_localization', None if cal_error is None or loc_error is None else cal_error + loc_error,
                 (['calibration_error_unknown'] if cal_error is None else []) + (['localization_error_unknown'] if loc_error is None else []))):
                blocked = issues + missing
                component = ({'status': 'unavailable', 'pixel_range_px': None, 'time_range_ms': None,
                              'source_frame_id_range': None, 'source_sample_count': 0,
                              'source_bracket_ms': None, 'reasons': blocked} if blocked else curve.project(q - error, q + error))
                components[name] = {'error_px_exact': ratio(error) if error is not None else None, **component}
            state = components['calibration_and_localization']['status']
            counts[state] += 1
            calibration_count += components['calibration']['status'] == 'bounded'
            endpoints.append({'id': point['id'], 'kind': point['kind'], 'q_px': point['q_px'],
                              'nominal_projection_status': point['status'], 'status': state,
                              'calibration_version': calibration['version'],
                              'components': components,
                              'unpropagated_evidence': {key: deepcopy(point[key]) for key in (
                                  'observed_range_px', 'endpoint_motion_error_px', 'endpoint_fusion_error_px', 'motion_error_px')}})
        output.append({'lane': lane['lane'], 'endpoints': endpoints, 'candidates': deepcopy(lane['candidates'])})
    return {'format': 'ez2rc.endpoint-time-ranges', 'schema_version': 1, 'algorithm_version': VERSION,
            **({'position_calibration': deepcopy(metadata['position_calibration'])} if 'position_calibration' in metadata else {}),
            'status': 'conditional', 'observation_cache_id': observations.id, 'motion_cache_id': movement.id,
            'decode_cache_id': projection['decode_cache_id'], 'endpoint_status_counts': counts,
            'calibration_component_bounded_count': calibration_count, 'lanes': output,
            'coordinate': 'Capture timestamp, before judgment correction and musical origin conversion.',
            'model': 'Exact inverse bounds of the closed pixel domain on the observed piecewise-linear representative scroll curve; holds include all inverse times. No extrapolation or clipping.',
            'dependence': 'Calibration plus localization uses a conservative sum with unspecified dependence. No cross-endpoint independence, shared error source, or duration cancellation is asserted.',
            'limits': 'Conditional on the representative motion curve. Motion/fusion errors and observed spread are retained but not added; unknown localization is not zero. These are not physical confidence intervals.',
            'completion': 'Candidate evidence only; no candidate selection, accepted event time, beat range, or chart completion is changed.'}
