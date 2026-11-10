"""Conditional beat ranges with explicitly shared timing error terms."""
from copy import deepcopy

from .config import JudgmentCalibration, RangeUncertainty
from .contracts import decimal_bpm, exact_ms, ratio, read_origin, read_ratio
from .origin import VERSION as ORIGIN_VERSION
from .timing import RESOLUTION_VERSION

VERSION = 'shared-origin-beat-ranges-1'
UNCERTAINTY_VERSION = 'shared-timing-beat-ranges-2'


def _closed(lo, hi):
    return {'lower': ratio(lo), 'upper': ratio(hi),
            'lower_inclusive': True, 'upper_inclusive': True}


def project_beat_ranges(events, resolution, adoption, *, uncertainty_model=None,
                       endpoint_ranges=None, pixels=None, projection=None):
    """Preserve common origin uncertainty without changing chart or event states.

    Times must already have passed pixel, endpoint, and judgment checks. These
    Default bounds include origin uncertainty conditional on accepted times.
    An explicit model can also include shared judgment calibration error.
    Endpoint bounds cannot be combined as independent errors.
    """
    if uncertainty_model is not None and not isinstance(uncertainty_model, RangeUncertainty):
        raise ValueError('range uncertainty requires a RangeUncertainty declaration')
    if (resolution['format'] != 'ez2rc.timing-resolution'
            or resolution['algorithm_version'] != RESOLUTION_VERSION
            or adoption['format'] != 'ez2rc.origin-adoption'
            or adoption['algorithm_version'] != ORIGIN_VERSION):
        raise ValueError('unsupported beat range parents')
    for key in ('observation_cache_id', 'motion_cache_id'):
        if resolution[key] != adoption[key]:
            raise ValueError('beat range parent lineage mismatch')
    reference = resolution['reference']
    resolved_origin = read_origin(reference, resolution['schema_version'])
    if any(reference[key] != adoption['supplied_reference'][key] for key in ('bpm', 'measure_beats')):
        raise ValueError('beat range musical reference mismatch')
    rows = {row['event_id']: row for row in resolution['events']}
    if len(rows) != len(resolution['events']) or len({e.id for e in events}) != len(events) or set(rows) != {e.id for e in events}:
        raise ValueError('beat range event identities differ')
    reasons, shared = [], None
    if adoption['status'] != 'bounded':
        # Also validate the scalar representation of inactive decisions.
        if read_origin(adoption, adoption['schema_version'], 'effective_beat_zero_ms') != resolved_origin:
            raise ValueError('beat range effective origin mismatch')
        reasons.append('bounded_origin_not_selected')
    else:
        if (type(adoption['schema_version']) is not int or adoption['schema_version'] != 3
                or adoption['effective_beat_zero_ms'] is not None
                or 'effective_beat_zero_ms_exact' in adoption
                or resolved_origin is not None or adoption['origin_source'] != 'anchor_intersection'
                or read_origin(adoption['supplied_reference'], 2) is not None
                or adoption['request']['policy'] != 'anchor_interval'
                or adoption['reasons']):
            raise ValueError('invalid bounded origin decision')
        interval = adoption['effective_origin_interval_ms']
        lo, hi = read_ratio(interval['lower']), read_ratio(interval['upper'])
        if (lo >= hi or interval['lower_inclusive'] is not True or interval['upper_inclusive'] is not True
                or interval != adoption['anchor_intersection_ms']):
            raise ValueError('bounded origin requires an unchanged closed interval of positive width')
        scale = decimal_bpm(reference['bpm']) / 60000
        shared = {'id': 'shared-origin', 'interval_ms': deepcopy(interval),
                  'shift_beats': _closed(-hi * scale, -lo * scale),
                  'anchor_ids': list(adoption['anchor_ids'])}
    if reference['measure_beats'] is None:
        reasons.append('measure_length_not_supplied')
    elif read_ratio(reference['measure_beats']) <= 0:
        raise ValueError('invalid beat range measure length')
    calibration = resolution['judgment_calibration']
    if calibration is None or calibration['status'] != 'verified':
        reasons.append('judgment_calibration_unverified')
    output = []
    for event in events:
        source = rows[event.id]
        if (source['accepted_start_ms'] != event.start_ms or source['accepted_end_ms'] != event.end_ms
                or source['status'] != event.status):
            raise ValueError('beat range event disagrees with timing decision')
        issues = [*reasons, *source['time_reasons']]
        if source['pixel_status'] != 'resolved':
            issues.append('pixel_choice_not_resolved')
        if event.status == 'ambiguous':
            issues.append('event_ambiguous')
        if (event.type not in ('tap', 'longnote') or event.start_ms is None
                or event.type == 'longnote' and event.end_ms is None):
            issues.append('complete_event_time_unavailable')
        row = {'event_id': event.id, 'lane': event.lane, 'type': event.type,
               'candidate_ids': list(event.candidate_ids), 'status': 'unavailable' if issues else 'bounded',
               'reasons': list(dict.fromkeys(issues)), 'shared_origin_id': None,
               'start_time_term_beats': None, 'end_time_term_beats': None,
               'start_beat_range': None, 'end_beat_range': None, 'duration_beats': None}
        if not issues:
            row['shared_origin_id'] = shared['id']
            for side, timestamp in (('start', event.start_ms), ('end', event.end_ms)):
                if timestamp is None:
                    continue
                term = exact_ms(timestamp) * scale
                row[side + '_time_term_beats'] = ratio(term)
                row[side + '_beat_range'] = _closed(term - hi * scale, term - lo * scale)
            if event.end_ms is not None:
                row['duration_beats'] = ratio((exact_ms(event.end_ms) - exact_ms(event.start_ms)) * scale)
        output.append(row)
    count = sum(row['status'] == 'bounded' for row in output)
    result = {
        'format': 'ez2rc.beat-ranges', 'schema_version': 1, 'algorithm_version': VERSION,
        'status': 'unavailable' if reasons else 'available', 'reasons': reasons,
        'bpm': reference['bpm'], 'measure_beats': deepcopy(reference['measure_beats']),
        'observation_cache_id': resolution['observation_cache_id'],
        'motion_cache_id': resolution['motion_cache_id'], 'decode_cache_id': resolution['decode_cache_id'],
        'origin_adoption_status': adoption['status'], 'shared_origin': shared,
        'bounded_event_count': count, 'unavailable_event_count': len(output) - count, 'events': output,
        'equation': 'beat = time_term_beats + shared_origin.shift_beats; choose the SAME shift for every endpoint in this artifact.',
        'uncertainty': 'Only shared origin uncertainty is propagated, conditional on accepted event times and supplied fixed BPM. Other timing and calibration errors are not included; these are not physical confidence intervals.',
        'duration_rule': 'The shared origin cancels in end minus start, and in any difference of accepted endpoint times. duration_beats is conditional on those times, not an independently verified physical duration.',
        'completion': 'Bounded events remain unresolved in chart.json. No midpoint or independent endpoint choice is permitted; this artifact never adds resolved notes.'}
    if uncertainty_model is not None:
        _apply_judgment_uncertainty(result, calibration, uncertainty_model)
    endpoint_requested = uncertainty_model is not None and uncertainty_model.endpoint_component is not None
    if endpoint_requested:
        from .endpoint_beats import apply_endpoint_ranges
        apply_endpoint_ranges(result, events, resolution, pixels, projection, endpoint_ranges)
    elif any(value is not None for value in (endpoint_ranges, pixels, projection)):
        raise ValueError('endpoint inputs require an explicit endpoint component request')
    return result


def _apply_judgment_uncertainty(result, calibration, model):
    """Add an explicitly common offset error, without assuming independence.

    Unknown relation gives an outer enclosure of every admissible coupling.
    Same-clock cancellation requires an explicit reference and equal bounds;
    inconsistent declarations cannot silently narrow either source interval.
    """
    reasons, judgment, combined = [], None, None
    scale = decimal_bpm(result['bpm']) / 60000
    if result['shared_origin'] is None:
        reasons.append('bounded_origin_not_selected')
    if calibration is None or calibration['status'] != 'verified':
        reasons.append('judgment_calibration_unverified')
    else:
        declared = JudgmentCalibration(**calibration)
        error = exact_ms(declared.error_ms)
        judgment = {'id': 'shared-judgment', 'scope': model.judgment_scope,
                    'calibration_version': declared.version, 'delta_ms': _closed(-error, error),
                    'shift_beats': _closed(-error * scale, error * scale)}
    if not reasons:
        interval = result['shared_origin']['interval_ms']
        lo, hi = read_ratio(interval['lower']), read_ratio(interval['upper'])
        if model.origin_relation == 'unknown':
            combined = _closed((-hi - error) * scale, (-lo + error) * scale)
        elif (lo, hi) != (model.origin_reference_ms - error, model.origin_reference_ms + error):
            reasons.append('shared_clock_bounds_mismatch')
        else:
            # O = reference + delta and judgment correction = nominal + delta.
            # The same delta cancels; no origin within its interval is selected.
            shift = -model.origin_reference_ms * scale
            combined = _closed(shift, shift)
    result.update(schema_version=2, algorithm_version=UNCERTAINTY_VERSION,
                  uncertainty_model={'request': model.snapshot(), 'status': 'blocked' if reasons else 'applied',
                                     'reasons': reasons, 'bounds_kind': 'outer_enclosure' if model.origin_relation == 'unknown' else 'declared_same_clock',
                                     'relation': 'unspecified dependence; no independence assumed' if model.origin_relation == 'unknown' else
                                                 'origin_ms = origin_reference_ms_exact + shared_judgment.delta_ms'},
                  shared_judgment=judgment, combined_shared_shift_beats=combined,
                  equation='beat = time_term_beats + shared_origin.shift_beats + shared_judgment.shift_beats; use the SAME source values for all endpoints and obey uncertainty_model.relation.',
                  uncertainty='Only origin and explicitly shared judgment offset errors are propagated. Unknown dependence uses a conservative outer enclosure, not a probability distribution. Other timing and localization errors remain outside these conditional bounds.',
                  duration_rule='Both shared terms cancel in differences of accepted endpoint times, regardless of their mutual dependence. Durations remain conditional on all unmodeled timing errors.')
    result['reasons'] = list(dict.fromkeys([*result['reasons'], *reasons]))
    if reasons:
        result['status'] = 'unavailable'
    for row in result['events']:
        row['shared_judgment_id'] = None
        for side in ('start', 'end'):
            row[side + '_origin_only_beat_range'] = deepcopy(row[side + '_beat_range'])
        if reasons:
            row.update(status='unavailable', start_beat_range=None, end_beat_range=None, duration_beats=None,
                       reasons=list(dict.fromkeys([*row['reasons'], *reasons])))
        elif row['status'] == 'bounded':
            row['shared_judgment_id'] = judgment['id']
            for side in ('start', 'end'):
                term = row[side + '_time_term_beats']
                if term is not None:
                    value = read_ratio(term)
                    row[side + '_beat_range'] = _closed(value + read_ratio(combined['lower']),
                                                       value + read_ratio(combined['upper']))
    result['bounded_event_count'] = sum(row['status'] == 'bounded' for row in result['events'])
    result['unavailable_event_count'] = len(result['events']) - result['bounded_event_count']
