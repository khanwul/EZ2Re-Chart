"""Link visual phases to supplied absolute anchors without accepting note times.

All arithmetic uses rational milliseconds. Marker calibration error is shared
by the whole channel, so it expands a phase hypothesis only after recurrence
has been checked. Source sample brackets remain evidence, not confidence bounds.
"""
from dataclasses import asdict
from fractions import Fraction
import math

from .config import validate_reference_inputs
from .contracts import Timing, decimal_bpm, exact_ms, origin_schema, ratio, read_origin, read_ratio
from .reference import VERSION as CANDIDATE_VERSION, phase_candidates

VERSION = 'anchored-reference-link-1'


def _fraction(value):
    return Fraction(str(value))


def _window(lo, hi, lower_inclusive=True, upper_inclusive=True):
    return {'lower': ratio(lo), 'upper': ratio(hi),
            'lower_inclusive': lower_inclusive, 'upper_inclusive': upper_inclusive}


def _intersecting_shifts(origin, window, period):
    """Compress all integer lifts that intersect a closed absolute interval.

    No guessed search horizon or enumeration is needed, even for wide anchors.
    A lift k means window + k * period, clipped to the origin interval.
    """
    lo, hi = origin
    a, b = read_ratio(window['lower']), read_ratio(window['upper'])
    first, last = math.ceil((lo - b) / period), math.floor((hi - a) / period)
    if b + first * period == lo and not window['upper_inclusive']:
        first += 1
    if a + last * period == hi and not window['lower_inclusive']:
        last -= 1
    return {'first': first, 'last': last} if first <= last else None


def link_reference_origin(candidates, *, marker_calibrations=None, beat_anchors=()):
    """Retain every anchor and phase, including unsupported/conflicting ones.

    Verified means asserted by the caller with a versioned basis. This function
    checks consistency; it neither authenticates that basis nor infers a meter.
    """
    calibrations = {} if marker_calibrations is None else marker_calibrations
    validate_reference_inputs(calibrations, beat_anchors)
    if (candidates['format'] != 'ez2rc.reference-candidates'
            or candidates['algorithm_version'] != CANDIDATE_VERSION):
        raise ValueError('unsupported reference candidates')
    supplied = candidates['supplied_reference']
    supplied_origin = read_origin(supplied, candidates['schema_version'])
    measure = read_ratio(supplied['measure_beats']) if supplied['measure_beats'] is not None else None
    Timing(supplied['bpm'], supplied_origin, measure)
    period = 60000 / decimal_bpm(supplied['bpm'])
    anchors, references, intervals = [], [], []
    for anchor in beat_anchors:
        row = {'input': anchor.snapshot(), 'origin_interval_ms': None,
               'reasons': [] if anchor.status == 'verified' else ['anchor_unverified']}
        if anchor.status == 'verified':
            center = _fraction(anchor.timestamp_ms) - anchor.beat * period
            error = _fraction(anchor.error_ms)
            interval = (center - error, center + error)
            row['origin_interval_ms'] = _window(*interval)
            intervals.append((anchor.id, interval))
            references.append(({'kind': 'anchor', 'id': anchor.id}, interval))
        anchors.append(row)
    common = None
    if intervals:
        lo = max(span[0] for _, span in intervals)
        hi = min(span[1] for _, span in intervals)
        if lo <= hi:
            common = (lo, hi)
    for row in anchors:
        interval = next((s for aid, s in intervals if aid == row['input']['id']), None)
        row['conflicting_anchor_ids'] = [aid for aid, span in intervals
                                        if interval is not None and max(interval[0], span[0]) > min(interval[1], span[1])]
    supplied_checks = []
    if supplied_origin is not None:
        zero = exact_ms(supplied_origin)
        references.insert(0, ({'kind': 'supplied_origin'}, (zero, zero)))
        supplied_checks = [{'anchor_id': aid, 'status': 'compatible' if lo <= zero <= hi else 'conflict'}
                           for aid, (lo, hi) in intervals]
    if common is not None:
        references.append(({'kind': 'anchor_intersection', 'anchor_ids': [aid for aid, _ in intervals]}, common))

    channels = {}
    for channel in ('beat_signal', 'barlines'):
        calibration = calibrations.get(channel)
        reasons = []
        if calibration is None or calibration.status != 'verified':
            reasons.append('marker_meaning_and_clock_unverified')
        channel_period = period if channel == 'beat_signal' else period * measure if measure is not None else None
        if channel_period is None:
            reasons.append('measure_length_not_supplied')
        result = {'calibration': asdict(calibration) if calibration is not None else None,
                  'status': 'unavailable', 'reasons': reasons, 'hypotheses': []}
        if not reasons:
            # Intersect before expanding the common offset error. Otherwise
            # contradictory marker phases could manufacture recurrent support.
            phase = phase_candidates(candidates[channel]['markers'], channel_period, exact=True)
            offset, error = _fraction(calibration.offset_ms), _fraction(calibration.error_ms)
            result.update(status=phase['status'], period_ms=ratio(channel_period),
                          excluded_markers=phase['excluded_markers'])
            if 2 * error >= channel_period:
                result['reasons'].append('clock_error_covers_full_period')
            for hypothesis in phase['hypotheses']:
                windows = [_window(read_ratio(w['lower']) + offset - error,
                                   read_ratio(w['upper']) + offset + error,
                                   w['lower_inclusive'], w['upper_inclusive'])
                           for w in hypothesis['phase_intervals_ms_exact']]
                comparisons = []
                for identity, origin in references:
                    matches = [{'window_index': index, 'integer_shifts': shifts}
                               for index, window in enumerate(windows)
                               if (shifts := _intersecting_shifts(origin, window, channel_period)) is not None]
                    comparisons.append({'reference': identity, 'origin_interval_ms': _window(*origin),
                                        'status': 'compatible' if matches else 'conflict', 'matches': matches})
                result['hypotheses'].append({
                    **{k: v for k, v in hypothesis.items() if k != 'phase_intervals_ms'},
                    'corrected_periodic_windows_ms': windows,
                    'origin_comparisons': comparisons,
                    'link_status': 'compared' if references else 'absolute_origin_unavailable'})
            if not references:
                result['reasons'].append('absolute_origin_unavailable')
            if phase['status'] == 'insufficient':
                result['reasons'].append('insufficient_recurrence')
        channels[channel] = result
    return {
        'format': 'ez2rc.reference-origin-link', 'schema_version': origin_schema(supplied_origin), 'algorithm_version': VERSION,
        'status': 'diagnostic', 'candidate_version': CANDIDATE_VERSION,
        'observation_cache_id': candidates['observation_cache_id'], 'motion_cache_id': candidates['motion_cache_id'],
        'supplied_reference': dict(supplied), 'inferred_beat_zero_ms': None, 'inferred_measure_beats': None,
        'anchors': anchors, 'anchor_status': 'unavailable' if not intervals else 'compatible' if common is not None else 'conflicting',
        'anchor_intersection_ms': _window(*common) if common is not None else None,
        'supplied_origin_comparisons': supplied_checks, 'channels': channels,
        'units': 'Every interval bound and period is an exact [numerator, denominator] in milliseconds. Input timestamps use the capture timeline.',
        'anchor_rule': 'origin = anchor timestamp - anchor beat * 60000 / BPM; expand by the supplied anchor error.',
        'marker_rule': 'musical time = observed marker time + channel offset. Integer-beat markers recur at BPM beats; measure-boundary markers recur at integer multiples of measure_beats from beat zero.',
        'lift_rule': 'Each corrected window repeats at all integer shifts times period. Matches give the inclusive range of shifts whose windows intersect the reference origin interval. Each intersection retains the window endpoint inclusivity.',
        'semantics': 'Versioned inputs are caller assertions, not independently verified truth. Offset error is common to the channel and is never reduced by recurrence. Compatibility is intersection with source evidence windows, not a physical confidence bound. Conflicting anchors and phase hypotheses remain separate; channels are not fused. Diagnostics never replace supplied timing or resolve notes.'}
