"""Explicit, conservative adoption of an anchor origin without estimating a point.

The default policy requires a singleton; the interval policy preserves a shared
range. Marker phases can veto adoption but never narrow an interval, average
alternatives, or cancel common clock error. Supplied timing remains authoritative.
"""
from copy import deepcopy
from dataclasses import asdict
from fractions import Fraction
import math

from .config import OriginAdoption
from .contracts import origin_payload, origin_schema, ratio, read_origin, read_ratio
from .reference_link import VERSION as LINK_VERSION

VERSION = 'explicit-origin-adoption-3'


def _covers_interval(interval, windows, period):
    """Check periodic union coverage in at most one cycle, without choosing a lift.

    A closed cycle suffices even for arbitrarily wide anchors. Each shorter
    window has at most three lifts intersecting that cycle, including endpoints.
    """
    lo, hi = read_ratio(interval['lower']), read_ratio(interval['upper'])
    right = min(hi, lo + period)
    spans = []
    for window in windows:
        a, b = read_ratio(window['lower']), read_ratio(window['upper'])
        ia, ib = window['lower_inclusive'], window['upper_inclusive']
        if b - a > period or b - a == period and (ia or ib):
            return True
        first, last = math.ceil((lo - b) / period), math.floor((right - a) / period)
        for k in range(first, last + 1):
            x, y = a + k * period, b + k * period
            lower, upper = max(lo, x), min(right, y)
            il, iu = x < lo or ia, y > right or ib
            if lower < upper or lower == upper and il and iu:
                spans.append((lower, upper, il, iu))
    merged = []
    for a, b, ia, ib in sorted(spans, key=lambda s: (s[0], not s[2])):
        if merged and (a < merged[-1][1] or a == merged[-1][1] and (ia or merged[-1][3])):
            if b > merged[-1][1]:
                merged[-1][1], merged[-1][3] = b, ib
            elif b == merged[-1][1]:
                merged[-1][3] |= ib
        else:
            merged.append([a, b, ia, ib])
    return any(a == lo and b == right and ia and ib for a, b, ia, ib in merged)


def adopt_reference_origin(linked, request=None):
    """Decide an opt-in request against the complete versioned link diagnostic.

    Exactness is conditional on the caller's anchor assertions, not a physical
    accuracy claim. No configured marker channel or alternative is discarded.
    """
    if (linked['format'] != 'ez2rc.reference-origin-link'
            or linked['algorithm_version'] != LINK_VERSION):
        raise ValueError('unsupported origin link')
    if request is not None and not isinstance(request, OriginAdoption):
        raise ValueError('origin adoption requires an OriginAdoption request')
    supplied = read_origin(linked['supplied_reference'], linked['schema_version'])
    interval = linked['anchor_intersection_ms']
    result = {
        'format': 'ez2rc.origin-adoption', 'schema_version': origin_schema(supplied), 'algorithm_version': VERSION,
        'link_version': LINK_VERSION, 'status': 'not_requested',
        'request': asdict(request) if request is not None else None,
        'supplied_reference': deepcopy(linked['supplied_reference']),
        'observation_cache_id': linked['observation_cache_id'], 'motion_cache_id': linked['motion_cache_id'],
        **origin_payload(supplied, 'effective_beat_zero_ms'),
        'origin_source': 'supplied' if supplied is not None else 'unresolved',
        'candidate_origin_ms_exact': None, 'anchor_intersection_ms': deepcopy(interval),
        'anchor_ids': [a['input']['id'] for a in linked['anchors'] if a['input']['status'] == 'verified'],
        'unverified_anchor_ids': [a['input']['id'] for a in linked['anchors'] if a['input']['status'] != 'verified'],
        'channel_checks': {}, 'reasons': [],
        'semantics': 'Explicit opt-in to a closed singleton intersection of all verified anchors. Every configured marker channel must have one recurrent compatible hypothesis. Markers never narrow the anchor interval or its shared uncertainty. No midpoint, majority vote, cycle choice, or meter inference. Supplied origin takes precedence even if adoption is blocked. Acceptance does not verify physical accuracy or bypass event calibration.'}
    if request is None:
        return result
    if request.policy == 'anchor_interval':
        result['semantics'] = ('Explicit anchor interval policy. All endpoints share the same origin interval; '
                               'no midpoint is selected. Configured marker channels must cover the whole interval '
                               'without narrowing it. Supplied origin takes precedence. Ranges remain conditional '
                               'on accepted event times, not physical confidence intervals or resolved notes.')
    reasons = result['reasons']
    zero = None
    bounded = False
    if linked['anchor_status'] == 'unavailable':
        reasons.append('verified_anchor_unavailable')
    elif linked['anchor_status'] != 'compatible' or interval is None:
        reasons.append('anchor_conflict')
    else:
        lo, hi = read_ratio(interval['lower']), read_ratio(interval['upper'])
        if lo > hi or not interval['lower_inclusive'] or not interval['upper_inclusive']:
            reasons.append('anchor_interval_not_closed')
        elif lo != hi and request.policy == 'anchor_interval':
            bounded = True
        elif lo != hi:
            reasons.append('anchor_origin_is_interval')
        else:
            zero = lo
            result['candidate_origin_ms_exact'] = ratio(zero)
    if any(c['status'] == 'conflict' for c in linked['supplied_origin_comparisons']):
        reasons.append('supplied_origin_conflict')

    for name, channel in linked['channels'].items():
        issues = []
        configured = channel['calibration'] is not None
        comparisons = []
        if configured:
            if channel['status'] != 'supported' or len(channel['hypotheses']) != 1:
                issues.append('unique_recurrent_phase_unavailable')
            if 'clock_error_covers_full_period' in channel['reasons']:
                issues.append('clock_error_covers_full_period')
            for index, hypothesis in enumerate(channel['hypotheses']):
                matches = [c for c in hypothesis['origin_comparisons']
                           if c['reference']['kind'] == 'anchor_intersection']
                status = matches[0]['status'] if len(matches) == 1 else 'unavailable'
                comparisons.append({'hypothesis_index': index, 'support_ids': list(hypothesis['support_ids']),
                                    'recurrence': hypothesis['status'], 'anchor_comparison': status})
                if hypothesis['status'] != 'recurring' or status != 'compatible':
                    issues.append('phase_not_compatible_with_anchors')
                if request.policy == 'anchor_interval':
                    if supplied is not None:
                        supplied_matches = [c for c in hypothesis['origin_comparisons']
                                            if c['reference']['kind'] == 'supplied_origin']
                        supplied_status = supplied_matches[0]['status'] if len(supplied_matches) == 1 else 'unavailable'
                        comparisons[-1]['supplied_comparison'] = supplied_status
                        if supplied_status != 'compatible':
                            issues.append('phase_not_compatible_with_supplied_origin')
                    elif bounded:
                        covered = _covers_interval(interval, hypothesis['corrected_periodic_windows_ms'],
                                                   read_ratio(channel['period_ms']))
                        comparisons[-1]['whole_anchor_interval_covered'] = covered
                        if not covered:
                            issues.append('phase_does_not_cover_anchor_interval')
            reasons.extend(f'{name}:{reason}' for reason in dict.fromkeys(issues))
        result['channel_checks'][name] = {
            'required': configured, 'status': 'not_configured' if not configured else 'blocked' if issues else 'compatible',
            'link_status': channel['status'], 'link_reasons': list(channel['reasons']),
            'hypotheses': comparisons}

    value = None
    if zero is not None:
        # Retain the legacy numeric form only when the decimal round trip is
        # exact. All other singletons travel as rational milliseconds.
        try:
            value = float(zero)
        except OverflowError:
            pass
        if value is None or not math.isfinite(value) or Fraction(str(value)) != zero:
            value = zero
    if reasons:
        result['status'] = 'blocked'
    elif supplied is not None:
        result['status'] = 'confirmed_supplied'
    elif bounded:
        result.update(status='bounded', schema_version=3, origin_source='anchor_intersection',
                      effective_origin_interval_ms=deepcopy(interval))
    else:
        result.update(status='adopted', schema_version=origin_schema(value),
                      **origin_payload(value, 'effective_beat_zero_ms'), origin_source='anchor_intersection')
    return result
