"""Optical landmark intervals measured from retained row projections.

Threshold crossings define a reproducible optical feature. Their sensitivity
envelope is not by itself a physical confidence interval or a musical timing
calibration; the caller must declare both the landmark relation and its error.
"""
from fractions import Fraction

import numpy as np

from .config import number, table
from .contracts import exact_ms, ratio, read_ratio

VERSION = 'projection-threshold-edge-1'
REGISTRATION_VERSION = 'projection-template-registration-1'


def validate_spec(value):
    fields = {'version', 'direction', 'window_offset_px', 'thresholds',
              'reference_offset_px', 'error_px'}
    table(value, fields, fields)
    if value['version'] != VERSION or value['direction'] not in ('rising', 'falling'):
        raise ValueError('unsupported projection edge definition')
    window = value['window_offset_px']
    if (not isinstance(window, list) or len(window) != 2
            or any(type(v) is not int for v in window) or window[0] >= window[1]):
        raise ValueError('edge window requires two increasing integer offsets')
    thresholds = value['thresholds']
    if (not isinstance(thresholds, list) or not thresholds
            or any(not 0 < number(v, 'edge threshold') < 255 for v in thresholds)
            or thresholds != sorted(set(thresholds))):
        raise ValueError('edge thresholds must be increasing and inside the signal range')
    offsets = value['reference_offset_px']
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError('edge reference offset requires two bounds')
    lower, upper = [read_ratio(v) if isinstance(v, list) else exact_ms(number(v, 'edge reference offset'))
                    for v in offsets]
    if lower > upper:
        raise ValueError('unordered edge reference offset')
    error = exact_ms(number(value['error_px'], 'edge measurement error', 0))
    return lower, upper, error


def measure_edge(signal, matched_top, roi_top, spec):
    """Retain every crossing; require exactly one at every threshold.

    Windows and returned row coordinates use the aligned viewport coordinate.
    Samples lie at integer row centers. No interpolation, nearest edge choice,
    boundary clamping, or missing-threshold substitution is performed.
    """
    validate_spec(spec)
    signal = np.asarray(signal)
    if signal.ndim != 1 or not np.isfinite(signal).all():
        raise ValueError('edge signal must be a finite row projection')
    if type(matched_top) is not int or type(roi_top) is not int:
        raise ValueError('edge window requires integer image coordinates')
    lower, upper = [matched_top - roi_top + v for v in spec['window_offset_px']]
    result = {'status': 'unavailable', 'window_px': [lower+roi_top, upper+roi_top],
              'row_interval_px': None, 'threshold_crossings': [], 'reason': None}
    if not 1 <= lower < upper <= len(signal):
        result['reason'] = 'edge_window_outside_observed_rows'
        return result
    before, after = signal[lower-1:upper-1], signal[lower:upper]
    for threshold in spec['thresholds']:
        hits = ((before < threshold) & (after >= threshold) if spec['direction'] == 'rising'
                else (before >= threshold) & (after < threshold))
        rows = np.flatnonzero(hits) + lower + roi_top
        result['threshold_crossings'].append({'threshold': threshold,
            'crossings': [[int(y-1), int(y)] for y in rows]})
    if any(len(r['crossings']) != 1 for r in result['threshold_crossings']):
        result['reason'] = 'missing_or_multiple_threshold_crossings'
        return result
    result.update(status='measured', row_interval_px=[
        min(r['crossings'][0][0] for r in result['threshold_crossings']),
        max(r['crossings'][0][1] for r in result['threshold_crossings'])])
    return result


def reference_interval(measurement, spec):
    """Declared downward offset and measurement error, kept separate."""
    lower, upper, error = validate_spec(spec)
    if measurement['status'] != 'measured':
        return None
    a, b = map(Fraction, measurement['row_interval_px'])
    return a + lower - error, b + upper + error


def validate_registration(spec):
    fields = {'version', 'direction', 'window_offset_px', 'source_crossings',
              'template_sha256', 'error_px'}
    table(spec, fields, fields)
    if spec['version'] != REGISTRATION_VERSION:
        raise ValueError('unsupported projection registration')
    checksum = spec['template_sha256']
    if (not isinstance(checksum, str) or len(checksum) != 64
            or any(c not in '0123456789abcdef' for c in checksum)):
        raise ValueError('registration requires a template checksum')
    rows = spec['source_crossings']
    if not isinstance(rows, list) or not rows:
        raise ValueError('registration requires source threshold crossings')
    for row in rows:
        table(row, {'threshold', 'row_interval_px'}, {'threshold', 'row_interval_px'})
        span = row['row_interval_px']
        if (not isinstance(span, list) or len(span) != 2 or any(type(v) is not int for v in span)
                or span[1] != span[0]+1):
            raise ValueError('source crossing must bracket adjacent row centers')
    edge = {'version':VERSION, 'direction':spec['direction'], 'window_offset_px':spec['window_offset_px'],
            'thresholds':[r['threshold'] for r in rows], 'reference_offset_px':[0,0], 'error_px':0}
    validate_spec(edge)
    error = exact_ms(number(spec['error_px'], 'registration error', 0))
    return edge, error


def measure_registration(signal, matched_top, roi_top, spec):
    """Preserve threshold identity when comparing a translated template.

    Each measured crossing is paired with the same source threshold. Taking
    independent envelopes first would lose that dependence. Incompatible
    threshold translations are a conflict, not an edge to silently discard.
    Source intervals are supplied with the template identity so reconstruction
    does not need the template image. Photometric validity remains conditional.
    """
    edge, error = validate_registration(spec)
    measured = measure_edge(signal, matched_top, roi_top, edge)
    measured.update(paired_top_intervals=[], template_top_interval_px=None)
    if measured['status'] != 'measured':
        return measured
    intervals = []
    for actual, source in zip(measured['threshold_crossings'], spec['source_crossings']):
        a,b = actual['crossings'][0]
        c,d = source['row_interval_px']
        lower,upper = Fraction(a-d)-error, Fraction(b-c)+error
        intervals.append((lower,upper))
        measured['paired_top_intervals'].append({'threshold':actual['threshold'],
                                                'top_interval_px':[ratio(lower),ratio(upper)]})
    lower,upper = max(a for a,_ in intervals), min(b for _,b in intervals)
    measured['template_top_interval_px'] = [ratio(lower),ratio(upper)]
    if lower > upper:
        measured.update(status='conflict', reason='threshold_registration_inconsistent')
    return measured
