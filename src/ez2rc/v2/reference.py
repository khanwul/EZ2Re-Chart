"""Conditional cyclic references from visual markers, never absolute beat zero.

Brightness peaks and moving horizontal bands have unverified musical meaning
and clock offsets. Their sample brackets are evidence windows, not confidence
intervals. Keep the two channels separate and retain incompatible hypotheses.
"""
from collections import defaultdict
from fractions import Fraction
import math

import numpy as np

from .contracts import Timing, decimal_bpm, finite_number, ratio
from .motion import STATUS, timeline
from .timing import ScrollTimeMap

VERSION = 'visual-reference-candidates-1'


def phase_candidates(markers, period, *, exact=False):
    """Intersect closed source windows modulo a supplied period.

    Maximal compatible evidence sets survive without majority selection. Two
    markers in the same cycle cannot establish recurrence. Gaps never imply a
    count of missing beats or a measure length.
    """
    if not isinstance(period, Fraction) or period <= 0:
        raise ValueError('phase period must be a positive Fraction')
    ids = [m['id'] for m in markers]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate reference marker ID')
    starts, ends = defaultdict(set), defaultdict(set)
    windows, excluded = {}, []
    for marker in markers:
        span = marker['timestamp_interval_ms']
        if span is None:
            excluded.append({'id': marker['id'], 'reason': 'no_usable_time_window'})
            continue
        if len(span) != 2 or not all(finite_number(v) for v in span) or span[1] < span[0]:
            raise ValueError('invalid reference marker time window')
        lo, hi = map(lambda v: Fraction(str(v)), span)
        if hi - lo >= period:
            excluded.append({'id': marker['id'], 'reason': 'window_covers_full_period'})
            continue
        windows[marker['id']] = (lo, hi)
        left, right = lo % period, lo % period + hi - lo
        arcs = [(left, min(right, period))]
        if right >= period:
            arcs.append((Fraction(0), right - period))
        for a, b in arcs:
            starts[a].add(marker['id']); ends[b].add(marker['id'])
    boundaries = sorted({Fraction(0), period, *starts, *ends})
    active, cells = set(), []
    for i, x in enumerate(boundaries[:-1]):
        at = active | starts[x]
        if at:
            cells.append((x, x, True, True, frozenset(at)))
        active = at - ends[x]
        if active:
            cells.append((x, boundaries[i + 1], False, False, frozenset(active)))
    sets = sorted({c[4] for c in cells}, key=lambda s: (-len(s), sorted(s)))
    maximal = []
    for support in sets:
        if not any(support < other for other in maximal):
            maximal.append(support)
    hypotheses = []
    for support in maximal:
        spans, cycles = [], []
        for a, b, inc_a, inc_b, members in cells:
            if members != support:
                continue
            phase = (a + b) / 2
            cycles.append(len({math.ceil((windows[mid][0] - phase) / period) for mid in support}))
            if spans and spans[-1][1] == a and (spans[-1][3] or inc_a):
                spans[-1][1], spans[-1][3] = b, inc_b
            else:
                spans.append([a, b, inc_a, inc_b])
        hypothesis = {
            'support_ids': sorted(support), 'conflicting_ids': sorted(set(windows) - support),
            'phase_intervals_ms': [{'lower': float(a), 'upper': float(b),
                                    'lower_inclusive': ia, 'upper_inclusive': ib} for a, b, ia, ib in spans],
            'distinct_cycles_min': min(cycles), 'distinct_cycles_max': max(cycles),
            'status': 'recurring' if min(cycles) >= 2 else 'insufficient_recurrence'}
        if exact:
            hypothesis['phase_intervals_ms_exact'] = [
                {'lower': ratio(a), 'upper': ratio(b), 'lower_inclusive': ia, 'upper_inclusive': ib}
                for a, b, ia, ib in spans]
        hypotheses.append(hypothesis)
    hypotheses.sort(key=lambda h: (h['phase_intervals_ms'][0]['lower'], h['support_ids']))
    recurring = any(h['status'] == 'recurring' for h in hypotheses)
    return {'period_ms': ratio(period),
            'status': 'insufficient' if not recurring else 'ambiguous' if len(hypotheses) > 1 else 'supported',
            'hypotheses': hypotheses, 'excluded_markers': excluded,
            'semantics': 'Conditional phase modulo the supplied period. Zero and period are the same phase; no absolute cycle index or physical clock calibration is inferred.'}


def _beat_peaks(ids, times, brightness):
    # Lane-image repetition says nothing about the separate beat ROI. Equal
    # brightness runs form a single episode even across storage chunk edges.
    cuts = [0, *(np.flatnonzero((np.diff(ids) != 1) | (np.diff(times) <= 0)) + 1), len(ids)]
    markers = []
    for start, stop in zip(cuts, cuts[1:]):
        i = start
        while i < stop:
            j = i + 1
            while j < stop and brightness[j] == brightness[i]:
                j += 1
            left, right = i > start, j < stop
            if ((not left or brightness[i] > brightness[i - 1])
                    and (not right or brightness[i] > brightness[j])):
                reasons = ([] if left else ['missing_rising_neighbor']) + ([] if right else ['missing_falling_neighbor'])
                markers.append({
                    'id': f'beat:f{ids[i]}-f{ids[j - 1]}', 'frame_ids': ids[i:j].tolist(),
                    'peak_span_ms': [float(times[i]), float(times[j - 1])],
                    'brightness': float(brightness[i]),
                    'rise': float(brightness[i] - brightness[i - 1]) if left else None,
                    'fall': float(brightness[i] - brightness[j]) if right else None,
                    'neighbor_frame_ids': [int(ids[i - 1]) if left else None, int(ids[j]) if right else None],
                    'timestamp_interval_ms': [float(times[i - 1]), float(times[j])] if not reasons else None,
                    'reasons': reasons})
            i = j
    return markers


def _barline_tracks(rows, groups, curve, profile, tolerance):
    good = {STATUS['origin'], STATUS['observed'], STATUS['repeated']}
    tracks, previous = [], []
    for i, fid in enumerate(curve.ids):
        current = []
        for j, (y, brightness) in enumerate(rows[int(fid)]):
            current.append({'id': f'bar:f{fid}:{j}', 'frame_id': int(fid), 'timestamp_ms': float(curve.ms[i]),
                            'repeat_group': int(groups[i]), 'y_px': y, 'brightness': brightness,
                            'q_px': float(curve.scroll[i] + profile['judge_y'] - y),
                            'motion_status': int(curve.status[i])})
        linked = (i > 0 and curve.ids[i] == curve.ids[i - 1] + 1 and curve.ms[i] > curve.ms[i - 1]
                  and curve.status[i] in good and curve.status[i - 1] in good)
        edges = []
        if linked:
            for a, track in enumerate(previous):
                for b, row in enumerate(current):
                    q = [p['q_px'] for p in track['observations']]
                    if (abs(q[-1] - row['q_px']) <= tolerance
                            and max(*q, row['q_px']) - min(*q, row['q_px']) <= 2 * tolerance):
                        edges.append((a, b))
        next_tracks = []
        for b, row in enumerate(current):
            matches = [a for a, target in edges if target == b]
            if len(matches) == 1 and sum(a == matches[0] for a, _ in edges) == 1:
                track = previous[matches[0]]
                track['observations'].append(row)
            else:
                track = {'id': row['id'], 'observations': [row], 'association_conflicts': []}
                tracks.append(track)
            if matches and (len(matches) != 1 or sum(a == matches[0] for a, _ in edges) != 1):
                track['association_conflicts'].append(row['id'])
                for a in matches:
                    previous[a]['association_conflicts'].append(row['id'])
            next_tracks.append(track)
        previous = next_tracks
    for track in tracks:
        observed = track['observations']
        q = [p['q_px'] for p in observed]
        y = [p['y_px'] for p in observed]
        support = len({p['repeat_group'] for p in observed})
        reasons = []
        if support < 2:
            reasons.append('insufficient_independent_frames')
        if max(y) - min(y) <= tolerance:
            reasons.append('no_observed_motion')
        if any(p['motion_status'] not in good for p in observed):
            reasons.append('uncertain_motion')
        if track['association_conflicts']:
            reasons.append('ambiguous_barline_association')
        projection = curve.project(float(np.median(q)))
        if projection['status'] not in ('sample', 'interpolated'):
            reasons.append('projection_' + projection['status'])
        track.update(q_px=float(np.median(q)), q_range_px=[min(q), max(q)],
                     independent_frame_groups=support, projection=projection,
                     timestamp_interval_ms=projection['bracket_ms'] if not reasons else None, reasons=reasons)
    return tracks


def reference_candidates(observations, movement, bpm, *, beat_zero_ms=None, measure_beats=None):
    """Build an auditable diagnostic without modifying accepted timing inputs."""
    Timing(bpm, beat_zero_ms, measure_beats)  # Validate optional supplied values.
    if movement.manifest['key']['parents'] != [observations.id]:
        raise ValueError('reference candidate parent lineage mismatch')
    curve = ScrollTimeMap(timeline(movement))
    chunks = list(observations.chunks())
    ids, times, brightness, groups = [np.concatenate([c[k] for c in chunks]) for k in
                                    ('frame_id', 'timestamp_ms', 'beat_brightness', 'repeat_candidate_group')]
    if (not np.array_equal(ids, curve.ids) or not np.array_equal(times, curve.ms)
            or brightness.shape != ids.shape or groups.shape != ids.shape
            or groups.dtype.kind not in 'iu' or np.any(groups < 0) or np.any(groups > ids)
            or not np.isfinite(brightness).all() or np.any(brightness < 0) or np.any(brightness > 255)):
        raise ValueError('invalid reference observation timeline or signals')
    profile = observations.manifest['metadata']['profile']
    tolerance = movement.manifest['key']['parameters']['match_tolerance_px']
    if not finite_number(tolerance) or tolerance <= 0 or not finite_number(profile['judge_y']):
        raise ValueError('invalid barline geometry or association tolerance')
    rows = defaultdict(list)
    for chunk in chunks:
        bars = chunk['barlines']
        if (bars.ndim != 2 or bars.shape[1] != 3 or not np.isfinite(bars).all()
                or not np.isin(bars[:, 0], chunk['frame_id']).all()
                or np.any(bars[:, 2] < 0) or np.any(bars[:, 2] > 255)):
            raise ValueError('invalid barline observations')
        for fid, y, strength in bars:
            rows[int(fid)].append((float(y), float(strength)))
    for values in rows.values():
        values.sort()
        if len({y for y, _ in values}) != len(values):
            raise ValueError('duplicate barline position')
    beats = _beat_peaks(ids, times, brightness)
    bars = _barline_tracks(rows, groups, curve, profile, tolerance)
    period = 60000 / decimal_bpm(bpm)
    return {
        'format': 'ez2rc.reference-candidates', 'schema_version': 1, 'algorithm_version': VERSION,
        'status': 'unresolved', 'observation_cache_id': observations.id, 'motion_cache_id': movement.id,
        'supplied_reference': {'bpm': bpm, 'beat_zero_ms': beat_zero_ms,
                               'measure_beats': ratio(measure_beats) if measure_beats is not None else None},
        'inferred_beat_zero_ms': None, 'inferred_measure_beats': None,
        'reasons': ['marker_musical_meaning_unverified', 'marker_clock_offsets_unverified',
                    'absolute_cycle_index_unknown'],
        'semantics': 'Conditional visual phase candidates only. Sample brackets and search tolerances are not physical confidence bounds. Channels are not fused or used to override supplied timing.',
        'beat_signal': {'assumption': 'Each brightness maximum occurs at an integer BPM beat, with an unknown visual clock offset.',
                        'markers': beats, 'phase': phase_candidates(beats, period)},
        'barlines': {'assumption': 'Each moving band is a measure boundary, with an unknown visual clock offset.',
                     'association_tolerance_px': tolerance, 'markers': bars,
                     'phase': phase_candidates(bars, period * measure_beats) if measure_beats is not None else
                              {'status': 'unavailable', 'reason': 'measure_length_not_supplied', 'hypotheses': []}}}
