"""Provisional video-time projections and exact fixed-BPM arithmetic.

Inverting the representative scroll curve does not calibrate endpoint timing,
determine a beat origin, or turn pixel candidates into musical notes.
"""

from dataclasses import asdict, replace
from fractions import Fraction

import numpy as np

from .contracts import Chart, Event, Metadata, Note, Timing, decimal_bpm, finite_number, ratio
from .motion import STATUS, timeline

PROJECTION_VERSION = "scroll-time-projection-1"
RESOLUTION_VERSION = "explicit-timing-1"


class ScrollTimeMap:
    """Invert a nondecreasing scroll curve without extrapolating across gaps.

    Brackets are intervals between source samples, not confidence intervals.
    Error envelopes stay in pixels because their physical coverage is unknown.
    """
    def __init__(self, clock):
        self.ids, self.ms, self.scroll, self.status, self.error = (
            np.array(clock[key], copy=True) for key in
            ('frame_id', 'timestamp_ms', 'scroll_px', 'status', 'error_px'))
        arrays = (self.ids, self.ms, self.scroll, self.status, self.error)
        if (any(a.ndim != 1 or a.dtype.kind not in 'iuf' for a in arrays)
                or not len(self.ids) or any(len(a) != len(self.ids) for a in arrays)
                or any(not np.isfinite(a).all() for a in arrays)
                or self.ids.dtype.kind not in 'iu' or self.status.dtype.kind not in 'iu'
                or np.any(self.ids < 0) or np.any(self.ids[1:] <= self.ids[:-1])
                or np.any(self.ms[1:] < self.ms[:-1]) or np.any(self.scroll[1:] < self.scroll[:-1])
                or np.any(self.error < 0) or not np.isin(self.status, list(STATUS.values())).all()):
            raise ValueError('invalid scroll timeline for time projection')

    def project(self, q):
        if not finite_number(q):
            raise ValueError('pixel coordinate must be finite')
        result = {'status': 'outside_range', 'nominal_timestamp_ms': None, 'bracket_ms': None,
                  'source_frame_ids': [], 'motion_error_px': None, 'reasons': []}
        if q < self.scroll[0] or q > self.scroll[-1]:
            result['reasons'] = ['outside_observed_scroll_range']
            return result
        lo = int(np.searchsorted(self.scroll, q, side='left'))
        hi = int(np.searchsorted(self.scroll, q, side='right'))
        exact = hi > lo
        left, right = (lo, hi - 1) if exact else (lo - 1, lo)
        ids, times, states = self.ids[left:right + 1], self.ms[left:right + 1], self.status[left:right + 1]
        result.update(bracket_ms=[float(times[0]), float(times[-1])],
                      source_frame_ids=ids.tolist(), motion_error_px=float(self.error[left:right + 1].max()))
        reasons = []
        if np.any(np.diff(ids) != 1):
            reasons.append('missing_frames')
        if not np.isin(states, [STATUS['origin'], STATUS['observed'], STATUS['repeated']]).all():
            reasons.append('uncertain_motion')
        if right > left and np.any(np.diff(times) <= 0):
            reasons.append('nonincreasing_sample_times')
        if reasons:
            result.update(status='unavailable', reasons=reasons)
        elif exact and right > left:
            result.update(status='held', reasons=['same_position_at_multiple_times'])
        else:
            timestamp = float(times[0]) if exact else float(
                times[0] + (q - self.scroll[left]) / (self.scroll[right] - self.scroll[left]) * (times[-1] - times[0]))
            result.update(status='sample' if exact else 'interpolated', nominal_timestamp_ms=timestamp)
        return result


def project_video_times(observations, movement, decoded):
    """Project competitive pixel endpoints into a separate provisional artifact.

    Selection, calibrated Event times, musical origin and chart completion are
    unchanged. Reconstruction needs only the existing cache parents.
    """
    from .decode import lanes

    if (movement.manifest['key']['parents'] != [observations.id]
            or decoded.manifest['key']['parents'][:2] != [observations.id, movement.id]):
        raise ValueError('time projection parent lineage mismatch')
    curve = ScrollTimeMap(timeline(movement))
    calibrations = observations.manifest['metadata']['profile']['calibration']
    output, counts = [], {}
    for lane in lanes(decoded):
        candidates = [c for c in lane['candidates'] if c['competitive']]
        used = {c[side] for c in candidates for side in ('start', 'end') if c[side]}
        endpoints = []
        for e in lane['endpoints']:
            if e['id'] not in used:
                continue
            projected = curve.project(e['q_px']) if e['q_px'] is not None else {
                'status': 'unlocalized', 'nominal_timestamp_ms': None, 'bracket_ms': None,
                'source_frame_ids': [], 'motion_error_px': None, 'reasons': ['endpoint_has_no_usable_location']}
            counts[projected['status']] = counts.get(projected['status'], 0) + 1
            endpoints.append({'id': e['id'], 'kind': e['kind'], 'q_px': e['q_px'], **projected,
                              'calibration_status': calibrations[e['kind']]['status'],
                              'observed_range_px': e['observed_range_px'],
                              'endpoint_motion_error_px': e['shared_motion_error_px'],
                              'endpoint_fusion_error_px': e['fusion_error_px'],
                              'localization_error_px': e['localization_error_px'],
                              'calibration_error_px': e['calibration_error_px']})
        statuses = {cid: e['status'] for e in lane['events'] for cid in e['candidate_ids']}
        output.append({'lane': lane['lane'], 'endpoints': endpoints,
                       'candidates': [{**{k: c[k] for k in ('id', 'type', 'selected', 'start', 'end')},
                                       'pixel_status': statuses[c['id']]} for c in candidates]})
    return {'format': 'ez2rc.time-projection', 'schema_version': 1, 'algorithm_version': PROJECTION_VERSION,
            'status': 'provisional', 'timing_status': 'pending',
            'observation_cache_id': observations.id, 'motion_cache_id': movement.id, 'decode_cache_id': decoded.id,
            'coordinate': 'q = representative scroll at the judgment line; timestamps use the capture timeline',
            'semantics': 'Nominal projections only. Sample brackets are not calibrated confidence intervals; endpoint calibration, motion error, beat origin and meter remain separate.',
            'endpoint_status_counts': counts, 'lanes': output}


def beat_to_ms(beat: Fraction, timing: Timing) -> Fraction:
    if timing.beat_zero_ms is None:
        raise ValueError("beat origin is unresolved")
    return Fraction(str(timing.beat_zero_ms)) + beat * 60000 / decimal_bpm(timing.bpm)


def ms_to_beat(timestamp_ms: Fraction, timing: Timing) -> Fraction:
    if timing.beat_zero_ms is None:
        raise ValueError("beat origin is unresolved")
    return (timestamp_ms - Fraction(str(timing.beat_zero_ms))) * decimal_bpm(timing.bpm) / 60000


class TimingResolver:
    """Convert already accepted event times using an explicit musical reference.

    No snapping or first-note origin assumption is made. Resolved here means
    the supplied reference defines a conversion, not independently measured
    timing accuracy.
    """
    def __init__(self, beat_zero_ms=None, measure_beats=None):
        self.beat_zero_ms, self.measure_beats = beat_zero_ms, measure_beats

    def resolve(self, events: tuple[Event, ...], metadata: Metadata, bpm: str) -> Chart:
        missing = [name for name, value in (('beat origin', self.beat_zero_ms),
                                             ('measure length', self.measure_beats)) if value is None]
        reference = Timing(bpm, self.beat_zero_ms, self.measure_beats,
                           'unresolved' if missing else 'resolved',
                           ', '.join(missing) + ' unresolved' if missing else None)
        notes, unresolved = [], []
        for event in events:
            if event.status == 'resolved' and not missing:
                notes.append(Note(event.id, event.lane, event.type,
                                  ms_to_beat(Fraction(str(event.start_ms)), reference),
                                  ms_to_beat(Fraction(str(event.end_ms)), reference) if event.end_ms is not None else None))
            else:
                unresolved.append(replace(event, status='incomplete', reason=reference.reason)
                                  if event.status == 'resolved' else event)
        return Chart(metadata, reference, tuple(notes), tuple(unresolved),
                     ('timing',) if missing or unresolved else ())


def resolve_pixel_timing(events, pixels, projection, metadata, bpm, *, beat_zero_ms=None,
                         measure_beats=None, judgment_calibration=None):
    """Apply explicit corrections without promoting uncertain pixel choices.

    Offset error alone is not a full event-time uncertainty interval. Unknown
    localization/motion accuracy remains in the linked projection artifact.
    """
    if projection['format'] != 'ez2rc.time-projection' or projection['algorithm_version'] != PROJECTION_VERSION:
        raise ValueError('unsupported time projection')
    by_event = {p['id']: p for p in pixels}
    if len(by_event) != len(pixels) or set(by_event) != {e.id for e in events}:
        raise ValueError('pixel timing event identities differ')
    endpoints = {e['id']: e for lane in projection['lanes'] for e in lane['endpoints']}
    verified = judgment_calibration is not None and judgment_calibration.status == 'verified'
    accepted, rows = [], []
    for event in events:
        pixel = by_event[event.id]
        chosen = pixel['selected_intervals']
        reasons, times = [], {'start': None, 'end': None}
        if pixel['status'] != 'resolved':
            reasons.append('pixel_' + pixel['status'])
        if len(chosen) != 1:
            reasons.append('no_unique_selected_candidate')
        else:
            candidate = chosen[0]
            # A uniquely selected incomplete interval may keep its visible end.
            # A competing pixel interpretation cannot acquire a resolved time.
            for side in ('start', 'end'):
                if side == 'end' and candidate['type'] == 'tap':
                    continue
                eid = candidate[side]
                if eid is None:
                    reasons.append(side + '_missing_endpoint')
                    continue
                point = endpoints[eid]
                if point['calibration_status'] != 'verified':
                    reasons.append(side + '_endpoint_calibration_unverified')
                if point['status'] not in ('sample', 'interpolated'):
                    reasons.append(side + '_' + point['status'])
                if (verified and pixel['status'] != 'ambiguous' and point['calibration_status'] == 'verified'
                        and point['status'] in ('sample', 'interpolated')):
                    value = point['nominal_timestamp_ms'] + judgment_calibration.offset_ms
                    if not finite_number(value):
                        raise ValueError('nonfinite corrected event time')
                    times[side] = value
        if not verified:
            reasons.append('judgment_calibration_unverified')
        proposed = dict(times)
        if times['start'] is not None and times['end'] is not None and times['end'] <= times['start']:
            reasons.append('nonpositive_duration')
            times['end'] = None
        state = 'ambiguous' if pixel['status'] == 'ambiguous' else 'incomplete' if reasons else 'resolved'
        accepted.append(replace(event, status=state, start_ms=times['start'], end_ms=times['end'],
                                start_interval=None, end_interval=None,
                                reason='; '.join(reasons) if reasons else None))
        rows.append({'event_id': event.id, 'pixel_status': pixel['status'], 'time_reasons': reasons,
                     'proposed_corrected_times_ms': proposed})
    chart = TimingResolver(beat_zero_ms, measure_beats).resolve(tuple(accepted), metadata, bpm)
    if not verified and 'timing' not in chart.pending_stages:
        chart = replace(chart, pending_stages=('timing',))
    unresolved = {e.id: e for e in chart.unresolved}
    final_events = tuple(unresolved.get(e.id, e) for e in accepted)
    for row, event in zip(rows, final_events):
        row.update(status=event.status, reason=event.reason,
                   accepted_start_ms=event.start_ms, accepted_end_ms=event.end_ms)
    report = {'format': 'ez2rc.timing-resolution', 'schema_version': 1, 'algorithm_version': RESOLUTION_VERSION,
              'status': chart.status, 'projection_version': projection['algorithm_version'],
              'observation_cache_id': projection['observation_cache_id'], 'motion_cache_id': projection['motion_cache_id'],
              'decode_cache_id': projection['decode_cache_id'],
              'reference': {'bpm': bpm, 'beat_zero_ms': beat_zero_ms,
                            'measure_beats': ratio(measure_beats) if measure_beats is not None else None,
                            'status': chart.timing.status},
              'judgment_calibration': asdict(judgment_calibration) if judgment_calibration is not None else None,
              'time_rule': 'accepted time = projected capture timestamp + verified judgment offset; beat zero uses the same capture timeline',
              'uncertainty': 'Calibration error is only the offset component. Source brackets and pixel envelopes are not calibrated time confidence bounds; event intervals remain unknown.',
              'resolved_note_count': len(chart.notes), 'unresolved_event_count': len(chart.unresolved), 'events': rows}
    return chart, final_events, report
