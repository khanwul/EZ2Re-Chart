"""Requested observation windows and explicitly auxiliary boundary evidence.

Scope is about visible evidence, not judgment time. A projected sprite crossing
is only a possible association with the requested window, never a new raw hit.
"""
from dataclasses import replace
import math

import numpy as np

VERSION = 'observation-window-1'
HOLD_REASON = 'requested_window_membership_uncertain'


def window_settings(start_ms, end_ms, before_ms=0., after_ms=0.):
    values = (start_ms, before_ms, after_ms)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in values):
        raise ValueError('invalid observation window')
    if end_ms is not None and (isinstance(end_ms, bool) or not isinstance(end_ms, (int, float))
                               or not math.isfinite(end_ms) or end_ms <= start_ms):
        raise ValueError('invalid observation window end')
    stop = None if end_ms is None else end_ms + after_ms
    if stop is not None and not math.isfinite(stop):
        raise ValueError('observation window overflow')
    return {'requested': {'start_ms': start_ms, 'end_ms': end_ms},
            'observation': {'start_ms': max(0., start_ms - before_ms), 'end_ms': stop},
            'context_before_ms': before_ms, 'context_after_ms': after_ms}


def song_window(song):
    return window_settings(song.start_ms, song.end_ms, song.context_before_ms, song.context_after_ms)


def has_context(window):
    return bool(window['context_before_ms'] or window['context_after_ms'])


def in_interval(timestamps, interval):
    values = np.asarray(timestamps)
    return (values >= interval['start_ms']) & (True if interval['end_ms'] is None else values < interval['end_ms'])


def validate_window(window):
    try:
        expected = window_settings(window['requested']['start_ms'], window['requested']['end_ms'],
                                   window['context_before_ms'], window['context_after_ms'])
        if window != expected:
            raise ValueError('inconsistent observation window')
    except (KeyError, TypeError) as exc:
        raise ValueError('invalid observation window declaration') from exc


def analysis_scope(observations, movement, decoded, window):
    """Retain direct members and possible boundary associations separately.

    Cache selections are never merged or changed. Context-only events stay in
    the diagnostic caches; uncertain associations remain unresolved in output.
    """
    from .decode import lanes
    from .motion import timeline
    from .observation import template_shapes

    validate_window(window)
    declared = observations.manifest['key']['parameters'].get('analysis_window')
    if declared is not None and declared != window:
        raise ValueError('analysis scope differs from the recorded observation window')
    if (movement.manifest['key']['parents'] != [observations.id]
            or decoded.manifest['key']['parents'][:2] != [observations.id, movement.id]):
        raise ValueError('analysis scope parent lineage mismatch')
    clock = timeline(movement)
    requested = in_interval(clock['timestamp_ms'], window['requested'])
    if not requested.any():
        raise ValueError('no frames in the requested analysis interval')
    if not in_interval(clock['timestamp_ms'], window['observation']).all():
        raise ValueError('observations outside declared observation window')
    roles = {int(f): bool(r) for f, r in zip(clock['frame_id'], requested)}
    raw = set()
    ids, times = [], []
    for chunk in observations.chunks():
        ids.extend(chunk['frame_id'].tolist()); times.extend(chunk['timestamp_ms'].tolist())
        raw.update(f'f{int(f)}:l{int(l)}:y{int(y)}' for f, l, y in chunk['candidate_index'])
    if ids != clock['frame_id'].tolist() or times != clock['timestamp_ms'].tolist():
        raise ValueError('analysis scope frame timeline differs from motion')
    core_ids = clock['frame_id'][requested]
    scroll = clock['scroll_px'][requested]
    profile = observations.manifest['metadata']['profile']
    shapes = template_shapes(observations.manifest['key']['parameters'])
    candidates, events = [], []
    for lane in lanes(decoded):
        endpoints = {e['id']: e for e in lane['endpoints']}
        roi = profile['lanes'][lane['lane']]['roi']
        rows = {}
        for c in lane['candidates']:
            if not c['competitive']:
                continue
            provenance = {}
            overlaps = []
            for side in ('start', 'end'):
                if c[side] is None:
                    continue
                e = endpoints[c[side]]
                if not set(e['observation_ids']) <= raw:
                    raise ValueError('analysis scope endpoint references missing observations')
                core, auxiliary = [], []
                for oid in e['observation_ids']:
                    (core if roles[int(oid.split(':', 1)[0][1:])] else auxiliary).append(oid)
                provenance[side] = {'endpoint_id': e['id'], 'requested_observation_ids': core,
                                    'auxiliary_observation_ids': auxiliary}
                if shapes is not None:
                    calibration = profile['calibration'][e['kind']]
                    offset = calibration['offset_px'] if calibration['status'] == 'verified' else 0.
                    # Undo the endpoint correction to recover template-top coordinates.
                    low, high = e['observed_range_px']
                    tops = scroll + profile['judge_y'] - high - offset
                    bottoms = scroll + profile['judge_y'] - low - offset + shapes[lane['lane']][e['kind']][0]
                    hit = (tops < roi[3]) & (bottoms > roi[1])
                    if hit.any():
                        frames = core_ids[hit]
                        overlaps.append({'side': side, 'requested_frame_bounds': [int(frames[0]), int(frames[-1])],
                                         'basis': 'representative_motion_and_template_footprint',
                                         'verified_membership': False})
            direct = any(p['requested_observation_ids'] for p in provenance.values())
            body = c['body']
            bounds = body['frame_bounds'] if body else [None, None]
            possible_body = (bounds[0] is not None and bounds[1] is not None
                             and bounds[0] <= int(core_ids[-1]) and bounds[1] >= int(core_ids[0]))
            # Body frame bounds are an envelope, not individual core witnesses.
            uncertain = bool(overlaps or possible_body or shapes is None)
            scope = 'requested' if direct else 'boundary_uncertain' if uncertain else 'auxiliary_only'
            row = {'candidate_id': c['id'], 'lane': c['lane'], 'selected': c['selected'], 'scope': scope,
                   'endpoints': provenance, 'projected_overlap': overlaps,
                   'body_frame_envelope_intersects': possible_body,
                   'template_geometry_available': shapes is not None}
            rows[c['id']] = row; candidates.append(row)
        for event in lane['events']:
            scopes = [rows[c]['scope'] for c in event['candidate_ids']]
            scope = ('requested' if 'requested' in scopes else 'boundary_uncertain'
                     if 'boundary_uncertain' in scopes else 'auxiliary_only')
            # An unrelated core alternative must not license a context-only selection.
            selected = [rows[c]['scope'] for c in event['selected_candidate_ids']]
            uncertain_selection = any(s != 'requested' for s in selected)
            events.append({'event_id': event['id'], 'scope': scope,
                           'included': scope != 'auxiliary_only',
                           'hold_resolution': scope != 'auxiliary_only' and (scope == 'boundary_uncertain' or uncertain_selection)})
    return {'format': 'ez2rc.analysis-scope', 'schema_version': 1, 'algorithm_version': VERSION,
            'window': window, 'interval_convention': 'start inclusive, end exclusive; original capture clock',
            'membership_basis': 'visible observation evidence, not note judgment time',
            'boundary_policy': 'projected overlap is uncertain; no fabricated raw associations or endpoint merging',
            'observation_cache_id': observations.id, 'motion_cache_id': movement.id, 'decode_cache_id': decoded.id,
            'observed_frame_bounds': [int(clock['frame_id'][0]), int(clock['frame_id'][-1]) + 1],
            'requested_frame_bounds': [int(core_ids[0]), int(core_ids[-1]) + 1],
            'requested_frame_count': int(requested.sum()), 'auxiliary_frame_count': int((~requested).sum()),
            'candidates': candidates, 'events': events}


def scope_events(events, pixels, scope):
    by_id = {e['event_id']: e for e in scope['events']}
    if {e.id for e in events} != set(by_id) or {p['id'] for p in pixels} != set(by_id):
        raise ValueError('analysis scope events differ from decode output')
    kept_events, kept_pixels = [], []
    for event, pixel in zip(events, pixels, strict=True):
        if event.id != pixel['id']:
            raise ValueError('analysis scope event order differs')
        row = by_id[event.id]
        if not row['included']:
            continue
        p = {**pixel, 'analysis_scope': row['scope'], 'decode_status': pixel['status'],
             'hold_resolution': row['hold_resolution']}
        if row['hold_resolution']:
            reason = 'requested-window membership is uncertain; auxiliary observations retained'
            p = {**p, 'status': 'ambiguous', 'reason': p['reason'] + '; ' + reason}
            event = replace(event, status='ambiguous', reason=event.reason + '; ' + reason)
        kept_events.append(event); kept_pixels.append(p)
    return tuple(kept_events), kept_pixels
