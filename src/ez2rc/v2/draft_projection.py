"""Editable screen estimates using the existing shared observation geometry.

No musical grid is imposed on screen endpoints. The midpoint of a conditional
coordinate range is labeled as an estimate and the full range is retained.
"""
from collections import defaultdict

from .barline_grid import spatial_reference
from .contracts import exact_ms, ratio, read_ratio
from .decode import KINDS, lanes
from .motion import STATUS, timeline
from .observation import template_shapes
from .relative_beats import project_relative_beats
from .window import HOLD_REASON

VERSION = 'screen-coordinate-draft-6'


def draft_timing_issues(draft):
    projection = draft['projection']
    issues = []
    if 'barline_beat_spacing_ambiguous' in projection['origin_issues']:
        issues.append('The recording does not establish unique beat positions for the measure lines.')
    elif projection['origin_issues']:
        issues.append('The first barline or measure numbering could not be established.')
    if projection['geometry_issues']:
        issues.append('Template geometry is unavailable for placing notes.')
    for left, right in projection.get('uncertain_measure_spans', []):
        issues.append(f'Check for missing measure lines between beats {read_ratio(left)} and {read_ratio(right)}.')
    if projection.get('unplaced_marker_ids'):
        issues.append(f"Measure lines without a unique beat position: {len(projection['unplaced_marker_ids'])}.")
    notes = draft['notes']
    if not notes:
        issues.append('No notes were detected in the requested interval.')
    missing = sum(n['start_beat'] is None or (n['type'] == 'longnote' and n['end_beat'] is None)
                  for n in notes)
    if missing:
        issues.append(f'Notes with missing beat positions: {missing}.')
    return issues


def draft_endpoint_offset(profile, kind, template_height=None):
    """Pixel offset from the matched top row to the draft's timing landmark."""
    overrides = profile.get('draft_offsets_px', {})
    if kind in overrides:
        return overrides[kind]
    calibration = profile['calibration'][kind]
    if calibration['status'] == 'verified':
        return calibration['offset_px']
    if kind in ('tap', 'head') and template_height is not None:
        return template_height - 1
    return 0


def project_draft(observations, movement, decoded, barlines, *, allowed_candidates=None, analysis_scope=None):
    if (movement.manifest['key']['parents'] != [observations.id]
            or decoded.manifest['key']['parents'][:2] != [observations.id, movement.id]
            or barlines['observation_cache_id'] != observations.id
            or barlines['motion_cache_id'] != movement.id):
        raise ValueError('draft projection parent mismatch')
    scoped = {r['event_id']: r for r in analysis_scope['events']} if analysis_scope is not None else {}
    if analysis_scope is not None and any(analysis_scope[k] != cache.id for k, cache in (
            ('observation_cache_id', observations), ('motion_cache_id', movement), ('decode_cache_id', decoded))):
        raise ValueError('draft analysis scope parent mismatch')
    profile = observations.manifest['metadata']['profile']
    shapes = template_shapes(observations.manifest['key']['parameters'])
    clock = timeline(movement)
    times = dict(zip(map(int, clock['frame_id']), map(float, clock['timestamp_ms'])))
    # This is a declared draft assumption, not a verified localization bound.
    error = profile.get('draft_localization_error_px', 1)
    calibration = {'format': 'ez2rc.spatial-calibration', 'schema_version': 1, 'version': VERSION,
                   'observation_cache_id': observations.id, 'motion_cache_id': movement.id,
                   'decode_cache_id': decoded.id, 'localization_error_px': error,
                   'endpoint_calibrations': []}
    requests, candidates, endpoint_data = defaultdict(list), [], {}
    for lane in lanes(decoded):
        eps = {e['id']: e for e in lane['endpoints']}
        events = {cid: e for e in lane['events'] for cid in e['candidate_ids']}
        for candidate in lane['candidates']:
            if not candidate['competitive']:
                continue
            event = events[candidate['id']]
            candidates.append({**candidate, 'lane': lane['lane'], 'event_id': event['id'],
                               'pixel_status': event['status'], 'pixel_reason': event['reason']})
            for eid in (candidate['start'], candidate['end']):
                if eid is None:
                    continue
                endpoint = eps[eid]
                kind = endpoint['kind']
                height = shapes[lane['lane']][kind][0] if shapes is not None else None
                offset = draft_endpoint_offset(profile, kind, height)
                if candidate['selected'] and endpoint['q_px'] is not None:
                    calibration['endpoint_calibrations'].append({
                        'endpoint_id': eid, 'candidate_id': candidate['id'], 'version': VERSION,
                        'offset_px': [offset, offset], 'position_error_px': error})
                if eid not in endpoint_data:
                    endpoint_data[eid] = {'offset_px': ratio(exact_ms(offset)), 'witnesses': []}
                    for oid in endpoint['observation_ids']:
                        frame, lane_id, seed = [int(v[1:]) for v in oid.split(':')]
                        requests[(frame, lane_id, seed, KINDS.index(kind))].append(eid)
    reference = spatial_reference(barlines, calibration)
    relative = (project_relative_beats(observations, movement, decoded, reference,
                                      measure_beats=(read_ratio(reference['measure_beats'])
                                                     if reference['measure_beats'] else None), draft_tap_anchors=True)
                if reference is not None and shapes is not None else None)
    relative_rows = {r['candidate_id']: r for r in relative['candidates']} if relative else {}
    origin_issues = list(barlines['reasons'])
    if reference is None:
        origin_issues.append('barline_origin_unavailable_or_ambiguous')
    geometry_issues = ['template_geometry_unavailable'] if shapes is None else []
    for chunk in observations.chunks():
        for i, (frame, lane, seed) in enumerate(chunk['candidate_index']):
            frame, lane, seed = int(frame), int(lane), int(seed)
            for kind in range(len(KINDS)):
                for eid in requests.get((frame, lane, seed, kind), ()):
                    endpoint_data[eid]['witnesses'].append({
                        'frame_id': frame, 'timestamp_ms': times[frame],
                        'y_px': float(chunk['positions'][i, kind, 1]),
                        **({'box_px': [int(chunk['positions'][i, kind, 0]), int(chunk['positions'][i, kind, 1]),
                                       *reversed(shapes[lane][KINDS[kind]])]} if shapes is not None else {}),
                        'observation_id': f'f{frame}:l{lane}:y{seed}',
                        'usable': not bool(int(chunk['flags'][i, kind]) & 3)})
    for data in endpoint_data.values():
        witnesses = sorted(data['witnesses'], key=lambda w: (not w['usable'], w['frame_id']))
        data['witness_count'] = len(witnesses)
        data['witnesses'] = ([witnesses[i] for i in sorted({0, len(witnesses)//2, len(witnesses)-1})]
                             if witnesses else [])
    notes, alternatives = [], []
    for candidate in candidates:
        if allowed_candidates is not None and candidate['id'] not in allowed_candidates:
            continue
        scope = scoped[candidate['event_id']] if analysis_scope is not None else None
        if scope is not None and not scope['included']:
            continue
        source = relative_rows.get(candidate['id'])
        evidence = {'event_id': candidate['event_id'], 'pixel_status': candidate['pixel_status'],
                    'pixel_reason': candidate['pixel_reason']}
        values, reasons = {}, [*origin_issues, *geometry_issues]
        if scope is not None:
            evidence.update(analysis_scope=scope['scope'], hold_resolution=scope['hold_resolution'])
            if scope['hold_resolution']:
                reasons.append(HOLD_REASON)
        for side in ('start', 'end'):
            point = source['endpoints'].get(side) if source else None
            domain = point['relative_beat']['beat_range'] if point and point['relative_beat'] else None
            values[side] = ratio((read_ratio(domain['lower'])+read_ratio(domain['upper']))/2) if domain else None
            raw = endpoint_data.get(candidate[side])
            evidence[side] = ({**raw, 'beat_range': domain, 'estimate': values[side],
                               'reasons': point['reasons'] if point else ['no_selected_coordinate_solution']}
                              if raw else None)
        if candidate['pixel_status'] != 'resolved':
            reasons.append('pixel_' + candidate['pixel_status'])
        if values['start'] is None:
            reasons.append('start_unavailable')
        if candidate['type'] == 'longnote':
            if values['end'] is None:
                reasons.append('end_unavailable')
            elif values['start'] is not None and read_ratio(values['end']) <= read_ratio(values['start']):
                values['end'] = None
                reasons.append('nonpositive_measured_duration')
        row = {'id': candidate['id'], 'lane': candidate['lane'], 'type': candidate['type'],
               'start_beat': values['start'], 'end_beat': values['end'],
               'status': 'unresolved' if reasons else 'draft', 'issues': sorted(set(reasons)),
               'evidence': evidence, 'origin': 'automatic'}
        (notes if candidate['selected'] else alternatives).append(row)
    gaps, current = [], None
    for fid, ms, state in zip(clock['frame_id'], clock['timestamp_ms'], clock['status']):
        if state not in (STATUS['origin'], STATUS['observed'], STATUS['repeated']):
            if current is None:
                current = {'first_frame': int(fid), 'start_ms': float(ms)}
                gaps.append(current)
            current.update(last_frame=int(fid), end_ms=float(ms))
        else:
            current = None
    geometry = relative.get('observed_geometry', {}) if relative else {}
    scale = geometry.get('scale_bounds_px_per_beat')
    return {'algorithm_version': VERSION, 'notes': notes, 'alternatives': alternatives,
            'default_start_reference': 'template_bottom',
            'measure_beats': reference['measure_beats'] if reference is not None else None,
            **({'barline_beats': barlines['barline_beats'] if reference else [],
                'unplaced_marker_ids': barlines['unplaced_marker_ids'],
                'uncertain_measure_spans': barlines['uncertain_measure_spans']} if 'barline_beats' in barlines else {}),
            'coordinate': 'first_numbered_barline_zero', 'origin_issues': sorted(set(origin_issues)),
            'scale_px_per_beat': scale, 'scale_basis': 'shared_observation_geometry',
            'geometry_status': geometry.get('status', 'unavailable'),
            'draft_tap_anchors': geometry.get('draft_tap_anchors', []),
            'geometry_issues': geometry_issues,
            'localization_assumption_px': error, 'unprocessed_motion': gaps, 'tail_beat_extension': [0, 1],
            'limitations': ['All automatic values need video review; missing notes may be undetected.',
                           'Screen landmarks are not verified musical timing.',
                           'Beats are unsnapped midpoints of conditional ranges, not exact musical values.',
                           'The localization allowance is a draft assumption, not a certified error bound.']}
