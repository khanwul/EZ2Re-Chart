"""First-visible-barline numbering from stored row signals and fixed meter.

Capture phase helps count cycles; it never supplies exact note beats or the
timestamp of beat zero. Moving-line identity and missing observations remain
visible hypotheses rather than being resolved by reference chart labels.
"""
from fractions import Fraction
import math

import numpy as np

from .contracts import decimal_bpm, exact_ms, ratio, read_ratio
from .motion import STATUS, timeline
from .reference import phase_candidates

VERSION = 'first-barline-grid-2'


def reconcile_cycles(hypotheses, marker_ids, measure):
    """Reconcile integer differences without requiring one capture-clock phase.

    Overlapping support sets identify the same marker in different phase
    witnesses. Require every witness difference to agree, and every crossing
    marker to be connected. Disconnected phases or inconsistent cycle counts
    remain alternatives. This does not change any timestamp tolerance.
    """
    edges = {identity:[] for identity in marker_ids}
    for h in hypotheses:
        rows = h['assignments']
        first = rows[0]
        for row in rows[1:]:
            delta = row['measure_index']-first['measure_index']
            a,b = first['marker_id'],row['marker_id']
            edges[a].append((b,delta));edges[b].append((a,-delta))
    values, components, conflicts = {},0,[]
    for identity in edges:
        if identity in values:
            continue
        components += 1
        values[identity] = 0
        stack = [identity]
        while stack:
            at = stack.pop()
            for other,delta in edges[at]:
                value = values[at]+delta
                if other in values:
                    if values[other] != value:
                        conflicts.append({'left':at,'right':other,'required_difference':delta,
                                          'existing_difference':values[other]-values[at]})
                else:
                    values[other] = value;stack.append(other)
    diagnostic = {'components':components,'conflicts':conflicts,'support_sets':len(hypotheses)}
    if not hypotheses or components!=1 or conflicts:
        return None,diagnostic
    first = min(values.values())
    numbered = sorted((n-first,identity) for identity,n in values.items())
    occupied = {n for n,_ in numbered}
    return {'first_marker_id':numbered[0][1],
            'assignments':[{'marker_id':identity,'measure_index':n,'beat':ratio(measure*n)} for n,identity in numbered],
            'missing_measure_indices':[n for n in range(numbered[-1][0]+1) if n not in occupied],
            'duplicate_cycle':len(occupied)!=len(numbered),'conflicting_marker_ids':[],
            'phase_witness_ms':None,'numbering_basis':'consistent_overlapping_phase_supports'},diagnostic


def thin_rows(projection, top, bottom, parameters):
    signal = np.min(projection.astype(float), axis=0)[:bottom-top]
    width = parameters['barline_max_thickness']
    radius = width+1
    if len(signal) <= radius:
        return []
    baseline = np.maximum(np.r_[np.repeat(signal[0],radius),signal[:-radius]],
                          np.r_[signal[radius:],np.repeat(signal[-1],radius)])
    possible = ((signal >= parameters['barline_min_brightness'])
                & (signal <= parameters['barline_max_brightness'])
                & (signal-baseline >= parameters['edge_threshold']))
    cuts = np.flatnonzero(np.diff(np.r_[False,possible,False].astype(int)))
    rows = []
    for start, stop in zip(cuts[::2],cuts[1::2]):
        if stop-start > width or start == 0 or stop == len(signal):
            continue
        peak = start+int(np.argmax(signal[start:stop]))
        rows.append({'y_px': float(top+peak), 'visible_range_px': [top+start-.5,top+stop-.5]})
    return rows


def number_barlines(markers, period, measure):
    """Retain distinct relative cycle assignments across all compatible phases."""
    phase = phase_candidates(markers,period,exact=True)
    by_id = {m['id']:m for m in markers}
    hypotheses, seen = [], set()
    for group in phase['hypotheses']:
        if group['status'] != 'recurring':
            continue
        for span in group['phase_intervals_ms_exact']:
            lo, hi = read_ratio(span['lower']),read_ratio(span['upper'])
            probes = {(lo+hi)/2}
            if span['lower_inclusive']: probes.add(lo)
            if span['upper_inclusive']: probes.add(hi)
            for at in sorted(probes):
                cycles = []
                for identity in group['support_ids']:
                    a,b = map(exact_ms,by_id[identity]['timestamp_interval_ms'])
                    cycle = math.ceil((a-at)/period)
                    if not a <= at+cycle*period <= b:
                        break
                    cycles.append((cycle,identity))
                else:
                    cycles.sort()
                    first = cycles[0][0]
                    normalized = tuple((identity,n-first) for n,identity in cycles)
                    key = (normalized,tuple(group['conflicting_ids']))
                    if key in seen:
                        continue
                    seen.add(key)
                    occupied = {n for n,_ in cycles}
                    hypotheses.append({'first_marker_id':cycles[0][1],
                        'assignments':[{'marker_id':identity,'measure_index':n,'beat':ratio(measure*n)}
                                       for identity,n in normalized],
                        'missing_measure_indices':[n-first for n in range(first,cycles[-1][0]+1) if n not in occupied],
                        'duplicate_cycle':len(occupied)!=len(cycles),
                        'conflicting_marker_ids':group['conflicting_ids'],
                        'phase_witness_ms':ratio(at)})
    consensus,diagnostic = reconcile_cycles(hypotheses,by_id,measure)
    phase['cycle_reconciliation'] = diagnostic
    if consensus is not None and len(hypotheses)>1:
        phase['numbering_alternatives'] = hypotheses
        hypotheses = [consensus]
    return phase,hypotheses


def infer_barline_grid(observations, movement, bpm, measure_beats):
    result = {'format':'ez2rc.barline-grid','schema_version':1,'algorithm_version':VERSION,
              'observation_cache_id':observations.id,'motion_cache_id':movement.id,
              'bpm':bpm,'measure_beats':ratio(measure_beats) if measure_beats else None,
              'coordinate':'first_barline_zero','status':'unavailable','accepted':False,
              'tracks':[],'hypotheses':[],'reasons':[]}
    period = 60000/decimal_bpm(bpm)
    if measure_beats is None:
        result['reasons'] = ['measure_length_not_supplied']
        return result
    if not isinstance(measure_beats,Fraction) or measure_beats <= 0:
        raise ValueError('invalid measure length')
    if movement.manifest['key']['parents'] != [observations.id]:
        raise ValueError('barline inference parent lineage mismatch')
    period *= measure_beats
    profile = observations.manifest['metadata']['profile']
    parameters = observations.manifest['key']['parameters']['algorithm']
    motion_parameters = movement.manifest['key']['parameters']
    clock = timeline(movement)
    clock_index = {int(f):i for i,f in enumerate(clock['frame_id'])}
    top, bottom = profile['lanes'][0]['roi'][1],int(profile['effect_top'])
    center, width = (top+bottom)/2,parameters['barline_max_thickness']
    previous, previous_frame, first_detection = [],None,None
    all_ids = []
    for chunk in observations.chunks():
        for j,frame in enumerate(chunk['frame_id']):
            frame = int(frame); i = clock_index[frame]
            timestamp, scroll = float(clock['timestamp_ms'][i]),float(clock['scroll_px'][i])
            if timestamp != float(chunk['timestamp_ms'][j]):
                raise ValueError('barline observation and motion timelines differ')
            all_ids.append(frame)
            rows = thin_rows(chunk['projection'][j],top,bottom,parameters)
            if rows and first_detection is None:
                first_detection = frame
            for row in rows:
                row.update(frame_id=frame,timestamp_ms=timestamp,q_px=scroll+profile['judge_y']-row['y_px'])
            edges = []
            if previous_frame is not None and frame == previous_frame+1:
                for a,track in enumerate(previous):
                    old = track['observations'][-1]
                    dt = timestamp-old['timestamp_ms']
                    for b,row in enumerate(rows):
                        supported = {STATUS['observed'],STATUS['repeated']}
                        if int(clock['status'][i]) in supported and int(clock['status'][i-1]) in supported:
                            matches = abs(row['q_px']-old['q_px']) <= 2*motion_parameters['match_tolerance_px']
                        else:
                            matches = 0 <= row['y_px']-old['y_px'] <= motion_parameters['max_speed_px_per_ms']*dt
                        if matches:
                            edges.append((a,b))
            current = []
            for b,row in enumerate(rows):
                matches = [a for a,k in edges if k==b]
                if len(matches)==1 and sum(a==matches[0] for a,_ in edges)==1:
                    track = previous[matches[0]]
                else:
                    track = {'id':f'barline:{len(result["tracks"])}','observations':[],'association_conflict':bool(matches)}
                    result['tracks'].append(track)
                    for a in matches:
                        previous[a]['association_conflict'] = True
                track['observations'].append(row)
                current.append(track)
            previous,previous_frame = current,frame
    if all_ids != list(map(int,clock['frame_id'])):
        raise ValueError('barline frame coverage differs from motion')
    markers = []
    for track in result['tracks']:
        rows = track['observations']
        below = [i for i,r in enumerate(rows) if r['visible_range_px'][1] < center]
        above = [i for i,r in enumerate(rows) if r['visible_range_px'][0] > center]
        crossing = [(rows[max(below)],rows[min(above)])] if below and above and max(below)<min(above) else []
        moving = max(r['y_px'] for r in rows)-min(r['y_px'] for r in rows)>width
        track['reasons'] = ([] if moving else ['no_supported_motion'])
        if len(crossing)!=1: track['reasons'].append('crossing_not_unique_or_unobserved')
        if track['association_conflict']: track['reasons'].append('association_conflict')
        if not track['reasons']:
            a,b = crossing[0]
            markers.append({'id':track['id'],'timestamp_interval_ms':[a['timestamp_ms'],b['timestamp_ms']]})
    phase,hypotheses = number_barlines(markers,period,measure_beats)
    tracks = {t['id']:t for t in result['tracks']}
    contiguous = bool(all_ids) and all_ids == list(range(all_ids[0],all_ids[-1]+1))
    for hypothesis in hypotheses:
        first = tracks[hypothesis['first_marker_id']]['observations'][0]
        index = clock_index[first['frame_id']]
        dt = float(clock['timestamp_ms'][index]-clock['timestamp_ms'][index-1]) if index else 0.
        leading = (contiguous and all_ids[0]==0 and index>0 and first['frame_id']==first_detection
                   and first['y_px']-top <= motion_parameters['max_speed_px_per_ms']*dt+width)
        hypothesis['leading_coverage'] = leading
        hypothesis['reasons'] = ([] if leading else ['first_line_leading_coverage_unestablished'])
        if hypothesis['conflicting_marker_ids']: hypothesis['reasons'].append('conflicting_marker_cycles')
        if hypothesis['duplicate_cycle']: hypothesis['reasons'].append('multiple_lines_in_one_cycle')
    result.update(phase=phase,hypotheses=hypotheses,crossing_markers=markers,screen_reference_y=center,
                  first_detection_frame_id=first_detection,
                  status='supported' if len(hypotheses)==1 and not hypotheses[0]['reasons'] else 'ambiguous' if hypotheses else 'unavailable',
                  reasons=[] if hypotheses else ['insufficient_recurring_barlines'],
                  semantics='Narrow common-row tracks and fixed-period cycle hypotheses. First-line numbering is relative to the first supported entering line. Phase is a cycle-count witness at a screen row, not a judgment timestamp. Detector completeness and physical calibration are not certified by recurrence.')
    return result


def spatial_reference(result, calibration):
    """Connect one supported numbering hypothesis to scoped visual calibration."""
    from .observed_geometry import VERSION as GEOMETRY_VERSION
    from .relative_beats import validate_reference
    if result['status'] != 'supported' or len(result['hypotheses']) != 1:
        return None
    tracks = {t['id']:t for t in result['tracks']}
    anchors, support = [], {}
    for assignment in result['hypotheses'][0]['assignments']:
        identity = assignment['marker_id']
        rows = tracks[identity]['observations']
        q = rows[0]['q_px']
        anchors.append({'id':identity,'beat':assignment['beat'],'q_px':q,'q_range_px':[q,q]})
        support[identity] = [{'frame_id':r['frame_id'],'y_px':r['y_px']} for r in rows]
    reference = {'format':'ez2rc.spatial-beat-reference','schema_version':1,
                 'version':VERSION,'origin':'first_barline_zero','measure_beats':result['measure_beats'],
                 'extrapolation':'conditional_linear', 'anchors':anchors,
                 'observation_cache_id':calibration['observation_cache_id'],
                 'motion_cache_id':calibration['motion_cache_id'],'decode_cache_id':calibration['decode_cache_id'],
                 'endpoint_calibrations':calibration['endpoint_calibrations'],
                 'observed_geometry':{'version':GEOMETRY_VERSION,'localization_error_px':calibration['localization_error_px'],
                     'scale':'constant','endpoint_position_relation':'same_raw_localization','anchor_observations':support}}
    if any(reference[k] != result[k] for k in ('observation_cache_id','motion_cache_id')):
        raise ValueError('automatic calibration parent lineage mismatch')
    validate_reference(reference)
    return reference
