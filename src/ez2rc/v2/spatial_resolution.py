"""Confirm musical events only within an existing verified profile's scope.

Algebraic uniqueness never verifies an empirical calibration. Scoped
conditional overrides must enclose the verified profile convention, and every
accepted label must participate in the joint geometry/grid feasibility proof.
Capture timestamps remain in their separate diagnostic artifacts.
"""
from collections import defaultdict
from dataclasses import replace

from .contracts import Chart, Note, Timing, exact_ms, read_ratio
from .relative_beats import bounds

VERSION = 'verified-relative-musical-events-5'


def resolve_spatial_chart(chart, events, relative, rhythm, barlines, profile, *, physical_review=None,
                          calibration_verification=None):
    audit={'format':'ez2rc.spatial-resolution','schema_version':1,'algorithm_version':VERSION,
           'coordinate':'first_barline_zero','status':'unavailable','accepted_note_count':0,
           'decisions':[],'reasons':[],
           'calibration_basis':'Existing caller-declared verified profile, without promoting conditional overrides.'}
    if physical_review is not None:
        audit['physical_review'] = physical_review.receipt()
    if calibration_verification is not None:
        audit['calibration_verification'] = calibration_verification.receipt()
    if relative is None:
        audit['reasons']=['spatial_reference_unavailable']
        return chart,events,audit
    geometry=relative.get('observed_geometry',{})
    proof=relative.get('rhythm_refinement',{})
    reviewed = {}
    if physical_review is not None:
        physical_review.check_relative(relative)
        if proof and proof.get('physical_review') != physical_review.receipt():
            raise ValueError('joint grid proof is not bound to the physical review')
        reviewed = {r['candidate_id']: physical_review.basis(r) for r in relative['candidates']}
        reviewed = {k:v for k,v in reviewed.items() if v is not None}
        audit['physical_review'] = physical_review.receipt()
    elif relative.get('physical_review') is not None:
        raise ValueError('physical review must be validated against visual caches')
    position=profile.get('position_calibration',{})
    prerequisites=[]
    scoped_verified=False
    if calibration_verification is not None:
        scope_reasons=calibration_verification.check_relative(relative)
        prerequisites.extend(scope_reasons)
        scoped_verified=not scope_reasons
        audit['calibration_basis']='Explicit measurement verification with matching code, observation and lane/kind scope.'
    if not geometry or geometry.get('status')!='conditional':prerequisites.append('shared_geometry_unavailable')
    if proof.get('status')!='conditional' or proof.get('grid')!=rhythm['grid']:
        prerequisites.append('joint_grid_feasibility_unavailable')
    # Selected endpoint unions participate in the common graph, including its
    # scale bounds. A local resolved label cannot certify that shared model
    # while another selected physical interpretation remains undecided.
    if any(r['pixel_status']!='resolved' and r['candidate_id'] not in reviewed for r in relative['candidates']):
        prerequisites.append('shared_geometry_has_unresolved_physical_dependencies')
    if barlines.get('status')!='supported' or len(barlines.get('hypotheses',[]))!=1:
        prerequisites.append('automatic_first_barline_unconfirmed')
    automatic=(barlines.get('hypotheses') or [{}])[0]
    expected={r['marker_id']:r['beat'] for r in automatic.get('assignments',[])}
    actual={a['id']:a['beat'] for a in relative['reference']['anchors']}
    if (relative['reference']['version']!=barlines.get('algorithm_version') or not expected or actual!=expected
            or any(relative['reference'][k]!=barlines.get(k) for k in ('observation_cache_id','motion_cache_id'))):
        prerequisites.append('reference_not_bound_to_automatic_numbering')
    if not scoped_verified and position.get('status')!='verified':prerequisites.append('profile_localization_unverified')
    elif not scoped_verified and geometry and exact_ms(position['error_px'])>exact_ms(relative['reference']['observed_geometry']['localization_error_px']):
        prerequisites.append('geometry_understates_verified_localization_bound')
    audit['reasons']=prerequisites
    by_event=defaultdict(list)
    for row in rhythm['candidates']:by_event[row['event_id']].append(row)
    spatial={r['candidate_id']:r for r in relative['candidates']}
    anchored=set(proof.get('anchored_endpoint_ids',[]))
    accepted,updated=[],[]
    for event in events:
        rows=by_event[event.id]
        reasons=list(prerequisites)
        row=rows[0] if len(rows)==1 else None
        if row is None:reasons.append('no_single_selected_physical_note')
        else:
            if row['pixel_status']!='resolved' and row['candidate_id'] not in reviewed:
                reasons.append('physical_selection_'+row['pixel_status'])
            if row['status']!='unique':reasons.append('rhythm_'+row['status'])
            source=spatial[row['candidate_id']]
            if source['status']!='conditional':reasons.append('spatial_endpoint_unavailable')
            for point in source['endpoints'].values():
                if point['endpoint_id'] not in anchored:reasons.append('endpoint_not_in_joint_grid_proof')
                if scoped_verified:continue
                declared=profile['calibration'].get(point.get('kind'),{})
                if declared.get('status')!='verified':
                    reasons.append('endpoint_convention_unverified');continue
                used=point.get('calibration',{})
                if point.get('optical_reference',{}).get('status')=='measured':
                    reasons.append('optical_reference_unverified')
                if point.get('template_registration',{}).get('status')=='measured':
                    reasons.append('template_registration_unverified')
                if used.get('source')=='profile':continue
                # An override can widen an established convention, never replace
                # it with an independently unverified correction or beat shift.
                if used.get('beat_shift',[0,1])!=[0,1]:
                    reasons.append('scoped_beat_shift_unverified');continue
                if 'offset_px' not in used:
                    reasons.append('endpoint_calibration_unavailable');continue
                lower,upper=bounds(used['offset_px'])
                offset,error=exact_ms(declared['offset_px']),exact_ms(declared['error_px'])
                if not lower<=offset-error<=offset+error<=upper:
                    reasons.append('scoped_offset_excludes_verified_convention')
                if (position.get('status')=='verified' and
                        (used.get('position_error_px') is None or
                         exact_ms(used['position_error_px'])<exact_ms(position['error_px']))):
                    reasons.append('scoped_localization_understates_verified_bound')
        reasons=sorted(set(reasons))
        decision={'event_id':event.id,'candidate_id':row['candidate_id'] if row else None,
                  'accepted':not reasons,'reasons':reasons,
                  'automatic_pixel_status':row['pixel_status'] if row else None,
                  'physical_basis':reviewed.get(row['candidate_id'], {'method':'automatic_selection'}) if row else None,
                  'calibration_basis':({'method':'scoped_verification','input_sha256':calibration_verification.input_sha256}
                                       if scoped_verified else {'method':'profile'}),
                  'observation_ids':list(event.observation_ids)}
        audit['decisions'].append(decision)
        if reasons:
            updated.append(event);continue
        value=row['note_candidates'][0]
        note=Note(event.id,row['lane'],row['type'],read_ratio(value['start_beat']),
                  read_ratio(value['end_beat']) if value['end_beat'] is not None else None)
        accepted.append(note)
        updated.append(replace(event,status='resolved',type=note.type,reason=None,
            start_ms=None,end_ms=None,start_interval=None,end_interval=None,
            start_beat=note.start_beat,end_beat=note.end_beat,beat_coordinate='first_barline_zero'))
    if not accepted:
        audit['status']='withheld'
        return chart,events,audit
    if chart.notes:
        # Recording-clock and first-line-relative notes need an explicit origin
        # relation before they can share a chart.
        raise ValueError('cannot combine clock-resolved and relative musical notes without an origin relation')
    timing=Timing(chart.timing.bpm,None,chart.timing.measure_beats,'resolved',None,'first_barline_zero')
    unresolved=tuple(e for e in updated if e.status!='resolved')
    result=Chart(chart.metadata,timing,tuple(accepted),unresolved,
                 tuple(s for s in chart.pending_stages if s!='timing'))
    audit.update(status='complete' if not unresolved else 'partial',accepted_note_count=len(accepted))
    return result,tuple(updated),audit
