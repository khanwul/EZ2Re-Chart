"""Versioned chart/events outputs and a static, escaped review report."""

from dataclasses import asdict
from html import escape
from pathlib import Path

from .contracts import (Chart, Event, Metadata, Note, TimeInterval, Timing,
                        origin_payload, origin_schema, ratio, read_origin, read_ratio)
from .storage import read_json, write_json


def chart_schema(chart):
    return 3 if chart.timing.coordinate=='first_barline_zero' else origin_schema(chart.timing.beat_zero_ms)


def chart_payload(chart: Chart) -> dict:
    return {
        "format": "ez2rc.chart", "schema_version": chart_schema(chart), "status": chart.status,
        "metadata": asdict(chart.metadata),
        "timing": {
            **origin_payload(chart.timing.beat_zero_ms), "status": chart.timing.status,
            "reason": chart.timing.reason,
            **({'coordinate':chart.timing.coordinate} if chart_schema(chart)==3 else {}),
            "bpm_timeline": [{"start_beat": [0, 1], "bpm": chart.timing.bpm}],
            "measure_lengths": ([{"start_beat": [0, 1], "length_beats": ratio(chart.timing.measure_beats)}]
                                if chart.timing.measure_beats is not None else None),
        },
        "notes": [{"id": n.id, "lane": n.lane, "type": n.type,
                   "start_beat": ratio(n.start_beat), "end_beat": ratio(n.end_beat) if n.end_beat is not None else None}
                  for n in sorted(chart.notes, key=lambda n: (n.start_beat, n.lane, n.id))],
        "unresolved": [{"id": e.id, "reason": e.reason, "status": e.status,
                        "candidate_refs": list(e.candidate_ids), "event_ref": f"artifacts/events.json#{e.id}"}
                       for e in sorted(chart.unresolved, key=lambda e: e.id)],
        "pending_stages": list(chart.pending_stages),
    }


def events_payload(events: tuple[Event, ...], *, motion_ref=None, observation_ref=None, fusion_ref=None,
                   decode_ref=None, time_projection_ref=None, timing_resolution_ref=None,
                   reference_candidates_ref=None, reference_origin_link_ref=None, origin_adoption_ref=None, beat_ranges_ref=None,
                   endpoint_time_ranges_ref=None, relative_beats_ref=None, analysis_scope_ref=None,
                   rhythm_candidates_ref=None, barline_grid_ref=None, spatial_resolution_ref=None, physical_review_ref=None,
                   calibration_verification_ref=None,
                   pixel_intervals=(), pending_stages=(), candidates=()) -> dict:
    ids = [event.id for event in events]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate event ID")
    musical = any(e.beat_coordinate is not None for e in events)
    rows=[]
    for event in sorted(events,key=lambda e:e.id):
        row=asdict(event)
        for key in ('start_beat','end_beat','beat_coordinate'):
            if not musical:row.pop(key)
            elif key!='beat_coordinate':row[key]=ratio(row[key]) if row[key] is not None else None
        rows.append(row)
    return {"format": "ez2rc.events", "schema_version": 2 if musical else 1,
            "motion_ref": motion_ref, "observation_ref": observation_ref,
            "fusion_ref": fusion_ref,
            "decode_ref": decode_ref, "time_projection_ref": time_projection_ref,
            **({'endpoint_time_ranges_ref': endpoint_time_ranges_ref} if endpoint_time_ranges_ref is not None else {}),
            "timing_resolution_ref": timing_resolution_ref, "pixel_intervals": list(pixel_intervals),
            "reference_candidates_ref": reference_candidates_ref,
            "reference_origin_link_ref": reference_origin_link_ref,
            "origin_adoption_ref": origin_adoption_ref,
            "beat_ranges_ref": beat_ranges_ref,
            **({'relative_beats_ref': relative_beats_ref} if relative_beats_ref is not None else {}),
            **({'rhythm_candidates_ref': rhythm_candidates_ref} if rhythm_candidates_ref is not None else {}),
            **({'barline_grid_ref': barline_grid_ref} if barline_grid_ref is not None else {}),
            **({'spatial_resolution_ref': spatial_resolution_ref} if spatial_resolution_ref is not None else {}),
            **({'physical_review_ref': physical_review_ref} if physical_review_ref is not None else {}),
            **({'calibration_verification_ref': calibration_verification_ref} if calibration_verification_ref is not None else {}),
            **({'analysis_scope_ref': analysis_scope_ref} if analysis_scope_ref is not None else {}),
            "pending_stages": list(pending_stages), "candidates": list(candidates),
            "events": rows}


def read_events(path: Path) -> dict:
    value = read_json(path, "ez2rc.events",schema_versions=(1,2))
    try:
        events = []
        for data in value["events"]:
            data = dict(data)
            if value['schema_version']==1 and any(k in data for k in ('start_beat','end_beat','beat_coordinate')):
                raise ValueError('relative musical events require schema 2')
            for key in ('start_beat','end_beat'):
                if data.get(key) is not None:data[key]=read_ratio(data[key])
            for key in ("start_interval", "end_interval"):
                if data[key] is not None:
                    data[key] = TimeInterval(**data[key])
            data["observation_ids"] = tuple(data["observation_ids"])
            data["candidate_ids"] = tuple(data["candidate_ids"])
            events.append(Event(**data))
        if len({e.id for e in events}) != len(events):
            raise ValueError("duplicate event ID")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid events contract: {exc}") from exc
    return value


def read_chart(path: Path) -> dict:
    value = read_json(path, "ez2rc.chart", schema_versions=(1, 2, 3))
    try:
        metadata = Metadata(**value["metadata"])
        notes = tuple(Note(n["id"], n["lane"], n["type"], read_ratio(n["start_beat"]),
                           read_ratio(n["end_beat"]) if n["end_beat"] is not None else None)
                      for n in value["notes"])
        timing = value["timing"]
        tempo = timing["bpm_timeline"]
        if len(tempo) != 1 or read_ratio(tempo[0]["start_beat"]) != 0:
            raise ValueError("only fixed-BPM timeline is implemented")
        measures = timing["measure_lengths"]
        if measures is not None and (len(measures) != 1 or read_ratio(measures[0]["start_beat"]) != 0):
            raise ValueError("only constant measure length is implemented")
        if value['schema_version']==3:
            if (timing.get('coordinate')!='first_barline_zero' or timing['beat_zero_ms'] is not None
                    or 'beat_zero_ms_exact' in timing):
                raise ValueError('relative musical chart requires a separate capture clock')
            origin=None
        else:
            if 'coordinate' in timing:raise ValueError('coordinate declaration requires chart schema 3')
            origin=read_origin(timing,value['schema_version'])
        clock = Timing(tempo[0]["bpm"], origin,
                       read_ratio(measures[0]["length_beats"]) if measures is not None else None,
                       timing["status"], timing["reason"],timing.get('coordinate','recording_timestamp_ms'))
        unresolved = value["unresolved"]
        ids = [n.id for n in notes] + [e["id"] for e in unresolved]
        if len(ids) != len(set(ids)) or any(n.lane >= metadata.lane_count for n in notes):
            raise ValueError("duplicate ID or invalid lane")
        for event in unresolved:
            if (not isinstance(event["id"], str) or not event["id"] or not event["reason"]
                    or event["status"] not in {"ambiguous", "incomplete"}
                    or not isinstance(event["candidate_refs"], list)
                    or event["event_ref"] != f"artifacts/events.json#{event['id']}"):
                raise ValueError("invalid unresolved event reference")
        chart = Chart(metadata, clock, notes, pending_stages=tuple(value["pending_stages"]))
        status = "partial" if unresolved else chart.status
        if value["status"] != status:
            raise ValueError("chart completion status contradicts its contents")
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid chart contract: {exc}") from exc
    return value


def write_outputs(root: Path, chart: Chart, events: tuple[Event, ...], *, observation_ref: str,
                  candidates=(), html_report=True, motion_ref=None, fusion_ref=None, decode_ref=None,
                  time_projection_ref=None, timing_resolution_ref=None, reference_candidates_ref=None,
                  reference_origin_link_ref=None, origin_adoption_ref=None, beat_ranges_ref=None,
                  endpoint_time_ranges_ref=None, relative_beats_ref=None, relative_beats=None,
                  rhythm_candidates_ref=None, barline_grid_ref=None, spatial_resolution_ref=None, physical_review_ref=None,
                  calibration_verification_ref=None,
                  analysis_scope_ref=None, analysis_scope=None,
                  pixel_intervals=(), diagnostics=None):
    event_ids = {e.id for e in events}
    if not {n.id for n in (*chart.notes, *chart.unresolved)} <= event_ids:
        raise ValueError("chart IDs must link to events")
    by_id = {e.id: e for e in events}
    for note in chart.notes:
        event = by_id[note.id]
        if event.status != "resolved" or event.lane != note.lane or event.type != note.type:
            raise ValueError("chart note disagrees with its observed event")
        if event.beat_coordinate is not None and (
                event.beat_coordinate!=chart.timing.coordinate
                or event.start_beat!=note.start_beat or event.end_beat!=note.end_beat):
            raise ValueError('chart musical note disagrees with its resolved event')
    for event in chart.unresolved:
        if by_id[event.id] != event:
            raise ValueError("unresolved chart event disagrees with evidence")
    payload = chart_payload(chart)
    if beat_ranges_ref is not None:
        payload['beat_ranges_ref'] = beat_ranges_ref
    if relative_beats_ref is not None:
        payload['relative_beats_ref'] = relative_beats_ref
    if rhythm_candidates_ref is not None:
        payload['rhythm_candidates_ref'] = rhythm_candidates_ref
    if barline_grid_ref is not None:
        payload['barline_grid_ref'] = barline_grid_ref
    if spatial_resolution_ref is not None:
        payload['spatial_resolution_ref'] = spatial_resolution_ref
    if physical_review_ref is not None:
        payload['physical_review_ref'] = physical_review_ref
    if calibration_verification_ref is not None:
        payload['calibration_verification_ref'] = calibration_verification_ref
    if analysis_scope_ref is not None:
        payload['analysis_scope_ref'] = analysis_scope_ref
    write_json(root / "chart.json", payload)
    write_json(root / "artifacts/events.json", events_payload(
        events, observation_ref=observation_ref, motion_ref=motion_ref, fusion_ref=fusion_ref,
        decode_ref=decode_ref, time_projection_ref=time_projection_ref, timing_resolution_ref=timing_resolution_ref,
        endpoint_time_ranges_ref=endpoint_time_ranges_ref,
        reference_candidates_ref=reference_candidates_ref,
        reference_origin_link_ref=reference_origin_link_ref,
        origin_adoption_ref=origin_adoption_ref,
        beat_ranges_ref=beat_ranges_ref,
        relative_beats_ref=relative_beats_ref,
        rhythm_candidates_ref=rhythm_candidates_ref,
        barline_grid_ref=barline_grid_ref,
        spatial_resolution_ref=spatial_resolution_ref,
        physical_review_ref=physical_review_ref,
        calibration_verification_ref=calibration_verification_ref,
        analysis_scope_ref=analysis_scope_ref,
        pixel_intervals=pixel_intervals,
        pending_stages=chart.pending_stages, candidates=candidates))
    if not html_report:
        return
    items = "".join(f"<li id='{escape(e.id, quote=True)}'>Lane {e.lane + 1}: "
                    f"{escape(e.status)} — {escape(e.reason or '')}</li>" for e in chart.unresolved)
    pending = ", ".join(chart.pending_stages)
    visual = ""
    if diagnostics is not None:
        states = ", ".join(f"{key}: {value}" for key, value in diagnostics["motion"].items())
        visual = (f"<p>Scroll frame states: {escape(states)}</p>"
                  f"<p>Endpoint timing calibration: {escape(diagnostics['endpoint_calibration'])}. "
                  "Visual scores are not probabilities; empty notes do not indicate an empty chart.</p>"
                  f"<p><a href='{escape(motion_ref, quote=True)}'>Scroll evidence</a> · "
                  f"<a href='{escape(fusion_ref, quote=True)}'>Accumulated evidence</a></p>")
        if diagnostics.get('position_calibration') is not None:
            position = diagnostics['position_calibration']
            bound = f"{position['error_px']} px" if 'error_px' in position else 'unknown'
            visual += (f"<p>Caller-declared observation localization: {escape(position['status'])}; "
                       f"bound: {escape(bound)}; version: {escape(position['version'])}; "
                       f"scope: {escape(position['scope'])}.</p>")
        if decode_ref:
            states = ", ".join(f"{key}: {value}" for key, value in diagnostics['decode'].items())
            coverage = ' (all observed frames)' if analysis_scope is not None else ''
            visual += (f"<p>Pixel interval states{coverage}: {escape(states)}.</p>"
                       f"<p><a href='{escape(decode_ref, quote=True)}'>Interval candidates and costs</a></p>")
    rows = "".join(f"<tr><td>{p['lane'] + 1}</td><td>{escape(p['status'])}</td>"
                   f"<td>{escape(', '.join(str(c['start_px']) + ' → ' + str(c['end_px']) for c in p['selected_intervals']) or 'noise alternative')}</td>"
                   f"<td>{p['selected_cost']:.3f} / {p['alternative_cost']:.3f}</td>"
                   f"<td>{escape(', '.join(p.get('review_causes', ())) or '—')}</td>"
                   + (f"<td>{escape(p['analysis_scope'])}</td>" if analysis_scope is not None else '')
                   + '</tr>' for p in pixel_intervals)
    if time_projection_ref:
        visual += (f"<p><a href='{escape(time_projection_ref, quote=True)}'>Provisional video-time projections</a>. "
                   "These estimates are not calibrated note times.</p>")
    if timing_resolution_ref:
        visual += (f"<p><a href='{escape(timing_resolution_ref, quote=True)}'>Timing decisions and supplied reference</a>. "
                   f"Resolved notes: {len(chart.notes)}; unresolved events: {len(chart.unresolved)}.</p>")
    if analysis_scope is not None:
        window = analysis_scope['window']
        requested, observed = window['requested'], window['observation']
        counts = {label: sum(c['selected'] and c['scope'] == label for c in analysis_scope['candidates'])
                  for label in ('requested', 'boundary_uncertain', 'auxiliary_only')}
        visual += (f"<p>Requested observation window: [{requested['start_ms']}, {requested['end_ms']}) ms; "
                   f"observation with context: [{observed['start_ms']}, {observed['end_ms']}) ms. "
                   f"Requested frames: {analysis_scope['requested_frame_count']}; "
                   f"auxiliary frames: {analysis_scope['auxiliary_frame_count']}. "
                   "Window membership describes visible evidence, not judgment time. "
                   f"Selected candidates: {counts['requested']} with requested-frame members, "
                   f"{counts['boundary_uncertain']} with uncertain boundary membership, "
                   f"{counts['auxiliary_only']} auxiliary only. "
                   "Context-only candidates remain in diagnostics; possible boundary associations remain unresolved. "
                   f"<a href='{escape(analysis_scope_ref, quote=True)}'>Window evidence and candidate scope</a>.</p>")
    if endpoint_time_ranges_ref:
        visual += (f"<p><a href='{escape(endpoint_time_ranges_ref, quote=True)}'>Conditional endpoint time ranges</a>. "
                   "Pixel error components use observed scroll support; missing error bounds stay unknown.</p>")
    if reference_candidates_ref:
        visual += (f"<p><a href='{escape(reference_candidates_ref, quote=True)}'>Visual musical-reference candidates</a>. "
                   "Raw cyclic phases require separate marker calibration and an absolute reference.</p>")
    if reference_origin_link_ref:
        visual += (f"<p><a href='{escape(reference_origin_link_ref, quote=True)}'>Marker calibration and absolute-anchor comparisons</a>. "
                   "Inspect compatible and conflicting origins and their calibration assumptions.</p>")
    if origin_adoption_ref:
        visual += (f"<p><a href='{escape(origin_adoption_ref, quote=True)}'>Origin adoption decision</a>. "
                   "Anchor use requires an explicit policy; exact origins, preserved intervals, and blocked requests are recorded.</p>")
    if beat_ranges_ref:
        visual += (f"<p><a href='{escape(beat_ranges_ref, quote=True)}'>Conditional beat ranges</a>. "
                   "Ranges share one origin shift across all endpoints and remain separate from resolved notes.</p>")
    if rhythm_candidates_ref:
        visual += (f"<p><a href='{escape(rhythm_candidates_ref, quote=True)}'>Rhythm grid candidates</a>. "
                   "The configured subdivisions bound the candidate search. A unique grid label still requires "
                   "calibration and physical evidence before acceptance.</p>")
    if barline_grid_ref:
        visual += (f"<p><a href='{escape(barline_grid_ref, quote=True)}'>First barline and measure hypotheses</a>. "
                   "Moving-line observations, leading coverage, missing lines and conflicting cycles remain inspectable.</p>")
    if calibration_verification_ref:
        visual += (f"<p><a href='{escape(calibration_verification_ref, quote=True)}'>Spatial measurement assessment and scope</a>. "
                   "This evidence applies to relative musical coordinates. Recording-clock calibration is reported separately.</p>")
    if spatial_resolution_ref:
        visual += f"<p><a href='{escape(spatial_resolution_ref, quote=True)}'>Musical acceptance decisions</a>.</p>"
    if relative_beats_ref:
        visual += (f"<p><a href='{escape(relative_beats_ref, quote=True)}'>First-barline relative beats and capture times</a>. "
                   "Beat zero is the first barline. Relative beats use spatial anchors; recording timestamps use observed scroll. "
                   "Conditional ranges, extrapolation and ambiguous selections remain separate from exact notes.</p>")
        if relative_beats is not None:
            def interval(value):
                if value is None:
                    return 'unavailable'
                return f"[{float(read_ratio(value['lower'])):.6f}, {float(read_ratio(value['upper'])):.6f}]"
            relative_rows = []
            for row in relative_beats['candidates']:
                for side, point in row['endpoints'].items():
                    beat = point['relative_beat']
                    capture = point['capture_time']
                    reasons = row['reasons'] + (['capture: ' + r for r in capture['reasons']] if capture else [])
                    relative_rows.append(f"<tr><td>{escape(row['candidate_id'])} / {escape(side)}</td>"
                        f"<td>{interval(beat['beat_range'] if beat else None)}</td>"
                        f"<td>{interval(capture['time_range_ms'] if capture else None)}</td>"
                        f"<td>{interval(row['duration_beat_range']) if row['type'] == 'longnote' else '—'}</td>"
                        f"<td>{escape(', '.join(reasons) or row['pixel_status'])}</td></tr>")
            visual += ("<table><caption>Conditional coordinates; capture milliseconds and relative beats are distinct</caption>"
                       "<tr><th>Candidate / endpoint</th><th>Relative beat</th><th>Capture ms</th><th>Length (beats)</th><th>State</th></tr>"
                       + ''.join(relative_rows) + '</table>')
    pixel_table = ("<table><caption>Relative pixel intervals; costs cover the whole lane</caption>"
                   "<tr><th>Lane</th><th>Pixel status</th><th>Start → end (px)</th><th>Selected / alternative</th>"
                   "<th>Review causes in competing hypotheses</th>"
                   + ('<th>Window evidence</th>' if analysis_scope is not None else '') + '</tr>'
                   f"{rows}</table>") if rows else ""
    report = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
              f"<title>{escape(chart.metadata.title)} — EZ2Re:Chart</title>"
              "<style>body{font:16px system-ui;max-width:70ch;margin:3rem auto;padding:0 1rem}"
              "li{margin:1rem 0}</style>"
              f"<h1>{escape(chart.metadata.title)}</h1><p>Result: {chart.status}</p>"
              f"<p>{escape(chart.timing.reason or 'Musical timeline resolved.')}</p>"
              f"<p>Pending stages: {escape(pending or 'none')}</p>"
              f"{visual}{pixel_table}"
              f"<ul>{items}</ul><p><a href='chart.json'>Chart</a> · "
              "<a href='artifacts/events.json'>Events and evidence references</a></p></html>")
    (root / "report.html").write_text(report, encoding="utf-8")
