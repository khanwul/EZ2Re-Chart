"""Versioned chart/events outputs and a static, escaped review report."""

from dataclasses import asdict
from html import escape
from pathlib import Path

from .contracts import Chart, Event, Metadata, Note, TimeInterval, Timing, ratio, read_ratio
from .storage import read_json, write_json


def chart_payload(chart: Chart) -> dict:
    return {
        "format": "ez2rc.chart", "schema_version": 1, "status": chart.status,
        "metadata": asdict(chart.metadata),
        "timing": {
            "beat_zero_ms": chart.timing.beat_zero_ms, "status": chart.timing.status,
            "reason": chart.timing.reason,
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


def events_payload(events: tuple[Event, ...], *, motion_ref=None, observation_ref=None,
                   pending_stages=(), candidates=()) -> dict:
    ids = [event.id for event in events]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate event ID")
    return {"format": "ez2rc.events", "schema_version": 1,
            "motion_ref": motion_ref, "observation_ref": observation_ref,
            "pending_stages": list(pending_stages), "candidates": list(candidates),
            "events": [asdict(e) for e in sorted(events, key=lambda e: e.id)]}


def read_events(path: Path) -> dict:
    value = read_json(path, "ez2rc.events")
    try:
        events = []
        for data in value["events"]:
            data = dict(data)
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
    value = read_json(path, "ez2rc.chart")
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
        clock = Timing(tempo[0]["bpm"], timing["beat_zero_ms"],
                       read_ratio(measures[0]["length_beats"]) if measures is not None else None,
                       timing["status"], timing["reason"])
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
                  candidates=(), motion_ref=None):
    event_ids = {e.id for e in events}
    if not {n.id for n in (*chart.notes, *chart.unresolved)} <= event_ids:
        raise ValueError("chart IDs must link to events")
    by_id = {e.id: e for e in events}
    for note in chart.notes:
        event = by_id[note.id]
        if event.status != "resolved" or event.lane != note.lane or event.type != note.type:
            raise ValueError("chart note disagrees with its observed event")
    for event in chart.unresolved:
        if by_id[event.id] != event:
            raise ValueError("unresolved chart event disagrees with evidence")
    write_json(root / "chart.json", chart_payload(chart))
    write_json(root / "artifacts/events.json", events_payload(
        events, observation_ref=observation_ref, motion_ref=motion_ref,
        pending_stages=chart.pending_stages, candidates=candidates))
    items = "".join(f"<li id='{escape(e.id, quote=True)}'>Lane {e.lane + 1}: "
                    f"{escape(e.status)} — {escape(e.reason or '')}</li>" for e in chart.unresolved)
    pending = ", ".join(chart.pending_stages)
    report = ("<!doctype html><html lang='en'><meta charset='utf-8'>"
              f"<title>{escape(chart.metadata.title)} — EZ2Re:Chart</title>"
              "<style>body{font:16px system-ui;max-width:70ch;margin:3rem auto;padding:0 1rem}"
              "li{margin:1rem 0}</style>"
              f"<h1>{escape(chart.metadata.title)}</h1><p>Result: {chart.status}</p>"
              f"<p>{escape(chart.timing.reason or 'Musical timeline resolved.')}</p>"
              f"<p>Pending stages: {escape(pending or 'none')}</p>"
              f"<ul>{items}</ul><p><a href='chart.json'>Chart</a> · "
              "<a href='artifacts/events.json'>Events and evidence references</a></p></html>")
    (root / "report.html").write_text(report, encoding="utf-8")
