"""Visual chart, video witnesses and a local review table for editable drafts."""
from html import escape
from pathlib import Path

import cv2

from .capture import VideoReader
from .contracts import read_ratio
from .draft import validate_draft
from .storage import file_hash
from .review_chart import chart_view


def write_review(path, draft):
    validate_draft(draft)
    def beat(value):
        if value is None:
            return '?'
        fraction = read_ratio(value)
        return f'{float(fraction):.6f} ({fraction})'
    rows = []
    selected_ids = {n['id'] for n in draft['notes']}
    alternatives = [n for n in draft['automatic']['alternatives'] if n['id'] not in selected_ids]
    for index, note in enumerate(draft['notes'] + alternatives):
        previews = []
        for side in ('start', 'end'):
            evidence = note['evidence'].get(side)
            if evidence and evidence['witnesses']:
                witness = evidence['witnesses'][0]
                seconds = witness['timestamp_ms']/1000
                previews.append(f'<button data-time="{seconds:.6f}">{side}: {seconds:.3f}s</button>')
        rows.append(f'<tr id="note-{index}">' + ''.join(f'<td>{escape(str(value))}</td>' for value in (
            note['id'], 'selected' if note['id'] in selected_ids else 'alternative',
            note['lane']+1, note['type'], beat(note['start_beat']), beat(note['end_beat']),
            note['status'], note['origin'], ', '.join(note['issues'])))
            + '<td>' + ' '.join(previews) + '</td></tr>')
    title = escape(draft['metadata']['title'])
    html = (f'<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{title} — draft review</title>'
            '<style>body{font:16px system-ui;margin:2rem}video{width:100%;max-height:60vh;background:#111}'
            'table{border-collapse:collapse;width:100%}td,th{padding:.5rem;border-bottom:1px solid #ccc;text-align:left}'
            'button,input{font:inherit}header{background:white;padding:.5rem}</style>'
            f'<h1>{title} — draft revision {draft["revision"]}</h1>'
            '<p>Compare every region with the original video: missing notes may be undetected. '
            'Values are unsnapped screen estimates, not verified musical timing. '
            'Witness buttons show the original detections, including after a manual beat edit.</p>'
            '<p>Choose the original local video below. Nothing is uploaded. If your browser cannot play its codec, '
            'use your video player or the <code>ez2rc frame</code> command.</p>'
            '<header><input id="file" type="file" accept="video/*"><video id="video" controls></video>'
            '<label>Find note, lane or issue <input id="filter" type="search"></label></header>'
            '<p>Edit using <code>ez2rc edit draft.json --id ID --lane 1 --start 4 --confirm</code>. '
            'Use add/delete/adopt for other changes, then regenerate this report with '
            '<code>ez2rc review draft.json --output review.html</code>.</p>'
            f'<p>{len(draft["notes"])} selected notes; {len(draft["automatic"]["alternatives"])} alternatives; '
            f'{len(draft["projection"]["unprocessed_motion"])} uncertain motion spans. '
            'An alternative is not an additional selected note.</p>'
            + chart_view(draft) +
            '<h2>노트 상세 · 대안 후보</h2><div style="overflow-x:auto"><table><thead><tr><th>ID</th><th>Selection</th><th>Lane</th><th>Type</th><th>Start beat</th><th>End beat</th>'
            '<th>Status</th><th>Origin</th><th>Issues</th><th>Video witnesses</th></tr></thead><tbody>'
            + ''.join(rows) + '</tbody></table></div><h2>Uncertain motion spans</h2><ul>'
            + ''.join(f'<li>{r["start_ms"]/1000:.3f}–{r["end_ms"]/1000:.3f}s '
                      f'(frames {r["first_frame"]}–{r["last_frame"]})</li>'
                      for r in draft['projection']['unprocessed_motion'])
            + '</ul><script>'
            'const video=document.getElementById("video");let url;'
            'document.querySelectorAll("a[href]").forEach(a=>a.addEventListener("click",()=>{'
            'if(!a.getAttribute("href").startsWith("#note-"))return;'
            'document.getElementById("filter").value="";document.querySelectorAll("tbody tr").forEach(r=>r.hidden=false);}));'
            'document.getElementById("file").addEventListener("change",e=>{'
            'if(!e.target.files.length)return;if(url)URL.revokeObjectURL(url);'
            'url=URL.createObjectURL(e.target.files[0]);video.src=url;});'
            'document.querySelectorAll("button[data-time]").forEach(b=>b.addEventListener("click",()=>{'
            'if(!video.src){alert("Choose the original video first.");return;}'
            'video.pause();video.currentTime=Number(b.dataset.time);}));'
            'document.getElementById("filter").addEventListener("input",e=>{'
            'const term=e.target.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>'
            'r.hidden=!r.textContent.toLowerCase().includes(term));});'
            '</script></html>')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(html, encoding='utf-8')


def export_frame(draft, output, *, identity=None, side='start', seconds=None, video=None, analysis_dir=None):
    validate_draft(draft)
    source = Path(video or draft['source']['video'])
    if file_hash(source) != draft['source']['video_sha256']:
        raise ValueError('video content does not match this draft')
    frame_id, corners = None, None
    if identity is not None:
        row = next((n for n in draft['notes'] + draft['automatic']['alternatives'] if n['id'] == identity), None)
        if row is None:
            raise ValueError('unknown note ID')
        point = row['evidence'].get(side)
        if not point or not point['witnesses']:
            raise ValueError('no video witness for this endpoint; use --time to inspect a region')
        witness = point['witnesses'][0]
        offset = draft['source'].get('alignment_offset_px')
        if offset is None:
            raise ValueError('draft has no source alignment; regenerate it or use --time to inspect the video')
        from .witness_box import draw_outline, matched_boxes, witness_kind
        original_type = next((n['type'] for group in ('notes', 'alternatives')
                              for n in draft['automatic'][group] if n['id'] == identity), row['type'])
        kind = witness_kind(original_type, side)
        boxes = matched_boxes([(witness, kind)], draft, analysis_dir)
        box = boxes.get((witness.get('observation_id'), kind))
        if box is None:
            raise ValueError('no matched note size available; use --time to view the original frame')
        x, y, width, height = box
        x, y = x + offset[0], y + offset[1]
        corners = ((x, y), (x+width-1, y+height-1))
        frame_id = witness['frame_id']
    elif seconds is None or seconds < 0:
        raise ValueError('supply a note ID or a nonnegative --time in seconds')
    output = Path(output)
    if output.exists():
        raise ValueError(f'preview output already exists: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    for frame in VideoReader(source, timestamp_policy=draft['source'].get('timestamp_policy', 'strict')):
        if (frame_id is not None and frame.stamp.frame_id == frame_id
                or frame_id is None and frame.stamp.timestamp_ms >= seconds*1000):
            if corners is not None:
                draw_outline(frame.bgr, *corners, kind)
            ok, encoded = cv2.imencode('.png', frame.bgr)
            if not ok:
                raise ValueError('cannot encode preview PNG')
            output.write_bytes(encoded.tobytes())
            return frame.stamp
    raise ValueError('requested frame is outside the video')
