"""Render numbered review photos from the original recording in one decoding pass."""
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from .capture import VideoReader
from .picture import Item, TapPlacementItem, beat_text, pending_items, witness_time
from .storage import digest, file_hash, read_json
from .witness_box import OUTLINE_DARK, OUTLINE_PINK, draw_outline, matched_boxes, witness_kind


def photo_folder(draft):
    name = "review-photos" if pending_items(draft) else "preview-photos"
    return f"{name}-r{draft['revision']:04d}"


def review_question(options):
    notes = list(options.values())
    lane = notes[0]['lane'] + 1
    if any(note['type'] != notes[0]['type'] for note in notes):
        return f'Lane {lane}: is this a tap or a hold?'
    sides = ('start', 'end') if notes[0]['type'] == 'longnote' else ('start',)
    uncertain = [side for side in sides if notes[0][side+'_beat'] is None or
                 any(note[side+'_beat'] != notes[0][side+'_beat'] for note in notes)]
    if uncertain == ['end']:
        return f'Lane {lane}: on which beat does the hold end?'
    if uncertain == ['start']:
        return f'Lane {lane}: on which beat does the note start?'
    if uncertain:
        return f'Lane {lane}: on which beats does the hold start and end?'
    return f'Does lane {lane} contain the note shown below?'


def _answers(options, number):
    lines = []
    for letter, note in options.items():
        parts = []
        flags = []
        for side in ('start', 'end') if note['type'] == 'longnote' else ('start',):
            value = note[side+'_beat']
            if value is None:
                flags.append(f'--{side} <beat>')
                parts.append(f'{side}: enter beat')
            else:
                parts.append(f'{side} beat {beat_text(value)}')
        kind = 'tap' if note['type'] == 'tap' else 'hold'
        command = ' '.join(['ez2rc', 'choose', str(number), letter, *flags])
        lines.append(f'{letter}   {kind}   {" / ".join(parts)}     {command}')
    lines.append(f'X   Remove the note     ez2rc choose {number} X')
    return lines


def _requests(draft):
    original_notes = {n['id']: n for group in ('notes', 'alternatives')
                      for n in draft['automatic'][group]}
    items = pending_items(draft)
    groups = list(items.values())
    if not groups:
        notes = sorted((n for n in draft['notes'] if witness_time(n) is not None), key=witness_time)
        if notes:
            groups = [Item(i+1, notes[index], {'': notes[index]}) for i, index in enumerate(
                sorted({0, len(notes)//2, len(notes)-1}))]
    requests = defaultdict(list)
    missing = []
    for item in groups:
        number = item.number
        frames = defaultdict(list)
        for letter, note in item.photo_notes():
            sides = ('start', 'end') if note['type'] == 'longnote' else ('start',)
            for side in sides:
                witnesses = (note['evidence'].get(side) or {}).get('witnesses', [])
                if not witnesses:
                    continue
                # Use two observed moments when available. Each marker belongs to that exact frame.
                selected = [witnesses[0]]
                if items and witnesses[-1]['frame_id'] != witnesses[0]['frame_id']:
                    selected.append(witnesses[-1])
                for w in selected:
                    original = original_notes.get(note['id'], note)
                    kind = witness_kind(original['type'], side)
                    frames[w['frame_id']].append((letter, side, original['lane'], w, kind))
        if not frames:
            missing.append(number)
        for sequence, (frame_id, marks) in enumerate(sorted(frames.items()), 1):
            prefix = 'check' if items else 'preview'
            name = f'{prefix}-{number:04d}-{sequence:02d}.png'
            requests[frame_id].append((name, number, marks, item))
    return requests, missing


def _photo(frame, number, marks, item, rois, offset, boxes):
    image = frame.bgr
    height, width = image.shape[:2]
    raw = offset is None or rois is None
    pointers = defaultdict(list)
    marked_count = 0
    if raw:
        shot = image.copy()
    else:
        dx, dy = offset
        left = max(0, min(r[0] for r in rois) + dx - 16)
        right = min(width, max(r[2] for r in rois) + dx + 16)
        ys = [round(w['y_px'] + dy) for _, _, _, w, _ in marks]
        footprints = [boxes.get((w.get('observation_id'), kind)) for _, _, _, w, kind in marks]
        known = [box for box in footprints if box is not None]
        left = max(0, min([left] + [x+dx-8 for x,y,w,h in known]))
        right = min(width, max([right] + [x+dx+w+8 for x,y,w,h in known]))
        bottom = min(height, max(ys + [y+dy+h for x,y,w,h in known])+111)
        top = max(0, min(ys)-110)
        if left >= right or top >= bottom:
            raise ValueError('The marked region is outside the source video frame.')
        shot = image[top:bottom, left:right].copy()
        for (letter, _, _, _, kind), box in zip(marks, footprints):
            if box is None:
                continue
            x, y, w, h = box
            x, y = x+dx-left, y+dy-top
            draw_outline(shot, (x, y), (x+w-1, y+h-1), kind)
            marked_count += 1
            if letter:
                pointers[x,y].append(letter)
    scale = min(4, 650 / shot.shape[1], 550 / shot.shape[0])
    shot = cv2.resize(shot, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST if scale>1 else cv2.INTER_AREA)
    for (x, y), letters in pointers.items():
        text = '/'.join(dict.fromkeys(letters))
        origin = (max(0, round(x*scale)-10), max(15, round(y*scale)-6))
        cv2.putText(shot,text,origin,cv2.FONT_HERSHEY_SIMPLEX,.45,OUTLINE_DARK,3,cv2.LINE_AA)
        cv2.putText(shot,text,origin,cv2.FONT_HERSHEY_SIMPLEX,.45,OUTLINE_PINK,1,cv2.LINE_AA)
    if isinstance(item, TapPlacementItem):
        heading = f'Check {number}: where does the tap group start?'
        footer = [*item.lines(), f'Apply all taps with ez2rc choose {number} A or ez2rc choose {number} B']
    elif 'A' in item.options:
        heading = f'Check {number}: {review_question(item.options)}'
        footer = _answers(item.options, number)
    else:
        heading = 'Preview only - no decision needed'
        footer = []
    labels = [f'{letter} lane {lane+1} {side}'.strip() for letter, side, lane, _, _ in marks]
    seconds = frame.stamp.timestamp_ms / 1000
    lines = [heading, f'{seconds:.3f}s   ' + ' / '.join(dict.fromkeys(labels))]
    if not marked_count:
        lines.append('Original frame without position markers')
    else:
        lines.append('The pink outline shows the detected note footprint.')
    if footer:
        footer.append('Unsure? Leave this item open and check another photo.')
    row_height = 32
    width = max(680, shot.shape[1] + 24)
    for line in lines + footer:
        width = max(width, cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,.58,1)[0][0]+36)
    top_margin = row_height * len(lines) + 10
    out = cv2.copyMakeBorder(shot, top_margin, row_height*len(footer)+10, 12,
                             width-shot.shape[1]-12, cv2.BORDER_CONSTANT, value=(32,32,32))
    def caption(line, y, color):
        cv2.putText(out,line,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.58,color,1,cv2.LINE_AA)
    for i, line in enumerate(lines):
        caption(line, 26+i*row_height, (130,225,255) if i == 0 else (240,240,240))
    for i, line in enumerate(footer):
        caption(line, top_margin+shot.shape[0]+26+i*row_height, (235,235,235))
    return out


def write_photos(destination, draft, *, video=None, analysis_dir=None, progress=None):
    requests, missing = _requests(draft)
    if not requests:
        return 0, missing
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f'Photo directory already exists: {destination}. Choose another directory with --output.')
    source = draft['source']
    path = Path(video or source.get('video', ''))
    if not path.is_file() or file_hash(path) != source.get('video_sha256'):
        raise ValueError('Use --video to specify the original recording that matches the draft.')
    rois = source.get('lane_rois')
    policy = source.get('timestamp_policy', 'strict')
    if analysis_dir and rois is None:
        manifest_path = Path(analysis_dir)/'artifacts/observations/manifest.json'
        if manifest_path.is_file():
            manifest = read_json(manifest_path, 'ez2rc.stage-cache')
            checksum = digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'})
            if manifest['manifest_sha256'] != checksum or manifest['cache_id'] != source.get('observation_cache_id'):
                raise ValueError('The observation data does not match the draft.')
            meta = manifest['metadata']
            rois = [lane['roi'] for lane in meta['profile']['lanes']]
            policy = meta['capture']['timestamp_policy']
    if rois is not None and len(rois) != draft['metadata']['lane_count']:
        raise ValueError('The lane geometry does not match the draft.')
    witnesses = ((w, kind) for rows in requests.values() for _, _, marks, _ in rows
                 for _, _, _, w, kind in marks)
    boxes = matched_boxes(witnesses, draft, analysis_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with TemporaryDirectory(dir=destination.parent) as tmp:
        stage = Path(tmp)/'photos'
        stage.mkdir()
        total = sum(map(len, requests.values()))
        for frame in VideoReader(path, timestamp_policy=policy):
            for name, number, marks, item in requests.pop(frame.stamp.frame_id, []):
                image = _photo(frame, number, marks, item, rois, source.get('alignment_offset_px'), boxes)
                if not cv2.imwrite(str(stage/name), image):
                    raise OSError(f'Could not save photo: {name}')
                count += 1
                if progress:
                    progress(f'Source video photos: {count}/{total}')
            if not requests:
                break
        if requests:
            raise ValueError('Some observed frames could not be found in the source video.')
        if file_hash(path) != source['video_sha256']:
            raise ValueError('The source video changed during photo generation.')
        stage.rename(destination)
    return count, missing
