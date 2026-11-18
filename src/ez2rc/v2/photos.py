"""Render numbered review photos from the original recording in one decoding pass."""
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import os
from tempfile import TemporaryDirectory

import cv2

from .capture import VideoReader
from .picture import beat_text, pending_items, witness_time
from .storage import digest, file_hash, read_json
from .witness_box import OUTLINE_DARK, OUTLINE_PINK, draw_outline, matched_boxes, witness_kind


def photo_folder(draft):
    name = "검토사진" if pending_items(draft) else "참고사진"
    return f"{name}-r{draft['revision']:04d}"


@lru_cache(maxsize=1)
def _korean_font():
    paths = [
        Path('/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
        Path('/usr/share/fonts/truetype/nanum/NanumGothic.ttf'),
        Path('/System/Library/Fonts/AppleSDGothicNeo.ttc'),
        Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/malgun.ttf',
    ]
    for path in paths:
        if path.is_file():
            try:
                return cv2.FontFace(str(path))
            except cv2.error:
                continue
    return None


def review_question(options, ko=True):
    notes = list(options.values())
    lane = notes[0]['lane'] + 1
    def say(korean, english):
        return korean if ko else english
    if any(note['type'] != notes[0]['type'] for note in notes):
        return say(f'{lane}번 레인에 보이는 것은 일반 노트인가요, 롱노트인가요?',
                   f'Lane {lane}: is this a tap or a hold?')
    sides = ('start', 'end') if notes[0]['type'] == 'longnote' else ('start',)
    uncertain = [side for side in sides if notes[0][side+'_beat'] is None or
                 any(note[side+'_beat'] != notes[0][side+'_beat'] for note in notes)]
    if uncertain == ['end']:
        return say(f'{lane}번 레인의 롱노트는 몇 박에서 끝나나요?', f'Lane {lane}: on which beat does the hold end?')
    if uncertain == ['start']:
        return say(f'{lane}번 레인의 노트는 몇 박에서 시작하나요?', f'Lane {lane}: on which beat does the note start?')
    if uncertain:
        return say(f'{lane}번 레인의 롱노트는 몇 박에서 시작하고 끝나나요?',
                   f'Lane {lane}: on which beats does the hold start and end?')
    return say(f'{lane}번 레인에 아래와 같은 노트가 보이나요?', f'Does lane {lane} contain the note shown below?')


def _answers(options, number, ko):
    lines = []
    for letter, note in options.items():
        parts = []
        flags = []
        for side in ('start', 'end') if note['type'] == 'longnote' else ('start',):
            value = note[side+'_beat']
            name = {'start':'시작', 'end':'끝'}[side] if ko else side
            position = beat_text(value)
            if value is None:
                flags.append(f'--{side} ' + ('<박자>' if ko else '<beat>'))
                position = '직접 입력' if ko else 'enter beat'
            elif ko:
                position += '박'
            parts.append(f'{name} {position}')
        kind = {'tap':'일반 노트','longnote':'롱노트'}[note['type']] if ko else note['type']
        command = ' '.join(['ez2rc', 'choose', str(number), letter, *flags])
        lines.append(f'{letter}   {kind}   {" / ".join(parts)}     {command}')
    remove = '노트가 아니라면 채보에서 삭제' if ko else 'No note here: remove it'
    lines.append(f'X   {remove}     ez2rc choose {number} X')
    return lines


def _requests(draft):
    original_notes = {n['id']: n for group in ('notes', 'alternatives')
                      for n in draft['automatic'][group]}
    items = pending_items(draft)
    groups = [(number, item.options) for number, item in items.items()]
    if not groups:
        notes = sorted((n for n in draft['notes'] if witness_time(n) is not None), key=witness_time)
        if notes:
            groups = [(i+1, {'': notes[index]}) for i, index in enumerate(
                sorted({0, len(notes)//2, len(notes)-1}))]
    requests = defaultdict(list)
    missing = []
    for number, options in groups:
        frames = defaultdict(list)
        for letter, note in options.items():
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
            requests[frame_id].append((name, number, marks, options))
    return requests, missing


def _photo(frame, number, marks, options, rois, offset, boxes):
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
            raise ValueError('원본 영상의 표시 위치가 화면 밖입니다.')
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
    font = _korean_font()
    ko = font is not None
    check = 'A' in options
    if check:
        heading = (f'확인 {number}번   ' if ko else f'Check {number}: ') + review_question(options, ko)
    else:
        heading = '대표 장면 · 선택할 항목이 없습니다' if ko else 'Preview only - no decision needed'
    labels = []
    for letter, side, lane, _, _ in marks:
        if ko:
            endpoint = '시작' if side == 'start' else '끝'
            labels.append(f'{letter} {lane+1}번 레인 {endpoint}'.strip())
        else:
            labels.append(f'{letter} lane {lane+1} {side}'.strip())
    seconds = frame.stamp.timestamp_ms / 1000
    lines = [heading, f'{seconds:.3f}s   ' + ' / '.join(dict.fromkeys(labels))]
    if not marked_count:
        lines.append('위치나 크기 정보가 없어 테두리 없이 표시합니다.' if ko else 'Original frame without position markers')
    else:
        lines.append('분홍색 테두리가 감지된 노트의 경계입니다.' if ko else
                     'The pink outline shows the detected note footprint.')
    footer = _answers(options, number, ko) if check else []
    if check:
        footer.append('판단하기 어려우면 입력하지 않고 다음 사진을 보세요.' if ko else
                      'Unsure? Leave this item open and check another photo.')
    row_height = 32
    width = max(680, shot.shape[1] + 24)
    for line in lines + footer:
        if ko:
            x, _, w, _ = cv2.getTextSize((0,0), line, (0,0), font, 20)
            width = max(width, x+w+36)
        else:
            width = max(width, cv2.getTextSize(line,cv2.FONT_HERSHEY_SIMPLEX,.58,1)[0][0]+36)
    top_margin = row_height * len(lines) + 10
    out = cv2.copyMakeBorder(shot, top_margin, row_height*len(footer)+10, 12,
                             width-shot.shape[1]-12, cv2.BORDER_CONSTANT, value=(32,32,32))
    def caption(line, y, color):
        if ko:
            cv2.putText(out, line, (12,y), color, font, 20)
        else:
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
        raise ValueError(f'사진 폴더가 이미 있습니다: {destination}. 다른 --output 폴더를 지정하세요.')
    source = draft['source']
    path = Path(video or source.get('video', ''))
    if not path.is_file() or file_hash(path) != source.get('video_sha256'):
        raise ValueError('초안과 같은 원본 영상을 --video로 지정하세요.')
    rois = source.get('lane_rois')
    policy = source.get('timestamp_policy', 'strict')
    if analysis_dir and rois is None:
        manifest_path = Path(analysis_dir)/'artifacts/observations/manifest.json'
        if manifest_path.is_file():
            manifest = read_json(manifest_path, 'ez2rc.stage-cache')
            checksum = digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'})
            if manifest['manifest_sha256'] != checksum or manifest['cache_id'] != source.get('observation_cache_id'):
                raise ValueError('초안과 관측 자료가 일치하지 않습니다.')
            meta = manifest['metadata']
            rois = [lane['roi'] for lane in meta['profile']['lanes']]
            policy = meta['capture']['timestamp_policy']
    if rois is not None and len(rois) != draft['metadata']['lane_count']:
        raise ValueError('초안과 레인 정보가 일치하지 않습니다.')
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
            for name, number, marks, options in requests.pop(frame.stamp.frame_id, []):
                image = _photo(frame, number, marks, options, rois, source.get('alignment_offset_px'), boxes)
                if not cv2.imwrite(str(stage/name), image):
                    raise OSError(f'사진을 저장하지 못했습니다: {name}')
                count += 1
                if progress:
                    progress(f'원영상 사진 {count}/{total}')
            if not requests:
                break
        if requests:
            raise ValueError('원영상에서 일부 관측 프레임을 찾을 수 없습니다.')
        if file_hash(path) != source['video_sha256']:
            raise ValueError('사진 생성 중 원본 영상이 변경되었습니다.')
        stage.rename(destination)
    return count, missing
