"""Recover the matched template footprint in aligned image coordinates."""
import cv2

from .observation import KINDS, template_shapes
from .storage import Cache


def witness_kind(note_type, side):
    if side == 'end':
        return 'tail'
    return 'tap' if note_type == 'tap' else 'head'


OUTLINE_PINK = (210, 72, 255)  # BGR
OUTLINE_DARK = (22, 12, 24)


def draw_outline(image, top_left, bottom_right, kind):
    """Outline the matched pixels, leaving the side facing a hold body open."""
    left, top = top_left
    right, bottom = bottom_right
    # Grow outward so small note interiors and the open end remain visible.
    for pad, color in ((0, OUTLINE_PINK), (1, OUTLINE_PINK), (2, OUTLINE_DARK)):
        x1, x2 = left - pad, right + pad
        y1 = top if kind == 'head' else top - pad
        y2 = bottom if kind == 'tail' else bottom + pad
        cv2.line(image, (x1, y1), (x1, y2), color)
        cv2.line(image, (x2, y1), (x2, y2), color)
        if kind != 'head':
            cv2.line(image, (x1, y1), (x2, y1), color)
        if kind != 'tail':
            cv2.line(image, (x1, y2), (x2, y2), color)


def matched_boxes(witnesses, draft, analysis_dir):
    boxes = {}
    missing = {}
    for witness, kind in witnesses:
        key = (witness.get('observation_id'), kind)
        if witness.get('box_px') is not None:
            boxes[key] = witness['box_px']
        elif key[0]:
            missing[key] = witness
    if not missing or analysis_dir is None:
        return boxes
    path = analysis_dir / 'artifacts/observations'
    if not path.is_dir():
        return boxes
    cache = Cache(path)
    if cache.id != draft['source'].get('observation_cache_id'):
        raise ValueError('초안과 관측 자료가 일치하지 않습니다.')
    shapes = template_shapes(cache.manifest['key']['parameters'])
    if shapes is None:
        return boxes
    frames = {w['frame_id'] for w in missing.values()}
    for chunk in cache.chunks():
        for i, (frame, lane, seed) in enumerate(chunk['candidate_index']):
            if frame not in frames:
                continue
            for k, kind in enumerate(KINDS):
                key = (f'f{frame}:l{lane}:y{seed}', kind)
                if key in missing:
                    x, y = map(int, chunk['positions'][i,k])
                    if y != missing[key]['y_px']:
                        raise ValueError('저장된 노트 위치와 관측 좌표가 일치하지 않습니다.')
                    height, width = shapes[lane][kind]
                    boxes[key] = [x, y, width, height]
                    del missing[key]
        if not missing:
            break
    return boxes
