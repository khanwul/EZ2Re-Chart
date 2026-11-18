"""Numbered draft decisions and a standalone PNG beat chart."""
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from statistics import median
import math
import os
import struct
import tempfile
import zlib

import cv2
import numpy as np

from .contracts import read_ratio
from .draft import change_note

IMAGE_NAME = 'chart.png'


@dataclass
class Item:
    number: int
    note: dict
    options: dict


def option_letter(index):
    """A ... W, Y, Z, AA ... (X is reserved for removing a note)."""
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWYZ'
    result = ''
    while True:
        index, digit = divmod(index, len(alphabet))
        result = alphabet[digit] + result
        if not index:
            return result
        index -= 1


def pending_items(draft):
    automatic = draft['automatic']
    by_event = defaultdict(list)
    selected_count = defaultdict(int)
    for row in automatic['notes']:
        selected_count[row['evidence'].get('event_id')] += 1
    for row in automatic['alternatives']:
        event = row['evidence'].get('event_id')
        if event is not None:
            by_event[event].append(row)
    current = {row['id']: row for row in draft['notes']}
    items = {}
    number = 0
    for original in automatic['notes']:
        event = original['evidence'].get('event_id')
        alternatives = by_event.get(event, []) if selected_count[event] == 1 else []
        timing = draft['projection'].get('timing_choices', {}).get(original['id'], [])
        if original['status'] != 'unresolved' and not original['issues'] and not alternatives and not timing:
            continue
        number += 1  # Allocate before filtering completed decisions.
        note = current.get(original['id'])
        if note is None or note['status'] == 'reviewed':
            continue
        options = {'A': note}
        for index, other in enumerate(sorted(alternatives, key=lambda r: r['id']), 1):
            if other['id'] not in current:
                options[option_letter(index)] = other
        for index, beats in enumerate(timing, len(alternatives) + 1):
            if any(beats[key] != note[key] for key in ('start_beat', 'end_beat')):
                options[option_letter(index)] = {**note, **beats}
        items[number] = Item(number, note, options)
    return items


def decide(draft, item, letter, fields):
    identity = item.note['id']
    if letter == 'X':
        if fields:
            raise ValueError('Omit beat values when removing a note.')
        return change_note(draft, 'delete', identity=identity)
    if letter not in item.options:
        raise ValueError('Enter a choice shown in the chart image.')
    candidate = item.options[letter]
    for side in ('start', 'end') if candidate['type'] == 'longnote' else ('start',):
        key = side + '_beat'
        if fields.get(key, candidate[key]) is None:
            raise ValueError(f'Check the position and supply the beat with --{side}.')
    if candidate['id'] == identity:
        beats = {key: candidate[key] for key in ('start_beat', 'end_beat')}
        return change_note(draft, 'edit', identity=identity, fields={**beats, **fields}, confirm=True)
    removed = change_note(draft, 'delete', identity=identity)
    return change_note(removed, 'adopt', identity=candidate['id'], fields=fields, confirm=True)


def beat_text(value):
    return '?' if value is None else f'{float(read_ratio(value)):.4f}'.rstrip('0').rstrip('.')


def description(note):
    start = beat_text(note['start_beat'])
    if note['type'] == 'tap':
        return f'tap {start}'
    return f'hold {start} -> {beat_text(note["end_beat"])}'


def witness_time(note):
    for side in ('start', 'end'):
        witnesses = (note['evidence'].get(side) or {}).get('witnesses', [])
        if witnesses:
            return witnesses[0]['timestamp_ms'] / 1000
    return None


def label(canvas, text, x, y, color=(35, 43, 55), size=.48):
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def _display_phase(draft):
    """Estimate a small shared offset from independent occupied quarter-beat slots."""
    slots = defaultdict(list)
    for note in draft.get('automatic', {}).get('notes', ()):
        if note['start_beat'] is not None:
            beat = read_ratio(note['start_beat'])
            slot = round(beat * 4)
            slots[slot].append(beat - Fraction(slot, 4))
    if len(slots) < 8:
        return Fraction(0)
    residuals = [median(values) for values in slots.values()]
    phase = median(residuals)
    agreeing = sum(abs(value - phase) <= Fraction(1, 96) for value in residuals)
    if abs(phase) <= Fraction(1, 16) and agreeing * 3 >= len(slots) * 2:
        return phase
    return Fraction(0)


def chart_strips(draft, items):
    lanes = draft['metadata']['lane_count']
    measure = draft['projection'].get('measure_beats')
    measure = read_ratio(measure) if measure is not None else None
    span = measure * 4 if measure is not None else Fraction(16)
    if span <= 0:
        raise ValueError('invalid measure length')
    lane_width = 28
    panel = 50 + lanes * lane_width
    grid_top, grid_bottom = 32, 880
    bottom_start = draft['projection'].get('default_start_reference') == 'template_bottom'
    phase = Fraction(0) if bottom_start else _display_phase(draft)
    tolerance = min(span / (grid_bottom-grid_top), Fraction(1, 96))

    def displayed(beat):
        corrected = beat - phase
        # Include straight and triplet subdivisions. Only absorb residual
        # jitter if the correction is at most one displayed pixel.
        for division in (1, 2, 3, 4, 6, 8, 12, 16):
            tick = Fraction(round(corrected * division), division)
            if abs(tick - corrected) <= tolerance:
                return tick
        return corrected

    blocks = defaultdict(list)
    def place(note, tag='', alternative=False, grid_choice=False):
        automatic = (note.get('origin') == 'automatic' and note.get('status') != 'reviewed'
                     and not grid_choice)
        start = read_ratio(note['start_beat']) if note['start_beat'] is not None else None
        end = read_ratio(note['end_beat']) if note['end_beat'] is not None else None
        if start is None and end is None:
            return
        if start is not None and automatic and not bottom_start:
            display_start = displayed(start)
            if end is None or display_start < end:
                start = display_start
        first = (start if start is not None else end) // span
        last = math.ceil(end / span) - 1 if start is not None and end is not None else first
        # Distant endpoints of a manually entered hold need no intervening empty panels.
        indices = range(first, last + 1) if last - first < 256 else (first, last)
        for block in indices:
            blocks[block].append((note['lane'], start, end, tag, alternative))
    indexed = {item.note['id']: item for item in items.values()}
    for note in draft['notes']:
        item = indexed.get(note['id'])
        place(note, f'{item.number}A' if item else '')
    for number, item in items.items():
        for letter, other in item.options.items():
            if letter != 'A':
                place(other, f'{number}{letter}', True, grid_choice=other['id'] == item.note['id'])

    # Keep rests in the score, including the lead-in from beat zero. Extremely
    # distant manually entered beats get an explicitly labelled jump instead.
    ordered = []
    previous = min(0, min(blocks, default=0)) - 1
    for block in sorted(blocks):
        if block - previous <= 256:
            ordered.extend(range(previous + 1, block + 1))
        else:
            ordered.append(block)
        previous = block

    cols = max(1, min(32, len(ordered), max(1, 12000 // panel)))
    width = max(720, cols * panel + 14)

    def blank(height):
        return np.full((height, width, 3), (248, 249, 251), dtype=np.uint8)

    header = blank(70)
    label(header, f'{draft["bpm"]} BPM   |   {len(draft["notes"])} notes   |   {len(items)} to check',
          14, 28, size=.7)
    label(header, 'Read upwards, then move right.   M = measure' if measure else
          'Read upwards, then move right.   Scale: beats', 14, 54)
    if phase:
        label(header, f'Auto-start display shift: {float(-phase):+.4f} beat', 370, 54)
    yield header

    for offset in range(0, len(ordered), cols):
        strip = blank(grid_bottom + 24)
        for column, block in enumerate(ordered[offset:offset + cols]):
            base = block * span
            left = column * panel + 43
            right = left + lanes * lane_width

            def y(beat):
                return grid_bottom - round((grid_bottom-grid_top) * (beat-base)/span)

            for lane in range(lanes):
                x = left + lane * lane_width
                shade = (27, 32, 40) if lane % 2 == 0 else (37, 44, 54)
                cv2.rectangle(strip, (x, grid_top), (x+lane_width-1, grid_bottom), shade, -1)
                label(strip, str(lane+1), x+8, grid_top-10, size=.4)

            # Keep guides for occupied subdivisions between quarter-beat lines.
            subdivisions = {p for _, start, end, _, _ in blocks[block] for p in (start, end)
                            if p is not None and base <= p < base+span and p.denominator in (3,6,8,12,16)}
            for point in subdivisions:
                cv2.line(strip, (left, y(point)), (right-1, y(point)), (73, 83, 96))
            # Draw stronger lines last when several ticks share a pixel.
            for division, color in ((4, (73, 83, 96)), (1, (106, 118, 133))):
                first = math.ceil(base * division)
                stop = math.ceil((base + span) * division)
                if span * division > grid_bottom - grid_top:
                    # Subpixel ticks cover consecutive rows even for very long measures.
                    rows = range(y(Fraction(stop - 1, division)), y(Fraction(first, division)) + 1)
                else:
                    rows = (y(Fraction(tick, division)) for tick in range(first, stop))
                for row in rows:
                    cv2.line(strip, (left, row), (right-1, row), color)
            major = measure or Fraction(4)
            for index in range(4):
                beat = base + index*major
                position = y(beat)
                caption = f'M{beat//major+1}' if measure else str(beat)
                label(strip, caption, left-42, position-4, size=.42)
                cv2.line(strip, (left, position), (right-1, position), (158, 168, 180))
            cv2.line(strip, (left, grid_top), (right-1, grid_top), (158, 168, 180))

            markers = defaultdict(list)
            for lane, start, end, tag, alternative in blocks[block]:
                x = left + lane * lane_width + 1
                top = y(min(base+span, end)) if end is not None else grid_top
                bottom = y(max(base, start)) if start is not None else grid_bottom
                color = (74, 197, 245) if lane % 2 else (232, 238, 247)
                if tag:
                    color = (106, 226, 208) if alternative else (255, 199, 86)
                if start is not None and end is not None:
                    body = tuple(round(channel * .72) for channel in color)
                    cv2.rectangle(strip, (x+1, top), (x+lane_width-4, bottom), body,
                                  1 if alternative else -1)
                for point, is_tail in ((start, False), (end, True)):
                    if point is not None and base <= point <= base+span:
                        position = y(point)
                        # Paint caps inward so a short hold ends at its observed row.
                        cap_top = position if is_tail else max(top, position-4)
                        cap_bottom = min(bottom, position+4) if is_tail else position
                        cv2.rectangle(strip, (x, cap_top), (x+lane_width-3, cap_bottom),
                                      color, 1 if alternative else -1)
                        if tag:
                            markers[x, position].append(tag)
            for (x, position), tags in markers.items():
                numbers = dict.fromkeys(tag.rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for tag in tags)
                caption = tags[0] if len(tags) == 1 else '/'.join(numbers)
                label(strip, caption, x, position-6, (255, 233, 150), .4)
        yield strip

    for item in items.values():
        for letter, candidate in item.options.items():
            strip = blank(25)
            seconds = witness_time(candidate)
            when = f'   video {seconds:.3f}s' if seconds is not None else ''
            label(strip, f'#{item.number}{letter}    lane {candidate["lane"]+1}    {description(candidate)}{when}', 16, 18)
            yield strip
    if items:
        strip = blank(36)
        label(strip,'Choose a letter to keep that note, or X to remove it.  ? = unknown beat.',16,22)
        yield strip


def write_picture(path, draft):
    """Encode a tall RGB chart one strip at a time, then atomically publish it."""
    items = pending_items(draft)
    path = Path(path)
    def chunk(stream, kind, payload):
        stream.write(struct.pack('>I', len(payload)) + kind + payload)
        stream.write(struct.pack('>I', zlib.crc32(payload, zlib.crc32(kind))))
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
    try:
        with temp.open('wb') as stream:
            stream.write(b'\x89PNG\r\n\x1a\n' + bytes(25))
            encoder = zlib.compressobj(4)
            height = 0
            for strip in chart_strips(draft, items):
                rows, width, _ = strip.shape
                height += rows
                scanlines = np.zeros((rows, width*3+1), dtype=np.uint8)
                scanlines[:,1:] = strip.reshape(rows,-1)
                data = encoder.compress(scanlines.tobytes())
                if data:
                    chunk(stream,b'IDAT',data)
            chunk(stream,b'IDAT',encoder.flush())
            chunk(stream,b'IEND',b'')
            stream.seek(8)
            chunk(stream,b'IHDR',struct.pack('>IIBBBBB',width,height,8,2,0,0,0))
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)
    return len(items)
