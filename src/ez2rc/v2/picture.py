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

from .chart_layout import (COLUMN_HEIGHT, COLUMN_MEASURES, KEY_COLORS, LANE_FILLS, MEASURE_RAIL,
                           chart_columns, fit_hold_columns, number_rail_width)
from .contracts import ratio, read_ratio
from .draft import beat_text, change_note, choose_tap_placement, start_choices
from .measure_grid import MeasureGrid
from .rhythm import DEFAULT_SUBDIVISIONS

IMAGE_NAME = 'chart.png'


@dataclass
class Item:
    number: int
    note: dict
    options: dict

    def frame_note(self, letter, side):
        return self.options.get(letter)

    def photo_notes(self):
        return self.options.items()


@dataclass
class TapPlacementItem:
    number: int
    placements: dict

    def lines(self):
        early, late = (self.placements[letter][0]['start_beat'] for letter in ('A', 'B'))
        shift = read_ratio(late) - read_ratio(early)
        yield f'#{self.number}    {len(self.placements["A"])} taps. B is {shift} beat later than A.'
        for letter, notes in self.placements.items():
            lanes = defaultdict(list)
            for note in notes:
                lanes[read_ratio(note['start_beat'])].append(note['lane'] + 1)
            for beat in sorted(lanes):
                numbers = ', '.join(str(lane) for lane in sorted(lanes[beat]))
                yield f'#{self.number}{letter}    beat {beat_text(ratio(beat))}    lanes {numbers}'

    def frame_note(self, letter, side):
        notes = self.placements.get(letter)
        if notes:
            return next((n for n in notes if (n['evidence'].get(side or 'start') or {}).get('witnesses')),
                        notes[0])

    def photo_notes(self):
        # Both placements refer to the same observed footprints.
        return (('A/B', note) for note in self.placements['A'])


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
    pairs = draft['projection'].get('tap_placements', {})
    if any(row != current.get(row['id']) for row in automatic['notes'] if row['id'] in pairs):
        pairs = {}  # A manual edit breaks the group. Keep the other taps individually accessible.
    items = {}
    number = 0
    group_shown = False
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
        if note['id'] in pairs:
            if not group_shown:
                placements = {letter: [{**current[key], 'start_beat': pair[i]} for key, pair in pairs.items()]
                              for i, letter in enumerate(('A', 'B'))}
                items[number] = TapPlacementItem(number, placements)
                group_shown = True
            continue
        options = {'A': note}
        for index, other in enumerate(sorted(alternatives, key=lambda r: r['id']), 1):
            if other['id'] not in current:
                options[option_letter(index)] = other
        for index, beats in enumerate(timing, len(alternatives) + 1):
            if any(beats[key] != note[key] for key in ('start_beat', 'end_beat')):
                options[option_letter(index)] = {**note, **beats}
        # Append grid choices after the letters already allocated to physical and timing alternatives.
        index = len(alternatives) + len(timing) + 1
        for candidate in list(options.values()):
            for start in start_choices(candidate):
                options[option_letter(index)] = {**candidate, 'start_beat': start}
                index += 1
        items[number] = Item(number, note, options)
    return items


def decide(draft, item, letter, fields):
    if isinstance(item, TapPlacementItem):
        if fields or letter not in item.placements:
            raise ValueError('Choose A or B to place all taps together, or leave the item open.')
        return choose_tap_placement(draft, letter)
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
    beats = {key: candidate[key] for key in ('start_beat', 'end_beat')}
    if candidate['id'] == identity:
        return change_note(draft, 'edit', identity=identity, fields={**beats, **fields}, confirm=True)
    removed = change_note(draft, 'delete', identity=identity)
    return change_note(removed, 'adopt', identity=candidate['id'], fields={**beats, **fields}, confirm=True)


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


def label(canvas, text, x, y, color=(208, 214, 222), size=.48):
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


def _start_labels(draft, grid):
    """Lay out one fractional beat label per occupied row, including chords."""
    positions = defaultdict(set)
    if 'start_ticks_per_beat' in draft['projection']:
        for note in draft['notes']:
            if note['start_beat'] is not None:
                beat = read_ratio(note['start_beat'])
                positions[grid.column(beat, COLUMN_MEASURES)].add(beat)
    labels = defaultdict(list)
    slot_width = slots_needed = 0
    for block, beats in positions.items():
        base = grid.boundary(block * COLUMN_MEASURES)
        end = grid.boundary((block + 1) * COLUMN_MEASURES)
        # Nearby starts can occupy adjacent pixel rows. Stagger their labels
        # horizontally and keep the guide for each at its actual beat.
        last_rows = []
        for beat in sorted(beats, reverse=True):
            row = round(COLUMN_HEIGHT * (end - beat) / (end - base))
            caption = beat_text(ratio(beat))
            width = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, .38, 1)[0][0]
            slot = next((i for i, last in enumerate(last_rows) if row-last >= 14), len(last_rows))
            if slot == len(last_rows):
                last_rows.append(row)
            else:
                last_rows[slot] = row
            labels[block].append((row, slot, caption))
            slot_width = max(slot_width, width + 7)
        slots_needed = max(slots_needed, len(last_rows))
    return labels, slots_needed, slot_width


def chart_strips(draft, items):
    lanes = draft['metadata']['lane_count']
    grid = MeasureGrid(draft['projection'])
    lane_width = 36
    grid_top, grid_height = 32, COLUMN_HEIGHT
    grid_bottom = grid_top + grid_height
    bottom_start = draft['projection'].get('default_start_reference') == 'template_bottom'
    phase = Fraction(0) if bottom_start else _display_phase(draft)

    def displayed(beat):
        corrected = beat - phase
        block = grid.column(corrected, COLUMN_MEASURES)
        span = grid.boundary((block+1)*COLUMN_MEASURES) - grid.boundary(block*COLUMN_MEASURES)
        tolerance = min(span / grid_height, Fraction(1, 96))
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
        first = grid.column(start if start is not None else end, COLUMN_MEASURES, end=start is None)
        last = grid.column(end, COLUMN_MEASURES, end=True) if start is not None and end is not None else first
        if first < last:
            start, end, first, last = fit_hold_columns(start, end, first, last, lambda block: grid.boundary(block * COLUMN_MEASURES))
        # Distant endpoints of a manually entered hold need no intervening empty panels.
        indices = range(first, last + 1) if last - first < 256 else (first, last)
        for block in indices:
            blocks[block].append((note['lane'], start, end, tag, alternative))
    indexed = {item.note['id']: item for item in items.values() if isinstance(item, Item)}
    for note in draft['notes']:
        item = indexed.get(note['id'])
        place(note, f'{item.number}A' if item else '')
    for number, item in items.items():
        if isinstance(item, TapPlacementItem):
            for letter, notes in item.placements.items():
                for note in notes:
                    place(note, f'{number}{letter}', letter == 'B', grid_choice=True)
            continue
        for letter, other in item.options.items():
            if letter != 'A':
                place(other, f'{number}{letter}', True, grid_choice=other['id'] == item.note['id'])

    ordered = list(chart_columns(blocks))
    rail_width = number_rail_width(blocks, grid if grid.numbered else None)
    start_labels, slots, slot_width = _start_labels(draft, grid)
    panel = rail_width + lanes * lane_width + 10 + slots * slot_width

    # The score runs in one row. Place the review key to its right so that
    # adding measures or review choices extends the image horizontally.
    legend = []
    for item in items.values():
        if isinstance(item, TapPlacementItem):
            legend.extend(item.lines())
            continue
        for letter, candidate in item.options.items():
            seconds = witness_time(candidate)
            when = f'   video {seconds:.3f}s' if seconds is not None else ''
            legend.append(f'#{item.number}{letter}    lane {candidate["lane"]+1}    {description(candidate)}{when}')
    line_height = 25
    rows_per_column = grid_height // line_height
    legend_columns = []
    right_edge = len(ordered) * panel
    for offset in range(0, len(legend), rows_per_column):
        lines = legend[offset:offset + rows_per_column]
        x = right_edge + 16
        legend_columns.append((x, lines))
        right_edge = x + max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, .48, 1)[0][0]
                             for line in lines)
    width = max(720, right_edge + 14)

    def blank(height):
        return np.full((height, width, 3), (6, 8, 11), dtype=np.uint8)

    header = blank(70)
    label(header, f'{draft["bpm"]} BPM   |   {len(draft["notes"])} notes   |   {len(items)} to check',
          14, 28, size=.7)
    label(header, 'Read upwards, then move right.' if grid.numbered else
          'Read upwards, then move right.   Scale: beats', 14, 54)
    if phase:
        label(header, f'Auto-start display shift: {float(-phase):+.4f} beat', 370, 54)
    yield header

    if ordered or legend_columns:
        height = grid_bottom + 24 if ordered else grid_top + min(len(legend), rows_per_column) * line_height
        strip = blank(height)
        for column, block in enumerate(ordered):
            first_measure = block * COLUMN_MEASURES
            base = grid.boundary(first_measure)
            span = grid.boundary(first_measure + COLUMN_MEASURES) - base
            left = column * panel + 3 + rail_width
            right = left + lanes * lane_width

            def y(beat):
                return grid_bottom - round(grid_height * (beat-base)/span)

            for lane in range(lanes):
                x = left + lane * lane_width
                shade = LANE_FILLS[lane % 2]
                cv2.rectangle(strip, (x, grid_top), (x+lane_width-1, grid_bottom), shade, -1)
                cv2.line(strip, (x, grid_top), (x, grid_bottom), (54, 60, 68))
                label(strip, str(lane+1), x+12, grid_top-10, size=.48)

            # Keep guides for occupied subdivisions between quarter-beat lines.
            subdivisions = {p for _, start, end, *_ in blocks[block] for p in (start, end)
                            if p is not None and base <= p < base+span and p.denominator in DEFAULT_SUBDIVISIONS}
            for point in subdivisions:
                cv2.line(strip, (left, y(point)), (right-1, y(point)), (73, 83, 96))
            # Draw stronger lines last when several ticks share a pixel.
            for division, color in ((4, (73, 83, 96)), (1, (106, 118, 133))):
                first = math.ceil(base * division)
                stop = math.ceil((base + span) * division)
                if span * division > grid_height:
                    # Subpixel ticks cover consecutive rows even for very long measures.
                    rows = range(y(Fraction(stop - 1, division)), y(Fraction(first, division)) + 1)
                else:
                    rows = (y(Fraction(tick, division)) for tick in range(first, stop))
                for row in rows:
                    cv2.line(strip, (left, row), (right-1, row), color)
            rail = strip[:, left-rail_width:left]
            boundary_rows = {grid_top, grid_bottom}
            cv2.rectangle(rail, (0, grid_top), (rail_width-1, grid_bottom), MEASURE_RAIL, -1)
            for index in range(first_measure, first_measure + COLUMN_MEASURES):
                if not grid.visible(index):
                    continue
                beat = grid.boundary(index)
                position = y(beat)
                boundary_rows.add(position)
                caption = grid.label(index) if grid.numbered else str(beat)
                label_y = (position + y(grid.boundary(index+1))) // 2 if grid.numbered else position - 9
                (w, h), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)
                label(rail, caption, (rail_width-w)//2, label_y+h//2, (245, 247, 250), .4)
                cv2.line(rail, (0, position), (rail_width-1, position), (185, 191, 200))
                cv2.line(strip, (left, position), (right-1, position), (158, 168, 180))
            if grid.visible(first_measure + COLUMN_MEASURES):
                cv2.line(rail, (0, grid_top), (rail_width-1, grid_top), (185, 191, 200))
                cv2.line(strip, (left, grid_top), (right-1, grid_top), (158, 168, 180))

            markers = defaultdict(list)
            for lane, start, end, tag, alternative in blocks[block]:
                x = left + lane * lane_width + 1
                top = y(min(base+span, end)) if end is not None else grid_top
                bottom = y(max(base, start)) if start is not None else grid_bottom
                color = KEY_COLORS[lane % 2]
                if tag:
                    color = (106, 226, 208) if alternative else (255, 199, 86)
                if start is not None and end is not None:
                    body = tuple(round(channel * .72) for channel in color)
                    if alternative:
                        # An outlined continuation has no end cap at the column break.
                        for side in (x+5, x+lane_width-8):
                            cv2.line(strip, (side, top), (side, bottom), body)
                    else:
                        cv2.rectangle(strip, (x+5, top), (x+lane_width-8, bottom), body, -1)
                for point, is_tail in ((start, False), (end, True)):
                    if point is None:
                        continue
                    in_column = base < point <= base+span if is_tail else base <= point < base+span
                    if in_column:
                        position = y(point)
                        # Paint caps inward so a short hold ends at its observed row.
                        cap_top = position if is_tail else max(top, position-3)
                        cap_bottom = min(bottom, position+3) if is_tail and position not in boundary_rows else position
                        cv2.rectangle(strip, (x, cap_top), (x+lane_width-3, cap_bottom),
                                      color, 1 if alternative else -1)
                        if tag:
                            markers[x, position].append(tag)
            for (x, position), tags in markers.items():
                numbers = dict.fromkeys(tag.rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for tag in tags)
                caption = tags[0] if len(tags) == 1 else '/'.join(numbers)
                label(strip, caption, x, position-8, (255, 233, 150), .48)
            for row, slot, caption in start_labels[block]:
                position = grid_top + row
                x = right + 5 + slot * slot_width
                cv2.line(strip, (x-4, position), (x-2, position), (123, 137, 154))
                label(strip, caption, x, position+4, (182, 194, 208), .38)
        for x, lines in legend_columns:
            for row, line in enumerate(lines):
                label(strip, line, x, grid_top + row * line_height)
        yield strip
    if items:
        strip = blank(36)
        hint = ('Choose A or B for the tap group, or leave it open. X removes individual notes.'
                if any(isinstance(item, TapPlacementItem) for item in items.values()) else
                'Choose a letter to keep that note, or X to remove it.  ? = unknown beat.')
        label(strip, hint, 16, 22)
        yield strip


def write_picture(path, draft):
    """Encode an RGB chart one scanline at a time, then atomically publish it."""
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
                scanline = np.empty(width*3+1, dtype=np.uint8)
                scanline[0] = 0  # PNG filter type
                for row in strip:
                    scanline[1:] = row.reshape(-1)
                    data = encoder.compress(scanline)
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
