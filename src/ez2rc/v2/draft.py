"""Portable editable drafts with atomic saves and replayable user changes."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
import re
import tempfile
from pathlib import Path
import uuid

from .contracts import Metadata, decimal_bpm, read_ratio, ratio
from .rhythm import DRAFT_GRID
from .storage import digest, read_json, write_json


def _note(row, lane_count):
    if not isinstance(row, dict) or set(row) != {
            'id', 'lane', 'type', 'start_beat', 'end_beat', 'status', 'issues', 'evidence', 'origin'}:
        raise ValueError('invalid draft note fields')
    if not isinstance(row['id'], str) or not row['id']:
        raise ValueError('note ID must be nonempty')
    if type(row['lane']) is not int or not 0 <= row['lane'] < lane_count:
        raise ValueError(f"lane must be between 1 and {lane_count}")
    if row['type'] not in ('tap', 'longnote'):
        raise ValueError('note type must be tap or longnote')
    start, end = (read_ratio(row[k]) if row[k] is not None else None for k in ('start_beat', 'end_beat'))
    if row['type'] == 'tap' and end is not None:
        raise ValueError('a tap cannot have an end beat; set --end none')
    if start is not None and end is not None and end <= start:
        raise ValueError('long-note end must follow its start')
    if row['status'] not in ('draft', 'unresolved', 'reviewed'):
        raise ValueError('invalid note review status')
    if (start is None or row['type'] == 'longnote' and end is None) and row['status'] != 'unresolved':
        raise ValueError('a note with missing beats must remain unresolved')
    if row['origin'] not in ('automatic', 'user', 'user_edit'):
        raise ValueError('invalid note origin')
    if (not isinstance(row['issues'], list) or any(not isinstance(s, str) or not s for s in row['issues'])
            or not isinstance(row['evidence'], dict)):
        raise ValueError('invalid note evidence or issues')


def validate_draft(value):
    try:
        if value['format'] != 'ez2rc.draft' or type(value['schema_version']) is not int or value['schema_version'] != 1:
            raise ValueError('unsupported draft schema')
        metadata = Metadata(**value['metadata'])
        decimal_bpm(value['bpm'])
        if value['status'] != 'draft' or value['projection']['tail_beat_extension'] != [0, 1]:
            raise ValueError('unsupported draft convention or status')
        if type(value['revision']) is not int or value['revision'] != len(value['history']):
            raise ValueError('draft revision disagrees with edit history')
        automatic = value['automatic']
        rows = automatic['notes'] + automatic['alternatives']
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('duplicate automatic candidate ID')
        for row in rows + value['notes']:
            _note(row, metadata.lane_count)
        replay = {r['id']: deepcopy(r) for r in automatic['notes']}
        if 'tap_placements' in value['projection']:
            placements = value['projection']['tap_placements']
            if not isinstance(placements, dict) or len(placements) < 2:
                raise ValueError('a placement needs at least two taps')
            shifts = set()
            for identity, pair in placements.items():
                note = replay.get(identity)
                if (not note or note['type'] != 'tap' or note['start_beat'] is not None
                        or not isinstance(pair, list) or len(pair) != 2):
                    raise ValueError('invalid placement for an unplaced automatic tap')
                early, late = map(read_ratio, pair)
                shifts.add(late - early)
            if len(shifts) != 1 or min(shifts) <= 0:
                raise ValueError('tap placements must have one positive beat shift')
        for change in value['history']:
            identity, before, after = change['id'], change['before'], change['after']
            if replay.get(identity) != before or before is None and after is None:
                raise ValueError('edit history does not match its previous note')
            if after is None:
                del replay[identity]
            else:
                _note(after, metadata.lane_count)
                if after['id'] != identity:
                    raise ValueError('edited note identity mismatch')
                replay[identity] = after
        current = {r['id']: r for r in value['notes']}
        if len(current) != len(value['notes']) or current != replay:
            raise ValueError('saved notes disagree with edit history')
        if value['document_sha256'] != digest({k: v for k, v in value.items() if k != 'document_sha256'}):
            raise ValueError('draft checksum mismatch; use the editing commands to change notes')
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f'invalid draft: {exc}') from exc
    return value


def seal(value):
    value['document_sha256'] = digest({k: v for k, v in value.items() if k != 'document_sha256'})
    return validate_draft(value)


def create_draft(projection, metadata, bpm, source):
    notes, alternatives = deepcopy(projection['notes']), deepcopy(projection['alternatives'])
    return seal({'format': 'ez2rc.draft', 'schema_version': 1, 'status': 'draft',
                 'metadata': metadata, 'bpm': bpm, 'source': source,
                 'projection': {k: v for k, v in projection.items() if k not in ('notes', 'alternatives')},
                 'automatic': {'notes': notes, 'alternatives': alternatives},
                 'notes': deepcopy(notes), 'revision': 0, 'history': []})


def read_draft(path):
    return validate_draft(read_json(Path(path), 'ez2rc.draft'))


@contextmanager
def draft_lock(path):
    lock = path.with_name('.' + path.name + '.lock')
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise ValueError(f'draft is locked by another save: {lock}') from exc
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink()


def save_draft(path, value, *, expected_sha=None, picture_path=None):
    """Lock and compare before replacing; failed writes preserve the old file."""
    validate_draft(value)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with draft_lock(path):
        if expected_sha is None:
            if path.exists():
                raise ValueError(f'output already exists: {path}')
        elif not path.is_file() or read_draft(path)['document_sha256'] != expected_sha:
            raise ValueError('draft changed since it was opened; reopen before saving')
        if picture_path is None and path.name == 'draft.json' and path.parent.name == '.ez2rc' and expected_sha:
            from .picture import IMAGE_NAME
            picture_path = path.parent.parent / IMAGE_NAME
        if picture_path is None:
            write_json(path, value)
        else:
            from .picture import write_picture
            with tempfile.TemporaryDirectory(dir=picture_path.parent) as tmp:
                staged = Path(tmp) / 'chart.png'
                write_picture(staged, value)
                write_json(path, value)
                try:
                    os.replace(staged, picture_path)
                except OSError as exc:
                    raise OSError('Your choice was saved. Run ez2rc chart to refresh the image.') from exc


def beat_text(value):
    if value is None:
        return '?'
    beat = read_ratio(value)
    if beat > 1 and beat.denominator != 1:
        whole, remainder = divmod(beat.numerator, beat.denominator)
        return f'{whole} + {remainder}/{beat.denominator}'
    return str(beat)


def start_choices(note):
    """Return the measured grid alternatives for an unplaced start."""
    if note['start_beat'] is not None:
        return []
    ticks = (note.get('evidence', {}).get('start') or {}).get('candidate_ticks', ())
    resolution = DRAFT_GRID.ticks_per_beat
    end = read_ratio(note['end_beat']) * resolution if note['end_beat'] is not None else None
    return [ratio(Fraction(tick, resolution)) for tick in ticks if end is None or tick < end]


def start_text(note):
    return ' or '.join(map(beat_text, start_choices(note))) or beat_text(note['start_beat'])


def parse_beat(text):
    if text == 'none':
        return None
    try:
        mixed = re.fullmatch(r'(\d+)\s*\+\s*(\d+/\d+)', text.strip())
        if mixed:
            return ratio(int(mixed[1]) + Fraction(mixed[2]))
        return ratio(Fraction(text))
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError('beat must be a number, a fraction such as 3/8, or none') from exc


def change_note(value, action, *, identity=None, fields=None, confirm=False):
    validate_draft(value)
    result = deepcopy(value)
    current = {r['id']: r for r in result['notes']}
    _apply_change(result, current, action, identity, fields, confirm)
    result['notes'] = list(current.values())
    return seal(result)


def choose_tap_placement(value, letter):
    validate_draft(value)
    result = deepcopy(value)
    pairs = result['projection']['tap_placements']
    current = {r['id']: r for r in result['notes']}
    original = {r['id']: r for r in result['automatic']['notes']}
    if letter not in ('A', 'B') or any(current.get(key) != original[key] for key in pairs):
        raise ValueError('Reopen the current chart to choose a tap placement.')
    index = ('A', 'B').index(letter)
    for identity, pair in pairs.items():
        _apply_change(result, current, 'edit', identity, {'start_beat': pair[index]}, True)
        result['history'][-1]['tap_placement'] = letter
    result['notes'] = list(current.values())
    return seal(result)


def _apply_change(result, current, action, identity, fields, confirm):
    before = current.get(identity)
    if action == 'add':
        identity = 'user:' + uuid.uuid4().hex
        after = {'id': identity, 'lane': 0, 'type': 'tap', 'start_beat': None, 'end_beat': None,
                 'status': 'unresolved', 'issues': [], 'evidence': {}, 'origin': 'user'}
    elif action == 'adopt':
        if before is not None:
            raise ValueError('candidate is already in the draft')
        after = next((deepcopy(r) for r in result['automatic']['alternatives'] if r['id'] == identity), None)
        if after is None:
            raise ValueError('unknown alternative candidate ID')
        after['origin'] = 'user_edit'
    elif action in ('edit', 'delete'):
        if before is None:
            raise ValueError('unknown note ID; use open to list notes')
        after = deepcopy(before) if action == 'edit' else None
    else:
        raise ValueError('unknown edit action')
    if after is not None:
        fields = fields or {}
        if set(fields) - {'lane', 'type', 'start_beat', 'end_beat'}:
            raise ValueError('unsupported note fields')
        after.update(fields)
        if fields.get('start_beat') is not None:
            after['issues'] = [issue for issue in after['issues']
                               if issue not in ('start_unavailable', 'start_grid_ambiguous', 'start_grid_no_candidate')]
        if after['origin'] == 'automatic':
            after['origin'] = 'user_edit'
        missing = after['start_beat'] is None or after['type'] == 'longnote' and after['end_beat'] is None
        after['status'] = 'reviewed' if confirm else 'unresolved' if missing else 'draft'
        _note(after, result['metadata']['lane_count'])
        current[identity] = after
    else:
        del current[identity]
    result['history'].append({'action': action, 'id': identity, 'before': deepcopy(before), 'after': deepcopy(after),
                              'timestamp': datetime.now(timezone.utc).isoformat()})
    result['revision'] += 1


def describe(value, *, identity=None, unresolved=False, limit=20):
    validate_draft(value)
    rows = value['notes']
    print(f"{value['metadata']['title']}: draft revision {value['revision']}; {len(rows)} notes, "
          f"{sum(r['status'] == 'unresolved' for r in rows)} unresolved, "
          f"{sum(r['status'] == 'reviewed' for r in rows)} reviewed")
    print('Coordinate: first numbered barline = 0; lane numbers below start at 1.')
    print('Automatic values need video review. Missing notes may be undetected.')
    if identity is not None:
        candidates = {r['id']: r for r in value['automatic']['alternatives']}
        candidates.update({r['id']: r for r in rows})
        rows = [candidates[identity]] if identity in candidates else []
        if not rows:
            raise ValueError('unknown note or alternative ID')
    elif unresolved:
        rows = [r for r in rows if r['status'] == 'unresolved']
    for row in rows[:limit]:
        print(f"{row['id']}  lane {row['lane']+1}  {row['type']}  "
              f"{start_text(row)} -> {beat_text(row['end_beat'])}  {row['status']}")
        if row['issues']:
            print('  ' + ', '.join(row['issues']))
        if identity is not None:
            print(json.dumps(row['evidence'], ensure_ascii=False, indent=2))
    if len(rows) > limit:
        print(f'{len(rows)-limit} more; use --limit or --id.')
    print(f"Competing alternatives: {len(value['automatic']['alternatives'])}; "
          f"uncertain motion spans: {len(value['projection']['unprocessed_motion'])}.")
