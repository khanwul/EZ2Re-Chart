from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ez2rc.v2.cli import main
from ez2rc.v2.draft import (change_note, create_draft, read_draft, save_draft, seal, describe)
from ez2rc.v2.observation import open_observations
from ez2rc.v2.runner import execute
from ez2rc.v2.storage import CacheWriter
from test_v2_barline_pipeline import marked_video


def empty_draft():
    return create_draft({'notes': [], 'alternatives': [], 'tail_beat_extension': [0, 1],
                         'unprocessed_motion': [], 'coordinate': 'first_numbered_barline_zero'},
                        {'game': 'generated', 'mode': '5k', 'lane_count': 5,
                         'title': '<generated>', 'difficulty': 'test'}, '150', {})


class DraftTest(unittest.TestCase):
    def test_preview_prefers_a_usable_coordinate_observation(self):
        from ez2rc.v2.draft_projection import _review_witnesses

        rows = [{'frame_id': i, 'observation_id': str(i), 'usable': i != 2}
                for i in range(5)]
        self.assertEqual(_review_witnesses(rows, {'2', '3'})[0], rows[3])
        self.assertEqual(_review_witnesses(rows, {'2'})[0], rows[0])
        self.assertEqual(_review_witnesses(rows, set()),
                         [rows[0], rows[3], rows[2]])
        self.assertEqual(_review_witnesses([], set()), [])

    def test_grid_starts_preserve_measurements_and_expose_all_choices(self):
        from ez2rc.v2.draft import beat_text, parse_beat
        from ez2rc.v2.draft_projection import apply_start_grid
        from ez2rc.v2.picture import decide, pending_items
        from ez2rc.v2.review import write_review
        from test_v2_rhythm import domain

        # The start range straddles ticks 3 and 4 above beat two.
        bounds = domain(F(129, 64), F(97, 48))
        midpoint = F(2) + F(7, 384)
        estimate = [midpoint.numerator, midpoint.denominator]
        def measured(identity, beat_range, end=None):
            lo, hi = (F(*beat_range[side]) for side in ('lower', 'upper'))
            center = (lo + hi) / 2
            raw = [center.numerator, center.denominator]
            note = {'id': identity, 'lane': 0, 'type': 'longnote' if end else 'tap',
                    'origin': 'automatic', 'start_beat': raw, 'end_beat': end, 'status': 'draft',
                    'issues': [], 'evidence': {'event_id': identity, 'start': {
                        'beat_range': beat_range, 'estimate': raw, 'witnesses': []}}}
            apply_start_grid(note)
            return note

        ambiguous = measured('two', bounds, [13, 5])
        self.assertEqual(ambiguous['evidence']['start']['estimate'], estimate)
        self.assertEqual(ambiguous['evidence']['start']['candidate_ticks'], [387, 388])
        self.assertIsNone(ambiguous['start_beat'])
        unique = measured('one', {**bounds, 'lower_inclusive': False})
        self.assertEqual(unique['start_beat'], [97, 48])
        # 1/192 is an internal tick, between allowed fractions.
        empty = measured('zero', domain(F(1, 192), F(1, 192)))
        self.assertEqual(empty['issues'], ['start_grid_no_candidate'])
        self.assertIsNone(empty['start_beat'])
        shared = measured('shared', domain(F(-1, 16), F(-1, 16)))
        self.assertEqual(shared['evidence']['start']['candidate_ticks'], [-12])
        self.assertEqual(shared['start_beat'], [-1, 16])
        short = measured('short', bounds, [97, 48])
        self.assertEqual(short['start_beat'], [129, 64])
        self.assertEqual(short['end_beat'], [97, 48])

        base = empty_draft()
        draft = create_draft({**base['projection'], 'notes': [ambiguous, unique, empty],
                              'alternatives': []}, base['metadata'], '120', {})
        items = pending_items(draft)
        self.assertEqual(len(items), 2)
        choices = items[1].options
        self.assertEqual([n['start_beat'] for n in choices.values()], [None, [129, 64], [97, 48]])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'review.html'
            write_review(path, draft)
            self.assertIn('2 + 1/64 or 2 + 1/48', path.read_text())
            chosen = decide(draft, items[1], 'C', {})
            path = Path(tmp) / 'draft.json'
            save_draft(path, chosen)
            saved = read_draft(path)
            self.assertEqual(list(pending_items(saved)), [2])
            self.assertEqual(saved['notes'][0]['start_beat'], [97, 48])
            self.assertEqual(saved['notes'][0]['end_beat'], [13, 5])
            self.assertEqual(saved['notes'][0]['issues'], [])
            self.assertEqual(saved['notes'][0]['evidence'], ambiguous['evidence'])
        for value in ([-9, 4], [-1, 48], [0, 1], [1, 64], [97, 48]):
            self.assertEqual(parse_beat(beat_text(value)), value)

    def test_default_starts_use_last_template_row_and_tails_keep_their_top(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = marked_video(root, longnotes=True)
            profile = root/'profile.toml'
            profile.write_text(profile.read_text().replace(
                'status = "verified"\noffset_px = 0.0\nerror_px = 0.0', 'status = "unverified"'))
            _, run = execute(song, output_root=root/'out')
            draft = read_draft(run/'draft.json')
            self.assertEqual(draft['projection']['default_start_reference'], 'template_bottom')
            expected_heads = {0: F(3, 2), 1: F(11, 4)}
            expected_tails = {0: F(5, 2), 1: F(17, 4)}
            self.assertEqual(len(draft['notes']), 6)
            for note in draft['notes']:
                for side in ('start', 'end') if note['type'] == 'longnote' else ('start',):
                    point = note['evidence'][side]
                    height = point['witnesses'][0]['box_px'][3]
                    offset = height - 1 if side == 'start' else 0
                    self.assertEqual(point['offset_px'], [offset, 1])
                    if note['type'] == 'longnote':
                        top_beat = (expected_heads if side == 'start' else expected_tails)[note['lane']]
                        expected = top_beat - F(offset, 60)
                        self.assertLessEqual(F(*point['beat_range']['lower']), expected)
                        self.assertGreaterEqual(F(*point['beat_range']['upper']), expected)

    def test_reconstruct_without_template_shapes_saves_unresolved_draft_and_chart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = marked_video(root, longnotes=True)
            _, first = execute(song, output_root=root/'out')
            observations = open_observations(first/'artifacts/observations')
            key = deepcopy(observations.manifest['key'])
            del key['parameters']['template_shapes']
            key['parameters'].pop('profile_sha256', None)
            with CacheWriter(root/'out/.cache', key) as writer:
                for chunk in observations.chunks():
                    writer.append(**chunk)
                legacy = writer.commit(observations.manifest['metadata'], validate=open_observations)
            for name in ('video.mkv', 'profile.toml', 'note.png', 'head.png', 'tail.png'):
                (root/name).unlink()
            with patch('ez2rc.v2.capture.av.open', side_effect=AssertionError('no media decoding')):
                code, run = execute(song, output_root=root/'out', observation_cache=legacy.path)
            self.assertEqual(code, 2, (run/'run.json').read_text())
            self.assertEqual(json.loads((run/'artifacts/barline-grid.json').read_text())['status'], 'supported')
            self.assertEqual(json.loads((run/'chart.json').read_text())['status'], 'partial')
            draft = read_draft(run/'draft.json')
            self.assertEqual(len(draft['notes']), 6)
            self.assertEqual(draft['projection']['geometry_status'], 'unavailable')
            self.assertIn('template_geometry_unavailable', draft['projection']['geometry_issues'])
            for note in draft['notes']:
                self.assertEqual(note['status'], 'unresolved')
                self.assertIsNone(note['start_beat'])
                self.assertIsNone(note['end_beat'])
                self.assertIn('template_geometry_unavailable', note['issues'])
                self.assertTrue(note['evidence']['start']['witnesses'])
            self.assertTrue((run/'review.html').is_file())

    def test_frame_marker_uses_original_video_coordinates_after_alignment_and_edit(self):
        from ez2rc.v2.review import export_frame
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = marked_video(root, translated=(3, 5))
            code, run = execute(song, output_root=root/'out')
            self.assertEqual(code, 2, (run/'run.json').read_text())
            draft = read_draft(run/'draft.json')
            self.assertEqual(draft['source']['alignment_offset_px'], [3, 5])
            original = draft['notes'][0]
            draft = change_note(draft, 'edit', identity=original['id'], fields={'start_beat': [10, 1]})
            path = root/'edited.json'
            save_draft(path, draft)
            draft = read_draft(path)
            witness = original['evidence']['start']['witnesses'][0]
            self.assertEqual(draft['notes'][0]['evidence'], original['evidence'])
            stamp = export_frame(draft, root/'marked.png', identity=original['id'])
            self.assertEqual(stamp.frame_id, witness['frame_id'])
            export_frame(draft, root/'raw.png', seconds=witness['timestamp_ms']/1000)
            raw, marked = (cv2.imread(str(root/name)) for name in ('raw.png', 'marked.png'))
            x, y, width, height = witness['box_px']
            self.assertEqual((height, width), cv2.imread(str(root/'note.png')).shape[:2])
            expected = raw.copy()
            from ez2rc.v2.witness_box import draw_outline
            draw_outline(expected, (x+3, y+5), (x+3+width-1, y+5+height-1), 'tap')
            np.testing.assert_array_equal(marked, expected)
            legacy_boxes = deepcopy(draft)
            for group in [legacy_boxes['automatic']['notes'], legacy_boxes['notes']]:
                for n in group:
                    for side in ('start', 'end'):
                        for w in (n['evidence'].get(side) or {}).get('witnesses', []):
                            w.pop('box_px', None)
            # Use the immutable old observation ID plus note kind to recover its footprint.
            from ez2rc.v2.witness_box import matched_boxes
            old = legacy_boxes['automatic']['notes'][0]['evidence']['start']['witnesses'][0]
            boxes = matched_boxes([(old,'tap')], legacy_boxes, run)
            self.assertEqual(boxes[old['observation_id'],'tap'], witness['box_px'])
            # Older drafts cannot safely assume a missing alignment is zero.
            legacy = deepcopy(draft)
            del legacy['source']['alignment_offset_px']
            seal(legacy)
            with self.assertRaisesRegex(ValueError, 'no source alignment'):
                export_frame(legacy, root/'legacy.png', identity=original['id'])
            export_frame(legacy, root/'legacy-raw.png', seconds=witness['timestamp_ms']/1000)

    def test_adopt_keeps_the_alternative_and_displays_the_current_revision_once(self):
        original = empty_draft()
        alternative = change_note(original, 'add', fields={'start_beat': [1, 1]})['notes'][0]
        alternative.update(id='alternative', origin='automatic')
        draft = create_draft({**original['projection'], 'notes': [], 'alternatives': [alternative]},
                             original['metadata'], original['bpm'], original['source'])
        adopted = change_note(draft, 'adopt', identity='alternative', fields={'start_beat': [2, 1]}, confirm=True)
        self.assertEqual(adopted['automatic'], draft['automatic'])
        self.assertEqual(adopted['notes'][0]['start_beat'], [2, 1])
        with patch('builtins.print') as output:
            describe(adopted, identity='alternative')
        rows = [c.args[0] for c in output.call_args_list if str(c.args[0]).startswith('alternative  ')]
        self.assertEqual(len(rows), 1)
        self.assertIn('reviewed', rows[0])

    def test_save_reopen_edit_delete_and_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'draft.json'
            original = empty_draft()
            save_draft(path, original)
            added = change_note(read_draft(path), 'add', fields={
                'lane': 4, 'type': 'longnote', 'start_beat': [7, 3], 'end_beat': [25, 8]})
            save_draft(path, added, expected_sha=original['document_sha256'])
            identity = added['notes'][0]['id']
            updated = change_note(read_draft(path), 'edit', identity=identity,
                                  fields={'lane': 1, 'start_beat': [5, 2]}, confirm=True)
            save_draft(path, updated, expected_sha=added['document_sha256'])
            self.assertEqual(read_draft(path), updated)
            self.assertEqual(updated['notes'][0]['status'], 'reviewed')
            self.assertEqual(updated['history'][1]['before'], added['notes'][0])
            deleted = change_note(updated, 'delete', identity=identity)
            save_draft(path, deleted, expected_sha=updated['document_sha256'])
            self.assertEqual(read_draft(path)['notes'], [])
            self.assertEqual(len(read_draft(path)['history']), 3)

    def test_invalid_edit_corruption_conflict_and_failed_save_preserve_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'draft.json'
            original = change_note(empty_draft(), 'add', fields={'start_beat': [2, 1]})
            save_draft(path, original)
            identity = original['notes'][0]['id']
            for fields in ({'lane': 5}, {'end_beat': [3, 1]}, {'start_beat': [2, 0]},
                           {'type': 'longnote', 'end_beat': [1, 1]}):
                with self.subTest(fields=fields), self.assertRaises(ValueError):
                    change_note(original, 'edit', identity=identity, fields=fields)
            with self.assertRaisesRegex(ValueError, 'missing beats'):
                change_note(original, 'edit', identity=identity, fields={'start_beat': None}, confirm=True)
            edited = change_note(original, 'edit', identity=identity, fields={'lane': 3})
            with self.assertRaisesRegex(ValueError, 'changed'):
                save_draft(path, edited, expected_sha='0'*64)
            with patch('ez2rc.v2.storage.os.replace', side_effect=OSError('disk error')):
                with self.assertRaises(OSError):
                    save_draft(path, edited, expected_sha=original['document_sha256'])
            self.assertEqual(read_draft(path), original)
            self.assertFalse(path.with_name('.draft.json.lock').exists())
            corrupt = deepcopy(original)
            corrupt['notes'][0]['lane'] = 3
            with self.assertRaisesRegex(ValueError, 'history'):
                seal(corrupt)

    def test_new_video_draft_and_cli_edit_reopen_with_original_video_witness(self):
        from ez2rc.v2.review import export_frame, write_review
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = marked_video(root, longnotes=True)
            code, run = execute(song, output_root=root/'out')
            self.assertEqual(code, 2)
            path = run/'draft.json'
            draft = read_draft(path)
            self.assertEqual(len(draft['notes']), 6)
            longs = [n for n in draft['notes'] if n['type'] == 'longnote']
            expected = {0: (F(3, 2), F(5, 2)), 1: (F(11, 4), F(17, 4))}
            for note in longs:
                for side, known in zip(('start', 'end'), expected[note['lane']]):
                    domain = note['evidence'][side]['beat_range']
                    lower, upper = F(*domain['lower']), F(*domain['upper'])
                    self.assertLessEqual(lower, known)
                    self.assertGreaterEqual(upper, known)
                    point = note['evidence'][side]
                    self.assertEqual(F(*point['estimate']), (lower+upper)/2)
                    if side == 'start':
                        self.assertIsNone(note['start_beat'])
                        self.assertIn(int(known * 192), point['candidate_ticks'])
                        self.assertGreater(len(point['candidate_ticks']), 1)
                    else:
                        self.assertEqual(note['end_beat'], point['estimate'])
                    # The independently generated screen endpoint is in the
                    # range, while an added musical extension is outside it.
                    self.assertLess(upper, known+F(1, 16))
            self.assertTrue(all('start_grid_ambiguous' in n['issues'] and n['status'] == 'unresolved'
                                for n in draft['notes']))
            self.assertTrue(draft['projection']['unprocessed_motion'])
            self.assertEqual(json.loads((run/'chart.json').read_text())['notes'], [])
            identity = longs[0]['id']
            with patch('builtins.print'):
                self.assertEqual(main(['edit', str(path), '--id', identity, '--lane', '2',
                                       '--start', '1/3', '--end', '8/3', '--confirm']), 0)
                self.assertEqual(main(['open', str(path), '--id', identity]), 0)
                self.assertEqual(main(['add', str(path), '--lane', '2', '--type', 'tap', '--start', '10']), 0)
            edited = read_draft(path)
            self.assertEqual(edited['automatic'], draft['automatic'])
            note = next(n for n in edited['notes'] if n['id'] == identity)
            self.assertEqual(note['start_beat'], [1, 3])
            self.assertEqual(note['end_beat'], [8, 3])
            export_frame(edited, root/'witness.png', identity=identity)
            self.assertTrue((root/'witness.png').is_file())
            write_review(root/'review.html', edited)
            self.assertIn(identity, (root/'review.html').read_text())
            # Reconstruction is a new automatic result and cannot overwrite a
            # user's revised draft, even when evidence caches are reused.
            _, again = execute(song, output_root=root/'out', observation_cache=run/'artifacts/observations')
            self.assertEqual(read_draft(again/'draft.json')['notes'], draft['notes'])
            self.assertEqual(read_draft(path), edited)
            song.write_text(song.read_text().replace('measure_beats = [1, 1]', ''))
            _, unknown = execute(song, output_root=root/'out', observation_cache=run/'artifacts/observations')
            incomplete = read_draft(unknown/'draft.json')
            self.assertEqual(len(incomplete['notes']), len(draft['notes']))
            self.assertTrue(all(n['start_beat'] is None and n['status'] == 'unresolved'
                                for n in incomplete['notes']))

    def test_tail_shift_is_removed_before_projection_but_pixel_landmark_is_preserved(self):
        from ez2rc.v2.decode import open_decode, lanes
        from ez2rc.v2.relative_beats import screen_end_reference
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = marked_video(root, longnotes=True)
            _, run = execute(song, output_root=root/'out')
            decoded = open_decode(run/'artifacts/decode')
            inputs = {'endpoint_calibrations': []}
            for lane in lanes(decoded):
                for e in lane['endpoints']:
                    inputs['endpoint_calibrations'].append({'endpoint_id': e['id'],
                        'offset_px': [2, 3], 'beat_shift': [1, 16]})
            normalized, receipt = screen_end_reference(inputs, decoded)
            tails = {e['id'] for lane in lanes(decoded) for e in lane['endpoints'] if e['kind'] == 'tail'}
            self.assertEqual({r['endpoint_id'] for r in receipt['removed_tail_shifts']}, tails)
            for old, new in zip(inputs['endpoint_calibrations'], normalized['endpoint_calibrations']):
                self.assertIn('beat_shift', old)
                self.assertEqual(new['offset_px'], old['offset_px'])
                self.assertEqual('beat_shift' in new, new['endpoint_id'] not in tails)


if __name__ == '__main__':
    unittest.main()
