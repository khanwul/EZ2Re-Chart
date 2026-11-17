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
            source_y = round(witness['y_px'] + 5)
            np.testing.assert_array_equal(marked[source_y], np.tile([0, 0, 255], (raw.shape[1], 1)))
            marked[source_y] = raw[source_y]
            np.testing.assert_array_equal(marked, raw)
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
                    self.assertEqual(F(*note[side+'_beat']), (lower+upper)/2)
                    # The independently generated screen endpoint is in the
                    # range, while an added musical extension is outside it.
                    self.assertLess(upper, known+F(1, 16))
            self.assertTrue(all(n['status'] == 'draft' for n in draft['notes']))
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
