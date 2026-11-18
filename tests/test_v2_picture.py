"""Exercise numbered PNG decisions through their public commands."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from ez2rc.v2.cli import main
from ez2rc.v2.draft import create_draft, read_draft, save_draft
from ez2rc.v2.picture import IMAGE_NAME, chart_strips, pending_items
from test_v2_barline_pipeline import marked_video


class PictureTest(unittest.TestCase):
    def test_every_beat_has_a_line_touching_the_bottom_of_on_beat_notes(self):
        # A 48-beat column used to skip alternate beats.
        notes = [
            {'id':'tap', 'lane':0, 'start_beat':[1,1], 'end_beat':None},
            {'id':'hold', 'lane':0, 'start_beat':[12,1], 'end_beat':[15,1]},
            {'id':'offbeat', 'lane':0, 'start_beat':[33,2], 'end_beat':None},
        ]
        draft = {'metadata':{'lane_count':2}, 'bpm':'120',
                 'projection':{'measure_beats':[12,1]}, 'notes':notes}
        _, score = chart_strips(draft, {})
        # Sample a lane containing no notes to locate the visible grid rows.
        empty_lane = score[:, 85]
        is_beat = (empty_lane == (106,118,133)).all(axis=1)
        is_measure = (empty_lane == (158,168,180)).all(axis=1)
        lines = np.flatnonzero(is_beat | is_measure)[::-1]
        self.assertEqual(len(lines), 49)
        self.assertTrue(is_measure[lines[::12]].all())
        # The cap is wider than the hold body. At its side, the grid line
        # must meet the cap on its last painted row.
        cap = (score[:,44] == (232,238,247)).all(axis=1)
        for beat in (1,12,15):
            row = lines[beat]
            self.assertTrue(cap[row])
            self.assertFalse(cap[row+1])
        self.assertTrue(cap[lines[17]+1:lines[16]].any())

    def test_systematic_capture_offset_aligns_auto_notes_but_keeps_manual_beats(self):
        from fractions import Fraction
        def note(name, beat, origin='automatic', end=None):
            return {'id':name, 'origin':origin, 'lane':0, 'start_beat':list(beat.as_integer_ratio()),
                    'end_beat':list(end.as_integer_ratio()) if end is not None else None}
        phase = Fraction(1, 27)
        automatic = [note(str(i), Fraction(i)+phase) for i in range(8)]
        automatic.append(note('hold', 8+phase+Fraction(1,1000), end=9+phase+Fraction(1,2)))
        manual = note('manual', 10+phase, 'user_edit')
        draft = {'metadata':{'lane_count':2}, 'bpm':'120', 'projection':{'measure_beats':[4,1]},
                 'notes':[*automatic, manual], 'automatic':{'notes':automatic}}
        _, score = chart_strips(draft, {})
        cap = (score[:,44] == (232,238,247)).all(axis=1)
        grid = score[:,85]
        def on_line(color):
            return np.flatnonzero((grid == color).all(axis=1) & cap & ~np.roll(cap,-1))
        self.assertEqual(len(on_line((158,168,180))), 3)  # beats 0, 4, 8
        self.assertEqual(len(on_line((106,118,133))), 6)
        self.assertEqual(len(on_line((73,83,96))), 1)     # hold tail at 9.5
        self.assertTrue(cap[880-round(53*(10+phase))])
        self.assertFalse(cap[880-53*10])

    def test_hold_outline_opens_toward_body(self):
        from ez2rc.v2.witness_box import OUTLINE_DARK, OUTLINE_PINK, draw_outline
        for kind in ('tap', 'head', 'tail'):
            with self.subTest(kind=kind):
                image = np.zeros((9, 13, 3), dtype=np.uint8)
                draw_outline(image, (2, 1), (10, 7), kind)
                np.testing.assert_array_equal(image[4, 0], OUTLINE_DARK)
                np.testing.assert_array_equal(image[4, 1], OUTLINE_PINK)
                marked = image.any(axis=2)
                self.assertTrue(marked[1:8, [2,10]].all())
                self.assertTrue((marked[1, 3:10] == (kind != 'head')).all())
                self.assertTrue((marked[7, 3:10] == (kind != 'tail')).all())
                self.assertFalse(marked[2:7, 3:10].any())

    def test_score_reads_up_then_right_and_hold_continues_after_column_break(self):
        def note(lane, start, end=None):
            return {'id': str(start), 'lane': lane, 'start_beat': [start,1],
                    'end_beat': [end,1] if end is not None else None}
        draft = {'metadata': {'lane_count': 3}, 'bpm': '120',
                 'projection': {'measure_beats': [4,1]},
                 'notes': [note(0,1), note(2,3), note(1,15,17), note(0,19)]}
        _, score = chart_strips(draft, {})
        ys, xs = np.where(np.all(score == (232,238,247), axis=2))
        groups = np.split(np.unique(xs), np.where(np.diff(np.unique(xs)) > 1)[0]+1)
        self.assertEqual(len(groups), 3)
        first, later, next_column = [ys[xs == group[0]].min() for group in groups]
        self.assertGreater(first, later)
        self.assertEqual(next_column, later)  # Both notes are three beats into their column.
        ys, xs = np.where(np.all(score == (53,142,176), axis=2))
        columns = np.split(np.unique(xs), np.where(np.diff(np.unique(xs))>1)[0]+1)
        self.assertEqual(len(columns), 2)
        self.assertLess(ys[xs == columns[0][0]].max(), ys[xs == columns[1][0]].min())

    def test_compact_extract_and_reconstruct(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            song = marked_video(root, longnotes=True)
            for command in ('extract', 'reconstruct'):
                out = root / command
                args = [command, str(song), '--output', str(out), '--no-progress']
                if command == 'reconstruct':
                    args.extend(['--observations', str(result / '.ez2rc/artifacts/observations')])
                self.assertIn(main(args), (0, 2))
                result = next(out.glob('*/*/*/' + IMAGE_NAME)).parent
                names = {p.name for p in result.iterdir()}
                from ez2rc.v2.photos import photo_folder
                folder = photo_folder(read_draft(result/'.ez2rc/draft.json'))
                if command == 'extract':
                    self.assertIn(folder, names)
                else:
                    self.assertNotIn(folder, names)
                    self.assertEqual(main(['photos','--run',str(result)]), 0)
                photos = sorted((result/folder).glob('*.png'))
                self.assertGreater(len(photos), 1)
                images = [cv2.imread(str(p)) for p in photos[:2]]
                self.assertTrue(all(im is not None for im in images))
                self.assertNotEqual(photos[0].read_bytes(), photos[1].read_bytes())
                self.assertIsNotNone(cv2.imread(str(result / IMAGE_NAME)))
                self.assertFalse((result / '.ez2rc/report.html').exists())

    def test_review_question_identifies_the_disputed_endpoint_and_action(self):
        from ez2rc.v2.photos import review_question, _answers
        a = {'lane': 2, 'type': 'longnote', 'start_beat': [1,1], 'end_beat': [3,1]}
        options = {'A': a, 'B': {**a, 'end_beat': [4,1]}}
        self.assertEqual(review_question(options), '3번 레인의 롱노트는 몇 박에서 끝나나요?')
        self.assertIn('끝 4박', _answers(options, 3, True)[1])
        self.assertIn('ez2rc choose 3 B', _answers(options, 3, True)[1])
        self.assertIn('--end <박자>', _answers({'A': {**a, 'end_beat': None}}, 3, True)[0])

    def test_review_photos_point_to_observed_lane_after_manual_lane_edit(self):
        from ez2rc.v2.photos import _requests
        original = {'id':'one', 'lane':0, 'type':'tap', 'issues':['pixel_ambiguous'],
                    'status':'unresolved', 'start_beat':[1,1], 'end_beat':None,
                    'evidence':{'start':{'witnesses':[
                        {'frame_id':8, 'timestamp_ms':160, 'y_px':15},
                        {'frame_id':10, 'timestamp_ms':200, 'y_px':21}]}}}
        other = {**original, 'id':'two', 'evidence':{}}
        draft = {'automatic':{'notes':[original, other], 'alternatives':[]},
                 'notes':[{**original,'lane':1}, other], 'projection':{}}
        requests, missing = _requests(draft)
        self.assertEqual(missing, [2])
        self.assertEqual(set(requests), {8, 10})
        self.assertEqual(requests[8][0][:3], ('check-0001-01.png',1,[('A','start',0,original['evidence']['start']['witnesses'][0],'tap')]))
        self.assertEqual(requests[10][0][:3], ('check-0001-02.png',1,[('A','start',0,original['evidence']['start']['witnesses'][1],'tap')]))

    def test_replacement_removal_and_missing_position_keep_item_numbers(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            one = {'id': 'one', 'lane': 0, 'start_beat': [1,1], 'end_beat': [2,1],
                   'type': 'longnote', 'status': 'unresolved', 'issues': ['pixel_ambiguous'],
                   'origin': 'automatic', 'evidence': {'event_id': 'group'}}
            two = {**one, 'id': 'two', 'lane': 1, 'start_beat': None, 'evidence': {}}
            alt = {**deepcopy(one), 'id': 'other-end', 'end_beat': [3,1]}
            proj = {'notes':[one,two], 'alternatives':[alt], 'tail_beat_extension':[0,1],
                    'timing_choices': {'two':[{'start_beat':[1,2], 'end_beat':[2,1]},
                                              {'start_beat':[3,4], 'end_beat':[2,1]}]}}
            meta = {'title':'sample','game':'test','mode':'5k','difficulty':'HD','lane_count':5}
            path = root / '.ez2rc/draft.json'
            save_draft(path, create_draft(proj, meta, '120', {}))
            def run(*args):
                return main([*args, '--run', str(root)])
            self.assertEqual(run('chart'), 0)
            before = (root / IMAGE_NAME).read_bytes()
            checksum = read_draft(path)['document_sha256']
            self.assertEqual(run('choose','2','A'), 1)
            self.assertEqual(read_draft(path)['document_sha256'], checksum)
            self.assertEqual(run('choose','1','B'), 0)
            self.assertNotEqual((root / IMAGE_NAME).read_bytes(), before)
            after = read_draft(path)
            self.assertEqual(list(pending_items(after)), [2])
            chosen = next(n for n in after['notes'] if n['id'] == 'other-end')
            self.assertEqual((chosen['end_beat'],chosen['status']), ([3,1], 'reviewed'))
            self.assertEqual(run('choose','1','X'), 1)
            self.assertEqual(run('choose','2','C'), 0)
            self.assertEqual(next(n['start_beat'] for n in read_draft(path)['notes'] if n['id'] == 'two'), [3,4])
            self.assertFalse(pending_items(read_draft(path)))
