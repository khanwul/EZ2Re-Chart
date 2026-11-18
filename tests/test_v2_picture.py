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
                self.assertIsNotNone(cv2.imread(str(result / IMAGE_NAME)))
                self.assertFalse((result / '.ez2rc/report.html').exists())

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
