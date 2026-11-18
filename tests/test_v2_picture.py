"""Exercise numbered PNG decisions through their public commands."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from fractions import Fraction
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
    def test_png_labels_close_fractional_starts_and_shows_a_chord_beat_once(self):
        from unittest.mock import patch
        from ez2rc.v2.picture import label

        rows = [{'id': str(lane), 'lane': lane, 'type': 'tap', 'end_beat': None,
                 'start_beat': start} for lane, start in enumerate(([65, 64], [49, 48], [65, 64]))]
        draft = {'metadata': {'lane_count': 3}, 'bpm': '120', 'notes': rows,
                 'projection': {'start_ticks_per_beat': 192, 'default_start_reference': 'template_bottom',
                                'measure_beats': [4, 1]}}
        with patch('ez2rc.v2.picture.label', wraps=label) as draw:
            list(chart_strips(draft, {}))
        positions = [(c.args[1], c.args[2]) for c in draw.call_args_list if ' + ' in c.args[1]]
        self.assertEqual([label for label, _ in positions], ['1 + 1/48', '1 + 1/64'])
        self.assertNotEqual(positions[0][1], positions[1][1])

    def test_measure_numbers_fit_the_rail_without_prefixes(self):
        from unittest.mock import patch
        from ez2rc.v2.picture import label

        for beat, first, width in ((0, 1, 15), (-16, -3, 23), (16000, 4001, 39)):
            with self.subTest(beat=beat):
                note = {'id': 'tap', 'lane': 0, 'start_beat': [beat, 1], 'end_beat': None}
                draft = {'metadata': {'lane_count': 5}, 'bpm': '120',
                         'projection': {'measure_beats': [4, 1]}, 'notes': [note]}
                with patch('ez2rc.v2.picture.label', wraps=label) as draw:
                    _, score = chart_strips(draft, {})
                labels = [call.args for call in draw.call_args_list if call.args[0].shape[1] < score.shape[1]]
                self.assertEqual([args[1] for args in labels], [str(i) for i in range(first, first + 4)])
                for rail, text, x, *_ in labels:
                    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)[0][0]
                    self.assertEqual(rail.shape[1], width)
                    self.assertGreater(x, 0)
                    self.assertLess(x + text_width, width)

    def test_long_chart_continues_to_the_right_after_eight_columns(self):
        # Beat 129 belongs one beat above the bottom of the ninth four-measure column.
        draft = {'metadata': {'lane_count': 5}, 'bpm': '120',
                 'projection': {'measure_beats': [4, 1]},
                 'notes': [{'id': 'tap', 'lane': 0, 'start_beat': [129, 1], 'end_beat': None}]}
        _, score = chart_strips(draft, {})
        measures = (score[:, 85] == (158, 168, 180)).all(axis=1)
        self.assertEqual(measures.sum(), 5)  # Both ends and three interior barlines.
        start = np.flatnonzero(measures)[-1]
        bars_across = (score[start] == (158, 168, 180)).all(axis=1)
        starts = np.flatnonzero(bars_across & ~np.roll(bars_across, 1))
        self.assertEqual(len(starts), 9)
        self.assertEqual(starts[0] - 3, 23)  # Room for the two digits in measure 36.
        grid = score[:, starts[-1]]
        for offset in (15, 30, 45):
            np.testing.assert_array_equal(grid[start - offset], (73, 83, 96))
        np.testing.assert_array_equal(score[start - 60, starts[-1] + 1], (232, 238, 247))

    def test_choices_without_beat_positions_continue_in_columns(self):
        from unittest.mock import patch
        from ez2rc.v2.picture import label

        notes = [{'id': str(i), 'type': 'tap', 'lane': 0, 'start_beat': None,
                  'end_beat': None, 'status': 'unresolved', 'issues': [], 'evidence': {}}
                 for i in range(80)]
        draft = {'metadata': {'lane_count': 5}, 'bpm': '120', 'projection': {},
                 'notes': notes, 'automatic': {'notes': notes, 'alternatives': []}}
        with patch('ez2rc.v2.picture.label', wraps=label) as draw_label:
            _, key, _ = chart_strips(draft, pending_items(draft))
        positions = [call.args[2:4] for call in draw_label.call_args_list if call.args[1].startswith('#')]
        self.assertEqual(len(positions), len(notes))
        self.assertGreater(positions[-1][0], positions[0][0])
        self.assertLess(max(y for x, y in positions), key.shape[0])
        self.assertLess(key.shape[0], 1024)

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
        # Every beat has three quieter guides, including beats during rests.
        is_quarter = (empty_lane == (73,83,96)).all(axis=1)
        expected = [lines[0] - round((lines[0] - lines[-1]) * tick / 192)
                    for tick in range(192) if tick % 4]
        np.testing.assert_array_equal(np.flatnonzero(is_quarter)[::-1], expected)
        # The cap is wider than the body. Both ends face inward.
        cap = (score[:,44] == (232,238,247)).all(axis=1)
        for beat in (1,12):
            row = lines[beat]
            self.assertTrue(cap[row])
            self.assertFalse(cap[row+1])
        self.assertTrue(cap[lines[15]])
        self.assertFalse(cap[lines[15]-1])
        self.assertTrue(cap[lines[17]+1:lines[16]].any())
        # The hold body sits inside both edges of its cap.
        cap_width = (score[lines[12]] == (232,238,247)).all(axis=1).sum()
        body_width = (score[lines[13]] == (167,171,178)).all(axis=1).sum()
        self.assertGreater(body_width, 0)
        self.assertLess(body_width, cap_width)
        # Each measure has a light number centered in its gray gutter.
        for beat in range(0, 48, 12):
            center = (lines[beat] + lines[beat+12]) // 2
            self.assertTrue((score[center-6:center+7, 3:18] > 210).all(axis=2).any())

    def test_legacy_display_phase_applies_only_to_automatic_starts(self):
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
        bars = np.flatnonzero((grid == (158,168,180)).all(axis=1))
        zero = int(bars[-1])
        pixels_per_beat = Fraction(zero - int(bars[-2]), 4)
        tail_row = zero-round(pixels_per_beat*(9+phase+Fraction(1,2)))
        self.assertTrue(cap[tail_row])
        self.assertFalse(cap[tail_row-1])
        self.assertTrue(cap[zero-round(pixels_per_beat*(10+phase))])
        self.assertFalse(cap[zero-round(pixels_per_beat*10)])

    def test_observed_tail_before_measure_preserves_a_visible_gap(self):
        # At this scale the off-grid tail is one raster row below the line.
        # A cap painted above the endpoint would overlap the line.
        for start in ([2,1], None):
            note = {'id':'hold', 'origin':'automatic', 'lane':0,
                    'start_beat':start, 'end_beat':[399,100]}
            draft = {'metadata':{'lane_count':2}, 'bpm':'120', 'notes':[note],
                     'projection':{'measure_beats':[4,1], 'default_start_reference':'template_bottom'}}
            _, score = chart_strips(draft, {})
            bars = np.flatnonzero((score[:,85] == (158,168,180)).all(axis=1))
            bar = bars[-2]  # Beat 4, counting up from the bottom.
            np.testing.assert_array_equal(score[bar,44], score[bar,85])
            np.testing.assert_array_equal(score[bar+1,44], (232,238,247))

    def test_boundary_heads_extend_inward_and_tails_stay_on_the_line(self):
        def render(start, end):
            notes = [{'id': 'hold', 'type': 'longnote', 'lane': 0, 'start_beat': start, 'end_beat': end},
                     {'id': 'tap', 'lane': 1, 'start_beat': [18, 1], 'end_beat': None}]
            draft = {'metadata': {'lane_count': 2}, 'bpm': '120', 'notes': notes,
                     'projection': {'measure_beats': [4, 1]}}
            _, score = chart_strips(draft, {})
            return score

        # A cap-only pixel at the left edge of the lane excludes the narrower body.
        for start, end, first_rows, second_rows in (
            ([4, 1], [8, 1], [512, 749, 750, 751, 752], []),
            ([12, 1], [16, 1], [32, 269, 270, 271, 272], []),
            ([16, 1], [20, 1], [], [752, 989, 990, 991, 992]),
            (None, [16, 1], [32], []),
            ([16, 1], None, [], [989, 990, 991, 992]),
        ):
            with self.subTest(start=start, end=end):
                score = render(start, end)
                for x, expected in ((19, first_rows), (116, second_rows)):
                    rows = np.flatnonzero((score[:, x] == (232, 238, 247)).all(axis=1))
                    np.testing.assert_array_equal(rows, expected)

        # A sliver rounding to zero pixels must not leave even a body-colored line.
        np.testing.assert_array_equal(render([15999, 1000], [17, 1]), render([16, 1], [17, 1]))
        np.testing.assert_array_equal(render([15, 1], [16001, 1000]), render([15, 1], [16, 1]))
        self.assertFalse(np.array_equal(render([15, 1], [1601, 100]), render([15, 1], [16, 1])))

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
        ys, xs = np.where(np.all(score == (30,104,184), axis=2))
        columns = np.split(np.unique(xs), np.where(np.diff(np.unique(xs))>1)[0]+1)
        self.assertEqual(len(columns), 2)
        self.assertLess(ys[xs == columns[0][0]].max(), ys[xs == columns[1][0]].min())

    def test_compact_extract_and_reconstruct(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = marked_video(root, bar_entries=(6, 26, 41, 61, 81, 101))
            # The procedural video places every template and line on an exact integer row.
            profile = root / 'profile.toml'
            profile.write_text('draft_localization_error_px = 0\n' + profile.read_text())
            song.write_text(song.read_text().replace('bpm = "150"', 'bpm = "600"')
                            .replace('measure_beats = [1, 1]', 'measure_beats = [4, 1]'))
            for command in ('extract', 'reconstruct'):
                out = root / command
                args = [command, str(song), '--output', str(out), '--no-progress']
                if command == 'reconstruct':
                    args.extend(['--observations', str(result / '.ez2rc/artifacts/observations')])
                with redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(args), 0, output.getvalue())
                self.assertIn('Draft complete:', output.getvalue())
                result = next(out.glob('*/*/*/' + IMAGE_NAME)).parent
                run = json.loads((result/'.ez2rc/run.json').read_text())
                self.assertEqual(run['status'], 'complete')
                self.assertEqual(run['stages']['timing']['status'], 'complete')
                self.assertEqual(run['stages']['timing']['confirmed_chart_status'], 'partial')
                self.assertEqual(run['picture']['pending_choices'], 0)
                grid = json.loads((result/'.ez2rc/artifacts/barline-grid.json').read_text())
                assignments = grid['hypotheses'][0]['assignments']
                expected = [[b, 1] for b in (0, 4, 7, 11, 15, 19)]
                self.assertEqual([a['beat'] for a in assignments], expected)
                draft = read_draft(result/'.ez2rc/draft.json')
                self.assertEqual(draft['projection']['barline_beats'], expected)
                self.assertEqual(draft['projection']['start_ticks_per_beat'], 192)
                # The last tap in lane 2 coincides with the sixth observed line.
                last = max((n for n in draft['notes'] if n['lane'] == 1),
                           key=lambda n: Fraction(*n['start_beat']))
                bounds = last['evidence']['start']['beat_range']
                self.assertEqual(last['start_beat'], [19, 1])
                self.assertLessEqual(Fraction(*bounds['lower']), 19)
                self.assertGreaterEqual(Fraction(*bounds['upper']), 19)
                self.assertFalse(list(result.glob('*-photos-r*')))
                self.assertNotIn('folder', run['picture'])
                self.assertNotIn('source video photos', output.getvalue())
                if command == 'reconstruct':
                    self.assertEqual(main(['photos','--run',str(result)]), 0)
                    photos = sorted(result.glob('preview-photos-r0000/preview-*.png'))
                    self.assertGreater(len(photos), 1)
                    images = [cv2.imread(str(p)) for p in photos[:2]]
                    self.assertTrue(all(im is not None for im in images))
                    self.assertNotEqual(photos[0].read_bytes(), photos[1].read_bytes())
                self.assertIsNotNone(cv2.imread(str(result / IMAGE_NAME)))
                self.assertFalse((result / '.ez2rc/report.html').exists())
            song.write_text(song.read_text().replace('measure_beats = [4, 1]', ''))
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(args), 0, output.getvalue())
            self.assertIn('Draft complete:', output.getvalue())

    def test_missing_barline_origin_leaves_partial_draft(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = marked_video(root, bar_entries=())
            with redirect_stdout(io.StringIO()) as output:
                self.assertEqual(main(['extract', str(song), '--output', str(root/'out'), '--no-progress']), 2)
            self.assertIn('Draft partial:', output.getvalue())
            self.assertIn('first barline', output.getvalue())
            self.assertIn('Notes with missing beat positions: 6.', output.getvalue())
            result = next((root/'out').glob('*/*/*/' + IMAGE_NAME)).parent
            self.assertTrue((result/'review-photos-r0000/check-0001-01.png').is_file())
            self.assertIn('source video photos', output.getvalue())

    def test_review_question_identifies_the_disputed_endpoint_and_action(self):
        from ez2rc.v2.photos import review_question, _answers
        a = {'lane': 2, 'type': 'longnote', 'start_beat': [1,1], 'end_beat': [3,1]}
        options = {'A': a, 'B': {**a, 'end_beat': [4,1]}}
        self.assertEqual(review_question(options), 'Lane 3: on which beat does the hold end?')
        self.assertIn('end beat 4', _answers(options, 3)[1])
        self.assertIn('ez2rc choose 3 B', _answers(options, 3)[1])
        self.assertIn('--end <beat>', _answers({'A': {**a, 'end_beat': None}}, 3)[0])

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
