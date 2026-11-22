"""Review translated tap placements from procedurally generated screen tracks."""
from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction as F
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import av
import cv2
import numpy as np

from ez2rc.v2.cli import main
from ez2rc.v2.contracts import ratio
from ez2rc.v2.draft import change_note, create_draft, read_draft, save_draft
from ez2rc.v2.draft_phase import recover_positions
from ez2rc.v2.draft_projection import apply_start_grid
from ez2rc.v2.endpoint_ranges import closed
from ez2rc.v2.picture import label, pending_items
from ez2rc.v2.storage import file_hash


def scene(positions=(66, 66, 42), tick_ranges=None, starts=(0, 0, 0), rival=False, scale=(96, 96),
          quarters=(9, 9, 10), indicator_only=False):
    notes, candidates, tracks, optical = [], [], {}, {}
    for lane, (y, first, quarter) in enumerate(zip(positions, starts, quarters, strict=True)):
        identity = f'key-{lane}'
        endpoint = f'edge-{lane}'
        base = F(quarter, 4)
        first_tick, last_tick = tick_ranges[lane] if tick_ranges else (3, 4)
        x = 2 + 12*lane
        witness = {'frame_id': first, 'timestamp_ms': first*20, 'y_px': y-5,
                   'observation_id': f'f{first}:l{lane}:y{y}', 'box_px': [x, y-5, 8, 6]}
        point = {'offset_px': [5, 1], 'beat_range': closed(base + F(first_tick, 192), base + F(last_tick, 192)),
                 'reasons': [], 'witnesses': [witness]}
        note = {'id': identity, 'type': 'tap', 'lane': lane, 'origin': 'automatic',
                'start_beat': None, 'end_beat': None, 'issues': [], 'status': 'draft',
                'evidence': {'start': point, 'pixel_status': 'resolved', 'event_id': identity}}
        if indicator_only:
            optical[endpoint] = {'pulse': {'range': point['beat_range']}}
            point.update(beat_range=None, reasons=['no_observed_path_to_numbered_barline'])
            note.update(status='unresolved', issues=['start_unavailable'])
        else:
            apply_start_grid(note)
        notes.append(note)
        candidates.append({'id': identity, 'event_id': identity, 'selected': True, 'start': endpoint})
        tracks[endpoint] = np.array([(f, y + 2*(f-first)) for f in range(first, first+10)])
    if rival:
        candidates.append({**candidates[-1], 'id': 'competing-hold', 'selected': False})
    shapes = [{'tap': (6, 8)} for _ in notes]
    with (patch('ez2rc.v2.draft_phase._read_tracks', return_value=(tracks, {}, {})),
          patch('ez2rc.v2.draft_phase._indicator_ranges', return_value=optical)):
        placements = recover_positions(notes, candidates, None, {}, shapes, {'hypotheses': []},
                                       [ratio(F(s)) for s in scale], 0)
    return notes, placements


def recording(path):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('ffv1', rate=50)
        stream.width, stream.height, stream.pix_fmt = 48, 96, 'bgr0'
        for f in range(10):
            image = np.zeros((96, 48, 3), np.uint8)
            for lane, y in enumerate((66, 66, 42)):
                bottom, left = y+2*f, 2+12*lane
                image[bottom-5:bottom+1, left:left+8] = 230
            frame = av.VideoFrame.from_ndarray(image, format='bgr24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


class TapPlacementsTest(unittest.TestCase):
    def test_two_translations_of_a_chord_and_later_tap(self):
        notes, pair = scene()
        for note, quarter in zip(notes, (9, 9, 10)):
            self.assertIsNone(note['start_beat'])
            self.assertEqual(pair[note['id']], [ratio(F(quarter, 4) + F(tick, 192)) for tick in (3, 4)])

    def test_pairwise_fits_must_share_a_scale(self):
        # Consecutive quarter beats span 20 and 30 pixels. Even the outer
        # pair fits somewhere in the scale range, but all three cannot fit.
        notes, pair = scene(positions=(76, 56, 26), scale=(72, 128), quarters=(9, 10, 11))
        self.assertTrue(all(len(n['evidence']['start']['candidate_ticks']) == 2 for n in notes))
        self.assertIsNone(pair)

    def test_third_placement_and_disconnected_groups_stay_individual(self):
        self.assertIsNone(scene(tick_ranges=[(3, 6)]*3)[1])
        self.assertIsNone(scene(positions=(66, 42, 66, 42), starts=(0, 0, 20, 20),
                                quarters=(9, 10, 13, 14))[1])

    def test_count_joint_placements_even_when_marginal_counts_differ(self):
        notes, pair = scene(tick_ranges=[(3, 6), (3, 6), (3, 4)])
        self.assertEqual(len(notes[0]['evidence']['start']['candidate_ticks']), 3)
        self.assertEqual(len(pair), 3)
        # Of four combinations, (3,4) and (4,4) fit. Only the first
        # note changes, so these are different spacings.
        self.assertIsNone(scene(positions=(66, 42), quarters=(9, 10), starts=(0, 0),
                                tick_ranges=[(3, 4), (4, 6)], scale=(94, 96))[1])

    def test_competing_shape_and_insufficient_overlap_keep_individual_options(self):
        for notes, pair in (scene(rival=True), scene(starts=(0, 0, 8))):
            self.assertEqual(set(pair), {n['id'] for n in notes[:2]})
            self.assertIsNone(notes[2]['start_beat'])
            self.assertEqual(len(notes[2]['evidence']['start']['candidate_ticks']), 2)

    def test_compare_and_apply_either_placement_from_the_chart(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root/'taps.mkv'
            recording(video)
            notes, pair = scene(indicator_only=True)
            single = {**deepcopy(notes[0]), 'id': 'individual', 'evidence': {'event_id': 'separate'}}
            draft = create_draft({'notes': [*notes, single], 'alternatives': [], 'tap_placements': pair,
                                  'tail_beat_extension': [0, 1], 'measure_beats': [4, 1]},
                                 {'title': 'Taps', 'mode': '3k', 'game': 'test', 'difficulty': 'test',
                                  'lane_count': 3}, '150', {'video': str(video), 'video_sha256': file_hash(video),
                                  'alignment_offset_px': [0, 0], 'lane_rois': [[12*i, 0, 12*i+12, 96] for i in range(3)]})
            manual = change_note(draft, 'edit', identity=notes[0]['id'], fields={'lane': 1})
            self.assertEqual(list(pending_items(manual)), [1, 2, 3, 4])
            for letter, index in [('A', 0), ('B', 1)]:
                run = root/letter
                path = run/'.ez2rc/draft.json'
                save_draft(path, draft)
                def cli(*args):
                    return main([*args, '--run', str(run)])
                with self.subTest(letter=letter), redirect_stdout(io.StringIO()) as output:
                    saved = path.read_bytes()
                    with patch('ez2rc.v2.picture.label', wraps=label) as draw:
                        self.assertEqual(cli('chart'), 0)
                    legend = [call.args[1] for call in draw.call_args_list if call.args[1].startswith('#1')]
                    self.assertEqual(legend, ['#1    3 taps. B is 1/192 beat later than A.',
                                              '#1A    beat 2 + 17/64    lanes 1, 2',
                                              '#1A    beat 2 + 33/64    lanes 3',
                                              '#1B    beat 2 + 13/48    lanes 1, 2',
                                              '#1B    beat 2 + 25/48    lanes 3'])
                    self.assertEqual(cli('choose', '1'), 0)
                    self.assertIn('\n'.join(legend), output.getvalue())
                    self.assertEqual(cli('photos'), 0)
                    self.assertIsNotNone(cv2.imread(str(run/'review-photos-r0000/check-0001-01.png')))
                    self.assertEqual(cli('choose', '1', letter, '--frame', str(run/'frame.png')), 0)
                    self.assertIsNotNone(cv2.imread(str(run/'frame.png')))
                    self.assertEqual(cli('choose', '1', 'X'), 1)
                    self.assertEqual(cli('choose', '1', letter, '--start', '3'), 1)
                    self.assertEqual(path.read_bytes(), saved)
                    self.assertEqual(list(pending_items(read_draft(path))), [1, 4])
                    with patch('ez2rc.v2.picture.write_picture', side_effect=OSError('cannot stage chart')):
                        self.assertEqual(cli('choose', '1', letter), 1)
                    self.assertEqual(path.read_bytes(), saved)
                    self.assertEqual(cli('choose', '1', letter), 0)
                    edited = read_draft(path)
                    self.assertEqual(edited['automatic'], draft['automatic'])
                    self.assertEqual(edited['notes'][-1], single)
                    self.assertEqual(list(pending_items(edited)), [4])
                    self.assertEqual(edited['revision'], len(notes))
                    for original, note, change in zip(notes, edited['notes'], edited['history']):
                        self.assertEqual(note['start_beat'], pair[original['id']][index])
                        self.assertEqual((note['status'], note['issues']), ('reviewed', []))
                        self.assertEqual((change['before'], change['after']), (original, note))
                    self.assertEqual(cli('choose', '4', 'A', '--start', '5/2'), 0)
                    self.assertFalse(pending_items(read_draft(path)))
