import unittest
from types import SimpleNamespace

import numpy as np

from ez2rc.v2.draft_hold_shape import confirm_holds


class HoldShapeTests(unittest.TestCase):
    def trace(self, *, split=False, hidden=None, rival=False, timing=False):
        frames = np.arange(40)
        image = np.zeros((40, 1, 120))
        indices, positions, flags, requests = [], [], [], {}
        for f in frames:
            bottom = 11+4*int(f)
            top = bottom-52
            image[f, 0, max(0, top):bottom+1] = 220
            # A one-row gap passes the scan row between sampled frames.
            if split and 0 <= bottom-26 < 120:
                image[f, 0, bottom-26] = 0
            for kind, edge, y in ((1, bottom, bottom-7), (2, top, top)):
                if not 0 < y < 112:
                    continue
                key = (int(f), 0, y)
                indices.append(key)
                requests[(*key, kind)] = 'start' if kind == 1 else 'end'
                positions.append([(0, y)]*3)
                flags.append([2 if hidden == kind else 4]*3)
        chunk = dict(frame_id=frames, timestamp_ms=frames, projection=image, candidate_index=indices,
                     scores=np.ones((len(indices), 3)), flags=np.array(flags), positions=np.array(positions))
        obs = SimpleNamespace(manifest={'metadata': {'profile': {'effect_top': 120, 'lanes': [{'roi': [0,0,10,120]}]}},
                                        'key': {'parameters': {'algorithm': {'brightness_threshold': 80}}}},
                              chunks=lambda: iter([chunk]))
        note = dict(id='hold', lane=0, issues=['pixel_ambiguous'], status='unresolved', start_beat=[0,1],
                    end_beat=[1,1], evidence={'start': {'candidate_ticks': [0]},
                                             'end': {'beat_range': {'lower': [9,10]}}})
        if timing:
            note.update(start_beat=None)
            note['issues'].append('start_grid_ambiguous')
        confirm_holds([note], {'hold': (requests, not rival)}, obs, [{'head': [8,10], 'tail': [8,10]}], 1,
                      {'max_speed_px_per_ms': 6, 'displacement_slack_px': 1})
        return note

    def test_connected_body_with_two_visible_ends(self):
        note = self.trace()
        self.assertEqual(note['issues'], [])
        self.assertEqual(note['evidence']['hold_shape']['body_frames'], [13,25])

    def test_separate_notes_and_missing_terminal_edges_stay_open(self):
        for options in (dict(split=True), dict(hidden=1), dict(hidden=2)):
            with self.subTest(options=options):
                self.assertEqual(self.trace(**options)['issues'], ['pixel_ambiguous'])

    def test_shape_does_not_override_a_rival_or_missing_start(self):
        for options in (dict(timing=True), dict(rival=True)):
            with self.subTest(options=options):
                note = self.trace(**options)
                self.assertIn('hold_shape', note['evidence'])
                self.assertEqual(note['status'], 'unresolved')
