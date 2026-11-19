import unittest

import numpy as np

from ez2rc.v2.draft_shape import confirm_taps, tap_requests


class TapShapeTests(unittest.TestCase):
    def test_requests_exclude_rivals_noise_and_search_limits(self):
        tap = dict(id='tap', type='tap', selected=True)
        event = dict(status='ambiguous', candidate_ids=['tap'], review_causes=['motion_alternatives'])
        endpoint = dict(observation_ids=['inside', 'limit'], search_boundary_observation_ids=['limit'])
        self.assertEqual(tap_requests(tap, event, endpoint), {'inside'})
        for changes in (dict(candidate_ids=['tap', 'hold']), dict(review_causes=['selection_alternative'])):
            self.assertFalse(tap_requests(tap, dict(event, **changes), endpoint))
        self.assertFalse(tap_requests(dict(tap, type='longnote'), event, endpoint))

    def run_tap(self, *, flag=4, body=False, alternate_beat=False, interrupted=False):
        class Observations:
            manifest = {'metadata': {'profile': {'lanes': [{'roi': [0, 0, 10, 100]}]}},
                        'key': {'parameters': {'algorithm': {'brightness_threshold': 40}}}}

            def chunks(self):
                yield chunk

        frames = np.arange(8)
        tops = 10 + frames
        fields = np.zeros((8, 1, 100))
        for f, top in enumerate(tops):
            fields[f, 0, top:top+8] = 200
            if body:
                fields[f, 0, top+8:] = 180
        flags = np.full((8, 1), flag)
        if interrupted:
            flags[3] = 6
        chunk = dict(frame_id=frames, candidate_index=[(f, 0, 0) for f in frames], flags=flags,
                     positions=np.array([[[0, top]] for top in tops]), scores=np.ones((8, 1)), projection=fields)
        rows = [dict(frame_id=int(f), visible_range_px=np.array([top+6.5, top+7.5])) for f, top in zip(frames, tops)]
        bars = dict(hypotheses=[{'assignments': [dict(marker_id='bar', beat=[4, 1])]}],
                    tracks=[dict(id='bar', observations=rows)])
        note = dict(id='tap', type='tap', issues=['pixel_ambiguous'], status='unresolved',
                    evidence={'start': {'offset_px': [7, 1]}}, start_beat=[5 if alternate_beat else 4, 1])
        confirm_taps([note], {k: 'tap' for k in chunk['candidate_index']}, Observations(),
                     [{'tap': [8, 10]}], bars, [[400, 1], [410, 1]], 0.75)
        return note

    def test_direct_shape_and_position_required(self):
        self.assertEqual(self.run_tap()['evidence']['tap_shape']['frames'], list(range(8)))
        for args in (dict(flag=5), dict(flag=6), dict(body=True), dict(interrupted=True), dict(alternate_beat=True)):
            with self.subTest(args=args):
                self.assertEqual(self.run_tap(**args)['issues'], ['pixel_ambiguous'])
