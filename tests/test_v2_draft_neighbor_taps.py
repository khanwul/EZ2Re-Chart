from fractions import Fraction as F
import unittest

from ez2rc.v2.draft_neighbor_taps import _segments, neighbor_position


class NeighborTapTests(unittest.TestCase):
    def placement(self, *, separated=False, conflict=False, scale=(F(400), F(401)), error=F(1,4)):
        target = {f: 4*f for f in range(16)}
        near = {f: 4*f+25 for f in range(5, 29)}
        samples = {f: {y} for f,y in near.items() if not separated or f != 17}
        markers = {f: {'bar': (F(4*f+100)-F(1,2), F(4*f+100)+F(1,2))} for f in range(19,29)}
        if conflict:
            markers[7] = {'bar': (0,1)}
        return neighbor_position(target, {'other': list(_segments(samples))}, markers,
                                 {'bar': F(3)}, scale, error)

    def test_two_overlapping_scenes_give_a_start_without_a_neighbor_beat(self):
        beat, proof = self.placement()
        self.assertEqual(beat, F(13,4))
        bridge, = proof['bridges']
        self.assertEqual(bridge['shared_frames'], list(range(5,16)))
        self.assertEqual(bridge['numbered_frames'], list(range(19,29)))

    def test_disconnected_conflicting_and_multiple_positions_stay_open(self):
        for params in (dict(separated=True), dict(conflict=True), dict(scale=(F(360),F(440)))):
            with self.subTest(params=params):
                self.assertIsNone(self.placement(**params))

    def test_all_paths_must_use_one_scale(self):
        target = {f: 2*f for f in range(24)}
        refs = {'a': [{f: 2*f+10 for f in range(8,40)}], 'b': [{f: 2*f+15 for f in range(8,40)}]}
        bars = {f: {'early': (2*f+100, 2*f+100), 'late': (2*f-98, 2*f-98)} for f in range(28,40)}
        # Each marker separately allows 1/4, but requires a different scale.
        for key in ('early', 'late'):
            subset = {f: {key: rows[key]} for f, rows in bars.items()}
            result = neighbor_position(target, refs, subset, {'early': F(0), 'late': F(1,2)},
                                       (F(390),F(405)), F(0))
            self.assertEqual(result[0], F(1,4))
        self.assertIsNone(neighbor_position(target, refs, bars, {'early': F(0), 'late': F(1,2)},
                                            (F(390),F(405)), F(0)))
