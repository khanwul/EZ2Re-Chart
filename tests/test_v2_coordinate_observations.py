"""Retain the actual order-statistic observations used by endpoint medians."""
from copy import deepcopy
import unittest

from ez2rc.v2.config import DecodeConfig
from ez2rc.v2.decode import _validate_lane
from ez2rc.v2.storage import CacheError
from v2_decode_fixtures import associate, scene


class CoordinateObservationsTest(unittest.TestCase):
    def test_odd_and_even_medians_bind_the_middle_observations(self):
        for positions in ((100, 101, 99), (100, 101, 99, 102)):
            _, endpoints = associate([(i, q, [.99, .01, .01]) for i, q in enumerate(positions)])
            tap = next(e for e in endpoints if e['kind'] == 'tap')
            support = tap['coordinate_observations']
            self.assertEqual(len(support), 1 if len(positions) % 2 else 2)
            self.assertEqual(sum(p['q_px']/len(support) for p in support), tap['q_px'])
            self.assertEqual([p['q_px'] for p in support], [100] if len(positions) % 2 else [100,101])
            self.assertTrue(all(p['observation_id'] in tap['observation_ids'] for p in support))

    def test_display_repetitions_do_not_get_more_median_weight(self):
        _, endpoints = associate([(0, 100, [.99,.01,.01]), (1,100,[.99,.01,.01]), (2,102,[.99,.01,.01])],
                                 groups=[0,0,2])
        tap = next(e for e in endpoints if e['kind'] == 'tap')
        self.assertEqual(tap['q_px'], 101)
        self.assertEqual([p['frame_id'] for p in tap['coordinate_observations']], [0,2])

    def test_corrupt_coordinate_identity_weight_or_value_is_rejected(self):
        data = scene([(80, 0)])
        _validate_lane(0, data, DecodeConfig(), 1.)
        for field,value in [('observation_id','not-observed'), ('weight',[1,2]), ('frame_id',900), ('q_px',81)]:
            bad = deepcopy(data)
            bad['endpoints'][0]['coordinate_observations'][0][field] = value
            with self.subTest(field=field), self.assertRaises(CacheError):
                _validate_lane(0, bad, DecodeConfig(), 1.)


if __name__ == '__main__':
    unittest.main()
