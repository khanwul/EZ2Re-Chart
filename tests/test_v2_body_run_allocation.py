"""Algebraic reward conservation on procedural brightness fields."""
from copy import deepcopy
import unittest

import numpy as np

from ez2rc.v2 import decode
from ez2rc.v2.config import DecodeConfig
from ez2rc.v2.storage import CacheError
from v2_decode_fixtures import scene, selected


def field(length, *, absent=(), unavailable=(), masks=None, weight=1.):
    q = np.arange(length)
    cells = {'index': np.column_stack((q * 0, q)),
             'score_sum': np.zeros((length, 4)), 'weight_sum': np.zeros((length, 4)),
             'visibility_weight': np.zeros((length, 4, 5))}
    cells['score_sum'][:, 3] = .6 * weight
    cells['weight_sum'][:, 3] = weight
    cells['visibility_weight'][:, 3, 1] = weight
    for a, b in absent:
        cells['score_sum'][a:b, 3] = 0
    for a, b in unavailable:
        cells['score_sum'][a:b, 3] = 0
        cells['weight_sum'][a:b, 3] = 0
        cells['visibility_weight'][a:b, 3, 1] = 0
    return decode.BodyEvidence(cells, 1., DecodeConfig(), endpoint_footprints=masks)


def body_reward(body, start, end, *, body_only=False):
    terms = decode.interval_cost([] if body_only else [{'utility': 0.}], 0,
                                 body.summary(start, end), body.config)
    return -sum(terms[k] for k in ('body_support', 'body_support_shortfall',
                                  'endpoint_body_overlap', 'body_run_allocation') if k in terms)


class BodyRunAllocationTest(unittest.TestCase):
    def test_partition_does_not_multiply_continuous_body_reward(self):
        for length in (12, 120, 1200):
            body = field(length)
            for count in (2, 3, 8):
                cuts = np.linspace(0, length, count + 1)
                reward = sum(body_reward(body, a, b) for a, b in zip(cuts, cuts[1:]))
                self.assertAlmostEqual(reward, body_reward(body, 0, length))
                self.assertLessEqual(reward, body.config.body_reward + 1e-12)

    def test_fractional_mask_and_sparse_support_share_the_same_scale(self):
        for weight in (.1, .5, 1., 3.):
            body = field(120, masks=[[0, 12.5], [50.5, 65.]], weight=weight)
            cuts = (0, 17.5, 60.25, 120)
            for body_only in (False, True):
                reward = sum(body_reward(body, a, b, body_only=body_only) for a, b in zip(cuts, cuts[1:]))
                self.assertAlmostEqual(reward, body_reward(body, 0, 120, body_only=body_only))

    def test_unobserved_cells_do_not_create_fresh_reward_runs(self):
        body = field(120, unavailable=[(40, 80)])
        self.assertEqual(body.runs, [(0., 120.)])
        self.assertAlmostEqual(body_reward(body, 0, 60) + body_reward(body, 60, 120),
                               body_reward(body, 0, 120))
        self.assertEqual(body.summary(0, 120)['unavailable_px'], 40.)

    def test_observed_darkness_separates_runs_without_changing_absence(self):
        body = field(120, absent=[(40, 80)])
        self.assertEqual(body.runs, [(0., 40.), (80., 120.)])
        self.assertAlmostEqual(body_reward(body, 0, 40), 1.)
        self.assertAlmostEqual(body_reward(body, 80, 120), 1.)
        self.assertEqual(body.summary(0, 120)['absent_px'], 40.)
        self.assertEqual(body.summary(0, 120)['reward_scale_px'], 120.)

    def test_half_open_intervals_do_not_borrow_adjacent_runs(self):
        scale = decode.BodyRunScale([[0, 20], [40, 160]])
        for a, b in ((0, 20), (20, 40), (40, 40), (-10, 0)):
            self.assertEqual(scale.scale(a, b, 24.), 24.)
        self.assertEqual(scale.scale(40, 41, 24.), 120.)
        self.assertEqual(scale.scale(19, 41, 24.), 120.)
        self.assertEqual(decode.BodyRunScale([]).scale(0, 1, 24.), 24.)

    def test_extent_includes_each_run_touching_the_search_radius(self):
        body = field(120, absent=[(40, 80)])
        for radius, left, right in ((19.5, 60, 60), (20, 0, 120)):
            with self.subTest(radius=radius):
                self.assertEqual(body.extent(60, -1, radius), left)
                self.assertEqual(body.extent(60, 1, radius), right)
        self.assertEqual(body.extent(-10, 1, 1), -10)
        self.assertEqual(body.extent(130, -1, 1), 130)

    def test_weak_internal_boundaries_do_not_gain_an_extra_body_reward(self):
        # Same procedural coordinates as the existing touching-endpoint contract.
        data = scene([(20, 1), (80, [.05, .05, .25]),
                      (80, [.05, .25, .05]), (140, 2)], [(20, 140)])
        self.assertEqual(selected(data), [('longnote', 20., 140.)])
        retained = {(c['start_px'], c['end_px']) for c in data['candidates']}
        self.assertIn((20., 80.), retained)
        self.assertIn((80., 140.), retained)

    def test_supported_touching_notes_still_keep_separate_endpoints(self):
        data = scene([(20, 1), (80, 2), (80, 1), (140, 2)], [(20, 140)])
        self.assertEqual(selected(data), [('longnote', 20., 80.), ('longnote', 80., 140.)])
        decode._validate_lane(0, data, DecodeConfig(), 1.)

    def test_validation_rejects_changed_runs_scales_and_allocation_costs(self):
        data = scene([(20, 1), (80, 2), (80, 1), (140, 2)], [(20, 140)])
        for field in ('runs', 'shortened_run', 'scale', 'cost'):
            bad = deepcopy(data)
            candidate = next(c for c in bad['candidates'] if c['selected'])
            if field == 'runs':
                bad['body_support_runs_px'] = [[20., 80.], [80., 140.]]
            elif field == 'shortened_run':
                bad['body_support_runs_px'] = [[20., 100.]]
            elif field == 'scale':
                candidate['body']['reward_scale_px'] *= 2
            else:
                candidate['cost_terms']['body_run_allocation'] = 0.
            with self.subTest(field=field), self.assertRaises(CacheError):
                decode._validate_lane(0, bad, DecodeConfig(), 1.)


if __name__ == '__main__':
    unittest.main()
