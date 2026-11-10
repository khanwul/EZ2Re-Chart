"""Procedural cadence variation and unsupported motion forecasts."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ez2rc.v2.config import MotionConfig
from ez2rc.v2.motion import STATUS, _forecast_rate, estimate, timeline
from v2_motion_fixtures import observations


def point(fid, time, scroll, *, segment=0, status='observed'):
    return {'id': fid, 'ms': time, 'path': (segment, 0., 0.),
            'row': {'scroll_px': scroll, 'status': STATUS[status]}}


class MotionForecastTest(unittest.TestCase):
    def test_rendered_step_phase_does_not_change_gap_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for phase in (-3, 3):
                for chunk in (1, 7):
                    truth = [8*i + phase*(i % 2) for i in range(20)]
                    obs = observations(root, truth, blank=range(12, 20), chunk_frames=chunk)
                    moved, _ = estimate(obs, MotionConfig(), root)
                    clock = timeline(moved)
                    np.testing.assert_allclose(clock['scroll_px'][:12], truth[:12])
                    np.testing.assert_allclose(np.diff(clock['scroll_px'][11:]), 8.)
                    self.assertTrue((clock['status'][12:] == STATUS['extrapolated']).all())
                    self.assertTrue((np.diff(clock['error_px'][11:]) > 0).all())
                    self.assertTrue((clock['reference_frame'][12:] == -1).all())

    def test_unequal_pts_and_coordinate_origin_preserve_constant_rate(self):
        times = [0., 5., 17., 18., 31.]
        for origin in (0., 10000.):
            rows = [point(i, t, origin+1.5*t) for i, t in enumerate(times)]
            self.assertEqual(_forecast_rate(rows, .25), 1.5)

    def test_disconnected_coordinates_do_not_vote_across_gap(self):
        history = [point(0, 0., 0.), point(1, 5., 10.),
                   point(2, 10., 100., segment=2), point(3, 15., 105., segment=2)]
        self.assertEqual(_forecast_rate(history, .25), 1.)

    def test_uncertain_or_repeated_support_requires_fallback(self):
        for status in ('single_lane', 'ambiguous', 'extrapolated', 'unresolved', 'repeated'):
            history = [point(0, 0., 0.), point(1, 5., 10., status=status), point(2, 10., 20.)]
            self.assertEqual(_forecast_rate(history, .25), .25)
        self.assertEqual(_forecast_rate([point(0, 0., 0.)], .25), .25)


if __name__ == '__main__':
    unittest.main()
