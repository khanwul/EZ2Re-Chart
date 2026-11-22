from fractions import Fraction as F
import unittest

from ez2rc.v2.barline_spacing import clock_bracket, recover_disconnected_crossings, row_crossings


class CrossingClockTests(unittest.TestCase):
    def readings(self, start, *, delay=0, sparse=False):
        frames = list(range(2 if sparse else 5))
        rows = [dict(frame_id=f, y_px=10*f, visible_range_px=[10*f-.5, 10*f+.5]) for f in frames]
        return row_crossings(rows, {f: F(start+4*f+(delay if f >= 3 else 0)) for f in frames})

    def test_two_local_clocks_place_a_fractional_measure(self):
        fit = clock_bracket(self.readings(250), self.readings(100), self.readings(450),
                            (F(2), F(11,2)), F(100), 20)
        self.assertEqual(fit[0], F(7,2))

    def test_inconsistent_anchor_motion_and_sparse_sweeps_stay_open(self):
        for right in (self.readings(450, delay=11), self.readings(450, sparse=True)):
            self.assertIsNone(clock_bracket(self.readings(250), self.readings(100), right,
                                           (F(2), F(11,2)), F(100), 20))


    def test_unavailable_motion_can_use_two_anchors_but_a_conflict_keeps_review(self):
        markers, tracks, times = [], [], {}
        for i, (name, time) in enumerate((('a', 50), ('b', 125), ('c', 250))):
            markers.append(dict(id=name, timestamp_interval_ms=[time, time+4]))
            rows = []
            for j in range(5):
                frame = 10*i+j
                times[frame] = F(time+4*j)
                rows.append(dict(frame_id=frame, y_px=8*j, visible_range_px=[8*j-.5, 8*j+.5]))
            tracks.append(dict(id=name, observations=rows))
        anchors = [dict(marker_id='a', beat=[0,1]), dict(marker_id='c', beat=[2,1])]
        evidence = dict(status='unavailable', no_measured_scale=True)
        self.assertEqual(recover_disconnected_crossings(markers, tracks, anchors, evidence, times, F(100), 15),
                         {'b': F(3,4)})
        self.assertEqual(evidence['crossings']['b']['local_crossing_clock']['anchors'], ['a','c'])
        self.assertFalse(recover_disconnected_crossings(markers, tracks, anchors, {'status':'conflict'},
                                                        times, F(100), 15))
