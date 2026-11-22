from fractions import Fraction
import unittest

from ez2rc.v2.barline_spacing import spatial_crossings


class LineSpacingTests(unittest.TestCase):
    def fit(self, steps, width=.5, conflict=False):
        def line(name, first, y):
            rows = [{'frame_id': first+i, 'y_px': y+i*6, 'visible_range_px': [y+i*6-width, y+i*6+width]}
                    for i in range(3)]
            return dict(id=name, observations=rows)
        tracks = [line('origin', 0, 60), line('numbered', 14, 44), line('moving', 6, 46)]
        if conflict:
            tracks[-1]['observations'][-1]['visible_range_px'] = [0, 1]
        anchors = [dict(marker_id='origin', beat=[0, 1]), dict(marker_id='numbered', beat=[3, 2])]
        return spatial_crossings(tracks, anchors, ['moving'], steps)

    def test_fractional_boundary_does_not_bisect_the_measure(self):
        steps = {f: (f-1, 6, 6) for f in range(1, 17)}
        self.assertEqual(self.fit(steps)[0], {'moving': Fraction(3, 4)})
        # Shift the movement before the middle line, keeping the total unchanged.
        for f in (3, 4, 5, 6):
            steps[f] = (f-1, 10, 11)
        for f in (10, 11, 12, 13):
            steps[f] = (f-1, 1, 2)
        self.assertEqual(self.fit(steps)[0], {'moving': Fraction(1)})

    def test_missing_inconsistent_and_multivalued_links_withhold_placement(self):
        steps = {f: (f-1, 6, 6) for f in range(1, 17)}
        for params in (({},), (steps, .5, True), (steps, 12)):
            with self.subTest(params=params):
                self.assertFalse(self.fit(*params)[0])
