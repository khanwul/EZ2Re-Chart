"""Video drafts can span a different number of quarter-note beats in each measure."""
from fractions import Fraction
import unittest

import numpy as np

from ez2rc.v2.barline_grid import observed_barline_beats
from ez2rc.v2.draft_projection import draft_timing_issues
from ez2rc.v2.overlay import BeatGrid
from ez2rc.v2.picture import chart_strips
from ez2rc.v2.review_chart import chart_view


def crossings(beats):
    return [{'id': str(i), 'timestamp_interval_ms': [500*b-2, 500*b+2]}
            for i, b in enumerate(beats)]


class ObservedMeterTest(unittest.TestCase):
    def test_different_and_fractional_lengths_and_an_uncertain_gap(self):
        expected = [0, 4, 7, 10.5, 14.5, 22.5, 26.5]
        hypotheses, gaps, unplaced, reasons = observed_barline_beats(crossings(expected), Fraction(500))
        self.assertEqual([Fraction(*a['beat']) for a in hypotheses[0]['assignments']], expected)
        self.assertEqual(gaps, [[[29, 2], [45, 2]]])
        self.assertEqual(unplaced, [])
        self.assertEqual(reasons, [])
        issues = draft_timing_issues({'projection': {'origin_issues': [], 'geometry_issues': [],
                                      'uncertain_measure_spans': gaps},
                                      'notes': [{'type': 'tap', 'start_beat': [4, 1]}]})
        self.assertIn('29/2', issues[0])

    def test_ambiguous_crossing_does_not_move_later_beats(self):
        markers = crossings([0, 4, 7, 11])
        markers[2]['timestamp_interval_ms'] = [3200, 3700]
        hypotheses, gaps, unplaced, _ = observed_barline_beats(markers, Fraction(500))
        self.assertEqual(unplaced, ['2'])
        self.assertEqual([a['beat'] for a in hypotheses[0]['assignments']], [[0, 1], [4, 1], [11, 1]])
        self.assertEqual(gaps, [[[4, 1], [11, 1]]])

    def test_picture_html_and_overlay_draw_the_saved_boundaries(self):
        projection = {'barline_beats': [[n, 1] for n in (0, 4, 7, 11, 15, 18)]}
        note = {'id': 'a', 'lane': 0, 'type': 'longnote', 'start_beat': [14, 1],
                'end_beat': [16, 1], 'status': 'draft', 'issues': []}
        draft = {'projection': projection, 'metadata': {'lane_count': 2}, 'notes': [note], 'bpm': '120'}
        _, png = chart_strips(draft, {})
        bars = np.flatnonzero((png[:, 70] == (158, 168, 180)).all(axis=1))[::-1]
        self.assertEqual(list(bars), [bars[0] - round((bars[0]-bars[-1])*b/15) for b in (0,4,7,11,15)])
        html = chart_view(draft)
        self.assertIn('0–15 beats', html)
        self.assertIn('Chart starting at beat 15', html)
        image = np.zeros((180,80,3), np.uint8)
        grid = BeatGrid({'lanes': [{'roi': [0, 0, 30, 180]}]}, (0,0), 10, None, projection=projection)
        grid.draw(image, 165)
        solid = np.flatnonzero((image[:, 2] == (50, 205, 255)).all(axis=1))
        self.assertEqual(list(solid), [15, 55, 95, 125, 165])

    def test_trailing_observed_lines_remain_visible_after_the_last_note(self):
        projection = {'barline_beats': [[b, 1] for b in (0, 2, 4, 6, 8, 11, 13, 16, 18)]}
        note = {'id': 'early', 'lane': 0, 'type': 'tap', 'start_beat': [1, 1], 'end_beat': None,
                'status': 'draft', 'issues': []}
        draft = {'projection': projection, 'metadata': {'lane_count': 2}, 'notes': [note], 'bpm': '100'}
        html = chart_view(draft)
        self.assertIn('Chart starting at beat 8', html)
        self.assertNotIn('Chart starting at beat 18', html)
        _, strip = chart_strips(draft, {})
        # Look beyond the first column for the five actual boundaries. The
        # last line closes the second column without creating a third.
        hits = np.all(strip == (158, 168, 180), axis=2)
        last_x = np.flatnonzero(np.any(hits, axis=0))[-1]
        rows = np.flatnonzero(hits[:, last_x])[::-1]
        self.assertEqual(list(rows), [rows[0] - round((rows[0]-rows[-1])*b/10) for b in (0,3,5,8,10)])
