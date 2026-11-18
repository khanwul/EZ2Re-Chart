"""Video drafts can span a different number of quarter-note beats in each measure."""
from fractions import Fraction
import unittest

from ez2rc.v2.barline_grid import observed_barline_beats
from ez2rc.v2.draft_projection import draft_timing_issues


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
