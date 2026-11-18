"""Chart drawing checks using independently constructed draft notes."""
from html.parser import HTMLParser
import unittest

from ez2rc.v2.review_chart import chart_view


def note(identity, start, end=None, lane=0):
    return {'id': identity, 'lane': lane, 'type': 'longnote' if end else 'tap',
            'start_beat': start, 'end_beat': end, 'status': 'draft', 'issues': []}


def draft(notes, measure=None):
    return {'notes': notes, 'metadata': {'lane_count': 5},
            'projection': {'measure_beats': measure}}


class Elements(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.elements = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))


class ReviewChartTest(unittest.TestCase):
    def test_short_hold_before_a_measure_stays_inside_both_endpoints(self):
        html = chart_view(draft([note('short', [159,40], [399,100])], [4,1]))
        marks = [a for t,a in Elements(html).elements if t == 'rect' and a['width'] in ('14','26')]
        self.assertEqual(len(marks), 3)
        for mark in marks:
            y = float(mark['y'])
            self.assertAlmostEqual(y, 34 + 159/40 * 540/16, places=3)
            self.assertAlmostEqual(y+float(mark['height']), 34 + 399/100 * 540/16, places=3)

    def test_long_note_continues_across_columns_without_duplicate_head(self):
        html = chart_view(draft([note('held', [15, 1], [17, 1])], [4, 1]))
        tags = Elements(html).elements
        self.assertEqual(sum(t == 'svg' for t, _ in tags), 2)
        self.assertEqual(sum(t == 'rect' and a.get('height') == '6' for t, a in tags), 1)
        self.assertEqual(sum(t == 'rect' and a.get('fill-opacity') == '.5' for t, a in tags), 2)
        self.assertIn('M1', html)
        self.assertIn('M5', html)

    def test_unknown_start_is_listed_without_inventing_position(self):
        html = chart_view(draft([note('unknown', None)]))
        self.assertNotIn('<svg', html)
        self.assertIn('Start beat needs review', html)
        self.assertIn('href="#note-0"', html)

    def test_negative_fractional_beats_and_untrusted_ids(self):
        html = chart_view(draft([note('<img src=x onerror=alert(1)>', [-1, 2])]))
        self.assertIn('-16–0 beats', html)
        self.assertIn('&lt;img', html)
        self.assertNotIn('<img', html)
        self.assertIn('shows beat ticks', html)

    def test_empty_and_unknown_tail_are_displayable(self):
        self.assertIn('No beat positions to display', chart_view(draft([])))
        row = {**note('partial', [2, 1]), 'type': 'longnote', 'status': 'unresolved'}
        html = chart_view(draft([row]))
        self.assertIn('>?</text>', html)
        self.assertNotIn('fill-opacity', html)
