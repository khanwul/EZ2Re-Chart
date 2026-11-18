"""Chart drawing checks using independently constructed draft notes."""
from html.parser import HTMLParser
import unittest
from xml.etree import ElementTree

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
    def test_quarter_guides_align_to_beats_in_a_fractional_panel(self):
        # This panel starts at beat 5/3. The measure line at 2.5 takes precedence.
        html = chart_view(draft([note('tap', [2, 1])], [5, 12]))
        tags = Elements(html).elements
        last_column = max(i for i, (tag, _) in enumerate(tags) if tag == 'svg')
        self.assertEqual(sum(t == 'svg' for t, _ in tags), 2)  # Includes the lead-in from beat zero.
        guides = [a['d'] for t, a in tags[last_column:] if t == 'path' and a['stroke'] == '#2c3748']
        self.assertEqual(guides, [f'M19 {y}h170' for y in (547, 385, 223, 61)])
        labels = [float(a['y']) for t, a in tags[last_column:] if t == 'text' and 'dominant-baseline' in a]
        self.assertEqual(labels, [506.5, 371.5, 236.5, 101.5])  # Centered in the four measures.

    def test_short_hold_before_a_measure_stays_inside_both_endpoints(self):
        html = chart_view(draft([note('short', [159,40], [399,100])], [4,1]))
        marks = [a for t,a in Elements(html).elements if t == 'rect' and a['width'] in ('16','26')]
        self.assertEqual(len(marks), 3)
        for mark in marks:
            y = float(mark['y'])
            self.assertAlmostEqual(y, 574 - 399/100 * 540/16, places=3)
            self.assertAlmostEqual(y+float(mark['height']), 574 - 159/40 * 540/16, places=3)

    def test_long_note_continues_across_columns_without_duplicate_head(self):
        html = chart_view(draft([note('held', [15, 1], [17, 1])], [4, 1]))
        tags = Elements(html).elements
        self.assertEqual(sum(t == 'svg' for t, _ in tags), 2)
        caps = [float(a['y']) for t, a in tags if t == 'rect' and a.get('width') == '26']
        self.assertEqual(caps, [574 - 15 * 540/16 - 3, 574 - 540/16])
        self.assertEqual(sum(t == 'rect' and a.get('fill-opacity') == '.72' for t, a in tags), 2)
        panels = ElementTree.fromstring(html[html.index('<div id="chart-panels">'):html.index('<h3>')])
        numbers = [label.text for label in panels.findall('.//svg/text[@dominant-baseline]')]
        self.assertEqual(numbers, list(map(str, range(1, 9))))

    def test_unknown_start_is_listed_without_inventing_position(self):
        html = chart_view(draft([note('unknown', None)]))
        self.assertNotIn('<svg', html)
        self.assertIn('Start beat needs review', html)
        self.assertIn('href="#note-0"', html)

    def test_hold_on_a_seam_keeps_its_head_thickness_in_its_own_column(self):
        for start, end, owner, head_y, tail_y in (([15999, 1000], [20, 1], 1, 574, 439),
                                                  ([12, 1], [16001, 1000], 0, 169, 34)):
            with self.subTest(start=start, end=end):
                html = chart_view(draft([note('hold', start, end), note('later', [18, 1], lane=1)], [4, 1]))
                root = ElementTree.fromstring(html[html.index('<div id="chart-panels">'):html.index('<h3>')])
                panels = root.findall('.//svg')
                self.assertIsNone(panels[1-owner].find('a[@href="#note-0"]'))
                hold = panels[owner].find('a[@href="#note-0"]')
                head = hold.find('rect[@width="26"]')
                tap = panels[1].find('a[@href="#note-1"]/rect')
                self.assertEqual(head.get('height'), tap.get('height'))
                self.assertEqual(float(head.get('y')) + float(head.get('height')), head_y)
                caps = hold.findall('path[@vector-effect="non-scaling-stroke"]')
                self.assertEqual(len(caps), 1)
                self.assertIn(f' {tail_y}h', caps[0].get('d'))

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
