"""Exact finite grids; uncertainty and physical ambiguity remain visible."""
from copy import deepcopy
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest

from ez2rc.v2.config import ConfigError, load_song
from ez2rc.v2.contracts import ratio, read_ratio
from ez2rc.v2.rhythm import DEFAULT_SUBDIVISIONS, RhythmGrid, grid_candidates, rhythm_candidates
from test_v2_relative_beats import project, scene
from test_v2_timing_resolver import moving_taps


def domain(lo, hi, left=True, right=True):
    return {'lower': ratio(F(lo)), 'upper': ratio(F(hi)),
            'lower_inclusive': left, 'upper_inclusive': right}


class RhythmTest(unittest.TestCase):
    def test_grid_union_deduplicates_equal_rationals_and_retains_support(self):
        actual = grid_candidates(domain(0, 1), RhythmGrid([2, 3]))
        self.assertEqual([read_ratio(c['beat']) for c in actual], [F(0), F(1,3), F(1,2), F(2,3), F(1)])
        self.assertEqual(actual[0]['subdivisions'], [2, 3])
        self.assertEqual(actual[2]['subdivisions'], [2])

    def test_negative_beats_and_open_boundaries(self):
        grid = RhythmGrid([3])
        self.assertEqual([read_ratio(c['beat']) for c in grid_candidates(domain(-1, F(1,3), False, False), grid)],
                         [F(-2,3), F(-1,3), F(0)])
        self.assertEqual(grid_candidates(domain(0, 0, False), grid), [])
        self.assertEqual(grid_candidates(domain(F(1,10), F(1,5)), grid), [])
        self.assertIsNone(grid_candidates(None, grid))

    def test_missing_endpoint_and_physical_ambiguity_do_not_become_accepted(self):
        data = scene(); data[4]['events'][0]['status'] = 'ambiguous'
        spatial = project(data=data)
        row = rhythm_candidates(spatial, RhythmGrid([2]))['candidates'][0]
        self.assertEqual(row['status'], 'unique')
        self.assertEqual(row['note_candidates'], [{'start_beat': [3,2], 'end_beat': [9,2]}])
        self.assertEqual(row['pixel_status'], 'ambiguous')
        self.assertFalse(row['accepted'])
        spatial['candidates'][0]['endpoints']['end']['relative_beat'] = None
        row = rhythm_candidates(spatial, RhythmGrid([2]))['candidates'][0]
        self.assertEqual(row['status'], 'unavailable')
        self.assertIsNone(row['note_candidates'])

    def test_joint_duration_can_exclude_all_marginal_pairs(self):
        spatial = project()
        spatial['candidates'][0]['duration_beat_range'] = domain(3, 4, False)
        row = rhythm_candidates(spatial, RhythmGrid([2]))['candidates'][0]
        self.assertEqual(row['status'], 'conflict')
        self.assertEqual(row['note_candidates'], [])
        self.assertTrue(all(p['status']=='unique' for p in row['endpoints'].values()))

    def test_widening_adds_hypotheses_instead_of_rounding(self):
        spatial = project()
        for p in spatial['candidates'][0]['endpoints'].values():
            value = read_ratio(p['relative_beat']['beat_range']['lower'])
            p['relative_beat']['beat_range'] = domain(value-F(1,32), value+F(1,32))
        row = rhythm_candidates(spatial, RhythmGrid())['candidates'][0]
        self.assertEqual(row['status'], 'ambiguous')
        self.assertGreater(len(row['note_candidates']), 1)
        self.assertFalse(row['accepted'])

    def test_configuration_rejects_invalid_grids_and_canonicalizes_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            song = moving_taps(Path(tmp)); original = song.read_text()
            self.assertEqual(load_song(song).rhythm_grid.subdivisions, DEFAULT_SUBDIVISIONS)
            song.write_text(original+'\n[timing.rhythm_grid]\nsubdivisions = [6, 3, 1]\n')
            self.assertEqual(load_song(song).snapshot()['rhythm_grid'], {'subdivisions': [1,3,6]})
            for value in ('[]', '[1,1]', '[0]', '[true]', '[1.5]'):
                song.write_text(original+f'\n[timing.rhythm_grid]\nsubdivisions = {value}\n')
                with self.subTest(value=value), self.assertRaises(ConfigError): load_song(song)

    def test_missing_spatial_reference_is_explicit(self):
        value = rhythm_candidates(None, RhythmGrid([4]))
        self.assertEqual(value['status'], 'unavailable')
        self.assertEqual(value['grid']['subdivisions'], [4])
        self.assertEqual(value['accepted_note_count'], 0)
