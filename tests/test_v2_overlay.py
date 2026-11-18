from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import av
import numpy as np

from ez2rc.v2.cli import main
from ez2rc.v2.draft import change_note, read_draft, save_draft
from ez2rc.v2.motion import STATUS
from ez2rc.v2.overlay import BeatGrid, frame_origins, Outlines
from ez2rc.v2.runner import execute
from test_v2_barline_pipeline import marked_video


class OverlayTest(unittest.TestCase):
    def test_scroll_requires_a_link_to_a_numbered_line(self):
        grid = {'status': 'supported', 'hypotheses': [{'assignments': [
            {'marker_id': 'line', 'beat': [4, 1]}]}], 'tracks': [
                {'id': 'line', 'observations': [{'frame_id': 11, 'y_px': 36}]}]}
        clock = {'frame_id': [10, 11, 12, 13, 14], 'scroll_px': [19, 20, 23, 25, 25],
                 'status': [STATUS[s] for s in ('origin', 'observed', 'ambiguous', 'observed', 'repeated')],
                 'reference_frame': [-1, 10, 11, 11, -1], 'repeat_group': [10, 11, 12, 13, 13]}
        self.assertEqual(frame_origins(clock, grid, 8), {10: 67, 11: 68, 13: 73, 14: 73})

    def test_moving_grid_clips_to_lanes_and_prefers_measure_at_shared_beat(self):
        profile = {'lanes': [{'roi': [40, 10, 80, 90]}, {'roi': [82, 30, 122, 90]}]}
        grid = BeatGrid(profile, (3, 2), 16, [3, 2])
        for origin in (64, 69):
            image = np.zeros((100, 180, 3), np.uint8)
            grid.draw(image, origin)
            y = origin + 2
            # Beat zero and the fractional measure boundary are solid lines.
            for row in (y, y-24):
                self.assertTrue(np.all(image[row, 43:83] == (50, 205, 255)))
                self.assertFalse(image[row, 83:85].any())
            # Beat one is dashed, and the second lane starts lower on screen.
            np.testing.assert_array_equal(image[y-16, 43], [245, 205, 105])
            self.assertFalse(image[y-16, 53].any())
            self.assertFalse(image[y-48, 85:125].any())

    def test_without_measure_length_only_whole_beat_guides_are_drawn(self):
        grid = BeatGrid({'lanes': [{'roi': [0, 0, 24, 24]}]}, (0, 0), 8, None)
        image = np.zeros((24, 24, 3), np.uint8)
        grid.draw(image, 16)
        np.testing.assert_array_equal(image[16, 2], [245, 205, 105])
        self.assertFalse(image[16, 10].any())

    def test_current_hold_bounds_include_offset_alignment_and_viewport_clipping(self):
        profile = {'calibration': {k: {'status': 'unverified'} for k in ('head', 'tail')},
                   'draft_offsets_px': {'tail': 2}, 'lanes': [{'roi': [4, 10, 16, 55]}]}
        notes = [{'start_beat': [1, 1], 'end_beat': [3, 1], 'lane': 0, 'type': 'longnote'}]
        outlines = Outlines({'notes': notes}, profile, [{'head': [4, 10]}], (3, 2), 10)
        image = np.zeros((60, 40, 3), np.uint8)
        outlines.draw(image, 40)
        self.assertFalse(image[:12].any())
        self.assertTrue(image[12, 8].any())
        self.assertFalse(image[16, 13].any())
        self.assertTrue(image[35, 13].any())

    def test_edited_run_exports_playable_video_without_overwriting_existing_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            _, run = execute(marked_video(root, translated=(3, 5)), output_root=root/'out')
            path = run/'draft.json'
            draft = read_draft(path)
            edited = change_note(draft, 'delete', identity=draft['notes'][0]['id'])
            save_draft(path, edited, expected_sha=draft['document_sha256'])
            log = StringIO()
            with redirect_stderr(log):
                self.assertEqual(main(['overlay', '--run', str(run)]), 0)
            messages = log.getvalue()
            for stage in ('reading chart draft', 'checking the source video', 'calculating note positions', '0.0%', '100.0%',
                          '126/126 frames', 'finalizing audio and video'):
                self.assertIn(stage, messages)
            self.assertNotIn('\r', messages)
            quiet_log = StringIO()
            with redirect_stderr(quiet_log):
                self.assertEqual(main(['overlay', '--run', str(run), '--no-progress',
                                       '--output', str(root/'quiet.mp4')]), 0)
            self.assertEqual(quiet_log.getvalue(), '')
            output = run/'overlay.mp4'
            with av.open(str(root/'video.mkv')) as original, av.open(str(output)) as rendered:
                self.assertEqual([f.pts*f.time_base for f in original.decode(video=0)],
                                 [f.pts*f.time_base for f in rendered.decode(video=0)])
            self.assertEqual(main(['overlay', '--run', str(run)]), 1)
            self.assertEqual(read_draft(path)['document_sha256'], edited['document_sha256'])
