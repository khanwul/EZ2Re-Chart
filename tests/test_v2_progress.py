from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ez2rc.v2.progress import ProgressDisplay, WorkProgress, frame_progress
from ez2rc.v2.config import load_song, load_profile, AlgorithmConfig
from ez2rc.v2.observation import collect
from ez2rc.v2.motion import estimate
from ez2rc.v2.fusion import accumulate
from ez2rc.v2.decode import infer, output_events
from v2_fixtures import fixture


class ProgressTest(unittest.TestCase):
    def test_video_progress_uses_whole_recording_duration_without_a_full_frame_count(self):
        self.assertIn('25.0%', frame_progress(5, 250, 0, 1000))
        self.assertIn('50.0%', frame_progress(5, 250, 0, 1000, total_frames=10))
        self.assertNotIn('%', frame_progress(5, 250, 0, None))
        self.assertIn('100.0%', frame_progress(10, 999, 0, None, complete=True, total_frames=10))

    def test_counts_are_throttled_but_completion_is_reported(self):
        messages = []
        with patch('ez2rc.v2.progress.time.monotonic', return_value=0):
            work = WorkProgress(messages.append, 'motion', 3, 'frames')
            work.update(1)
            work.update(3)
        self.assertEqual(len(messages), 2)
        self.assertIn('0/3 frames', messages[0])
        self.assertIn('100.0%', messages[1])

    def test_heartbeat_and_shutdown_leave_clean_output(self):
        for terminal in (False, True):
            stream = StringIO()
            stream.isatty = lambda: terminal
            display = ProgressDisplay(stream)
            with patch('ez2rc.v2.progress.time.monotonic', return_value=0):
                display('decode / lane 1: associating endpoints')
            with patch.object(display.stopped, 'wait', side_effect=[False, True]), \
                    patch('ez2rc.v2.progress.time.monotonic', return_value=20):
                display._heartbeat()
            display.start()
            display.finish()
            self.assertFalse(display.worker.is_alive())
            self.assertIn('working', stream.getvalue())
            self.assertEqual('\r' in stream.getvalue(), terminal)
            self.assertTrue(stream.getvalue().endswith('\n'))

    def test_pipeline_progress_preserves_results_and_cache_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song = load_song(fixture(root)[0])
            algo = AlgorithmConfig()
            _, obs, _ = collect(song, load_profile(song.profile), algo, root/'obs')
            messages = []
            results = []
            for name, report in [('silent', None), ('visible', messages.append)]:
                dest = root/name
                motion, _ = estimate(obs, algo.motion, dest, progress=report)
                fusion, _ = accumulate(obs, motion, algo.fusion, dest, progress=report)
                decoded, _ = infer(obs, motion, fusion, algo.decode, dest, progress=report)
                results.append((motion.id, fusion.id, decoded.id, output_events(decoded)))
            self.assertEqual(results[0], results[1])
            for stage in ('motion / frames', 'fusion / frames', 'decode / lanes'):
                self.assertTrue(any(stage in m and '100.0%' in m for m in messages))
            self.assertTrue(any('associating endpoints' in m for m in messages))
