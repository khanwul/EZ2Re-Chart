from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ez2rc.v2.config import AlgorithmConfig, FusionConfig, MotionConfig, load_algorithm, load_profile, load_song
from ez2rc.v2.fusion import visibility
from ez2rc.v2.motion import landmarks
from ez2rc.v2.observation import cache_keys, collect, observe_frame, open_observations
from ez2rc.v2.storage import CacheError
from v2_decode_fixtures import scene, selected
from v2_fixtures import fixture, top_edge_scene


class TopContinuationTest(unittest.TestCase):
    def test_bright_top_continuation_retains_internal_maximum_as_clipped(self):
        frame, profile = top_edge_scene()
        observed = observe_frame(frame, profile, AlgorithmConfig(), (0, 0))
        self.assertEqual(observed['candidate_index'][0].tolist(), [0, 0, 0])
        self.assertEqual(observed['positions'][0, :, 1].tolist(), [4.] * 3)
        self.assertTrue(np.all(observed['scores'][0, :3] > .95))
        self.assertEqual(observed['flags'][0].tolist(), [5] * 3)
        # No score/position editing or seed deletion: the raw image maximum is
        # still available to all three typed interpretations.
        for k, template in enumerate(profile.lanes[0].templates.values()):
            result = cv2.matchTemplate(frame.bgr[:20, 12:24], template, cv2.TM_CCOEFF_NORMED)
            _, score, _, xy = cv2.minMaxLoc(result)
            self.assertAlmostEqual(observed['scores'][0, k], score)
            self.assertEqual(observed['positions'][0, k].tolist(), [12 + xy[0], xy[1]])
        category, weight = visibility(observed['flags'][:1], FusionConfig())
        self.assertEqual(category.tolist(), [[2] * 3])
        self.assertEqual(weight.tolist(), [[.25] * 3])
        motion = landmarks(observed, {'lanes': [{}], 'effect_top': 65}, MotionConfig())
        self.assertFalse(any(row[2] == 0 for row in motion))

    def test_real_endpoint_with_edges_near_top_keeps_usable_location(self):
        for top, offset in ((0, (0, 0)), (9, (3, 5))):
            with self.subTest(top=top, offset=offset):
                frame, profile = top_edge_scene(endpoint=True, top=top, offset=offset)
                observed = observe_frame(frame, profile, AlgorithmConfig(), offset)
                self.assertEqual(observed['positions'][0, :, 1].tolist(), [top + 4.] * 3)
                self.assertTrue(np.all(observed['scores'][0, :3] > .999))
                self.assertEqual(observed['flags'][0].tolist(), [4] * 3)
                # A dark gap above a standalone endpoint also establishes its
                # visible boundary, even when its search window reaches y=0.
                dx, dy = offset
                frame.bgr[top + dy:top + dy + 4, 12 + dx:24 + dx] = 0
                observed = observe_frame(frame, profile, AlgorithmConfig(), offset)
                matches = np.all(observed['positions'][:, :, 1] == top + 4, axis=1)
                self.assertTrue(matches.any())
                self.assertTrue(np.all((observed['flags'][matches] & 1) == 0))

    def test_continuation_requires_row_after_whole_kind_specific_template(self):
        frame, profile = top_edge_scene()
        # An edge immediately following the 12-row template is observed. The
        # shorter template still lies entirely in the top-censored body.
        frame.bgr[16:70, 12:24] = 0
        profile.lanes[0].templates['tail'] = profile.lanes[0].templates['tail'][:6]
        observed = observe_frame(frame, profile, AlgorithmConfig(), (0, 0))
        self.assertEqual(observed['positions'][0, :, 1].tolist(), [4.] * 3)
        self.assertEqual(observed['flags'][0].tolist(), [4, 4, 5])

    def test_repeated_and_skipped_displays_do_not_create_top_endpoints(self):
        observations = []
        for fid, extent in ((100, 30), (101, 30), (102, 55)):
            frame, profile = top_edge_scene(body_end=extent, frame_id=fid)
            observed = observe_frame(frame, profile, AlgorithmConfig(), (0, 0))
            observations.append(observed)
            self.assertEqual(observed['flags'][0].tolist(), [5] * 3)
            self.assertEqual(observed['candidate_index'][0].tolist(), [fid, 0, 0])
        self.assertEqual(observations[0]['signature'], observations[1]['signature'])
        self.assertNotEqual(observations[1]['signature'], observations[2]['signature'])
        # This is frame-local evidence, independent of where analysis starts.
        frame, profile = top_edge_scene(body_end=55, frame_id=102)
        alone = observe_frame(frame, profile, AlgorithmConfig(), (0, 0))
        for key in ('candidate_index', 'positions', 'scores', 'flags'):
            np.testing.assert_array_equal(alone[key], observations[-1][key])

    def test_top_flags_prevent_two_complete_intervals_and_keep_short_notes(self):
        body, profile = top_edge_scene()
        end, _ = top_edge_scene(endpoint=True)
        clipped = observe_frame(body, profile, AlgorithmConfig(), (0, 0))['flags'][0].tolist()
        usable = observe_frame(end, profile, AlgorithmConfig(), (0, 0))['flags'][0].tolist()
        points = [(10, 1), (30, 1), (50, 2), (70, 1), (90, 2)]
        before = scene(points, [(10, 110)], errors=(2, 4), bounds=(0, 110))
        self.assertEqual(selected(before), [('longnote', 10., 50.), ('longnote', 70., 90.)])
        after = scene(points, [(10, 110)], flags=[usable, *[clipped] * 4],
                      errors=(2, 4), bounds=(0, 110))
        self.assertEqual(selected(after), [('longnote', 10., None)])
        self.assertTrue(all(e['status'] == 'incomplete' for e in after['events']))
        self.assertEqual({r for e in before['endpoints'] for r in e['observation_ids']},
                         {r for e in after['endpoints'] for r in e['observation_ids']})
        # Independent resources at a touching tail/onset remain legal, and
        # genuine endpoints still permit a one-pixel longnote.
        short = scene([(20, 1), (21, 2)], [(20, 21)], flags=[usable] * 2)
        self.assertEqual(selected(short), [('longnote', 20., 21.)])
        touching = scene([(20, 1), (80, 2), (80, 0)], [(20, 80)], flags=[usable] * 3)
        self.assertEqual(set(selected(touching)), {('longnote', 20., 80.), ('tap', 80., None)})

    def test_observation_version_invalidates_old_flags_without_changing_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root)
            song = load_song(path)
            profile, algorithm = load_profile(song.profile), load_algorithm(song.algorithm)
            current_keys = cache_keys(song, profile, algorithm, 'video-hash')
            with patch('ez2rc.v2.observation.VERSION', 'visual-observations-1'):
                old_keys = cache_keys(song, profile, algorithm, 'video-hash')
                capture, old, _ = collect(song, profile, algorithm, root / 'cache')
            self.assertEqual(current_keys[0], old_keys[0])
            self.assertNotEqual(current_keys[1], old_keys[1])
            with self.assertRaisesRegex(CacheError, 'unsupported observation'):
                open_observations(old.path)
            new_capture, new, reused = collect(song, profile, algorithm, root / 'cache')
            self.assertFalse(reused)
            self.assertEqual(capture.manifest, new_capture.manifest)
            self.assertNotEqual(new.id, old.id)
            self.assertTrue(old.path.exists())
            self.assertEqual(current_keys, cache_keys(replace(song, bpm='200'), profile, algorithm, 'video-hash'))
