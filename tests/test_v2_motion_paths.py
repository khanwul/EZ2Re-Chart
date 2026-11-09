"""Procedural periodic landmarks and interrupted displacement paths."""
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ez2rc.v2.config import MotionConfig
from ez2rc.v2.motion import (STATUS, _consistent_paths, _step_interval, estimate,
                             open_motion, timeline)
from ez2rc.v2.storage import CacheError, CacheWriter, digest, file_hash, stage_key, write_json
from v2_motion_fixtures import observations


def periodic(root, scroll, *, chunk=3, times=None, blank=(), one_lane=()):
    """Regular procedural dots admit aliases over long, but not short, gaps."""
    times = np.arange(len(scroll)) * 10. if times is None else np.asarray(times, float)
    profile = {'effect_top': 140., 'lanes': [{'roi': [lane * 20, 0, lane * 20 + 20, 160]} for lane in range(3)]}
    frames, group = [], 0
    for fid, displacement in enumerate(scroll):
        if fid == 0 or displacement != scroll[fid - 1]:
            group = fid
        index, positions = [], []
        projection = np.zeros((3, 160), np.float16)
        if fid not in blank:
            for lane in range(1 if fid in one_lane else 3):
                for q in range(-160, int(max(scroll)) + 160, 40):
                    y = displacement + 5 + lane - q
                    if 1 < y < 140:
                        index.append([fid, lane, int(y)])
                        positions.append([[lane * 20., y]] * 3)
                        projection[lane, int(y)] = 200
        frames.append({'frame_id': np.array([fid]), 'timestamp_ms': np.array([times[fid]]),
                       'repeat_candidate_group': np.array([group]), 'projection': projection[None],
                       'edge': np.abs(np.diff(projection, axis=1, prepend=0)).astype(np.float16)[None],
                       'candidate_index': np.asarray(index, np.int64).reshape(-1, 3),
                       'positions': np.asarray(positions, np.float32).reshape(-1, 3, 2),
                       'scores': np.tile(np.array([.99, .1, .1, .1], np.float32), (len(index), 1)),
                       'flags': np.zeros((len(index), 3), np.uint8)})
    key = stage_key('observation', 'periodic-procedural', {'scroll': list(scroll), 'times': times.tolist(),
                    'chunk': chunk, 'blank': list(blank), 'one_lane': list(one_lane)})
    with CacheWriter(root, key) as writer:
        for at in range(0, len(frames), chunk):
            batch = frames[at:at + chunk]
            writer.append(**{k: np.concatenate([f[k] for f in batch]) for k in batch[0]})
        return writer.commit({'profile': profile})


class MotionPathTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = MotionConfig(max_speed_px_per_ms=2.)

    def run_periodic(self, scroll, **kwargs):
        cache = periodic(self.root, scroll, **kwargs)
        result, _ = estimate(cache, self.config, self.root)
        checks = np.concatenate([c['path_checks'] for c in result.chunks()])
        return result, timeline(result), checks

    def test_long_anchor_alias_is_rejected_by_complete_adjacent_path(self):
        truth = list(range(0, 43, 6))
        result, clock, checks = self.run_periodic(truth)
        np.testing.assert_allclose(clock['scroll_px'], truth)
        self.assertTrue((clock['status'][1:] == STATUS['observed']).all())
        self.assertGreater(result.manifest['metadata']['path_rejected_models'], 0)
        self.assertTrue(np.any(checks[:, 7] == 0))
        alternatives = np.concatenate([c['alternatives'] for c in result.chunks()])
        for frame, _, scroll, _, _ in alternatives:
            self.assertAlmostEqual(scroll, truth[int(frame)])

    def test_acceleration_uses_observed_displacements(self):
        truth = [0, 3, 11, 23, 29, 44, 52]
        _, clock, _ = self.run_periodic(truth)
        np.testing.assert_allclose(clock['scroll_px'], truth)
        self.assertTrue((clock['status'][1:] == STATUS['observed']).all())

    def test_variable_pts_gap_keeps_multiple_adjacent_modes(self):
        _, clock, checks = self.run_periodic([0, 6, 12, 48, 54, 60], times=[0, 10, 20, 80, 90, 100])
        np.testing.assert_allclose(clock['scroll_px'], [0, 6, 12, 48, 54, 60])
        self.assertEqual(clock['status'][3], STATUS['ambiguous'])
        self.assertFalse(np.any((checks[:, 1] < 3) & (checks[:, 0] >= 3)))

    def test_repeat_blank_and_single_lane_break_path_constraints(self):
        for changes, truth, broken in (
            ({}, [0, 6, 12, 12, 18, 24], 3),
            ({'blank': (3,)}, [0, 6, 12, 18, 24, 30], 3),
            ({'one_lane': (3,)}, [0, 6, 12, 18, 24, 30], 3),
        ):
            with self.subTest(changes=changes):
                _, clock, checks = self.run_periodic(truth, **changes)
                self.assertFalse(np.any((checks[:, 1] < broken) & (checks[:, 0] >= broken)))
                if not changes:
                    self.assertEqual(clock['status'][broken], STATUS['repeated'])

    def test_locally_ambiguous_periodic_motion_keeps_aliases(self):
        obs = periodic(self.root, [0, 6, 12, 18])
        result, _ = estimate(obs, MotionConfig(), self.root)
        self.assertTrue((timeline(result)['status'][1:] == STATUS['ambiguous']).all())
        self.assertEqual(result.manifest['metadata']['path_checked_models'], 0)

    def test_path_evidence_is_invariant_to_input_and_output_chunks(self):
        results = []
        for chunk in (1, 17):
            result, _, _ = self.run_periodic(list(range(0, 780, 6)), chunk=chunk)
            pieces = list(result.chunks())
            results.append({k: np.concatenate([c[k] for c in pieces]) for k in pieces[0]})
        for key in results[0]:
            np.testing.assert_array_equal(results[0][key], results[1][key], err_msg=key)

    def test_invalid_path_decision_is_rejected_even_with_updated_checksums(self):
        result, _, checks = self.run_periodic([0, 6, 12, 18, 24])
        self.assertGreater(len(checks), 0)
        chunk = result.path / 'chunk-000000.npz'
        data = next(result.chunks())
        data['path_checks'][0, 7] = 1 - data['path_checks'][0, 7]
        np.savez_compressed(chunk, **data)
        manifest = result.manifest
        manifest['chunks'][0]['sha256'] = file_hash(chunk)
        manifest['manifest_sha256'] = digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'})
        write_json(result.path / 'manifest.json', manifest)
        with self.assertRaisesRegex(CacheError, 'path checks'):
            open_motion(result.path)

    def test_nonperiodic_acceleration_and_large_unobserved_gap(self):
        for truth, times in (([0, 8, 19, 25, 42, 60, 69], None),
                             ([0, 8, 16, 256, 264], [0, 16, 32, 512, 528])):
            with self.subTest(truth=truth):
                obs = observations(self.root, truth, times=times)
                result, _ = estimate(obs, MotionConfig(), self.root)
                clock = timeline(result)
                np.testing.assert_allclose(clock['scroll_px'], truth, atol=2.)
                if times:
                    self.assertEqual(clock['status'][3], STATUS['extrapolated'])

    def test_unique_mode_requires_multilane_support_and_clear_quality_margin(self):
        model = {'delta': 9., 'quality': 1., 'lanes': 2}
        self.assertEqual(_step_interval([model], self.config), (7., 11.))
        self.assertIsNone(_step_interval([{**model, 'lanes': 1}], self.config))
        self.assertIsNone(_step_interval([model, {**model, 'delta': 19., 'quality': .9}], self.config))
        self.assertIsNone(_step_interval([], self.config))

    def test_direct_and_path_intervals_may_touch(self):
        ref = {'id': 0, 'path': (0, 0., 0.)}
        models = [{'ref': ref, 'delta': d, 'scroll': d, 'quality': 1., 'lanes': 2} for d in (14., 15.)]
        checks = []
        kept = _consistent_paths(models, (0, 8., 12.), 2, self.config, checks)
        self.assertEqual([m['delta'] for m in kept], [14.])
        self.assertEqual([r[7] for r in checks], [1, 0])
        self.assertEqual(len(models), 2)


if __name__ == '__main__':
    unittest.main()
