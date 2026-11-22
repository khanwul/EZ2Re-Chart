from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from ez2rc.v2.capture import CaptureError, TimestampClock, VideoReader
from ez2rc.v2.config import ConfigError, load_algorithm, load_profile, load_song
from ez2rc.v2.observation import cache_keys, collect, observe_frame, open_observations
from ez2rc.v2.runner import execute
from ez2rc.v2.storage import Cache, CacheError, CacheWriter, digest, file_hash, stage_key, write_json
from v2_fixtures import fixture


class V2CaptureCacheTest(unittest.TestCase):
    def test_stage_validators_decompress_each_chunk_once(self):
        from ez2rc.v2.fusion import open_fusion
        from ez2rc.v2.motion import open_motion

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song, _ = fixture(root)
            code, run = execute(song, output_root=root / 'out')
            self.assertEqual(code, 2)
            for stage, open_stage in (('observations', open_observations), ('motion', open_motion),
                                      ('fusion', open_fusion)):
                with self.subTest(stage=stage), patch.object(
                        Cache, '_read_chunk', autospec=True, side_effect=Cache._read_chunk) as read:
                    cache = open_stage(run / 'artifacts' / stage)
                    self.assertEqual([call.args[1] for call in read.call_args_list], cache.manifest['chunks'])

    def test_profile_identity_is_from_parsed_bytes_even_if_file_changes_during_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root)
            song = load_song(path)
            original = song.profile.read_bytes()
            original_hash = file_hash(song.profile)
            def mutate_profile(*args, **kwargs):
                song.profile.write_bytes(original.replace(b'judge_y = 72.0', b'judge_y = 73.0'))
                return observe_frame(*args, **kwargs)
            with patch('ez2rc.v2.observation.observe_frame', side_effect=mutate_profile):
                code, run = execute(path, output_root=root/'out')
            self.assertEqual(code, 2, (run/'run.json').read_text())
            cache = run/'artifacts/observations'
            manifest = open_observations(cache).manifest
            metadata = manifest['metadata']
            self.assertEqual(metadata['profile']['judge_y'], 72.0)
            self.assertEqual(metadata['profile_sha256'], original_hash)
            self.assertEqual(manifest['key']['parameters']['profile_sha256'], original_hash)
            self.assertEqual(json.loads((run/'run.json').read_text())['input']['profile_sha256'], original_hash)
            self.assertNotEqual(metadata['profile_sha256'], file_hash(song.profile))
            code, rejected = execute(path, output_root=root/'out', observation_cache=cache)
            self.assertEqual(code, 1)
            self.assertIn('recording profile changed', json.loads((rejected/'run.json').read_text())['failure']['message'])
            song.profile.write_bytes(original)
            code, replay = execute(path, output_root=root/'out', observation_cache=cache)
            self.assertEqual(code, 2, (replay/'run.json').read_text())

    def test_profile_hash_does_not_reread_file_after_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root)
            source = load_song(path).profile
            original_hash = file_hash(source)
            imread = cv2.imread
            def mutate_profile(*args, **kwargs):
                source.write_text(source.read_text().replace('judge_y = 72.0', 'judge_y = 73.0'))
                return imread(*args, **kwargs)
            with patch('ez2rc.v2.config.cv2.imread', side_effect=mutate_profile):
                profile = load_profile(source)
            self.assertEqual(profile.judge_y, 72.0)
            self.assertEqual(profile.source_sha256, original_hash)
            self.assertNotEqual(profile.source_sha256, file_hash(source))

    def test_vfr_nonzero_pts_and_partial_range_share_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, pts = fixture(root)
            reader = VideoReader(root / "video.mkv")
            frames = list(reader)
            self.assertEqual([f.stamp.pts for f in frames], pts)
            self.assertEqual([f.stamp.timestamp_ms for f in frames], [0, 17, 33, 83, 100, 133])
            self.assertEqual(reader.metadata["first_pts"], 1000)
            self.assertEqual(reader.metadata["first_time_base"], [1, 1000])
            subset = list(VideoReader(root / "video.mkv", start_ms=30, end_ms=101))
            self.assertEqual([f.stamp.frame_id for f in subset], [2, 3, 4])
            self.assertEqual([f.stamp.timestamp_ms for f in subset], [33, 83, 100])
            self.assertTrue(np.array_equal(frames[2].bgr[:70], frames[3].bgr[:70]))

    def test_missing_and_decreasing_pts_do_not_silently_become_cfr(self):
        clock = TimestampClock(Fraction(60))
        with self.assertRaisesRegex(CaptureError, "missing"):
            clock.stamp(0, None, Fraction(1, 1000))
        clock = TimestampClock(Fraction(60), "fps_fallback")
        self.assertEqual(clock.stamp(0, None, Fraction(1, 1000)).source, "fps_fallback")
        self.assertEqual(clock.stamp(1, 50, Fraction(1, 1000)).source, "fps_fallback")
        clock = TimestampClock(Fraction(60))
        clock.stamp(0, 100, Fraction(1, 1000))
        clock.stamp(1, 100, Fraction(1, 1000))  # preserve equal presentation time
        with self.assertRaisesRegex(CaptureError, "nonmonotonic"):
            clock.stamp(2, 99, Fraction(1, 1000))

    def test_cache_integrity_interruption_and_recovery(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            key = stage_key("capture", "test", {"x": 1})
            with self.assertRaises(RuntimeError):
                with CacheWriter(root, key) as writer:
                    writer.append(frame_id=np.arange(3))
                    raise RuntimeError("interrupt")
            self.assertFalse((root / "capture" / digest(key)).exists())
            with CacheWriter(root, key) as writer:
                writer.append(frame_id=np.arange(3))
                cache = writer.commit({"n": 3})
            self.assertEqual(next(cache.chunks())["frame_id"].tolist(), [0, 1, 2])
            (cache.path / "chunk-000000.npz").write_bytes(b"broken")
            with self.assertRaises(CacheError):
                Cache(cache.path)
            with CacheWriter(root, key) as writer:
                writer.append(frame_id=np.arange(3))
                repaired = writer.commit({"n": 3})
            self.assertEqual(next(repaired.chunks())["frame_id"].tolist(), [0, 1, 2])
            self.assertEqual(len(list(repaired.path.parent.glob(".*.corrupt-*"))), 1)

    def test_observations_keep_scores_weak_candidates_repeats_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root, weak=True)
            song = load_song(path)
            profile, algo = load_profile(song.profile), load_algorithm(song.algorithm)
            capture, cache, reused = collect(song, profile, algo, root / "cache")
            self.assertFalse(reused)
            cache = open_observations(cache.path)
            chunks = list(cache.chunks())
            self.assertEqual(len(chunks), 3)
            self.assertTrue(all(len(c["frame_id"]) <= 2 for c in chunks))
            self.assertGreater(sum(len(c["scores"]) for c in chunks), 0)
            self.assertEqual(chunks[0]["scores"].shape[1], 4)
            self.assertEqual(chunks[1]["repeat_candidate_group"].tolist(), [2, 2])
            self.assertEqual(cache.manifest["metadata"]["position_error_px"], None)
            with patch("ez2rc.v2.observation.VideoReader", side_effect=AssertionError("should reuse")):
                _, again, reused = collect(song, profile, algo, root / "cache")
            self.assertTrue(reused)
            self.assertEqual(again.manifest, cache.manifest)
            _, fresh, _ = collect(song, profile, algo, root / "fresh")
            self.assertEqual(fresh.manifest, cache.manifest)

    def test_relevant_changes_invalidate_only_the_affected_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root)
            song = load_song(path)
            profile, algo = load_profile(song.profile), load_algorithm(song.algorithm)
            before = cache_keys(song, profile, algo, "video-hash")
            self.assertEqual(before, cache_keys(replace(song, bpm="200", title="Renamed"), profile, algo, "video-hash"))
            changed = cache_keys(song, profile, replace(algo, edge_threshold=20), "video-hash")
            self.assertEqual(before[0], changed[0])
            self.assertNotEqual(before[1], changed[1])
            changed = cache_keys(replace(song, start_ms=10), profile, algo, "video-hash")
            self.assertNotEqual(before[0], changed[0])
            self.assertNotEqual(before[1], changed[1])
            changed = cache_keys(song, profile, algo, "changed-content")
            self.assertNotEqual(before[0], changed[0])
            self.assertNotEqual(before[1], changed[1])

    def test_changed_template_invalidates_observations_but_not_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root)
            song = load_song(path)
            profile, algo = load_profile(song.profile), load_algorithm(song.algorithm)
            before = cache_keys(song, profile, algo, "video")
            import cv2
            image = cv2.imread(str(root / "note.png"))
            image[0, 0, 0] ^= 1
            cv2.imwrite(str(root / "note.png"), image)
            after = cache_keys(song, load_profile(song.profile), algo, "video")
            self.assertEqual(before[0], after[0])
            self.assertNotEqual(before[1], after[1])

    def test_manifest_damage_and_false_timestamp_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root)
            song = load_song(path)
            _, observations, _ = collect(song, load_profile(song.profile), load_algorithm(song.algorithm), root / "cache")
            manifest = observations.path / "manifest.json"
            data = json.loads(manifest.read_text())
            data["metadata"]["frame_count"] = 999
            write_json(manifest, data)
            with self.assertRaisesRegex(CacheError, "manifest checksum"):
                Cache(observations.path)
            # Even a consistently rehashed archive must satisfy PTS semantics.
            data["metadata"]["frame_count"] = 6
            chunk = observations.path / "chunk-000000.npz"
            with np.load(chunk, allow_pickle=False) as archive:
                arrays = {k: archive[k] for k in archive.files}
            arrays["timestamp_ms"][1] += 1
            np.savez_compressed(chunk, **arrays)
            data["chunks"][0]["sha256"] = file_hash(chunk)
            data["manifest_sha256"] = digest({k: v for k, v in data.items() if k != "manifest_sha256"})
            write_json(manifest, data)
            with self.assertRaisesRegex(CacheError, "original PTS"):
                open_observations(observations.path)

    def test_translated_profile_keeps_aligned_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root, translated=(4, 3))
            song = load_song(path)
            _, obs, _ = collect(song, load_profile(song.profile), load_algorithm(song.algorithm), root / "cache")
            self.assertEqual(obs.manifest["metadata"]["alignment_offset_px"], [4, 3])
            first = next(obs.chunks())
            first_scores = first["scores"][first["candidate_index"][:, 0] == 0, 0]
            positions = first["positions"][first["candidate_index"][:, 0] == 0, 0]
            self.assertTrue(any(np.allclose(xy, [22, 10]) and score > .99
                                for xy, score in zip(positions, first_scores)))

    def test_extract_reconstruct_and_failure_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song_path, _ = fixture(root)
            output = root / "out"
            code, run = execute(song_path, output_root=output)
            self.assertEqual(code, 2)
            first_chart = (run / "chart.json").read_bytes()
            status = json.loads((run / "run.json").read_text())
            self.assertEqual(status["status"], "partial")
            self.assertFalse(status["stages"]["observation"]["reused"])
            self.assertEqual(status["input"]["capture"]["first_pts"], 1000)
            (root / "video.mkv").unlink()
            (root / "note.png").unlink()
            with patch("ez2rc.v2.capture.av.open", side_effect=AssertionError("decoder must stay unused")):
                code, again = execute(song_path, output_root=output,
                                      observation_cache=run / "artifacts/observations")
            self.assertEqual(code, 2)
            self.assertNotEqual(again, run)
            self.assertEqual((again / "chart.json").read_bytes(), first_chart)
            self.assertEqual((again / "artifacts/events.json").read_bytes(), (run / "artifacts/events.json").read_bytes())
            code, failed = execute(song_path, output_root=output)
            self.assertEqual(code, 1)
            self.assertEqual(json.loads((failed / "run.json").read_text())["status"], "failed")
            self.assertEqual((run / "chart.json").read_bytes(), first_chart)

    def test_reconstruct_rejects_changed_media_and_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            song_path, _ = fixture(root)
            code, run = execute(song_path, output_root=root / "out")
            self.assertEqual(code, 2)
            original = (root / "note.png").read_bytes()
            (root / "note.png").write_bytes(original + b"changed")
            code, failed = execute(song_path, output_root=root / "out",
                                   observation_cache=run / "artifacts/observations")
            self.assertEqual(code, 1)
            self.assertIn("template content", json.loads((failed / "run.json").read_text())["failure"]["message"])
            (root / "note.png").write_bytes(original)
            with (root / "video.mkv").open("ab") as file:
                file.write(b"changed")
            code, failed = execute(song_path, output_root=root / "out",
                                   observation_cache=run / "artifacts/observations")
            self.assertEqual(code, 1)
            self.assertIn("video content", json.loads((failed / "run.json").read_text())["failure"]["message"])

    def test_invalid_profile_lane_count_and_parameter_types_fail_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            path = root / "profile.toml"
            original = path.read_text()
            for bad in (original.replace('mode = "2k"', 'mode = "5k"'),
                        original.replace('channel = "gray"', 'channel = []')):
                path.write_text(bad)
                with self.assertRaises(ConfigError):
                    load_profile(path)
            (root / "algo.toml").write_text('schema_version = 1\ntimestamp_policy = []\n')
            with self.assertRaises(ConfigError):
                load_algorithm(root / "algo.toml")

    def test_resolution_and_alignment_fail_before_observation_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _ = fixture(root)
            profile_path = root / "profile.toml"
            profile_path.write_text(profile_path.read_text().replace("[96, 96]", "[100, 100]"))
            code, run = execute(path, output_root=root / "out")
            self.assertEqual(code, 1)
            self.assertFalse((run / "artifacts/observations").exists())
            self.assertIn("resolution", json.loads((run / "run.json").read_text())["failure"]["message"])


if __name__ == "__main__":
    unittest.main()
