from fractions import Fraction
from importlib.resources import files
from pathlib import Path
import tempfile
import unittest

import av
import cv2
import numpy as np

from ez2rc.v2.config import load_profile, load_song
from ez2rc.v2.setup import check_video, initialize


class SetupTest(unittest.TestCase):
    def test_packaged_profile_and_copied_assets_work_without_a_checkout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video = root/'recording.mkv'
            with av.open(str(video), 'w') as container:
                stream = container.add_stream('ffv1', rate=60)
                stream.width, stream.height, stream.pix_fmt = 1920, 1080, 'bgr0'
                for i in range(2):
                    frame = av.VideoFrame.from_ndarray(np.zeros((1080, 1920, 3), np.uint8), format='bgr24')
                    frame.pts, frame.time_base = i, Fraction(1, 60)
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode():
                    container.mux(packet)
            assets = root/'assets'
            assets.mkdir()
            for color in ('white', 'cyan'):
                for suffix in ('', '_lnhead', '_lntail'):
                    image = np.random.default_rng(len(suffix)+len(color)).integers(0, 255, (22, 82, 3), dtype=np.uint8)
                    cv2.imwrite(str(assets/f'note_{color}{suffix}.png'), image)
            song = initialize(root/'project', video=video, bpm='150', templates=assets)
            settings = load_song(song)
            profile = load_profile(settings.profile)
            self.assertEqual(settings.difficulty, 'SHD')
            self.assertEqual(profile.resolution, (1920, 1080))
            self.assertEqual(len(profile.asset_hashes), 15)
            for p in assets.glob('*.png'):
                p.unlink()
            self.assertEqual(load_profile(settings.profile).asset_hashes, profile.asset_hashes)
            with self.assertRaisesRegex(ValueError, 'FPS'):
                check_video(video, (1920, 1080), 50)
            with self.assertRaisesRegex(ValueError, 'resolution'):
                check_video(video, (1280, 720), 60)
            with self.assertRaisesRegex(ValueError, 'missing template'):
                initialize(root/'incomplete', video=video, bpm='150', templates=assets)
            self.assertFalse((root/'incomplete').exists())
            self.assertIn('expected_fps = 60', files('ez2rc.v2').joinpath(
                'resources/ez2on_5k_1080p_speed8.toml').read_text())


if __name__ == '__main__':
    unittest.main()
