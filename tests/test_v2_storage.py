from pathlib import Path
import tempfile
import unittest

import numpy as np

from ez2rc.v2.storage import Cache, CacheError, CacheWriter, stage_key


def positive_cache(path):
    def check(manifest, index, arrays):
        if np.any(arrays['values'] < 0):
            raise ValueError('negative value')
    return Cache(path, validate_chunk=check)


class CachePublicationTest(unittest.TestCase):
    def test_semantic_failure_does_not_publish_or_replace_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); key = stage_key('decode', 'test', {})
            with CacheWriter(root, key) as writer:
                writer.append(values=np.array([1]))
                valid = writer.commit({}, validate=positive_cache)
            for parameters in ({}, {'other': True}):
                with CacheWriter(root, stage_key('decode', 'test', parameters)) as writer:
                    writer.append(values=np.array([1]))
                    writer.append(values=np.array([-1]))
                    with self.assertRaisesRegex(CacheError, 'negative value'):
                        writer.commit({}, validate=positive_cache)
                    pending, final = writer.temp, writer.final
                self.assertFalse(pending.exists())
                self.assertEqual(final.exists(), parameters == {})
            np.testing.assert_array_equal(next(valid.chunks())['values'], [1])

    def test_existing_publisher_is_semantically_checked_before_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); key = stage_key('decode', 'test', {})
            # Valid checksums alone do not make this an acceptable winner.
            with CacheWriter(root, key) as writer:
                writer.append(values=np.array([-1]))
                writer.commit({})
            with CacheWriter(root, key) as writer:
                writer.append(values=np.array([2]))
                valid = writer.commit({}, validate=positive_cache)
            np.testing.assert_array_equal(next(valid.chunks())['values'], [2])
            quarantined = list(valid.path.parent.glob('.*.corrupt-*'))
            self.assertEqual(len(quarantined), 1)
            with self.assertRaisesRegex(CacheError, 'negative value'):
                positive_cache(quarantined[0])
            with CacheWriter(root, key) as writer:
                writer.append(values=np.array([3]))
                winner = writer.commit({}, validate=positive_cache)
                pending = writer.temp
            self.assertFalse(pending.exists())
            self.assertEqual(winner.path, valid.path)
            np.testing.assert_array_equal(next(winner.chunks())['values'], [2])
