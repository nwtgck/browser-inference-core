"""The frozen reference input cannot silently follow the live checkout."""
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tests.support import historical_webgpu as fixture
from tests.support import frozen_webgpu_source as reference


class HistoricalFixture(unittest.TestCase):
    def test_live_source_environment_is_not_an_input_and_each_copy_is_private(self):
        with patch.dict(os.environ, {'LCB_TEST_LLAMA_SOURCE': '/missing/not-a-historical-input'}):
            with fixture.historical_source() as first:
                original = (first / reference.SOURCE_PATH).read_bytes()
                (first / reference.SOURCE_PATH).write_bytes(b'mutated only in private test directory')
                with fixture.historical_source() as second:
                    self.assertNotEqual(first, second)
                    self.assertEqual((second / reference.SOURCE_PATH).read_bytes(), original)
                    self.assertTrue((second / 'patches' / reference.PARAM_PATCH_NAME).is_file())
            self.assertFalse(first.exists())

    def test_corrupt_archive_is_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shutil.copyfile(fixture.FIXTURE / 'manifest.json', root / 'manifest.json')
            (root / 'source.zip').write_bytes(b'not the recorded archive')
            with patch.object(fixture, 'FIXTURE', root), self.assertRaisesRegex(ValueError, 'integrity'):
                with fixture.historical_source():
                    self.fail('Corrupted input was yielded')

    def test_frozen_normal_and_diagnostic_compositions_keep_their_reference_identities(self):
        with fixture.historical_source() as upstream, tempfile.TemporaryDirectory() as temporary:
            for profiled in (False, True):
                with self.subTest(profiled=profiled):
                    out = Path(temporary) / str(profiled)
                    reference.prepare(upstream, out, patch_root=upstream / 'patches', gpu_profile=profiled)
                    for name, expected in reference.FINAL_OUTPUTS[profiled].items():
                        self.assertEqual(reference.tensor_copy.digest(out / name), expected)
