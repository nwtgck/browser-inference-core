"""Revision-selection unit checks; these do not substitute for vendor integration."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_moe_direct_slot as moe


class ReviewedMoeSources(unittest.TestCase):
    def test_each_reviewed_revision_uses_its_own_exact_input_map(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary).resolve()
            for revision, inputs in moe.REVIEWED_REVISIONS.items():
                with self.subTest(revision=revision), \
                     patch.object(moe.subprocess, 'check_output', side_effect=[str(source), revision]), \
                     patch.object(moe, 'digest', side_effect=lambda path: inputs[path.relative_to(source).as_posix()]) as digest:
                    self.assertEqual(moe.verify_reviewed_source(source), revision)
                    self.assertEqual([call.args[0].relative_to(source).as_posix() for call in digest.call_args_list], list(inputs))

    def test_parent_repository_cannot_supply_upstream_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / 'vendor/llama.cpp'
            source.mkdir(parents=True)
            with patch.object(moe.subprocess, 'check_output', return_value=temporary) as git, \
                 patch.object(moe, 'digest') as digest, \
                 self.assertRaisesRegex(ValueError, 'real upstream Git checkout'):
                moe.verify_reviewed_source(source)
            git.assert_called_once()
            digest.assert_not_called()

    def test_unknown_revision_and_changed_inputs_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary).resolve()
            with patch.object(moe.subprocess, 'check_output', side_effect=[str(source), '0' * 40]), \
                 self.assertRaisesRegex(ValueError, 'semantic review'):
                moe.verify_reviewed_source(source)
            for revision in moe.REVIEWED_REVISIONS:
                with self.subTest(revision=revision), \
                     patch.object(moe.subprocess, 'check_output', side_effect=[str(source), revision]), \
                     patch.object(moe, 'digest', return_value='0' * 64), \
                     self.assertRaisesRegex(ValueError, 'semantic review'):
                    moe.verify_reviewed_source(source)

    def test_changed_patch_failure_is_independent_of_ambient_vendor_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / 'upstream'; source.mkdir()
            bad_patch = root / 'changed.patch'; bad_patch.write_text('changed patch')
            with patch.object(moe, 'verify_reviewed_source') as verify, \
                 self.assertRaisesRegex(ValueError, 'patch identity'):
                moe.prepare(source, root / 'output', bad_patch)
            verify.assert_not_called()
            self.assertFalse((root / 'output').exists())

    def test_common_shader_identity_does_not_hide_backend_differences(self):
        old, new = [moe.REVIEWED_REVISIONS[revision] for revision in (
            '7fe450e19305b828c199d602c23a8337aaa1f03b',
            'd81235049384534c167caea52b85a694f6103d14')]
        self.assertEqual(old[moe.SHADER_PATH], new[moe.SHADER_PATH])
        for relative in ('ggml/src/ggml-webgpu/ggml-webgpu.cpp',
                         'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'):
            self.assertNotEqual(old[relative], new[relative])


if __name__ == '__main__':
    unittest.main()
