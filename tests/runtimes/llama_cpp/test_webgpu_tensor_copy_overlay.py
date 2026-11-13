"""Real upstream target configuration and actual hook tests, not GPU execution."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path.insert(0, str(ROOT.parents[1]))
from tests.support.llama_cpp.webgpu_stages import tensor_copy as copy
class TensorCopyOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.source / copy.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        cls.patch = ROOT / copy.PATCH_DIRECTORY / copy.PATCH_NAME

    def test_checked_idempotent_copy_preserves_upstream(self):
        before = (self.source / copy.SOURCE_PATH).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'overlay'
            result = copy.prepare(self.source, out, self.patch)
            stamp = result.stat().st_mtime_ns
            self.assertEqual(copy.digest(result), copy.PATCHED_SOURCE_SHA256)
            self.assertIn('src_ctx->buffer.Get() == dst_ctx->buffer.Get()', result.read_text())
            self.assertEqual(copy.prepare(self.source, out, self.patch).stat().st_mtime_ns, stamp)
        self.assertEqual(before, (self.source / copy.SOURCE_PATH).read_bytes())

    def test_unrelated_commit_is_allowed_but_changed_dependency_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)
            for relative in next(iter(copy.REVIEWED_REVISIONS.values())):
                destination = source / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.source / relative, destination)
            def git(*args):
                return subprocess.check_output(['git', '-c', 'user.name=Fixture',
                    '-c', 'user.email=fixture@example.invalid', *args], cwd=source, text=True).strip()
            git('init', '-q'); git('add', '.'); git('commit', '-qm', 'Reviewed inputs')
            first = copy.verify_reviewed_source(source)
            (source / 'unrelated.txt').write_text('Unrelated upstream change\n')
            git('add', '.'); git('commit', '-qm', 'Unrelated change')
            self.assertNotEqual(copy.verify_reviewed_source(source), first)
            (source / 'ggml/src/ggml-backend.cpp').write_text('Changed copy contract\n')
            with self.assertRaisesRegex(ValueError, 'semantic review'):
                copy.verify_reviewed_source(source)

    def test_changed_source_patch_and_overlap_fail_closed(self):
        with patch.object(copy, 'git', side_effect=[str(self.source), '0' * 40]):
            self.assertEqual(copy.verify_reviewed_source(self.source), '0' * 40)
        with patch.object(copy, 'git', side_effect=[str(self.source), '0' * 40]), \
                patch.object(copy, 'digest', return_value='0' * 64), \
                self.assertRaisesRegex(ValueError, 'semantic review'):
            copy.verify_reviewed_source(self.source)
        for out in (self.source, self.source / 'overlay', self.source.parent):
            with self.assertRaisesRegex(ValueError, 'outside'):
                copy.prepare(self.source, out, self.patch)
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / 'bad.patch'; bad.write_bytes(self.patch.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'patch identity'):
                copy.prepare(self.source, Path(tmp) / 'out', bad)
            self.assertFalse((Path(tmp) / 'out').exists())

    def test_dawn_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                copy.verify_dawn(Path(tmp))
        with patch.object(copy, 'digest', return_value='0' * 64), self.assertRaisesRegex(ValueError, 'Dawn'):
            copy.verify_dawn(Path('/unused'))

    def test_actual_prepared_hook_mock_and_optional_real_headers(self):
        command = [sys.executable, str(runtime_test_root('llama-cpp') / 'webgpu-tensor-copy/check_hook.py'), str(self.source)]
        if os.environ.get('LCB_TEST_DAWN_PACKAGE'):
            command += ['--dawn-package', os.environ['LCB_TEST_DAWN_PACKAGE']]
        subprocess.run(command, check=True)

if __name__ == '__main__':
    unittest.main()
