"""Real upstream target configuration and actual hook tests, not GPU execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_tensor_copy as copy
import build as builder

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
        with patch.object(copy.subprocess, 'check_output', side_effect=[str(self.source), '0' * 40]):
            self.assertEqual(copy.verify_reviewed_source(self.source), '0' * 40)
        with patch.object(copy, 'digest', return_value='0' * 64), self.assertRaisesRegex(ValueError, 'semantic review'):
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
        command = [sys.executable, str(ROOT / 'tests/webgpu-tensor-copy/check_hook.py'), str(self.source)]
        if os.environ.get('LCB_TEST_DAWN_PACKAGE'):
            command += ['--dawn-package', os.environ['LCB_TEST_DAWN_PACKAGE']]
        subprocess.run(command, check=True)

class TensorCopyBuildCommand(unittest.TestCase):
    def test_existing_profile_selects_backend_without_optimization_flags(self):
        profiles = json.loads((ROOT / 'config/profiles.json').read_text())
        toolchain = {'llamaCommit': 'a' * 40, 'emsdkVersion': 'fixture', 'emscriptenAsyncifyBigIntPatch': {}}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'llama-cpp'
            (root / 'config').mkdir(parents=True)
            for name in ('profiles.json', 'variants.json'):
                shutil.copyfile(ROOT / 'config' / name, root / 'config' / name)
            (root / 'vendor/llama.cpp/include').mkdir(parents=True)
            (root / 'vendor/llama.cpp/include/llama.h').touch()
            dawn = root.parent / '.tools/emdawnwebgpu_pkg'
            dawn.mkdir(parents=True); (dawn / 'emdawnwebgpu.port.py').touch()
            def output(*args, **kwargs):
                return 'emcc fixture' if args[0] == 'emcc' else 'a' * 40
            for profile, config in profiles.items():
                for variant in ('browser', 'test'):
                    argv = ['build.py', '--profile', profile, '--variant', variant]
                    with self.subTest(profile=profile, variant=variant), patch.object(builder, 'ROOT', root), \
                         patch.object(builder, 'runtime_toolchain', return_value=toolchain), \
                         patch.object(builder, 'source_status', return_value=[]), \
                         patch.object(builder, 'output', side_effect=output), \
                         patch.object(builder, 'verify_asyncify_bigint_patch'), \
                         patch.object(builder.shutil, 'which', return_value='/fixture/emcc'), \
                         patch.object(builder.subprocess, 'run') as run, patch.object(sys, 'argv', argv):
                        builder.main()
                        command = run.call_args_list[0].args[0]
                        self.assertIn('-DLCB_WEBGPU=' + ('ON' if config['webgpu'] else 'OFF'), command)
                        self.assertFalse(any('TENSOR_COPY' in a or 'PARAM_UPLOAD_BATCHING' in a for a in command))
                        recorded = json.loads((root / 'build' / profile / variant / 'provenance.json').read_text())
                        self.assertEqual(recorded['cmakeCommand'], command)

if __name__ == '__main__':
    unittest.main()
