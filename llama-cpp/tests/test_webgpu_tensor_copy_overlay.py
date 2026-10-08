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

    def test_wrong_revision_source_patch_and_overlap_fail_closed(self):
        with patch.object(copy.subprocess, 'check_output', side_effect=[str(self.source), '0' * 40]), self.assertRaisesRegex(ValueError, 'semantic review'):
            copy.verify_reviewed_source(self.source)
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

    @unittest.skipUnless(shutil.which('cmake'), 'CMake required')
    def test_actual_target_enabled_disabled_reconfigure_and_coexisting_overlays(self):
        dawn = os.environ.get('LCB_TEST_DAWN_PACKAGE')
        if not dawn:
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE to test enabled target configuration')
        before = (self.source / copy.SOURCE_PATH).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / 'build'
            for enabled, moe in [(False, False), (True, False), (True, True), (False, True), (False, False), (True, True), (False, False)]:
                command = ['cmake', '-S', str(ROOT / 'tests/webgpu-tensor-copy'), '-B', str(build),
                           '-DLCB_LLAMA_SOURCE=' + str(self.source), '-DEMDAWNWEBGPU_DIR=' + dawn,
                           '-DLCB_WEBGPU_TENSOR_COPY=' + ('ON' if enabled else 'OFF'),
                           '-DLCB_WEBGPU_MOE_DIRECT_SLOT=' + ('ON' if moe else 'OFF')]
                subprocess.run(command, check=True, capture_output=True, text=True)
                original = (build / 'before.txt').read_text().split(';')
                actual = (build / 'after.txt').read_text().split(';')
                commands = json.loads((build / 'compile_commands.json').read_text())
                copies = [c for c in commands if Path(c['file']).name == 'ggml-webgpu.cpp']
                self.assertEqual(len(copies), 1)
                compiled = Path(copies[0]['file'])
                if enabled:
                    self.assertEqual(set(original) - set(actual), {'ggml-webgpu.cpp'})
                    self.assertEqual(compiled, build / 'webgpu-source-overlay/ggml-webgpu.cpp')
                    self.assertEqual(copy.digest(compiled), copy.PATCHED_SOURCE_SHA256)
                else:
                    self.assertEqual(original, actual)
                    self.assertEqual(compiled.read_bytes(), before)
                    self.assertEqual(compiled, self.source / copy.SOURCE_PATH)
                includes = (build / 'includes.txt').read_text().split(';')
                if moe:
                    self.assertEqual(includes[0], str(build / 'moe-direct-slot-overlay'))
                self.assertTrue((build / 'mtmd-overlay/clip.cpp').is_file())
                self.assertTrue((build / 'mtmd-audio-overlay/mtmd-audio.cpp').is_file())
        self.assertEqual(before, (self.source / copy.SOURCE_PATH).read_bytes())

    def test_dawn_identity_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                copy.verify_dawn(Path(tmp))
        with patch.object(copy, 'digest', return_value='0' * 64), self.assertRaisesRegex(ValueError, 'Dawn'):
            copy.verify_dawn(Path('/unused'))

    def test_cli_rejects_cpu_before_toolchain_probe(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/build.py'), '--profile',
                                 'cpu-wasm32', '--variant', 'browser', '--webgpu-tensor-copy'],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires a WebGPU profile', result.stderr)

    def test_actual_prepared_hook_mock_and_optional_real_headers(self):
        command = [sys.executable, str(ROOT / 'tests/webgpu-tensor-copy/check_hook.py'), str(self.source)]
        if os.environ.get('LCB_TEST_DAWN_PACKAGE'):
            command += ['--dawn-package', os.environ['LCB_TEST_DAWN_PACKAGE']]
        subprocess.run(command, check=True)

class TensorCopyBuildCommand(unittest.TestCase):
    def test_opt_in_and_explicit_off_are_recorded_for_every_profile(self):
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
                for enabled, batching in ([(False, False), (True, False), (False, True), (True, True)] if config['webgpu'] else [(False, False)]):
                    argv = ['build.py', '--profile', profile, '--variant', 'browser']
                    if enabled:
                        argv.append('--webgpu-tensor-copy')
                    if batching:
                        argv.append('--webgpu-param-upload-batching')
                    with self.subTest(profile=profile, enabled=enabled, batching=batching), patch.object(builder, 'ROOT', root), \
                         patch.object(builder, 'runtime_toolchain', return_value=toolchain), \
                         patch.object(builder, 'source_status', return_value=[]), \
                         patch.object(builder, 'output', side_effect=output), \
                         patch.object(builder, 'verify_asyncify_bigint_patch'), \
                         patch.object(builder.shutil, 'which', return_value='/fixture/emcc'), \
                         patch.object(builder.subprocess, 'run') as run, patch.object(sys, 'argv', argv):
                        builder.main()
                        command = run.call_args_list[0].args[0]
                        flag = '-DLCB_WEBGPU_TENSOR_COPY=' + ('ON' if enabled else 'OFF')
                        self.assertEqual([a for a in command if 'LCB_WEBGPU_TENSOR_COPY=' in a], [flag])
                        batch_flag = '-DLCB_WEBGPU_PARAM_UPLOAD_BATCHING=' + ('ON' if batching else 'OFF')
                        self.assertEqual([a for a in command if 'LCB_WEBGPU_PARAM_UPLOAD_BATCHING=' in a], [batch_flag])
                        recorded = json.loads((root / 'build' / profile / 'browser/provenance.json').read_text())
                        self.assertEqual(recorded['cmakeCommand'], command)

if __name__ == '__main__':
    unittest.main()
