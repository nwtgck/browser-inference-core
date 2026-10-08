"""Check composed source preparation and real target configuration, not GPU execution."""
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
import prepare_webgpu_source as source
import prepare_webgpu_tensor_copy as copy
import publish_artifacts as publisher


class WebgpuSourceOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def test_four_source_modes_and_reuse_preserve_input(self):
        before = (self.upstream / source.SOURCE_PATH).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            for mode in [(False, False), (True, False), (False, True), (True, True), (False, False)]:
                with self.subTest(mode=mode):
                    result = source.prepare(self.upstream, Path(tmp), copy=mode[0], batch=mode[1])
                    self.assertEqual(copy.digest(result), source.OUTPUT_SHA256[mode])
                    stamp = result.stat().st_mtime_ns
                    self.assertEqual(source.prepare(self.upstream, Path(tmp), copy=mode[0], batch=mode[1]).stat().st_mtime_ns, stamp)
        self.assertEqual(before, (self.upstream / source.SOURCE_PATH).read_bytes())

    def test_patch_result_and_option_identity_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); patches = root / 'patches'; patches.mkdir()
            for name in [copy.PATCH_NAME, source.PARAM_PATCH_NAME]:
                shutil.copyfile(ROOT / copy.PATCH_DIRECTORY / name, patches / name)
            (patches / source.PARAM_PATCH_NAME).write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'patch identity'):
                source.prepare(self.upstream, root / 'out', copy=False, batch=True, patch_root=patches)
            self.assertFalse((root / 'out').exists())
            with self.assertRaisesRegex(ValueError, 'booleans'):
                source.prepare(self.upstream, root / 'out', copy='OFF', batch=False)
            with patch.dict(source.OUTPUT_SHA256, {(True, True): '0' * 64}), self.assertRaisesRegex(ValueError, 'combined'):
                source.prepare(self.upstream, root / 'out', copy=True, batch=True)
        with self.assertRaisesRegex(ValueError, 'outside'):
            source.prepare(self.upstream, self.upstream / 'out', copy=True, batch=True)

    @unittest.skipUnless(shutil.which('cmake'), 'CMake required')
    def test_actual_target_all_modes_definitions_and_moe_reconfiguration(self):
        dawn = os.environ.get('LCB_TEST_DAWN_PACKAGE')
        if not dawn:
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        before = (self.upstream / source.SOURCE_PATH).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            build = Path(tmp) / 'build'
            for moe in (False, True):
                for copy_on, batch_on in [(False, False), (True, False), (False, True), (True, True), (True, False), (False, False)]:
                    with self.subTest(copy=copy_on, batch=batch_on, moe=moe):
                        command = ['cmake', '-S', str(ROOT / 'tests/webgpu-tensor-copy'), '-B', str(build),
                                   '-DLCB_LLAMA_SOURCE=' + str(self.upstream), '-DEMDAWNWEBGPU_DIR=' + dawn,
                                   '-DLCB_WEBGPU_TENSOR_COPY=' + ('ON' if copy_on else 'OFF'),
                                   '-DLCB_WEBGPU_PARAM_UPLOAD_BATCHING=' + ('ON' if batch_on else 'OFF'),
                                   '-DLCB_WEBGPU_MOE_DIRECT_SLOT=' + ('ON' if moe else 'OFF')]
                        subprocess.run(command, check=True, capture_output=True, text=True)
                        entries = json.loads((build / 'compile_commands.json').read_text())
                        compiled = [entry for entry in entries if Path(entry['file']).name == 'ggml-webgpu.cpp']
                        self.assertEqual(len(compiled), 1)
                        path = Path(compiled[0]['file'])
                        self.assertEqual(copy.digest(path), source.OUTPUT_SHA256[(copy_on, batch_on)])
                        self.assertEqual('GGML_WEBGPU_BATCH_PARAM_UPLOADS' in compiled[0]['command'], batch_on)
                        expected = build / 'webgpu-source-overlay/ggml-webgpu.cpp' if copy_on or batch_on else self.upstream / source.SOURCE_PATH
                        self.assertEqual(path, expected)
                        if moe:
                            self.assertEqual((build / 'includes.txt').read_text().split(';')[0], str(build / 'moe-direct-slot-overlay'))
        self.assertEqual(before, (self.upstream / source.SOURCE_PATH).read_bytes())

    def test_cmake_options_are_off_by_default(self):
        text = (ROOT / 'CMakeLists.txt').read_text()
        self.assertIn('option(LCB_WEBGPU_TENSOR_COPY "Enable experimental same-device WebGPU copies" OFF)', text)
        self.assertIn('option(LCB_WEBGPU_PARAM_UPLOAD_BATCHING "Batch experimental WebGPU parameter uploads" OFF)', text)

    def test_cpu_cli_rejected_before_toolchain(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/build.py'), '--profile', 'cpu-wasm32',
                                 '--variant', 'browser', '--webgpu-param-upload-batching'], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('requires a WebGPU profile', result.stderr)

    def test_publisher_rejects_either_experiment_or_ambiguous_flags(self):
        for name in ('LCB_WEBGPU_TENSOR_COPY', 'LCB_WEBGPU_PARAM_UPLOAD_BATCHING'):
            for flags in ([f'-D{name}=ON'], [f'-D{name}=OFF', f'-D{name}=ON'],
                          [f'-D{name}:BOOL=ON'], [f'-D{name}=OFF', f'-D{name}:BOOL=ON']):
                manifest = {'profiles': {'webgpu': {'variants': {'browser': {'cmakeCommand': flags}}}}}
                with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, 'artifact-only'):
                    publisher.require_publishable_tensor_copy(manifest)
        publisher.require_publishable_tensor_copy({'profiles': {'cpu': {'variants': {'browser': {'cmakeCommand': [
            '-DLCB_WEBGPU_TENSOR_COPY=OFF', '-DLCB_WEBGPU_PARAM_UPLOAD_BATCHING=OFF']}}}}})


if __name__ == '__main__':
    unittest.main()
