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


class WebgpuSourceOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def test_combined_source_and_reuse_preserve_input(self):
        before = (self.upstream / source.SOURCE_PATH).read_bytes()
        loader_before = (self.upstream / source.LOADER_PATH).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            result = source.prepare(self.upstream, Path(tmp))
            self.assertEqual(copy.digest(result), source.OUTPUT_SHA256)
            self.assertEqual(copy.digest(result.parent / 'ggml-webgpu-shader-lib.hpp'), source.SSM_HEADER_OUTPUT_SHA256)
            self.assertEqual(copy.digest(result.parent / 'llama-model-loader.cpp'), source.LOADER_OUTPUT_SHA256)
            header_stamp = (result.parent / 'ggml-webgpu-shader-lib.hpp').stat().st_mtime_ns
            stamp = result.stat().st_mtime_ns
            loader_stamp = (result.parent / 'llama-model-loader.cpp').stat().st_mtime_ns
            self.assertEqual(source.prepare(self.upstream, Path(tmp)).stat().st_mtime_ns, stamp)
            self.assertEqual((result.parent / 'ggml-webgpu-shader-lib.hpp').stat().st_mtime_ns, header_stamp)
            self.assertEqual((result.parent / 'llama-model-loader.cpp').stat().st_mtime_ns, loader_stamp)
        self.assertEqual(before, (self.upstream / source.SOURCE_PATH).read_bytes())
        self.assertEqual(loader_before, (self.upstream / source.LOADER_PATH).read_bytes())

    def test_repository_nested_builds_apply_all_patches_without_touching_git_state(self):
        upstream_before = {name: (self.upstream / name).read_bytes()
                           for name in (source.SOURCE_PATH, source.LOADER_PATH)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkout = root / 'checkout'
            subprocess.run(['git', 'init', '--quiet', str(checkout)], check=True)
            (checkout / '.gitignore').write_text('llama-cpp/build/\n')
            subprocess.run(['git', 'add', '.gitignore'], cwd=checkout, check=True)
            subprocess.run(['git', '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                            'commit', '--quiet', '-m', 'fixture'], cwd=checkout, check=True)
            worktree = root / 'worktree'
            subprocess.run(['git', 'worktree', 'add', '--quiet', '--detach', str(worktree)],
                           cwd=checkout, check=True)
            self.assertTrue((worktree / '.git').is_file())
            for parent in (checkout, worktree):
                (parent / 'keep-untracked.txt').write_text('must remain unchanged')
                status = subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=parent)
                index = subprocess.check_output(['git', 'ls-files', '--stage'], cwd=parent)
                with self.subTest(worktree=parent == worktree):
                    build = parent / 'llama-cpp/build/webgpu-wasm32-jspi/test/webgpu-source-overlay'
                    result = source.prepare(self.upstream, build)
                    self.assertEqual(copy.digest(result), source.OUTPUT_SHA256)
                    self.assertEqual(copy.digest(build / 'llama-model-loader.cpp'), source.LOADER_OUTPUT_SHA256)
                    stamp = result.stat().st_mtime_ns
                    self.assertEqual(source.prepare(self.upstream, build).stat().st_mtime_ns, stamp)
                    self.assertEqual(subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=parent), status)
                    self.assertEqual(subprocess.check_output(['git', 'ls-files', '--stage'], cwd=parent), index)
                    self.assertEqual((parent / 'keep-untracked.txt').read_text(), 'must remain unchanged')
        for name, before in upstream_before.items():
            self.assertEqual((self.upstream / name).read_bytes(), before)

    def test_patch_environment_does_not_inherit_repository_selectors(self):
        context = {key: '/enclosing/repository' for key in
                   ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE')}
        with patch.dict(os.environ, {**context, 'GIT_CEILING_DIRECTORIES': '/wrong', 'KEEP_ME': 'value'}):
            env = source.patch_environment(Path('/build/overlay/scratch'))
            self.assertEqual(env['GIT_CEILING_DIRECTORIES'], '/build/overlay')
            self.assertEqual(env['KEEP_ME'], 'value')
            for key in context:
                self.assertNotIn(key, env)
                self.assertEqual(os.environ[key], context[key])

    def test_patch_result_and_overlap_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); patches = root / 'patches'; patches.mkdir()
            for name in [copy.PATCH_NAME, source.PARAM_PATCH_NAME, source.LOADER_PATCH_NAME, source.SSM_PATCH_NAME]:
                shutil.copyfile(ROOT / copy.PATCH_DIRECTORY / name, patches / name)
            (patches / source.PARAM_PATCH_NAME).write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'patch identity'):
                source.prepare(self.upstream, root / 'out', patch_root=patches)
            self.assertFalse((root / 'out').exists())
            with patch.object(source, 'OUTPUT_SHA256', '0' * 64), self.assertRaisesRegex(ValueError, 'combined'):
                source.prepare(self.upstream, root / 'out')
        with self.assertRaisesRegex(ValueError, 'outside'):
            source.prepare(self.upstream, self.upstream / 'out')

    @unittest.skipUnless(shutil.which('cmake'), 'CMake required')
    def test_actual_target_combined_source_and_moe_reconfiguration(self):
        dawn = os.environ.get('LCB_TEST_DAWN_PACKAGE')
        if not dawn:
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        before = (self.upstream / source.SOURCE_PATH).read_bytes()
        loader_before = (self.upstream / source.LOADER_PATH).read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(['git', 'init', '--quiet', tmp], check=True)
            build = Path(tmp) / 'llama-cpp/build/overlay-check'
            for moe in (False, True, False):
                command = ['cmake', '-S', str(ROOT / 'tests/webgpu-tensor-copy'), '-B', str(build),
                           '-DLCB_LLAMA_SOURCE=' + str(self.upstream), '-DEMDAWNWEBGPU_DIR=' + dawn,
                           '-DLCB_WEBGPU_MOE_DIRECT_SLOT=' + ('ON' if moe else 'OFF')]
                subprocess.run(command, check=True, capture_output=True, text=True)
                entries = json.loads((build / 'compile_commands.json').read_text())
                compiled = [entry for entry in entries if Path(entry['file']).name == 'ggml-webgpu.cpp']
                self.assertEqual(len(compiled), 1)
                self.assertEqual(Path(compiled[0]['file']), build / 'webgpu-source-overlay/ggml-webgpu.cpp')
                self.assertEqual(copy.digest(Path(compiled[0]['file'])), source.OUTPUT_SHA256)
                self.assertIn('GGML_WEBGPU_BATCH_PARAM_UPLOADS', compiled[0]['command'])
                self.assertEqual(copy.digest(build / 'webgpu-source-overlay/ggml-webgpu-shader-lib.hpp'), source.SSM_HEADER_OUTPUT_SHA256)
                self.assertFalse((build / 'webgpu-source-overlay/ggml-wgsl-shaders.hpp').exists())
                loaders = [entry for entry in entries if Path(entry['file']).name == 'llama-model-loader.cpp']
                self.assertEqual(len(loaders), 1)
                self.assertEqual(Path(loaders[0]['file']), build / 'webgpu-source-overlay/llama-model-loader.cpp')
                self.assertEqual(copy.digest(Path(loaders[0]['file'])), source.LOADER_OUTPUT_SHA256)
                if moe:
                    self.assertEqual((build / 'includes.txt').read_text().split(';')[0], str(build / 'moe-direct-slot-overlay'))
                self.assertTrue((build / 'mtmd-overlay/clip.cpp').is_file())
                self.assertTrue((build / 'mtmd-audio-overlay/mtmd-audio.cpp').is_file())
        self.assertEqual(before, (self.upstream / source.SOURCE_PATH).read_bytes())
        self.assertEqual(loader_before, (self.upstream / source.LOADER_PATH).read_bytes())

    def test_loader_inputs_patch_and_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in source.LOADER_INPUTS:
                inputs = dict(source.LOADER_INPUTS); inputs[relative] = '0' * 64
                with patch.object(source, 'LOADER_INPUTS', inputs), self.assertRaisesRegex(ValueError, 'semantic review'):
                    source.prepare(self.upstream, root / 'out')
                self.assertFalse((root / 'out').exists())
            with patch.object(source, 'LOADER_PATCH_SHA256', '0' * 64), self.assertRaisesRegex(ValueError, 'patch identity'):
                source.prepare(self.upstream, root / 'out')
            self.assertFalse((root / 'out').exists())
            with patch.object(source, 'LOADER_OUTPUT_SHA256', '0' * 64), self.assertRaisesRegex(ValueError, 'loader source identity'):
                source.prepare(self.upstream, root / 'out')
            self.assertFalse((root / 'out/llama-model-loader.cpp').exists())

    def test_ssm_shader_patch_and_header_fail_closed_before_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'out'
            for name, message in [('SSM_SHADER_SHA256', 'upstream shader'),
                                  ('SSM_PATCH_SHA256', 'patch identity'),
                                  ('SSM_HEADER_OUTPUT_SHA256', 'header identity')]:
                with self.subTest(guard=name), patch.object(source, name, '0' * 64):
                    with self.assertRaisesRegex(ValueError, message):
                        source.prepare(self.upstream, output)
                self.assertFalse((output / 'ggml-webgpu.cpp').exists())
                self.assertFalse((output / 'llama-model-loader.cpp').exists())
                self.assertFalse((output / 'ggml-webgpu-shader-lib.hpp').exists())

    def test_webgpu_is_the_only_source_overlay_build_switch(self):
        text = (ROOT / 'CMakeLists.txt').read_text()
        self.assertIn('if(LCB_WEBGPU)\n    include(cmake/WebgpuSourceOverlay.cmake)', text)
        self.assertNotIn('LCB_WEBGPU_TENSOR_COPY', text)
        self.assertNotIn('LCB_WEBGPU_PARAM_UPLOAD_BATCHING', text)

    def test_actual_arena_and_graph_regressions(self):
        command = [sys.executable, str(ROOT / 'tests/webgpu-param-upload-batching/check_uploads.py'), str(self.upstream)]
        if os.environ.get('LCB_TEST_DAWN_PACKAGE'):
            command += ['--dawn-package', os.environ['LCB_TEST_DAWN_PACKAGE']]
        subprocess.run(command, check=True)


if __name__ == '__main__':
    unittest.main()
