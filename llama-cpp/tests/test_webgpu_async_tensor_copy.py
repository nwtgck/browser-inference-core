"""Source-bound optional copy hook: real helpers, deferred queues and full TU syntax."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tests/webgpu-async-tensor-copy'
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
import prepare_webgpu_async_tensor_copy as async_copy
import prepare_webgpu_tensor_copy as tensor_copy


def function(text, name):
    for match in re.finditer(r'\b' + re.escape(name) + r'\s*\(', text):
        start = text.rfind('\n', 0, match.start()) + 1
        brace = text.index('{', match.end())
        prefix = text[start:match.start()].strip()
        if ';' in text[match.end():brace] or not re.match(r'^(static|void|bool|size_t|ggml_backend_dev_t|enum)\b', prefix):
            continue
        depth, end = 1, brace + 1
        while depth:
            depth += (text[end] == '{') - (text[end] == '}')
            end += 1
        return text[start:end] + '\n'
    raise ValueError('Missing actual helper: ' + name)


class AsyncTensorCopy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        cls.identity = json.loads((HERE / 'inputs.json').read_text())
        for relative, expected in cls.identity['upstream_sha256'].items():
            if tensor_copy.digest(cls.upstream / relative) != expected:
                raise ValueError('Async-copy fixture needs semantic review: ' + relative)

    def prepare(self, directory):
        return source.prepare(self.upstream, directory)

    def test_reviewed_identity_and_pristine_vendor(self):
        self.assertEqual(self.identity['upstream_sha256'], async_copy.CONTRACT_INPUTS)
        self.assertEqual(self.identity['patch_sha256'], async_copy.PATCH_SHA256)
        self.assertEqual(self.identity['dawn_header_sha256'], tensor_copy.DAWN_HEADERS)
        before = {p: (self.upstream / p).read_bytes() for p in async_copy.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory(prefix='async-copy-identity-') as temporary:
            result = self.prepare(Path(temporary))
            self.assertEqual(tensor_copy.digest(result), source.OUTPUT_SHA256)
            self.assertEqual(source.iq3s_subgroup.INPUTS['ggml-webgpu.cpp'], async_copy.OUTPUT_SHA256)
            self.assertRegex(result.read_text(), r'/\* \.cpy_tensor_async\s+= \*/\s*ggml_backend_webgpu_cpy_tensor_async,')
        self.assertEqual(before, {p: (self.upstream / p).read_bytes() for p in before})

    def test_contract_input_patch_output_and_reapplication_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix='async-copy-guards-') as temporary:
            root = Path(temporary)
            with patch.object(async_copy, 'prepare'), patch.object(source.iq3s_subgroup, 'prepare', return_value={}), \
                    patch.object(source.host_input_snapshot, 'prepare', return_value={}), \
                    patch.object(source.ssm_conv_silu, 'prepare', return_value={}), \
                    patch.object(source.moe_group_skip, 'prepare', return_value={}), \
                patch.object(source.router_tail, 'prepare', return_value={}), \
                    patch.object(source, 'OUTPUT_SHA256', async_copy.INPUT_SHA256), \
                    patch.object(source, 'SSM_HEADER_OUTPUT_SHA256', source.rms_scale.OUTPUTS['ggml-webgpu-shader-lib.hpp']):
                baseline = source.prepare(self.upstream, root / 'base')
            self.assertEqual(tensor_copy.digest(baseline), async_copy.INPUT_SHA256)
            upstream = root / 'upstream'
            for relative in async_copy.CONTRACT_INPUTS:
                target = upstream / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.upstream / relative, target)
            patches = root / 'patches'
            patches.mkdir()
            patch_file = patches / async_copy.PATCH_NAME
            shutil.copyfile(ROOT / source.PATCH_DIRECTORY / async_copy.PATCH_NAME, patch_file)
            work = root / 'work'
            target = work / source.SOURCE_PATH
            target.parent.mkdir(parents=True)
            shutil.copyfile(baseline, target)
            env = source.patch_environment(work)
            for relative in async_copy.CONTRACT_INPUTS:
                dependency = upstream / relative
                original = dependency.read_bytes()
                dependency.write_bytes(original + b'\n')
                with self.subTest(relative=relative), self.assertRaisesRegex(ValueError, 'contract changed'):
                    async_copy.prepare(work, upstream, patches, env)
                self.assertEqual(target.read_bytes(), baseline.read_bytes())
                dependency.write_bytes(original)
            target.write_bytes(b'changed input\n')
            with self.assertRaisesRegex(ValueError, 'composition input changed'):
                async_copy.prepare(work, upstream, patches, env)
            shutil.copyfile(baseline, target)
            patch_file.write_bytes(patch_file.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'patch identity changed'):
                async_copy.prepare(work, upstream, patches, env)
            shutil.copyfile(ROOT / source.PATCH_DIRECTORY / async_copy.PATCH_NAME, patch_file)
            with patch.object(async_copy, 'OUTPUT_SHA256', '0' * 64), self.assertRaisesRegex(ValueError, 'Unexpected async tensor-copy output'):
                async_copy.prepare(work, upstream, patches, env)
            shutil.copyfile(baseline, target)
            async_copy.prepare(work, upstream, patches, env)
            self.assertEqual(tensor_copy.digest(target), async_copy.OUTPUT_SHA256)
            with self.assertRaisesRegex(ValueError, 'composition input changed'):
                async_copy.prepare(work, upstream, patches, env)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_copy_hook_helpers_generic_fallback_and_scheduler(self):
        with tempfile.TemporaryDirectory(prefix='async-copy-native-') as temporary:
            root = Path(temporary)
            actual = self.prepare(root / 'overlay').read_text()
            generic = (self.upstream / 'ggml/src/ggml-backend.cpp').read_text()
            registration = re.search(r'/\* \.cpy_tensor_async\s+= \*/\s*(\w+)\s*,', actual)
            self.assertIsNotNone(registration)
            code = (HERE / 'mock-harness.cpp').read_text()
            groups = {
                'ACTUAL_OFFSET': (actual, ['ggml_webgpu_tensor_offset']),
                'ACTUAL_BUFFER_BASE': (actual, ['ggml_backend_webgpu_buffer_get_base']),
                'ACTUAL_GENERIC_BASE': (generic, ['ggml_backend_buffer_get_base']),
                'ACTUAL_DEVICE_ACCESS': (generic, ['ggml_backend_get_device', 'ggml_backend_dev_type']),
                'ACTUAL_WEBGPU_SYNCHRONIZE': (actual, ['ggml_backend_webgpu_synchronize']),
                'ACTUAL_SYNCHRONIZE': (generic, ['ggml_backend_synchronize']),
                'ACTUAL_WEBGPU_SET': (actual, ['ggml_backend_webgpu_buffer_set_tensor']),
                'ACTUAL_GENERIC_SET_GET': (generic, ['ggml_backend_tensor_set', 'ggml_backend_tensor_get']),
                'ACTUAL_BUFFER_COPY': (actual, ['ggml_backend_webgpu_buffer_cpy_tensor']),
                'ACTUAL_GENERIC_BUFFER_COPY': (generic, ['ggml_backend_buffer_copy_tensor']),
                'ACTUAL_HOOK': (actual, ['ggml_backend_webgpu_cpy_tensor_async']),
                'ACTUAL_GENERIC_COPY': (generic, ['ggml_backend_tensor_copy']),
                'ACTUAL_GENERIC_ASYNC_COPY': (generic, ['ggml_backend_tensor_copy_async']),
                'ACTUAL_GENERIC_USAGE': (generic, ['ggml_backend_buffer_get_usage']),
                'ACTUAL_HOST_WEIGHT': (generic, ['ggml_backend_sched_is_host_weight']),
                'ACTUAL_SCHEDULER_COPY': (generic, ['ggml_backend_sched_copy_input']),
            }
            for marker, (text, names) in groups.items():
                code = code.replace('// @' + marker + '@', '\n'.join(function(text, name) for name in names))
            code = code.replace('// @REGISTERED_HOOK@', registration.group(1) + ',')
            self.assertNotIn('// @', code)
            harness = root / 'async-copy.cpp'
            harness.write_text(code)
            for macros in ([], ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'],
                           ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS', '-DGGML_WEBGPU_GPU_PROFILE', '-DGGML_WEBGPU_CPU_PROFILE']):
                binary = root / ('async-copy-' + str(len(macros)))
                subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O0', *macros, str(harness), '-o', str(binary)], check=True)
                subprocess.run([str(binary)], check=True)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_complete_composed_source_syntax_with_pinned_dawn_headers(self):
        configured = os.environ.get('LCB_TEST_DAWN_PACKAGE')
        dawn = Path(configured) if configured else ROOT.parent / '.tools/emdawnwebgpu_pkg'
        if not dawn.is_dir():
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE for the exact local Dawn package')
        for relative, expected in self.identity['dawn_header_sha256'].items():
            self.assertEqual(tensor_copy.digest(dawn / relative), expected)
        with tempfile.TemporaryDirectory(prefix='async-copy-syntax-') as temporary:
            root = Path(temporary)
            target = self.prepare(root)
            upstream_webgpu = self.upstream / 'ggml/src/ggml-webgpu'
            subprocess.run([sys.executable, str(upstream_webgpu / 'wgsl-shaders/embed_wgsl.py'), '--input_dir',
                            str(upstream_webgpu / 'wgsl-shaders'), '--output_file', str(root / 'ggml-wgsl-shaders.hpp')], check=True)
            includes = [self.upstream / name for name in ('ggml/include', 'ggml/src')]
            includes += [upstream_webgpu, root]
            command = ['c++', '-std=c++20', '-fsyntax-only', '-D__wasi__', '-DGGML_SCHED_MAX_COPIES=4']
            command += ['-I' + str(path) for path in includes]
            command += ['-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include')]
            for macros in ([], ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'],
                           ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS', '-DGGML_WEBGPU_GPU_PROFILE', '-DGGML_WEBGPU_CPU_PROFILE']):
                subprocess.run(command + macros + [str(target)], check=True)


if __name__ == '__main__':
    unittest.main()
