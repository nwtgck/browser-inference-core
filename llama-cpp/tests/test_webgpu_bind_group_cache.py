"""Execute actual bounded-cache and graph helpers, with pinned-header syntax checks."""
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
HERE = ROOT / 'tests/webgpu-bind-group-cache'
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
import prepare_webgpu_bind_group_cache as cache
import prepare_webgpu_tensor_copy as tensor_copy


def definition(text, declaration, offset=0):
    start = text.index(declaration, offset)
    brace = text.index('{', start)
    depth, end = 1, brace + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}')
        end += 1
    if text[start:start + 7] == 'struct ':
        end += 1
    return text[start:end] + '\n'


def function(text, name):
    for match in re.finditer(r'\b' + re.escape(name) + r'\s*\(', text):
        start = text.rfind('\n', 0, match.start()) + 1
        brace = text.index('{', match.end())
        if ';' in text[match.end():brace] or not text[start:match.start()].lstrip().startswith('static '):
            continue
        return definition(text, text[start:match.start()] + name + '(', start)
    raise ValueError('Missing actual helper: ' + name)


class BindGroupCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        cls.identity = json.loads((HERE / 'inputs.json').read_text())

    def prepare(self, directory):
        return source.prepare(self.upstream, directory)

    def test_checked_source_patch_output_and_pristine_upstream(self):
        before = (self.upstream / source.SOURCE_PATH).read_bytes()
        with tempfile.TemporaryDirectory(prefix='bind-group-cache-') as temporary:
            result = self.prepare(Path(temporary))
            self.assertEqual(tensor_copy.digest(result), source.OUTPUT_SHA256)
            self.assertEqual(source.scale_unary.INPUTS['ggml-webgpu.cpp'], cache.OUTPUT_SHA256)
            actual = result.read_text()
            graph = function(actual, 'ggml_backend_webgpu_graph_compute')
            self.assertEqual(graph.count('bind_group_cache.begin_graph('), 1)
            self.assertEqual(graph.count('bind_group_cache.end_graph()'), 1)
            self.assertLess(graph.index('bind_group_cache.begin_graph('), graph.index('while (node_idx'))
            self.assertLess(graph.index('bind_group_cache.end_graph()'), graph.index('return GGML_STATUS_SUCCESS;'))
            free = function(actual, 'ggml_backend_webgpu_buffer_free_buffer')
            self.assertLess(free.index('buffer_generation.fetch_add('), free.index('ctx->buffer.Destroy()'))
        self.assertEqual(before, (self.upstream / source.SOURCE_PATH).read_bytes())

    def test_changed_contract_input_patch_and_output_fail_closed(self):
        with tempfile.TemporaryDirectory(prefix='bind-group-cache-guards-') as temporary:
            root = Path(temporary)
            # Construct and verify the actual preceding composition stage, while
            # retaining the final production identity guard in normal operation.
            with patch.object(cache, 'prepare'), patch.object(source.scale_unary, 'prepare', return_value={}), \
                    patch.object(source.q8_subgroup, 'prepare', return_value={}), \
                    patch.object(source.rms_scale, 'prepare', return_value={}), \
                    patch.object(source.async_tensor_copy, 'prepare'), \
                    patch.object(source.iq3s_subgroup, 'prepare', return_value={}), \
                    patch.object(source.host_input_snapshot, 'prepare', return_value={}), \
                    patch.object(source.ssm_conv_silu, 'prepare', return_value={}), \
                    patch.object(source, 'OUTPUT_SHA256', cache.INPUT_SHA256), \
                    patch.object(source, 'SSM_HEADER_OUTPUT_SHA256', source.q4_short.OUTPUT_SHA256):
                baseline = source.prepare(self.upstream, root / 'base')
            self.assertEqual(tensor_copy.digest(baseline), cache.INPUT_SHA256)
            upstream = root / 'upstream'
            for relative in cache.CONTRACT_INPUTS:
                target = upstream / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.upstream / relative, target)
            patches = root / 'patches'
            patches.mkdir()
            shutil.copyfile(ROOT / source.PATCH_DIRECTORY / cache.PATCH_NAME, patches / cache.PATCH_NAME)
            work = root / 'work'
            target = work / source.SOURCE_PATH
            target.parent.mkdir(parents=True)
            shutil.copyfile(baseline, target)
            env = source.patch_environment(work)
            dependency = upstream / next(iter(cache.CONTRACT_INPUTS))
            original = dependency.read_bytes()
            dependency.write_bytes(original + b'\n')
            with self.assertRaisesRegex(ValueError, 'contract changed'):
                cache.prepare(work, upstream, patches, env)
            dependency.write_bytes(original)
            target.write_bytes(b'changed composed input\n')
            with self.assertRaisesRegex(ValueError, 'composition input changed'):
                cache.prepare(work, upstream, patches, env)
            shutil.copyfile(baseline, target)
            patch_file = patches / cache.PATCH_NAME
            patch_file.write_bytes(patch_file.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'patch identity changed'):
                cache.prepare(work, upstream, patches, env)
            shutil.copyfile(ROOT / source.PATCH_DIRECTORY / cache.PATCH_NAME, patch_file)
            with unittest.mock.patch.object(cache, 'OUTPUT_SHA256', '0' * 64), self.assertRaisesRegex(ValueError, 'Unexpected bind-group cache output'):
                cache.prepare(work, upstream, patches, env)
            shutil.copyfile(baseline, target)
            cache.prepare(work, upstream, patches, env)
            self.assertEqual(tensor_copy.digest(target), cache.OUTPUT_SHA256)
            with self.assertRaisesRegex(ValueError, 'composition input changed'):
                cache.prepare(work, upstream, patches, env)

    def test_production_guards_and_reconfiguration_inputs(self):
        with tempfile.TemporaryDirectory(prefix='bind-group-cache-production-') as temporary:
            root = Path(temporary)
            for guard in ('INPUT_SHA256', 'PATCH_SHA256', 'OUTPUT_SHA256'):
                output = root / guard
                with self.subTest(guard=guard), patch.object(cache, guard, '0' * 64):
                    with self.assertRaises(ValueError):
                        source.prepare(self.upstream, output)
                self.assertFalse((output / 'ggml-webgpu.cpp').exists())
                self.assertFalse((output / 'ggml-webgpu-shader-lib.hpp').exists())
            for relative in cache.CONTRACT_INPUTS:
                output = root / ('contract-' + str(list(cache.CONTRACT_INPUTS).index(relative)))
                with self.subTest(contract=relative), patch.object(cache, 'CONTRACT_INPUTS', {
                    **cache.CONTRACT_INPUTS, relative: '0' * 64,
                }):
                    with self.assertRaisesRegex(ValueError, 'Bind-group cache contract'):
                        source.prepare(self.upstream, output)
                self.assertFalse((output / 'ggml-webgpu.cpp').exists())
            text = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
            self.assertIn('scripts/prepare_webgpu_bind_group_cache.py', text)
            self.assertIn(cache.PATCH_NAME, text)
            for relative in cache.CONTRACT_INPUTS:
                self.assertIn('${LCB_LLAMA_SOURCE}/' + relative, text)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_cache_build_graph_free_and_queue_ownership(self):
        with tempfile.TemporaryDirectory(prefix='bind-group-cache-native-') as temporary:
            root = Path(temporary)
            actual = self.prepare(root / 'overlay').read_text()
            code = (HERE / 'mock-harness.cpp').read_text()
            functions = {
                'ACTUAL_CREATE_BUFFER': 'ggml_webgpu_create_buffer',
                'ACTUAL_MAKE_ENTRY': 'ggml_webgpu_make_bind_group_entry',
                'ACTUAL_BUILD_MULTI': 'ggml_backend_webgpu_build_multi',
                'ACTUAL_SUBMIT': 'ggml_backend_webgpu_submit_commands',
                'ACTUAL_GRAPH_COMPUTE': 'ggml_backend_webgpu_graph_compute',
                'ACTUAL_BUFFER_FREE': 'ggml_backend_webgpu_buffer_free_buffer',
            }
            structures = {
                'ACTUAL_PARAM_ARENA': 'webgpu_param_arena',
                'ACTUAL_CACHE': 'webgpu_bind_group_cache',
                'ACTUAL_ENCODED_OP': 'webgpu_encoded_op',
                'ACTUAL_DISPATCH_DESC': 'webgpu_dispatch_desc',
            }
            for marker, name in functions.items():
                code = code.replace('// @' + marker + '@', function(actual, name))
            for marker, name in structures.items():
                code = code.replace('// @' + marker + '@', definition(actual, 'struct ' + name + ' {'))
            harness = root / 'bind-group-cache.cpp'
            harness.write_text(code)
            for batching in (False, True):
                for profiling in (False, True):
                    with self.subTest(batching=batching, profiling=profiling):
                        binary = root / ('cache-' + str(batching) + '-' + str(profiling))
                        macros = (['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'] if batching else [])
                        macros += ['-DGGML_WEBGPU_GPU_PROFILE'] if profiling else []
                        subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O0', *macros, str(harness), '-o', str(binary)], check=True)
                        subprocess.run([str(binary)], check=True)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_complete_composed_source_syntax_with_pinned_headers(self):
        configured = os.environ.get('LCB_TEST_DAWN_PACKAGE')
        dawn = Path(configured) if configured else ROOT.parent / '.tools/emdawnwebgpu_pkg'
        if not dawn.is_dir():
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE for the exact local Dawn package')
        for relative, expected in self.identity['dawn_header_sha256'].items():
            self.assertEqual(hashlib.sha256((dawn / relative).read_bytes()).hexdigest(), expected)
        with tempfile.TemporaryDirectory(prefix='bind-group-cache-syntax-') as temporary:
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
