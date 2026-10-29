"""Source-bound SSM convolution/SiLU graph, allocator, host and shader contracts."""
import hashlib
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
HERE = ROOT / 'tests/webgpu-ssm-conv-silu'
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
import prepare_webgpu_ssm_conv_silu as fusion


def function(text, marker):
    start = text.index(marker)
    return text[start:text.index('\n}', start) + 2]


def factory(upstream, prepared, directory):
    directory.mkdir(parents=True, exist_ok=True)
    text = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    actual = function(text, '    webgpu_pipeline get_ssm_conv_pipeline(')
    # The class method ends at four-space indentation rather than column zero.
    actual = actual[:actual.index('\n    }') + 6]
    keys = []
    for marker in ('struct ggml_webgpu_ssm_conv_pipeline_key {',
                   'struct ggml_webgpu_ssm_conv_pipeline_key_hash {',
                   'struct ggml_webgpu_ssm_conv_shader_decisions {'):
        start = text.index(marker)
        keys.append(text[start:text.index('\n};', start) + 3])
    cpp = directory / 'factory.cpp'
    cpp.write_text((HERE / 'factory-prefix.cpp').read_text().replace('// @ACTUAL_KEYS@', '\n'.join(keys)) +
                   actual + '\n' + (HERE / 'factory-cases.cpp').read_text())
    executable = directory / 'factory'
    subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-I' + str(prepared),
                    '-I' + str(directory), '-isystem', str(upstream / fusion.BASE), '-isystem', str(upstream / 'ggml/include'),
                    str(cpp), '-o', str(executable)], check=True)
    shaders = directory / 'shaders'
    shaders.mkdir()
    subprocess.run([str(executable), str(shaders)], check=True)
    return shaders


class SsmConvSilu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        cls.temp = tempfile.TemporaryDirectory(prefix='ssm-conv-silu-tests-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.prepared = source.prepare(cls.upstream, cls.directory / 'prepared').parent
        cls.library = None
        # Canonical shaders are always generated from untouched upstream sources.
        cls.canonical = cls.directory / 'canonical'
        cls.canonical.mkdir()
        subprocess.run([sys.executable, str(cls.upstream / fusion.BASE / 'wgsl-shaders/embed_wgsl.py'),
                        '--input_dir', str(cls.upstream / fusion.BASE / 'wgsl-shaders'),
                        '--output_file', str(cls.canonical / 'ggml-wgsl-shaders.hpp')], check=True)

    @classmethod
    def native_library(cls):
        if cls.library:
            return cls.library
        configured = os.environ.get('LCB_TEST_GGML_BASE_LIBRARY')
        if configured:
            cls.library = Path(configured).resolve()
        else:
            build = cls.directory / 'native-build'
            subprocess.run(['cmake', '-S', str(cls.upstream), '-B', str(build), '-DBUILD_SHARED_LIBS=OFF',
                            '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF', '-DGGML_CCACHE=OFF', '-DLLAMA_BUILD_TESTS=OFF',
                            '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_TOOLS=OFF', '-DLLAMA_BUILD_COMMON=OFF', '-DLLAMA_CURL=OFF'],
                           check=True, capture_output=True)
            subprocess.run(['cmake', '--build', str(build), '--target', 'ggml-base', '-j', '2'], check=True, capture_output=True)
            cls.library = build / 'ggml/src/libggml-base.a'
        return cls.library

    def native(self, cpp, destination):
        subprocess.run(['c++', '-std=c++17', '-O1', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror',
                        '-I' + str(self.prepared), '-I' + str(HERE), '-isystem', str(self.upstream / 'ggml/include'),
                        '-isystem', str(self.upstream / 'ggml/src'), str(cpp), str(self.native_library()),
                        '-pthread', '-ldl', '-lm', '-o', str(destination)], check=True)
        subprocess.run([str(destination)], check=True)

    def test_checked_composition_and_unchanged_dot(self):
        identity = json.loads((HERE / 'inputs.json').read_text())
        self.assertEqual(hashlib.sha256((ROOT / 'scripts/prepare_webgpu_ssm_conv_silu.py').read_bytes()).hexdigest(), identity['module_sha256'])
        self.assertFalse((self.prepared / 'ggml-wgsl-shaders.hpp').exists())
        for name, expected in fusion.CONTRACT_INPUTS.items():
            self.assertEqual(hashlib.sha256((self.upstream / name).read_bytes()).hexdigest(), expected)
        final = {'ggml-webgpu.cpp': source.FINAL_OUTPUTS[False]['ggml-webgpu.cpp'],
                 'ggml-webgpu-shader-lib.hpp': source.FINAL_OUTPUTS[False]['ggml-webgpu-shader-lib.hpp'],
                 'ggml-webgpu-ssm-conv-silu.hpp': fusion.OUTPUTS['ggml-webgpu-ssm-conv-silu.hpp']}
        for name, expected in final.items():
            self.assertEqual(hashlib.sha256((self.prepared / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(hashlib.sha256((self.prepared / fusion.HEADER_NAME).read_bytes()).hexdigest(), fusion.HEADER_SHA256)
        embedded = (self.prepared / fusion.HEADER_NAME).read_text().split('R"wgsl(', 1)[1].split(')wgsl"', 1)[0]
        original = (self.upstream / fusion.BASE / 'wgsl-shaders/ssm_conv.wgsl').read_text()
        self.assertEqual(embedded, original.replace('    dst[dst_idx] = sum;',
                         '    let result = sum / (1.0 + exp(-sum));\n    dst[dst_idx] = result;'))
        unary = (self.upstream / fusion.BASE / 'wgsl-shaders/unary.wgsl').read_text()
        self.assertIn('/ (1.0 + exp(-src[params.offset_src + src_idx]))', unary)
        cpp = (self.prepared / 'ggml-webgpu.cpp').read_text()
        self.assertIn('if (auto fused = ggml_webgpu_ssm_conv_silu(ctx, cgraph, node_idx, num_encoded_ops)) { return fused; }', cpp)
        self.assertIn('return ggml_webgpu_ssm_conv(ctx, src0, src1, node);', cpp)

    def test_contract_patch_output_and_embedding_drift(self):
        for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS'):
            values = getattr(fusion, field)
            for name in values:
                with self.subTest(field=field, path=name), patch.object(fusion, field, {**values, name: '0' * 64}):
                    with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                        source.prepare(self.upstream, Path(tmp))
        for field in ('PATCH_SHA256', 'HEADER_SHA256'):
            with self.subTest(field=field), patch.object(fusion, field, '0' * 64):
                with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
                    source.prepare(self.upstream, Path(tmp))

    @unittest.skipUnless(shutil.which('c++') and shutil.which('cmake'), 'native tools required')
    def test_real_graph_allocator_visibility_and_layout(self):
        self.native(HERE / 'admission.cpp', self.directory / 'admission')

    @unittest.skipUnless(shutil.which('c++') and shutil.which('cmake'), 'native tools required')
    def test_actual_runtime_bindings_params_and_dispatch(self):
        text = (self.prepared / 'ggml-webgpu.cpp').read_text()
        actual = []
        markers = ('static size_t ggml_webgpu_tensor_offset(', 'static wgpu::Buffer ggml_webgpu_tensor_buf(',
                   'static size_t ggml_webgpu_tensor_align_offset(const ggml_tensor *',
                   'static size_t ggml_webgpu_tensor_misalignment(const ggml_tensor *',
                   'static size_t ggml_webgpu_tensor_misalignment(webgpu_context &',
                   'static size_t ggml_webgpu_tensor_align_offset(webgpu_context &',
                   'static size_t ggml_webgpu_tensor_binding_size(const ggml_tensor *',
                   'static size_t ggml_webgpu_tensor_binding_size(webgpu_context &',
                   'static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(',
                   'static wgpu::BindGroupEntry ggml_webgpu_make_tensor_bind_group_entry(',
                   'static webgpu_encoded_op ggml_webgpu_ssm_conv(')
        for marker in markers:
            actual.append(function(text, marker))
        marker = 'static void * ggml_backend_webgpu_buffer_get_base('
        actual.append(function(text[text.rindex(marker):], marker))
        start = text.index('struct webgpu_ssm_conv_silu_binding {')
        actual.append(text[start:text.index('\n};', start) + 3])
        for marker in ('static bool ggml_webgpu_ssm_conv_silu_binding(',
                       'static std::optional<webgpu_encoded_op> ggml_webgpu_ssm_conv_silu('):
            actual.append(function(text, marker))
        cpp = self.directory / 'dispatch.cpp'
        cpp.write_text((HERE / 'dispatch-prefix.cpp').read_text().replace('// @ACTUAL_FUNCTIONS@', '\n'.join(actual)) +
                       '\n' + (HERE / 'dispatch-cases.cpp').read_text())
        self.native(cpp, self.directory / 'dispatch')

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_factory_cache_and_scalar_vec4_both_workgroups(self):
        directory = self.directory / 'factory'
        directory.mkdir()
        shutil.copyfile(self.canonical / 'ggml-wgsl-shaders.hpp', directory / 'ggml-wgsl-shaders.hpp')
        shaders = factory(self.upstream, self.prepared, directory)
        self.assertEqual(len(list(shaders.glob('*.wgsl'))), 8)
        for path in shaders.glob('fused-*.wgsl'):
            old = (shaders / path.name.replace('fused-', 'plain-')).read_text()
            self.assertEqual(path.read_text(), old.replace('    dst[dst_idx] = sum;',
                             '    let result = sum / (1.0 + exp(-sum));\n    dst[dst_idx] = result;'))

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_full_cpp_normal_and_optional_profile_syntax(self):
        dawn = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parent / '.tools/emdawnwebgpu_pkg'))
        if not dawn.is_dir():
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        identity = json.loads((HERE / 'inputs.json').read_text())
        for name, expected in identity['dawn_header_sha256'].items():
            self.assertEqual(hashlib.sha256((dawn / name).read_bytes()).hexdigest(), expected)
        includes = [self.prepared, self.canonical, self.upstream / fusion.BASE, self.upstream / 'ggml/include', self.upstream / 'ggml/src']
        command = ['c++', '-std=c++20', '-fsyntax-only', '-D__wasi__', '-DGGML_SCHED_MAX_COPIES=4']
        command += ['-I' + str(path) for path in includes]
        command += ['-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include')]
        for flags in ([], ['-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE']):
            subprocess.run(command + ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'] + flags + [str(self.prepared / 'ggml-webgpu.cpp')], check=True)


if __name__ == '__main__':
    unittest.main()
