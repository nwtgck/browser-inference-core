"""Checked RMS/SCALE graph, view, allocator, binding and factory contracts."""
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
HERE = ROOT / 'tests/webgpu-rms-scale'
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
import prepare_webgpu_rms_scale as fusion


def function(text, marker):
    start = text.index(marker)
    return text[start:text.index('\n}', start) + 2]


def prepare(upstream, directory):
    source.prepare(upstream, directory)
    return directory


def emit_factory(upstream, prepared, directory, reference_scale_wg_size=128):
    directory.mkdir(parents=True, exist_ok=True)
    text = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    start = text.index('    webgpu_pipeline get_rms_scale_pipeline(')
    actual = text[start:text.index('\n    }', start) + len('\n    }')]
    code = (HERE / 'factory-prefix.cpp').read_text() + actual + (HERE / 'factory-cases.cpp').read_text()
    cpp = directory / 'factory.cpp'
    cpp.write_text(code)
    executable = directory / 'factory'
    subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-I' + str(prepared),
                    '-isystem', str(upstream / fusion.BASE), str(cpp), '-o', str(executable)], check=True)
    shaders = directory / 'shaders'
    shaders.mkdir()
    subprocess.run([str(executable), str(shaders), str(upstream / fusion.BASE / 'wgsl-shaders/row_norm.wgsl'),
                    str(upstream / fusion.BASE / 'wgsl-shaders/scale.wgsl'), str(reference_scale_wg_size)], check=True)
    return shaders


class RmsScale(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        cls.temporary = tempfile.TemporaryDirectory(prefix='rms-scale-tests-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.prepared = prepare(cls.upstream, cls.directory / 'prepared')
        cls.library = None

    @classmethod
    def native_library(cls):
        if cls.library:
            return cls.library
        configured = os.environ.get('LCB_TEST_GGML_BASE_LIBRARY')
        if configured:
            cls.library = Path(configured).resolve()
            return cls.library
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
                        '-I' + str(self.prepared), '-isystem', str(self.upstream / 'ggml/include'),
                        '-isystem', str(self.upstream / 'ggml/src'), str(cpp), str(self.native_library()),
                        '-pthread', '-ldl', '-lm', '-o', str(destination)], check=True)
        subprocess.run([str(destination)], check=True)

    def test_checked_composition_upstream_and_exact_reduction_source(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(source.async_tensor_copy, 'prepare'), \
                patch.object(source.iq3s_subgroup, 'prepare', return_value={}), \
                patch.object(source.host_input_snapshot, 'prepare', return_value={}), \
                patch.object(source.ssm_conv_silu, 'prepare', return_value={}), \
                patch.object(source.moe_group_skip, 'prepare', return_value={}), \
                patch.object(source.router_tail, 'prepare', return_value={}), \
                patch.object(source.host_encode_assembly, 'prepare'), \
                patch.object(source.f32_small_m, 'prepare'), \
                patch.object(source, 'OUTPUT_SHA256', fusion.OUTPUTS['ggml-webgpu.cpp']), \
                patch.object(source, 'SSM_HEADER_OUTPUT_SHA256', fusion.OUTPUTS['ggml-webgpu-shader-lib.hpp']):
            stage = source.prepare(self.upstream, Path(temporary)).parent
            for name in ('ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp', 'ggml-webgpu-rms-scale.hpp'):
                self.assertEqual(hashlib.sha256((stage / name).read_bytes()).hexdigest(), fusion.OUTPUTS[name])
        self.assertEqual(hashlib.sha256((self.prepared / fusion.HEADER_NAME).read_bytes()).hexdigest(), fusion.HEADER_SHA256)
        for name, expected in fusion.CONTRACT_INPUTS.items():
            self.assertEqual(hashlib.sha256((self.upstream / name).read_bytes()).hexdigest(), expected)
        self.assertFalse((self.prepared / 'ggml-wgsl-shaders.hpp').exists())
        old = (self.upstream / fusion.BASE / 'wgsl-shaders/row_norm.wgsl').read_text()
        embedded = (self.prepared / fusion.HEADER_NAME).read_text()
        new = embedded.split('R"wgsl(', 1)[1].split(')wgsl"', 1)[0]
        self.assertEqual(hashlib.sha256(new.encode()).hexdigest(), fusion.OUTPUTS['wgsl-shaders/row_norm.wgsl'])
        begin = old.index('    var sum =')
        end = old.index('        let v = src[i_src];')
        self.assertEqual(old[begin:end], new[new.index('    var sum ='):new.index('        let v = src[i_src];')])
        self.assertIn('let normalized = scale * (v - mean_val);', new)
        self.assertIn('let result = normalized * params.post_scale + params.post_bias;', new)

    def test_all_contract_patch_outputs_and_embedding_guards(self):
        for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS'):
            values = getattr(fusion, field)
            for name in values:
                with self.subTest(field=field, path=name), patch.object(fusion, field, {**values, name: '0' * 64}):
                    with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                        prepare(self.upstream, Path(temporary))
        for field in ('PATCH_SHA256', 'HEADER_SHA256'):
            with self.subTest(field=field), patch.object(fusion, field, '0' * 64):
                with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                    prepare(self.upstream, Path(temporary))

    def test_gpu_checker_rejects_invalid_commands_before_preparing(self):
        script = HERE / 'check_gpu.py'
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'missing'
            for command in ('[]', 'null', '{}', '"node"', '[""]', '["node",7]', '[invalid'):
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, str(script), '--source', str(out), '--work', str(out),
                                             '--webgpu-module', str(out), '--node-command-json', command],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('--node-command-json must', result.stderr)
                    self.assertFalse(out.exists())

            for reference_wg in ('0', '1025', 'bad'):
                with self.subTest(reference_wg=reference_wg):
                    result = subprocess.run([sys.executable, str(script), '--source', str(out), '--work', str(out),
                                             '--webgpu-module', str(out), '--reference-scale-wg-size', reference_wg],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('--reference-scale-wg-size', result.stderr)
                    self.assertFalse(out.exists())

    @unittest.skipUnless(shutil.which('c++') and shutil.which('cmake'), 'native build tools required')
    def test_actual_graph_views_and_allocator_source_lifetime(self):
        self.native(HERE / 'admission.cpp', self.directory / 'admission')

    @unittest.skipUnless(shutil.which('c++') and shutil.which('cmake'), 'native build tools required')
    def test_actual_dispatch_binding_guards_params_and_fallback(self):
        text = (self.prepared / 'ggml-webgpu.cpp').read_text()
        functions = []
        for marker in ('static inline uint32_t ggml_webgpu_u32_from_f32(', 'static void * ggml_backend_webgpu_buffer_get_base(',
                       'static size_t ggml_webgpu_tensor_offset(', 'static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry('):
            position = text.rindex(marker) if 'get_base' in marker else text.index(marker)
            functions.append(function(text[position:], marker))
        start = text.index('struct webgpu_rms_scale_binding {')
        functions.append(text[start:text.index('\n};', start) + 3])
        functions += [function(text, marker) for marker in ('static bool ggml_webgpu_rms_scale_binding(',
                                                          'static std::optional<webgpu_encoded_op> ggml_webgpu_rms_scale(')]
        prefix = (HERE / 'dispatch-prefix.cpp').read_text().replace('// @ACTUAL_FUNCTIONS@', '\n'.join(functions))
        cpp = self.directory / 'dispatch.cpp'
        cpp.write_text(prefix + '\n' + (HERE / 'fixture.cpp').read_text() + '\n' + (HERE / 'dispatch-cases.cpp').read_text())
        self.native(cpp, self.directory / 'dispatch')

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_pipeline_cache_variants_and_staged_references(self):
        shaders = emit_factory(self.upstream, self.prepared, self.directory / 'factory')
        self.assertEqual(len(list(shaders.glob('*.wgsl'))), 6)
        for path in shaders.glob('fused-*.wgsl'):
            text = path.read_text()
            self.assertNotIn('pow(', text)
            self.assertIn('sum += v * v;', text)
            self.assertIn('@workgroup_size(128)', text)
            self.assertIn('let normalized = scale * (v - mean_val);', text)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_complete_source_with_pinned_dawn_headers(self):
        dawn = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parent / '.tools/emdawnwebgpu_pkg'))
        if not dawn.is_dir(): self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        identity = json.loads((HERE / 'inputs.json').read_text())
        for name, expected in identity['dawn_header_sha256'].items():
            self.assertEqual(hashlib.sha256((dawn / name).read_bytes()).hexdigest(), expected)
        subprocess.run([sys.executable, str(self.upstream / fusion.BASE / 'wgsl-shaders/embed_wgsl.py'),
                        '--input_dir', str(self.upstream / fusion.BASE / 'wgsl-shaders'), '--output_file',
                        str(self.directory / 'ggml-wgsl-shaders.hpp')], check=True)
        includes = [self.prepared, self.directory, self.upstream / fusion.BASE, self.upstream / 'ggml/include', self.upstream / 'ggml/src']
        command = ['c++', '-std=c++20', '-fsyntax-only', '-D__wasi__', '-DGGML_SCHED_MAX_COPIES=4']
        command += ['-I' + str(path) for path in includes]
        command += ['-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include')]
        for macros in (['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'], ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS', '-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE']):
            subprocess.run(command + macros + [str(self.prepared / 'ggml-webgpu.cpp')], check=True)


if __name__ == '__main__': unittest.main()
