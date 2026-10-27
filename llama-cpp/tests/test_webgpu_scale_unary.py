"""Source-bound graph/allocator/dispatch/factory checks; software GPU is separate."""
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
HERE = ROOT / 'tests/webgpu-scale-unary'
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
import prepare_webgpu_scale_unary as fusion


def function(text, marker):
    start = text.index(marker)
    return text[start:text.index('\n}', start) + 2]


def prepare(upstream, directory):
    source.prepare(upstream, directory)
    return directory


def emit_factory(upstream, prepared, directory):
    directory.mkdir(parents=True, exist_ok=True)
    factory = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    start = factory.index('    webgpu_pipeline get_scale_unary_pipeline(')
    actual = factory[start:factory.index('\n    }', start) + len('\n    }')]
    code = r'''
#include "pre_wgsl.hpp"
#include "ggml-webgpu-scale-unary-shaders.hpp"
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
struct webgpu_pipeline { unsigned id; std::string code, name; };
static unsigned made = 0;
static webgpu_pipeline ggml_webgpu_create_pipeline(int, const std::string & code, const std::string & label) {
    return {++made, code, label};
}
class Library {
    int device = 0;
    pre_wgsl::Preprocessor preprocessor;
    std::unordered_map<int, webgpu_pipeline> scale_unary_pipelines;
public:
''' + actual + r'''
};
static std::string contents(const char * path) {
    std::ifstream stream(path);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}
int main(int argc, char ** argv) {
    if (argc != 4) throw std::runtime_error("output, scale source, unary source required");
    Library library;
    std::set<unsigned> ids;
    for (bool sigmoid : {false, true}) for (bool final : {false, true}) for (bool inplace : {false, true}) {
        const int key = int(sigmoid) | (int(final) << 1) | (int(inplace) << 2);
        auto a = library.get_scale_unary_pipeline(sigmoid, final, inplace);
        auto b = library.get_scale_unary_pipeline(sigmoid, final, inplace);
        if (a.id != b.id || a.code.find('#') != std::string::npos || a.code.find("@workgroup_size(128)") == std::string::npos)
            throw std::runtime_error("actual factory/cache mismatch");
        ids.insert(a.id);
        std::ofstream(std::string(argv[1]) + "/fused-" + std::to_string(key) + ".wgsl") << a.code;
    }
    if (made != 8 || ids.size() != 8) throw std::runtime_error("incomplete pipeline key");
    pre_wgsl::Preprocessor preprocessor;
    for (bool inplace : {false, true}) {
        std::vector<std::string> defines{"TYPE_F32", "WG_SIZE=128"};
        if (inplace) defines.push_back("INPLACE");
        std::ofstream(std::string(argv[1]) + (inplace ? "/scale-inplace.wgsl" : "/scale.wgsl"))
            << preprocessor.preprocess(contents(argv[2]), defines);
        for (const std::string op : {"SILU", "SIGMOID"}) {
            auto unary_defines = defines; unary_defines.push_back(op);
            std::ofstream(std::string(argv[1]) + "/" + op + (inplace ? "-inplace.wgsl" : ".wgsl"))
                << preprocessor.preprocess(contents(argv[3]), unary_defines);
        }
    }
    std::cout << "8 actual factory/cache variants and 6 upstream reference shaders emitted\n";
}
'''
    cpp = directory / 'factory.cpp'
    cpp.write_text(code)
    executable = directory / 'factory'
    subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror', '-I' + str(prepared),
                    '-isystem', str(upstream / fusion.BASE), str(cpp), '-o', str(executable)], check=True)
    shaders = directory / 'shaders'
    shaders.mkdir()
    subprocess.run([str(executable), str(shaders), str(upstream / fusion.BASE / 'wgsl-shaders/scale.wgsl'),
                    str(upstream / fusion.BASE / 'wgsl-shaders/unary.wgsl')], check=True)
    return shaders


class ScaleUnary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        cls.temporary = tempfile.TemporaryDirectory(prefix='scale-unary-tests-')
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

    def test_checked_composition_and_upstream_preservation(self):
        self.assertEqual(hashlib.sha256((self.prepared / 'ggml-webgpu.cpp').read_bytes()).hexdigest(), fusion.OUTPUTS['ggml-webgpu.cpp'])
        self.assertEqual(hashlib.sha256((self.prepared / 'ggml-webgpu-scale-unary.hpp').read_bytes()).hexdigest(), fusion.OUTPUTS['ggml-webgpu-scale-unary.hpp'])
        with tempfile.TemporaryDirectory() as temporary, patch.object(source.q8_subgroup, 'prepare', return_value={}), \
                patch.object(source, 'SSM_HEADER_OUTPUT_SHA256', fusion.OUTPUTS['ggml-webgpu-shader-lib.hpp']):
            stage = source.prepare(self.upstream, Path(temporary)).parent
            self.assertEqual(hashlib.sha256((stage / 'ggml-webgpu-shader-lib.hpp').read_bytes()).hexdigest(), fusion.OUTPUTS['ggml-webgpu-shader-lib.hpp'])
        self.assertEqual(hashlib.sha256((self.prepared / fusion.HEADER_NAME).read_bytes()).hexdigest(), fusion.HEADER_SHA256)
        for name, expected in fusion.CONTRACT_INPUTS.items():
            self.assertEqual(hashlib.sha256((self.upstream / name).read_bytes()).hexdigest(), expected)
        self.assertFalse((self.prepared / 'ggml-wgsl-shaders.hpp').exists())

    def test_changed_contract_patch_outputs_and_embedding_fail_closed(self):
        for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS'):
            name = next(iter(getattr(fusion, field)))
            with self.subTest(field=field), patch.object(fusion, field, {**getattr(fusion, field), name: '0' * 64}):
                with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                    prepare(self.upstream, Path(temporary))
        for field in ('PATCH_SHA256', 'HEADER_SHA256'):
            with self.subTest(field=field), patch.object(fusion, field, '0' * 64):
                with tempfile.TemporaryDirectory() as temporary, self.assertRaises(ValueError):
                    prepare(self.upstream, Path(temporary))

    @unittest.skipUnless(shutil.which('c++') and shutil.which('cmake'), 'native build tools required')
    def test_actual_graph_fusion_and_allocator_source_lifetime(self):
        self.native(HERE / 'admission.cpp', self.directory / 'admission')

    @unittest.skipUnless(shutil.which('c++') and shutil.which('cmake'), 'native build tools required')
    def test_actual_dispatch_guards_params_and_fallback(self):
        actual = (self.prepared / 'ggml-webgpu.cpp').read_text()
        markers = ['static inline uint32_t ggml_webgpu_u32_from_f32(', 'static void * ggml_backend_webgpu_buffer_get_base(',
                   'static size_t ggml_webgpu_tensor_offset(', 'static wgpu::BindGroupEntry ggml_webgpu_make_bind_group_entry(']
        # Select the definition rather than the narrow forward declaration.
        functions = []
        for marker in markers:
            position = actual.rindex(marker) if 'get_base' in marker else actual.index(marker)
            functions.append(function(actual[position:], marker))
        start = actual.index('struct webgpu_scale_unary_binding {')
        functions.append(actual[start:actual.index('\n};', start) + 3])
        functions += [function(actual, marker) for marker in (
            'static bool ggml_webgpu_scale_unary_binding(', 'static std::optional<webgpu_encoded_op> ggml_webgpu_scale_unary(')]
        code = (HERE / 'dispatch-prefix.cpp').read_text().replace('// @ACTUAL_FUNCTIONS@', '\n'.join(functions))
        cpp = self.directory / 'dispatch.cpp'; cpp.write_text(code + '\n' + (HERE / 'dispatch-cases.cpp').read_text())
        self.native(cpp, self.directory / 'dispatch')

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_eight_pipeline_keys_and_upstream_shader_references(self):
        shaders = emit_factory(self.upstream, self.prepared, self.directory / 'factory')
        self.assertEqual(len(list(shaders.glob('*.wgsl'))), 14)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_complete_composed_source_with_pinned_dawn_headers(self):
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


if __name__ == '__main__':
    unittest.main()
