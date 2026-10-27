"""Source-bound short-Q8 selection tests; no GPU/model inference is claimed."""
from pathlib import Path
import os
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
import prepare_webgpu_q8_short as q8


def function(text, marker):
    start = text.index(marker)
    return text[start:text.index('\n}', start) + 2]


def build_host(upstream, prepared, output):
    text = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    here = ROOT / 'tests/webgpu-q8-short'
    chunks = [
        (here / 'host-prefix.cpp').read_text(),
        '#include "ggml-webgpu-q8-subgroup-shaders.hpp"',
        function(text, 'template <typename T> inline void ggml_webgpu_hash_combine('),
        function(text, 'struct ggml_webgpu_shader_lib_context {') + ';',
        function(text, 'inline uint32_t ggml_webgpu_q8_short_workgroup('),
        function(text, 'inline uint32_t ggml_webgpu_q4_short_workgroup('),
        function(text, 'inline uint32_t ggml_webgpu_q8_matvec_subgroup_size('),
        text[text.index('struct ggml_webgpu_mul_mat_vec_pipeline_key {'):
             text.index('struct ggml_webgpu_quantize_q8_pipeline_key {')],
        function(text, 'inline bool ggml_webgpu_can_use_mmvq('),
        '''class RecordingLibrary {
            int device = 0;
            pre_wgsl::Preprocessor preprocessor;
            std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_key,webgpu_pipeline,
                               ggml_webgpu_mul_mat_vec_pipeline_key_hash> mul_mat_vec_pipelines;
            public:
        ''',
        text[text.index('    webgpu_pipeline get_mul_mat_vec_pipeline('):
             text.index('    webgpu_pipeline get_mul_mat_fast_pipeline(')],
        '};',
        (here / 'host-cases.cpp').read_text(),
    ]
    output.mkdir(parents=True, exist_ok=True)
    shader_dir = upstream / 'ggml/src/ggml-webgpu/wgsl-shaders'
    subprocess.run([sys.executable, str(shader_dir / 'embed_wgsl.py'), '--input_dir', str(shader_dir),
                    '--output_file', str(output / 'ggml-wgsl-shaders.hpp')], check=True)
    code = output / 'host.cpp'
    code.write_text('\n'.join(chunks))
    exe = output / 'host'
    subprocess.run(['g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(upstream / 'ggml/include'),
                    '-I' + str(upstream / 'ggml/src/ggml-webgpu'),
                    '-I' + str(prepared),
                    '-I' + str(output), str(code), '-o', str(exe)], check=True)
    return exe


class Q8ShortOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / q8.HEADER_PATH).is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def test_preparation_does_not_copy_shaders_or_modify_upstream(self):
        before = {p: copy.digest(self.upstream / p) for p in q8.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source.prepare(self.upstream, out)
            header = out / 'ggml-webgpu-shader-lib.hpp'
            stamp = header.stat().st_mtime_ns
            self.assertEqual(copy.digest(header), source.SSM_HEADER_OUTPUT_SHA256)
            self.assertFalse((out / 'ggml-wgsl-shaders.hpp').exists())
            source.prepare(self.upstream, out)
            self.assertEqual(header.stat().st_mtime_ns, stamp)
        self.assertEqual(before, {p: copy.digest(self.upstream / p) for p in before})

    def test_all_preparation_contracts_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'out'
            for name in q8.CONTRACT_INPUTS:
                with self.subTest(path=name), patch.object(q8, 'CONTRACT_INPUTS', {
                    **q8.CONTRACT_INPUTS, name: '0' * 64,
                }):
                    with self.assertRaisesRegex(ValueError, 'Q8 short matvec contract changed'):
                        source.prepare(self.upstream, out)
                self.assertFalse((out / 'ggml-webgpu-shader-lib.hpp').exists())
            for name in ['INPUT_SHA256', 'OUTPUT_SHA256', 'PATCH_SHA256']:
                with self.subTest(guard=name), patch.object(q8, name, '0' * 64):
                    with self.assertRaises(ValueError):
                        source.prepare(self.upstream, out)
                self.assertFalse((out / 'ggml-webgpu.cpp').exists())
                self.assertFalse((out / 'ggml-webgpu-shader-lib.hpp').exists())

    def test_every_contract_is_a_cmake_reconfiguration_dependency(self):
        text = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        for name in q8.CONTRACT_INPUTS:
            self.assertIn('${LCB_LLAMA_SOURCE}/' + name, text)
        self.assertIn(q8.PATCH_NAME, text)
        self.assertIn('scripts/prepare_webgpu_q8_short.py', text)

    def test_checker_rejects_invalid_node_commands_before_preparing_or_running(self):
        script = ROOT / 'tests/webgpu-q8-short/check_gpu.py'
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not-created'
            for command in ['[]', 'null', '{}', '"node"', '[""]', '["node", 7]', '[invalid']:
                with self.subTest(command=command):
                    result = subprocess.run([
                        sys.executable, str(script), '--llama-source', str(out),
                        '--work', str(out), '--webgpu-module', str(out),
                        '--node-command-json', command,
                    ], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('--node-command-json must', result.stderr)
                    self.assertFalse(out.exists())

    @unittest.skipUnless(shutil.which('g++'), 'C++ compiler required')
    def test_actual_selector_cache_factory_and_preprocessor(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source.prepare(self.upstream, out / 'prepared')
            exe = build_host(self.upstream, out / 'prepared', out / 'host')
            result = subprocess.run([str(exe), str(out / 'emitted')], check=True,
                                    capture_output=True, text=True)
            self.assertIn('actual selector/cache/factory checks passed', result.stdout)
            for size in (32, 64, 128, 256):
                shader = (out / 'emitted' / f'q8-{size}.wgsl').read_text()
                self.assertIn(f'@workgroup_size({size})', shader)
                self.assertNotIn('#include', shader)
            print(result.stdout.strip())


if __name__ == '__main__':
    unittest.main()
