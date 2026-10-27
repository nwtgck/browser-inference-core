"""Source-bound Q4_K workgroup checks; no browser/model inference claim."""
from pathlib import Path
import importlib.util
import struct
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
import prepare_webgpu_q4_short as q4
from test_webgpu_q8_short_overlay import function


def prepare(upstream, output):
    source.prepare(upstream, output)


def build_host(upstream, prepared, output):
    text = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    original = (upstream / q4.HEADER_PATH).read_text()
    here = ROOT / 'tests/webgpu-q4-short'
    chunks = [(here / 'host-prefix.cpp').read_text(), '#include "ggml-webgpu-q8-subgroup-shaders.hpp"\n#include "ggml-webgpu-iq3s-id-subgroup-shaders.hpp"',
              function(text, 'template <typename T> inline void ggml_webgpu_hash_combine('),
              function(text, 'struct ggml_webgpu_shader_lib_context {') + ';']
    for name in ('q8', 'expert', 'q4'):
        chunks.append(function(text, 'inline uint32_t ggml_webgpu_' + name + '_short_workgroup('))
    chunks.append(function(text, 'inline uint32_t ggml_webgpu_q8_matvec_subgroup_size('))
    chunks.append(function(text, 'inline uint32_t ggml_webgpu_iq3s_id_subgroup_size('))
    chunks += [text[text.index('struct ggml_webgpu_mul_mat_vec_pipeline_key {'):text.index('struct ggml_webgpu_quantize_q8_pipeline_key {')],
               text[text.index('struct ggml_webgpu_mul_mat_id_pipeline_key {'):text.index('/** Cpy **/')],
               function(text, 'inline bool ggml_webgpu_can_use_mmvq(')]
    for name, body in [('RecordingLibrary', text), ('ReferenceLibrary', original)]:
        chunks += ['class ' + name + ''' {
            int device = 0;
            pre_wgsl::Preprocessor preprocessor;
            std::unordered_map<ggml_webgpu_mul_mat_vec_pipeline_key, webgpu_pipeline,
                ggml_webgpu_mul_mat_vec_pipeline_key_hash> mul_mat_vec_pipelines;
            std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_key, webgpu_pipeline,
                ggml_webgpu_mul_mat_id_pipeline_key_hash> mul_mat_id_vec_pipelines;
            public:
        ''']
        for start, end in [('    webgpu_pipeline get_mul_mat_vec_pipeline(', '    webgpu_pipeline get_mul_mat_fast_pipeline('),
                           ('    webgpu_pipeline get_mul_mat_id_vec_pipeline(', '    webgpu_pipeline get_unary_pipeline(')]:
            chunks.append(body[body.index(start):body.index(end)])
        chunks += ['webgpu_pipeline get_mul_mat_fast_pipeline(const ggml_webgpu_shader_lib_context &, bool) {throw std::runtime_error("unexpected matrix path");}', '};']
    cpp = (prepared / 'ggml-webgpu.cpp').read_text()
    chunks += [(here / 'dispatch-prefix.cpp').read_text(),
               function(cpp, 'static inline void compute_2d_workgroups('),
               function(cpp, 'static webgpu_encoded_op ggml_webgpu_mul_mat_id_vec('),
               function(cpp, 'static webgpu_encoded_op ggml_webgpu_mul_mat('),
               (here / 'dispatch-cases.cpp').read_text(),
               (here / 'host-cases.cpp').read_text()]
    output.mkdir(parents=True, exist_ok=True)
    shader_dir = upstream / 'ggml/src/ggml-webgpu/wgsl-shaders'
    subprocess.run([sys.executable, str(shader_dir / 'embed_wgsl.py'), '--input_dir', str(shader_dir),
                    '--output_file', str(output / 'ggml-wgsl-shaders.hpp')], check=True)
    code = output / 'host.cpp'
    code.write_text('\n'.join(chunks))
    exe = output / 'host'
    subprocess.run(['g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(upstream / 'ggml/include'), '-I' + str(upstream / 'ggml/src/ggml-webgpu'),
                    '-I' + str(prepared),
                    '-I' + str(output), str(code), '-o', str(exe)], check=True)
    return exe


class Q4ShortOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / q4.HEADER_PATH).is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def test_preparation_preserves_upstream_and_shader_files(self):
        before = {p: copy.digest(self.upstream / p) for p in q4.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            prepare(self.upstream, out)
            self.assertEqual(copy.digest(out / 'ggml-webgpu-shader-lib.hpp'), source.SSM_HEADER_OUTPUT_SHA256)
            self.assertFalse((out / 'ggml-wgsl-shaders.hpp').exists())
            stamp = (out / 'ggml-webgpu-shader-lib.hpp').stat().st_mtime_ns
            prepare(self.upstream, out)
            self.assertEqual(stamp, (out / 'ggml-webgpu-shader-lib.hpp').stat().st_mtime_ns)
        self.assertEqual(before, {p: copy.digest(self.upstream / p) for p in before})

    def test_stage_contracts_fail_closed(self):
        for guard in ('CONTRACT_INPUTS', 'INPUT_SHA256', 'OUTPUT_SHA256', 'PATCH_SHA256'):
            values = getattr(q4, guard)
            cases = ({**values, name: '0' * 64} for name in values) if isinstance(values, dict) else ['0' * 64]
            for value in cases:
                with self.subTest(guard=guard, value=value), tempfile.TemporaryDirectory() as tmp, patch.object(q4, guard, value):
                    with self.assertRaises(ValueError): prepare(self.upstream, Path(tmp))
                    self.assertFalse((Path(tmp) / 'ggml-webgpu.cpp').exists())
                    self.assertFalse((Path(tmp) / 'ggml-webgpu-shader-lib.hpp').exists())

    def test_build_inputs_and_multitoken_matrix_factory_remain_source_bound(self):
        text = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn(q4.PATCH_NAME, text)
        self.assertIn('scripts/prepare_webgpu_q4_short.py', text)
        for path in q4.CONTRACT_INPUTS:
            self.assertIn('${LCB_LLAMA_SOURCE}/' + path, text)
        self.assertEqual(source.scale_unary.INPUTS['ggml-webgpu-shader-lib.hpp'], q4.OUTPUT_SHA256)
        with tempfile.TemporaryDirectory() as tmp:
            prepare(self.upstream, Path(tmp))
            prepared = (Path(tmp) / 'ggml-webgpu-shader-lib.hpp').read_text()
        original = (self.upstream / q4.HEADER_PATH).read_text()
        def matrix_factory(body):
            start = body.index('    webgpu_pipeline get_mul_mat_id_pipeline(')
            return body[start:body.index('    webgpu_pipeline get_mul_mat_id_vec_pipeline(', start)]
        # The later matrix-only group overlay changes this shader selection.
        self.assertEqual(matrix_factory(prepared), matrix_factory(original).replace(
            'preprocessor.preprocess(wgsl_mul_mat_id, defines)',
            'preprocessor.preprocess(wgsl_mul_mat_id_group, defines)'))

    def test_runner_rejects_invalid_commands_and_overlapping_work(self):
        runner = ROOT / 'tests/webgpu-q4-short/check_gpu.py'
        with tempfile.TemporaryDirectory() as tmp:
            for value in ('invalid', '{}', '[]', '[1]', '["", "node"]', '"node"'):
                with self.subTest(value=value):
                    result = subprocess.run([sys.executable, str(runner), '--llama-source', str(self.upstream),
                        '--work', tmp, '--webgpu-module', 'unused', '--node-command-json', value],
                        capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('--node-command-json', result.stderr)
                    self.assertFalse((Path(tmp) / 'prepared').exists())
            result = subprocess.run([sys.executable, str(runner), '--llama-source', str(self.upstream),
                '--work', str(self.upstream), '--webgpu-module', 'unused'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('must not overlap', result.stderr)

    @unittest.skipUnless(shutil.which('g++'), 'C++ compiler required')
    def test_actual_upstream_dequantizer_reference_known_blocks_and_contract_guards(self):
        spec = importlib.util.spec_from_file_location('q4_gpu_checker', ROOT / 'tests/webgpu-q4-short/check_gpu.py')
        checker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(checker)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            for path in checker.REFERENCE_INPUTS:
                with self.subTest(path=path), patch.object(checker, 'REFERENCE_INPUTS', {
                        **checker.REFERENCE_INPUTS, path: '0' * 64}):
                    with self.assertRaisesRegex(ValueError, 'CPU reference contract changed'):
                        checker.build_reference(self.upstream, out)
            exe = checker.build_reference(self.upstream, out)
            params = [0,0,0,1,1,256,1,256,1,256,1,256,1,1,1,1]
            header = struct.pack('<5I', 0,144,256,0,16)
            for scales, qs, d, dmin, expected in [
                ([1]*4+[0]*4+[1]*4, 0x21, 0x3c00, 0, 384.0),
                ([255]*12, 255, 0x2c00, 0x2800, 14616.0),
                ([0]*12, 0, 0x3c00, 0, 0.0),
            ]:
                weights = struct.pack('<2H', d,dmin) + bytes(scales) + bytes([qs]*128)
                payload = header + weights + struct.pack('<256f', *([1.0]*256)) + struct.pack('<16I', *params)
                result = subprocess.run([str(exe)], input=payload, check=True, capture_output=True)
                self.assertEqual(struct.unpack('<2d', result.stdout), (expected, expected))
            result = subprocess.run([str(exe)], input=header, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'truncated fixture', result.stderr)

    @unittest.skipUnless(shutil.which('g++'), 'C++ compiler required')
    def test_actual_selector_cache_factory_dispatch_and_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            prepare(self.upstream, out / 'prepared')
            exe = build_host(self.upstream, out / 'prepared', out / 'host')
            result = subprocess.run([str(exe), str(out / 'shaders')], check=True, capture_output=True, text=True)
            self.assertIn('actual Q4 selector/cache/factory checks passed', result.stdout)
            for kind in ('ordinary', 'indexed'):
                for size in (32, 64, 128, 256):
                    shader = (out / 'shaders' / f'{kind}-{size}.wgsl').read_text()
                    self.assertIn(f'@workgroup_size({size})', shader)
                    self.assertNotIn('#include', shader)
            print(result.stdout.strip())


if __name__ == '__main__': unittest.main()
