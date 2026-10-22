"""Source-bound expert workgroup selection; not browser/model inference."""
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
import prepare_webgpu_expert_short as expert
from test_webgpu_q8_short_overlay import function


def build_host(upstream, prepared, output):
    text = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
    original = (upstream / source.SSM_HEADER_PATH).read_text()
    here = ROOT / 'tests/webgpu-expert-short'
    factory_start = '    webgpu_pipeline get_mul_mat_id_vec_pipeline('
    factory_end = '    webgpu_pipeline get_unary_pipeline('
    chunks = [(here / 'host-prefix.cpp').read_text(),
              function(text, 'template <typename T> inline void ggml_webgpu_hash_combine('),
              function(text, 'struct ggml_webgpu_shader_lib_context {') + ';',
              function(text, 'inline uint32_t ggml_webgpu_expert_short_workgroup('),
              function(text, 'struct ggml_webgpu_mul_mat_vec_shader_decisions {') + ';',
              text[text.index('struct ggml_webgpu_mul_mat_id_pipeline_key {'):text.index('/** Cpy **/')]]
    for name, body in [('RecordingLibrary', text), ('ReferenceLibrary', original)]:
        chunks += ['class ' + name + ''' {
            int device = 0;
            pre_wgsl::Preprocessor preprocessor;
            std::unordered_map<ggml_webgpu_mul_mat_id_pipeline_key, webgpu_pipeline,
                ggml_webgpu_mul_mat_id_pipeline_key_hash> mul_mat_id_vec_pipelines;
            public:
        ''', body[body.index(factory_start):body.index(factory_end)], '};']
    cpp = (prepared / 'ggml-webgpu.cpp').read_text()
    chunks += [(here / 'dispatch-prefix.cpp').read_text(),
               function(cpp, 'static inline void compute_2d_workgroups('),
               function(cpp, 'static webgpu_encoded_op ggml_webgpu_mul_mat_id_vec('),
               (here / 'dispatch-cases.cpp').read_text()]
    chunks.append((here / 'host-cases.cpp').read_text())
    output.mkdir(parents=True, exist_ok=True)
    shader_dir = upstream / 'ggml/src/ggml-webgpu/wgsl-shaders'
    subprocess.run([sys.executable, str(shader_dir / 'embed_wgsl.py'), '--input_dir', str(shader_dir),
                    '--output_file', str(output / 'ggml-wgsl-shaders.hpp')], check=True)
    code = output / 'host.cpp'
    code.write_text('\n'.join(chunks))
    exe = output / 'host'
    subprocess.run(['g++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                    '-I' + str(upstream / 'ggml/include'), '-I' + str(upstream / 'ggml/src/ggml-webgpu'),
                    '-I' + str(output), str(code), '-o', str(exe)], check=True)
    return exe


class ExpertShortOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SSM_HEADER_PATH).is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def test_preparation_is_source_preserving_and_idempotent(self):
        before = {p: copy.digest(self.upstream / p) for p in expert.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source.prepare(self.upstream, out)
            for name in ('ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp'):
                self.assertEqual(copy.digest(out / name), expert.COMPOSITION_OUTPUTS['ggml/src/ggml-webgpu/' + name])
            stamp = (out / 'ggml-webgpu-shader-lib.hpp').stat().st_mtime_ns
            source.prepare(self.upstream, out)
            self.assertEqual(stamp, (out / 'ggml-webgpu-shader-lib.hpp').stat().st_mtime_ns)
            self.assertFalse((out / 'ggml-wgsl-shaders.hpp').exists())
        self.assertEqual(before, {p: copy.digest(self.upstream / p) for p in before})

    def test_contracts_and_composition_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            for group in ('CONTRACT_INPUTS', 'COMPOSITION_INPUTS', 'COMPOSITION_OUTPUTS'):
                values = getattr(expert, group)
                for name in values:
                    with self.subTest(group=group, name=name), patch.object(expert, group, {**values, name: '0' * 64}):
                        out = Path(tmp) / (group + '-' + str(list(values).index(name)))
                        with self.assertRaises(ValueError): source.prepare(self.upstream, out)
                        self.assertFalse((out / 'ggml-webgpu.cpp').exists())
                        self.assertFalse((out / 'ggml-webgpu-shader-lib.hpp').exists())
            with patch.object(expert, 'PATCH_SHA256', '0' * 64):
                with self.assertRaisesRegex(ValueError, 'patch identity'): source.prepare(self.upstream, Path(tmp) / 'patch')

    def test_build_reconfiguration_covers_every_contract(self):
        text = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        for path in {*expert.CONTRACT_INPUTS, *expert.COMPOSITION_INPUTS}:
            self.assertIn('${LCB_LLAMA_SOURCE}/' + path, text)
        self.assertIn(expert.PATCH_NAME, text)
        self.assertIn('scripts/prepare_webgpu_expert_short.py', text)

    def test_multitoken_factory_is_unchanged_and_key_defaults_to_zero(self):
        prepare_stage = expert.prepare
        signature = '    webgpu_pipeline get_mul_mat_id_pipeline('
        compared = []

        def inspect_stage(work, upstream, patch_root, env):
            header = work / source.SSM_HEADER_PATH
            def matrix_factory(text):
                start = text.index(signature)
                return text[start:text.index('    webgpu_pipeline get_mul_mat_id_vec_pipeline(', start)]

            before = matrix_factory(header.read_text())
            prepare_stage(work, upstream, patch_root, env)
            after = matrix_factory(header.read_text())
            self.assertEqual(before, after)
            self.assertIn('ggml_webgpu_mul_mat_id_pipeline_key key = {};', after)
            self.assertNotIn('key.workgroup_size =', after)
            compared.append(True)

        with tempfile.TemporaryDirectory() as tmp, patch.object(expert, 'prepare', inspect_stage):
            source.prepare(self.upstream, Path(tmp))
        self.assertEqual(compared, [True])

    def test_runner_rejects_invalid_commands_before_preparation(self):
        runner = ROOT / 'tests/webgpu-expert-short/check_gpu.py'
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
    def test_actual_selector_cache_factory_and_upstream_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source.prepare(self.upstream, out / 'prepared')
            exe = build_host(self.upstream, out / 'prepared', out / 'host')
            result = subprocess.run([str(exe), str(out / 'shaders')], check=True, capture_output=True, text=True)
            self.assertIn('actual expert selector/cache/factory checks passed', result.stdout)
            for quant in ('Q5_1', 'IQ4_NL'):
                for size in (128, 256):
                    shader = (out / 'shaders' / f'{quant}-{size}.wgsl').read_text()
                    self.assertIn(f'@workgroup_size({size})', shader)
                    self.assertNotIn('#include', shader)
            print(result.stdout.strip())


if __name__ == '__main__': unittest.main()
