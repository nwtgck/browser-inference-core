"""Source-bound expert workgroup selection; not browser/model inference."""
from pathlib import Path
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path.insert(0, str(ROOT.parents[1]))
from build_tools.runtimes.llama_cpp import source as source
# Historical byte identities belong to the reference fixture, not production.
from tests.support import legacy_webgpu_source as reference_source
from tests.support.llama_cpp.webgpu_stages import tensor_copy as copy
from tests.support.llama_cpp.webgpu_stages import expert_short as expert
from tests.runtimes.llama_cpp.test_webgpu_q8_short_overlay import function

from tests.runtimes.llama_cpp.matmul_vec4_test_support import explicit_layout_argument


from tests.support import legacy_webgpu_source as legacy_source

def build_host(upstream, prepared, output):
    text = explicit_layout_argument((prepared / 'ggml-webgpu-shader-lib.hpp').read_text())
    original = explicit_layout_argument((upstream / reference_source.SSM_HEADER_PATH).read_text())
    here = runtime_test_root('llama-cpp') / 'webgpu-expert-short'
    factory_start = '    webgpu_pipeline get_mul_mat_id_vec_pipeline('
    factory_end = '    webgpu_pipeline get_unary_pipeline('
    chunks = [(here / 'host-prefix.cpp').read_text(), '#include "ggml-webgpu-iq3s-id-subgroup-shaders.hpp"',
              function(text, 'template <typename T> inline void ggml_webgpu_hash_combine('),
              function(text, 'struct ggml_webgpu_shader_lib_context {') + ';',
              function(text, 'inline uint32_t ggml_webgpu_expert_short_workgroup('),
              function(text, 'inline uint32_t ggml_webgpu_q4_short_workgroup('),
              function(text, 'inline uint32_t ggml_webgpu_iq3s_id_subgroup_size('),
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
               function(cpp, 'static bool ggml_webgpu_mul_mat_vec4_aligned('),
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
                    '-I' + str(prepared), '-I' + str(output), str(code), '-o', str(exe)], check=True)
    return exe


class ExpertShortOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / reference_source.SSM_HEADER_PATH).is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def test_preparation_is_source_preserving_and_idempotent(self):
        before = {p: copy.digest(self.upstream / p) for p in expert.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            source.prepare(self.upstream, out)
            for name in ('ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp'):
                expected = (reference_source.FINAL_OUTPUTS[False]['ggml-webgpu.cpp'] if name == 'ggml-webgpu.cpp'
                            else reference_source.FINAL_OUTPUTS[False]['ggml-webgpu-shader-lib.hpp'])
                self.assertEqual(copy.digest(out / name), expected)
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
                        with self.assertRaises(ValueError): legacy_source.prepare(self.upstream, out)
                        self.assertFalse((out / 'ggml-webgpu.cpp').exists())
                        self.assertFalse((out / 'ggml-webgpu-shader-lib.hpp').exists())
            with patch.object(expert, 'PATCH_SHA256', '0' * 64):
                with self.assertRaisesRegex(ValueError, 'patch identity'): legacy_source.prepare(self.upstream, Path(tmp) / 'patch')


    def test_multitoken_factory_is_unchanged_and_key_defaults_to_zero(self):
        prepare_stage = expert.prepare
        signature = '    webgpu_pipeline get_mul_mat_id_pipeline('
        compared = []

        def inspect_stage(work, upstream, patch_root, env):
            header = work / reference_source.SSM_HEADER_PATH
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
            legacy_source.prepare(self.upstream, Path(tmp))
        self.assertEqual(compared, [True])

    def test_runner_rejects_invalid_commands_before_preparation(self):
        runner = runtime_test_root('llama-cpp') / 'webgpu-expert-short/check_gpu.py'
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
