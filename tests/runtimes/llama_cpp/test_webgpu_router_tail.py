"""Source-bound router tail graph, allocator, binding and actual factory checks."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
HERE = runtime_test_root('llama-cpp') / 'webgpu-router-tail'
sys.path[:0] = [str(ROOT.parents[1]), str(HERE)]
from build_tools.runtimes.llama_cpp import source as source
# Historical byte identities belong to the reference fixture, not production.
from tests.support import legacy_webgpu_source as reference_source
from tests.support.llama_cpp.webgpu_stages import router_tail as router
import router_factory


from tests.support import legacy_webgpu_source as legacy_source

class RouterTail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temporary = tempfile.TemporaryDirectory(prefix='router-tail-tests-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.normal = source.prepare(cls.upstream, cls.directory / 'normal').parent
        cls.profile = source.prepare(cls.upstream, cls.directory / 'profile', gpu_profile=True).parent

    def test_canonical_sum_rows_uniform_and_2d_guard_match_gpu_baseline(self):
        shader = (self.upstream / router.BASE / 'wgsl-shaders/sum_rows.wgsl').read_text()
        params = re.search(r'struct Params \{(.*?)\};', shader, re.S).group(1)
        fields = re.findall(r'\b(\w+)\s*:\s*u32', params)
        self.assertEqual(fields, ['offset_src', 'offset_dst', 'stride_src1', 'stride_src2',
                                  'stride_src3', 'ne0', 'ne1', 'ne2', 'nrows'])
        identity = json.loads((HERE / 'inputs.json').read_text())
        self.assertEqual(identity['sum_rows_source_sha256'], hashlib.sha256(shader.encode()).hexdigest())
        runner = (HERE / 'run_gpu.mjs').read_text()
        values = re.search(r'sum_rows: uniform\(\[([^]]+)\]', runner).group(1).split(', ')
        self.assertEqual(values[8:], ['1', '0', '0', '0'])
        self.assertEqual(len(values) * 4 % 16, 0)
        # Evaluate the actual shader's row and guard expressions across a padded
        # 2D grid. This is source arithmetic, not WGSL/GPU numerical execution.
        row_expression = re.search(r'let row = ([^;]+);', shader).group(1)
        guard = re.search(r'if \((row >= params.nrows)\) \{\s*return;', shader).group(1)
        rows, skipped = [], []
        for y in range(2):
            for x in range(3):
                context = {'wid': SimpleNamespace(x=x, y=y), 'num_wg': SimpleNamespace(x=3),
                           'params': SimpleNamespace(nrows=5)}
                context['row'] = eval(row_expression, {'__builtins__': {}}, context)
                (skipped if eval(guard, {'__builtins__': {}}, context) else rows).append(context['row'])
        self.assertEqual(rows, [0, 1, 2, 3, 4])
        self.assertEqual(skipped, [5])
        self.assertIn('dst[params.offset_dst + row] = shared_sum[0];', shader)
        self.assertIn('dst[params.offset_dst + row] = shared_sum[0] / f32(params.ne0);', shader)

    def test_checked_stage_before_profile_and_exact_shader_headers(self):
        self.assertEqual(reference_source.host_encode_assembly.INPUT_SHA256, router.OUTPUTS['ggml-webgpu.cpp'])
        self.assertEqual(reference_source.tensor_copy.digest(self.normal / 'ggml-webgpu.cpp'), reference_source.FINAL_OUTPUTS[False]['ggml-webgpu.cpp'])
        for name in ('ggml-webgpu-shader-lib.hpp', 'ggml-webgpu-router-tail.hpp'):
            self.assertEqual(reference_source.tensor_copy.digest(self.normal / name), (reference_source.FINAL_OUTPUTS[False][name] if name == 'ggml-webgpu-shader-lib.hpp' else router.OUTPUTS[name]))
        for name, expected in reference_source.FINAL_OUTPUTS[True].items():
            self.assertEqual(reference_source.tensor_copy.digest(self.profile / name), expected)
        for prepared in (self.normal, self.profile):
            self.assertEqual(reference_source.tensor_copy.digest(prepared / router.HEADER_NAME), router.HEADER_SHA256)
            self.assertFalse((prepared / 'ggml-wgsl-shaders.hpp').exists())
            # Usage scopes cover the whole GPUBuffer, even when writable
            # binding ranges are disjoint. Match the ordinary ggml shaders.
            embedded = (prepared / router.HEADER_NAME).read_text()
            self.assertNotIn('var<storage, read> probs:', embedded)
            self.assertNotIn('var<storage, read> ids:', embedded)
            self.assertEqual(embedded.count('var<storage, read_write> ids: array<i32>;'), 2)
        stamps = {p.name: p.stat().st_mtime_ns for p in self.normal.iterdir() if p.is_file()}
        source.prepare(self.upstream, self.normal)
        self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in self.normal.iterdir() if p.is_file()})

    def test_every_stage_identity_rejects_without_publishing_partial_outputs(self):
        before = {p.name: p.read_bytes() for p in self.normal.iterdir() if p.is_file()}
        for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS'):
            values = getattr(router, field)
            for name in values:
                with self.subTest(field=field, name=name), patch.object(router, field, {**values, name: '0' * 64}):
                    with self.assertRaises(ValueError):
                        legacy_source.prepare(self.upstream, self.normal)
                self.assertEqual(before, {p.name: p.read_bytes() for p in self.normal.iterdir() if p.is_file()})
        for field in ('PATCH_SHA256', 'HEADER_SHA256'):
            with self.subTest(field=field), patch.object(router, field, '0' * 64):
                with self.assertRaises(ValueError):
                    legacy_source.prepare(self.upstream, self.normal)
                self.assertEqual(before, {p.name: p.read_bytes() for p in self.normal.iterdir() if p.is_file()})

    def test_actual_graph_allocator_and_extracted_checked_dispatch(self):
        library = os.environ.get('LCB_TEST_GGML_BASE_LIBRARY')
        if not library:
            self.skipTest('Set LCB_TEST_GGML_BASE_LIBRARY to an upstream ggml-base archive')
        text = (self.normal / 'ggml-webgpu.cpp').read_text()
        start = text.index('struct webgpu_router_tail_binding {')
        end = text.index('struct webgpu_ssm_conv_silu_binding {', start)
        (self.directory / 'dispatch.inc').write_text(text[start:end])
        executable = self.directory / 'host'
        subprocess.run(['c++', '-std=c++17', '-O1', '-Wall', '-Wextra', '-Werror',
                        '-I' + str(self.normal), '-I' + str(self.directory),
                        '-isystem', str(self.upstream / 'ggml/include'), '-isystem', str(self.upstream / 'ggml/src'),
                        str(HERE / 'host-cases.cpp'), library, '-pthread', '-ldl', '-lm', '-o', str(executable)], check=True)
        subprocess.run([str(executable)], check=True)

    def test_actual_factory_normal_profile_cache_and_shader_identity(self):
        identity = json.loads((HERE / 'inputs.json').read_text())
        previous = {}
        for label, prepared in (('normal', self.normal), ('profile', self.profile)):
            directory = self.directory / label / 'factory'
            executable = router_factory.build(self.upstream, prepared, directory)
            for wg in (256, 512, 1024):
                target = directory / str(wg)
                target.mkdir()
                subprocess.run([str(executable), str(target), str(wg)], check=True, capture_output=True)
                for name, expected in identity['processed_shader_sha256'][str(wg)].items():
                    data = (target / name).read_bytes()
                    self.assertEqual(hashlib.sha256(data).hexdigest(), expected)
                    if label == 'normal':
                        previous[(wg, name)] = data
                    else:
                        self.assertEqual(data, previous[(wg, name)])

    def test_optional_gpu_cli_rejects_invalid_json_before_preparing(self):
        script = HERE / 'check_gpu.py'
        for command in ('[]', 'null', '{}', '["node",7]', '[invalid'):
            with tempfile.TemporaryDirectory() as temporary:
                destination = Path(temporary) / 'absent'
                result = subprocess.run([sys.executable, str(script), '--llama-source', str(destination),
                                         '--work', str(destination), '--webgpu-module', str(destination),
                                         '--node-command-json', command], text=True, capture_output=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('--node-command-json must', result.stderr)
                self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
