"""Source-bound router tail graph, allocator, binding and actual factory checks."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tests/webgpu-router-tail'
sys.path[:0] = [str(ROOT / 'scripts'), str(HERE)]
import prepare_webgpu_source as source
import prepare_webgpu_router_tail as router
import router_factory


class RouterTail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temporary = tempfile.TemporaryDirectory(prefix='router-tail-tests-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.normal = source.prepare(cls.upstream, cls.directory / 'normal').parent
        cls.profile = source.prepare(cls.upstream, cls.directory / 'profile', gpu_profile=True).parent

    def test_checked_stage_before_profile_and_exact_shader_headers(self):
        self.assertEqual(source.host_encode_assembly.INPUT_SHA256, router.OUTPUTS['ggml-webgpu.cpp'])
        self.assertEqual(source.tensor_copy.digest(self.normal / 'ggml-webgpu.cpp'), source.OUTPUT_SHA256)
        for name in ('ggml-webgpu-shader-lib.hpp', 'ggml-webgpu-router-tail.hpp'):
            self.assertEqual(source.tensor_copy.digest(self.normal / name), router.OUTPUTS[name])
        for name, expected in source.q8_diagnostic.OUTPUTS.items():
            self.assertEqual(source.tensor_copy.digest(self.profile / name), expected)
        for prepared in (self.normal, self.profile):
            self.assertEqual(source.tensor_copy.digest(prepared / router.HEADER_NAME), router.HEADER_SHA256)
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
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        for name in router.CONTRACT_INPUTS:
            self.assertIn('${LCB_LLAMA_SOURCE}/' + name, cmake)
        self.assertIn(router.PATCH_NAME, cmake)
        self.assertIn('scripts/prepare_webgpu_router_tail.py', cmake)

    def test_every_stage_identity_rejects_without_publishing_partial_outputs(self):
        before = {p.name: p.read_bytes() for p in self.normal.iterdir() if p.is_file()}
        for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS'):
            values = getattr(router, field)
            for name in values:
                with self.subTest(field=field, name=name), patch.object(router, field, {**values, name: '0' * 64}):
                    with self.assertRaises(ValueError):
                        source.prepare(self.upstream, self.normal)
                self.assertEqual(before, {p.name: p.read_bytes() for p in self.normal.iterdir() if p.is_file()})
        for field in ('PATCH_SHA256', 'HEADER_SHA256'):
            with self.subTest(field=field), patch.object(router, field, '0' * 64):
                with self.assertRaises(ValueError):
                    source.prepare(self.upstream, self.normal)
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
