"""Compare actual staged host builders, without GPU or target-speed assertions."""
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
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import prepare_webgpu_source as source
from webgpu_prefix_context import without_final_matvec_stages
import prepare_webgpu_host_encode_assembly as assembly
import prepare_webgpu_tensor_copy as tensor_copy
from test_webgpu_bind_group_cache import definition, function

HERE = ROOT / 'tests/webgpu-host-encode-assembly'


def extracted_model(actual, header):
    model = (ROOT / 'tests/webgpu-bind-group-cache/mock-harness.cpp').read_text().split('static void assert_values(')[0]
    for marker, name in {
        'ACTUAL_CREATE_BUFFER': 'ggml_webgpu_create_buffer',
        'ACTUAL_MAKE_ENTRY': 'ggml_webgpu_make_bind_group_entry',
        'ACTUAL_SUBMIT': 'ggml_backend_webgpu_submit_commands',
        'ACTUAL_GRAPH_COMPUTE': 'ggml_backend_webgpu_graph_compute',
        'ACTUAL_BUFFER_FREE': 'ggml_backend_webgpu_buffer_free_buffer',
    }.items():
        value = function(actual, name)
        if marker == 'ACTUAL_GRAPH_COMPUTE' and 'static ggml_status ggml_backend_webgpu_graph_compute_impl(' in actual:
            value = (definition(actual, 'struct webgpu_profile_operation_guard {') + 'template<class Encode>\n' +
                     function(actual, 'ggml_backend_webgpu_graph_compute_impl') + value)
            model = model.replace('bool batch_compute_passes = true;', 'bool batch_compute_passes = true; bool operation_in_progress = false;')
            model = model.replace('enum ggml_status { GGML_STATUS_SUCCESS };', 'enum ggml_status { GGML_STATUS_SUCCESS, GGML_STATUS_FAILED };')
        model = model.replace('// @' + marker + '@', value)
    helpers = function(actual, 'ggml_backend_webgpu_build_multi') + function(actual, 'ggml_backend_webgpu_build')
    if 'struct webgpu_dispatch_view {' in actual:
        helpers = (definition(actual, 'struct webgpu_dispatch_view {') + 'template<class Dispatch>\n' +
                   function(actual, 'ggml_backend_webgpu_build_impl') + helpers)
    model = model.replace('// @ACTUAL_BUILD_MULTI@', helpers)
    for marker, name in {'ACTUAL_PARAM_ARENA': 'webgpu_param_arena', 'ACTUAL_CACHE': 'webgpu_bind_group_cache',
                         'ACTUAL_ENCODED_OP': 'webgpu_encoded_op', 'ACTUAL_DISPATCH_DESC': 'webgpu_dispatch_desc'}.items():
        model = model.replace('// @' + marker + '@', definition(actual, 'struct ' + name + ' {'))
    model = model.replace('struct webgpu_pipeline { wgpu::ComputePipeline pipeline; std::string name; };',
                          definition(header, 'struct webgpu_pipeline {'))
    replacements = [
        ('void SetPipeline(ComputePipeline value) { pipeline = std::move(value); }',
         "void SetPipeline(ComputePipeline value) { event('P', uintptr_t(value.Get())); pipeline = std::move(value); }"),
        ('void SetBindGroup(uint32_t, BindGroup value) { group = std::move(value); }',
         "void SetBindGroup(uint32_t, BindGroup value) { event('B', uintptr_t(value.Get())); group = std::move(value); }"),
        ('void DispatchWorkgroups(uint32_t, uint32_t, uint32_t) { assert(encoded && group); encoded->push_back({group, pipeline}); }',
         "void DispatchWorkgroups(uint32_t x, uint32_t y, uint32_t z) { event('D', 0, x, y, z); assert(encoded && group); encoded->push_back({group, pipeline}); }"),
        ('void End() {}', "void End() { event('E'); }"),
        ('ComputePassEncoder BeginComputePass(const ComputePassDescriptor * = nullptr) { return ComputePassEncoder(encoded); }',
         "ComputePassEncoder BeginComputePass(const ComputePassDescriptor * = nullptr) { event('C'); return ComputePassEncoder(encoded); }"),
    ]
    for before, after in replacements:
        if model.count(before) != 1:
            raise ValueError('Host allocation model API changed: ' + before)
        model = model.replace(before, after)
    if '// @ACTUAL_' in model:
        raise ValueError('Incomplete actual helper extraction')
    return model


class HostEncodeAssembly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temporary = tempfile.TemporaryDirectory(prefix='host-encode-assembly-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        with without_final_matvec_stages(source), patch.object(source.moe_tight_tile_bound, 'prepare'), patch.object(source.matmul_vec4_layout, 'prepare'), patch.object(assembly, 'prepare'), patch.object(source, 'OUTPUT_SHA256', assembly.INPUT_SHA256):
            cls.baseline = source.prepare(cls.upstream, cls.directory / 'baseline').read_text()
        # Isolate the assembly prefix; the later layout stage is checked separately.
        with without_final_matvec_stages(source), patch.object(source.moe_tight_tile_bound, 'prepare'), patch.object(source.matmul_vec4_layout, 'prepare'):
            cls.normal = source.prepare(cls.upstream, cls.directory / 'normal')
            cls.profile = source.prepare(cls.upstream, cls.directory / 'profile', gpu_profile=True)

    def test_real_composition_preserves_all_other_bytes_and_profile_fields(self):
        actual = self.normal.read_text()
        base_end = self.baseline.index('static void ggml_backend_webgpu_buffer_memset(')
        new_end = actual.index('static void ggml_backend_webgpu_buffer_memset(')
        base_start = self.baseline.index('static webgpu_encoded_op ggml_backend_webgpu_build_multi(')
        new_start = actual.index('// These references are used only while build_impl')
        self.assertEqual(actual[:new_start], self.baseline[:base_start])
        self.assertEqual(actual[new_end:], self.baseline[base_end:])
        self.assertEqual(tensor_copy.digest(self.normal), assembly.OUTPUT_SHA256)
        self.assertEqual(source.OUTPUT_SHA256, assembly.OUTPUT_SHA256)
        self.assertEqual(source.profile_query_budget.INPUT_SHA256, assembly.OUTPUT_SHA256)
        self.assertEqual(source.q8_diagnostic.INPUTS['ggml-webgpu.cpp'], source.profile_query_budget.OUTPUT_SHA256)
        for filename in ('ggml-webgpu-shader-lib.hpp', 'ggml-backend.cpp', 'llama-model-loader.cpp', 'llama-sampler.cpp'):
            self.assertEqual((self.normal.parent / filename).read_bytes(), (self.directory / 'baseline' / filename).read_bytes())
        profile = self.profile.read_text()
        for name in ('ggml_backend_webgpu_build_impl', 'ggml_backend_webgpu_build_multi', 'ggml_backend_webgpu_build'):
            self.assertEqual(function(actual, name), function(profile, name))
        profile_header = (self.profile.parent / 'ggml-webgpu-shader-lib.hpp').read_text()
        self.assertIn('std::array<uint8_t, 32> diagnostic_sha256', definition(profile_header, 'struct webgpu_pipeline {'))
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_host_encode_assembly.py', cmake)
        self.assertIn(assembly.PATCH_NAME, cmake)

    def test_production_guards_fail_before_publishing(self):
        for field in ('INPUT_SHA256', 'OUTPUT_SHA256', 'PATCH_SHA256'):
            destination = self.directory / field
            with self.subTest(field=field), patch.object(assembly, field, '0' * 64):
                with self.assertRaises(ValueError):
                    source.prepare(self.upstream, destination)
            self.assertFalse((destination / 'ggml-webgpu.cpp').exists())
        for index, relative in enumerate(assembly.CONTRACT_INPUTS):
            destination = self.directory / ('contract-' + str(index))
            values = {**assembly.CONTRACT_INPUTS, relative: '0' * 64}
            with self.subTest(contract=relative), patch.object(assembly, 'CONTRACT_INPUTS', values):
                with self.assertRaisesRegex(ValueError, 'Host encode assembly contract'):
                    source.prepare(self.upstream, destination)
            self.assertFalse((destination / 'ggml-webgpu.cpp').exists())

    def test_actual_patch_drift_and_duplicate_application_fail_closed(self):
        work = self.directory / 'drift'
        target = work / tensor_copy.SOURCE_PATH
        target.parent.mkdir(parents=True)
        target.write_text(self.baseline)
        patches = self.directory / 'patches'
        patches.mkdir()
        patch_file = patches / assembly.PATCH_NAME
        original = (ROOT / source.PATCH_DIRECTORY / assembly.PATCH_NAME).read_bytes()
        patch_file.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'patch identity changed'):
            assembly.prepare(work, self.upstream, patches, source.patch_environment(work))
        self.assertEqual(target.read_text(), self.baseline)
        patch_file.write_bytes(original)
        assembly.prepare(work, self.upstream, patches, source.patch_environment(work))
        self.assertEqual(target.read_bytes(), self.normal.read_bytes())
        with self.assertRaisesRegex(ValueError, 'composition input changed'):
            assembly.prepare(work, self.upstream, patches, source.patch_environment(work))

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_warm_allocations_and_deferred_ownership_relative_to_baseline(self):
        template = (HERE / 'allocation-cases.cpp').read_text()
        for profiling, prepared in ((False, self.normal), (True, self.profile)):
            # Use each real final header, including the profile diagnostic array.
            header = (prepared.parent / 'ggml-webgpu-shader-lib.hpp').read_text()
            results = []
            for variant, actual in (('baseline', self.baseline), ('prepared', prepared.read_text())):
                model = extracted_model(actual, header)
                code = '#include <cstdlib>\n#include <new>\n#include <cassert>\n#include <cstdint>\n#include <cstddef>\n#include <array>\n'
                code += template.replace('// @MODEL_AND_ACTUAL@', model)
                name = variant + ('-profile' if profiling else '-normal')
                cpp, binary = self.directory / (name + '.cpp'), self.directory / name
                cpp.write_text(code)
                macros = ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'] + (['-DGGML_WEBGPU_GPU_PROFILE'] if profiling else [])
                build = subprocess.run(['c++', '-std=c++17', '-O3', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                                        *macros, str(cpp), '-o', str(binary)], capture_output=True, text=True)
                self.assertEqual(build.returncode, 0, build.stderr)
                result = subprocess.run([str(binary)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                results.append(json.loads(result.stdout))
            old, new = results
            self.assertEqual(len(old['cases']), 164)
            self.assertEqual(len(new['cases']), 164)
            for before, after in zip(old['cases'], new['cases']):
                allocation_fields = ('allocationsPerOp', 'allocationBytesPerOp', 'allocationSizesSorted')
                self.assertEqual({k: v for k, v in before.items() if k not in allocation_fields},
                                 {k: v for k, v in after.items() if k not in allocation_fields})
                self.assertLessEqual(after['allocationsPerOp'], before['allocationsPerOp'])
                self.assertLessEqual(after['allocationBytesPerOp'], before['allocationBytesPerOp'])
                if before['storageEntries'] and before['dispatches']:
                    self.assertLess(after['allocationsPerOp'], before['allocationsPerOp'])
            self.assertEqual(old['mockBindGroupEntryBytes'], new['mockBindGroupEntryBytes'])
            self.assertEqual(old['actualPipelineStructBytesInThisMock'], new['actualPipelineStructBytesInThisMock'])


if __name__ == '__main__':
    unittest.main()
