"""Frozen copy-stage regressions; live-input checks: test_current_copy_recording."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
HERE = Path(__file__).resolve().parents[2] / 'fixtures/llama-webgpu-history-cd959bae/checks/webgpu-host-input-snapshot'
sys.path.insert(0, str(ROOT.parents[1]))
from tests.support.historical_webgpu import historical_source, FIXTURE
# Historical byte identities belong to the reference fixture, not production.
from tests.support import frozen_webgpu_source as reference_source
from tests.support.llama_cpp.frozen_prefix_context import without_final_matvec_stages
from tests.support.llama_cpp.frozen_webgpu_stages import host_input_snapshot as snapshot
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy as tensor_copy
spec = importlib.util.spec_from_file_location('host_snapshot_harness', HERE / 'emit_harness.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class HistoricalHostInputSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = cls.enterClassContext(historical_source())
        cls.patch_root = cls.upstream / 'patches'
        cls.identity = json.loads((HERE / 'inputs.json').read_text())
        for name, expected in cls.identity['upstream_sha256'].items():
            if tensor_copy.digest(cls.upstream / name) != expected:
                raise ValueError('Host snapshot fixture needs semantic review: ' + name)

    def prepare(self, directory):
        return reference_source.prepare(self.upstream, directory, patch_root=self.patch_root)

    def test_reviewed_stage_protocol_identity_and_pristine_upstream(self):
        self.assertEqual(self.identity['contract_inputs'], snapshot.CONTRACT_INPUTS)
        self.assertEqual(self.identity['inherited_harness_sha256'], tensor_copy.digest(FIXTURE / 'checks/webgpu-async-tensor-copy/mock-harness.cpp'))
        self.assertEqual(self.identity['patch_sha256'], snapshot.PATCH_SHA256)
        self.assertEqual(self.identity['input_sha256'], snapshot.INPUT_SHA256)
        self.assertEqual(self.identity['output_sha256'], snapshot.OUTPUT_SHA256)
        self.assertEqual(self.identity['backend_output_sha256'], snapshot.BACKEND_OUTPUT_SHA256)
        self.assertEqual(self.identity['header_output_sha256'], snapshot.HEADER_OUTPUT_SHA256)
        before = {name: tensor_copy.digest(self.upstream / name) for name in self.identity['upstream_sha256']}
        with tempfile.TemporaryDirectory(prefix='host-snapshot-identity-') as temporary:
            directory = Path(temporary)
            target = self.prepare(directory)
            self.assertEqual(tensor_copy.digest(target), reference_source.FINAL_OUTPUTS[False]['ggml-webgpu.cpp'])
            self.assertEqual(tensor_copy.digest(directory / 'ggml-backend.cpp'), snapshot.BACKEND_OUTPUT_SHA256)
            self.assertEqual(tensor_copy.digest(directory / 'ggml-backend-host-snapshot.h'), snapshot.HEADER_OUTPUT_SHA256)
            self.assertEqual(tensor_copy.digest(self.patch_root / snapshot.PATCH_NAME), snapshot.PATCH_SHA256)
            provider = target.read_text()
            self.assertRegex(provider, r'/\* \.get_proc_address\s+= \*/\s*ggml_backend_webgpu_reg_get_proc_address,')
            contract = (directory / 'ggml-backend-host-snapshot.h').read_text()
            self.assertIn('ggml_backend_copy_host_input_snapshot_v1', contract)
            self.assertIn('false: unsupported, with no synchronization, enqueue, or tensor mutation.', contract)
        self.assertEqual(before, {name: tensor_copy.digest(self.upstream / name) for name in before})

    def test_module_guards_reject_contract_patch_input_output_and_existing_source_drift(self):
        with tempfile.TemporaryDirectory(prefix='host-snapshot-drift-') as temporary:
            root = Path(temporary)
            with without_final_matvec_stages(reference_source), patch.object(reference_source.moe_tight_tile_bound, 'prepare'), patch.object(reference_source.matmul_vec4_layout, 'prepare'), patch.object(snapshot, 'prepare', return_value={}), \
                    patch.object(reference_source.ssm_conv_silu, 'prepare', return_value={}), \
                    patch.object(reference_source.moe_group_skip, 'prepare', return_value={}), \
                patch.object(reference_source.router_tail, 'prepare', return_value={}), \
                    patch.object(reference_source.host_encode_assembly, 'prepare'), \
                    patch.object(reference_source.f32_small_m, 'prepare'), \
                    patch.object(reference_source.moe_gather_logical32, 'prepare', return_value={}), \
                    patch.object(reference_source, 'OUTPUT_SHA256', snapshot.INPUT_SHA256), \
                    patch.object(reference_source, 'SSM_HEADER_OUTPUT_SHA256', reference_source.ssm_conv_silu.INPUTS['ggml-webgpu-shader-lib.hpp']):
                baseline = reference_source.prepare(self.upstream, root / 'base', patch_root=self.patch_root)
            self.assertEqual(tensor_copy.digest(baseline), snapshot.INPUT_SHA256)
            reviewed = root / 'upstream'
            for name in snapshot.CONTRACT_INPUTS:
                destination = reviewed / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.upstream / name, destination)
            patches = root / 'patches'; patches.mkdir()
            patch_file = patches / snapshot.PATCH_NAME
            original_patch = self.patch_root / snapshot.PATCH_NAME
            shutil.copyfile(original_patch, patch_file)

            def stage(name):
                work = root / name
                target = work / reference_source.SOURCE_PATH
                target.parent.mkdir(parents=True)
                shutil.copyfile(baseline, target)
                return work, target

            for index, name in enumerate(snapshot.CONTRACT_INPUTS):
                dependency = reviewed / name
                original = dependency.read_bytes()
                dependency.write_bytes(original + b'\n')
                work, target = stage('contract-' + str(index))
                with self.subTest(path=name), self.assertRaisesRegex(ValueError, 'contract changed'):
                    snapshot.prepare(work, reviewed, patches, reference_source.patch_environment(work))
                self.assertEqual(target.read_bytes(), baseline.read_bytes())
                self.assertFalse((work / snapshot.BACKEND_SOURCE_PATH).exists())
                dependency.write_bytes(original)
            work, target = stage('input'); target.write_bytes(b'changed input\n')
            with self.assertRaisesRegex(ValueError, 'composition input changed'):
                snapshot.prepare(work, reviewed, patches, reference_source.patch_environment(work))
            work, target = stage('patch'); patch_file.write_bytes(patch_file.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'patch identity changed'):
                snapshot.prepare(work, reviewed, patches, reference_source.patch_environment(work))
            self.assertEqual(target.read_bytes(), baseline.read_bytes())
            shutil.copyfile(original_patch, patch_file)
            for index, path in enumerate((snapshot.BACKEND_SOURCE_PATH, snapshot.HEADER_PATH)):
                work, target = stage('existing-' + str(index))
                existing = work / path; existing.parent.mkdir(parents=True, exist_ok=True); existing.write_bytes(b'unreviewed\n')
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'unreviewed source'):
                    snapshot.prepare(work, reviewed, patches, reference_source.patch_environment(work))
                self.assertEqual(existing.read_bytes(), b'unreviewed\n')
                self.assertEqual(target.read_bytes(), baseline.read_bytes())
            new_upstream_header = reviewed / snapshot.HEADER_PATH
            new_upstream_header.write_text('upstream now provides this contract\n')
            work, _ = stage('upstream-extension')
            with self.assertRaisesRegex(ValueError, 'unreviewed source'):
                snapshot.prepare(work, reviewed, patches, reference_source.patch_environment(work))
            new_upstream_header.unlink()
            for field in ('OUTPUT_SHA256', 'BACKEND_OUTPUT_SHA256', 'HEADER_OUTPUT_SHA256'):
                work, _ = stage(field)
                with self.subTest(field=field), patch.object(snapshot, field, '0' * 64), self.assertRaisesRegex(ValueError, 'output identity'):
                    snapshot.prepare(work, reviewed, patches, reference_source.patch_environment(work))
            work, _ = stage('success')
            extra = snapshot.prepare(work, reviewed, patches, reference_source.patch_environment(work))
            self.assertEqual(set(extra), {'ggml-backend.cpp', 'ggml-backend-host-snapshot.h'})
            with self.assertRaisesRegex(ValueError, 'composition input changed'):
                snapshot.prepare(work, reviewed, patches, reference_source.patch_environment(work))

    def test_other_scheduler_lifecycle_barriers_callbacks_and_existing_copy_paths_are_unchanged(self):
        with tempfile.TemporaryDirectory(prefix='host-snapshot-lifecycle-') as temporary:
            directory = Path(temporary)
            actual_provider = self.prepare(directory).read_text()
            actual = (directory / 'ggml-backend.cpp').read_text()
            upstream = (self.upstream / snapshot.BACKEND_SOURCE_PATH).read_text()
            for name in ('ggml_backend_sched_compute_splits', 'ggml_backend_sched_alloc_splits',
                         'ggml_backend_sched_reset', 'ggml_backend_sched_free', 'ggml_backend_sched_synchronize',
                         'ggml_backend_tensor_copy', 'ggml_backend_tensor_copy_async', 'ggml_backend_tensor_get',
                         'ggml_backend_sched_is_host_weight'):
                with self.subTest(helper=name):
                    self.assertEqual(harness.function(actual, name), harness.function(upstream, name))
            # Compare the previous composed provider, not pristine upstream: this
            # stage must retain checked async-copy and ordered-readback behavior.
            with without_final_matvec_stages(reference_source), patch.object(reference_source.moe_tight_tile_bound, 'prepare'), patch.object(reference_source.matmul_vec4_layout, 'prepare'), patch.object(snapshot, 'prepare', return_value={}), \
                    patch.object(reference_source.ssm_conv_silu, 'prepare', return_value={}), \
                    patch.object(reference_source.moe_group_skip, 'prepare', return_value={}), \
                patch.object(reference_source.router_tail, 'prepare', return_value={}), \
                    patch.object(reference_source.host_encode_assembly, 'prepare'), \
                    patch.object(reference_source.f32_small_m, 'prepare'), \
                    patch.object(reference_source.moe_gather_logical32, 'prepare', return_value={}), \
                    patch.object(reference_source, 'OUTPUT_SHA256', snapshot.INPUT_SHA256), \
                    patch.object(reference_source, 'SSM_HEADER_OUTPUT_SHA256', reference_source.ssm_conv_silu.INPUTS['ggml-webgpu-shader-lib.hpp']):
                before = reference_source.prepare(self.upstream, directory / 'before', patch_root=self.patch_root).read_text()
            for name in ('ggml_backend_webgpu_cpy_tensor_async', 'ggml_backend_webgpu_get_tensor_async',
                         'ggml_backend_webgpu_synchronize', 'ggml_backend_webgpu_buffer_cpy_tensor'):
                self.assertEqual(harness.function(actual_provider, name), harness.function(before, name))
            constructor = harness.block(actual, 'ggml_backend_sched_t ggml_backend_sched_new(')
            self.assertIn('calloc(1, sizeof(struct ggml_backend_sched))', constructor)
            discovery = harness.block(constructor, '        if (sched->n_copies == 1) {')
            previous_constructor = harness.block(upstream, 'ggml_backend_sched_t ggml_backend_sched_new(')
            self.assertEqual(constructor.replace(discovery + '\n', '', 1), previous_constructor)
            scheduler_copy = harness.function(actual, 'ggml_backend_sched_copy_input')
            self.assertLess(scheduler_copy.index('snapshot(input_backend'), scheduler_copy.index('if (input->flags'))
            self.assertIn('input != split->inputs[0]', scheduler_copy)
            self.assertNotIn('get_proc_address', scheduler_copy)
            properties = harness.function(actual_provider, 'ggml_backend_webgpu_device_get_props')
            self.assertRegex(properties, r'/\* \.async\s+= \*/\s*false')
            self.assertRegex(properties, r'/\* \.events\s+= \*/\s*false')

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_registry_cache_provider_scheduler_order_guards_and_failures(self):
        with tempfile.TemporaryDirectory(prefix='host-snapshot-native-') as temporary:
            directory = Path(temporary)
            self.prepare(directory / 'prepared')
            code = harness.emit(self.upstream, directory / 'prepared')
            emitted = directory / 'snapshot.cpp'; emitted.write_text(code)
            for macros in ([], ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'],
                           ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS', '-DGGML_WEBGPU_GPU_PROFILE', '-DGGML_WEBGPU_CPU_PROFILE']):
                binary = directory / ('snapshot-' + str(len(macros)))
                subprocess.run(['c++', '-std=c++17', '-Wall', '-Wextra', '-Werror', '-O0', *macros,
                                str(emitted), '-o', str(binary)], check=True)
                result = subprocess.run([str(binary)], check=True, capture_output=True, text=True)
                self.assertIn('source-bound snapshot/registry/cache/scheduler/queue/guard checks', result.stdout)
                print(result.stdout.strip())



class InheritedHarnessFormatting(unittest.TestCase):
    def test_source_composition_accepts_clang_formatted_harness(self):
        code = harness.before_main((FIXTURE / 'checks/webgpu-async-tensor-copy/mock-harness.cpp').read_text())
        self.assertNotIn('int main()', code)
        # The inherited harness may be formatted separately from the builder.
        for old in (
            'struct ggml_backend;using ggml_backend_t=ggml_backend*;',
            'GGML_BACKEND_DEVICE_TYPE_CPU,GGML_BACKEND_DEVICE_TYPE_GPU',
            'device_iface iface;enum ggml_backend_dev_type type;',
            'struct ggml_backend_sched_split { int backend_id=0;ggml_cgraph graph; };',
            'std::vector<std::function<void()>> pending;',
            'pending.push_back([=]{std::copy(snapshot.begin(),snapshot.end(),b.p->bytes.begin()+offset);});',
            'ggml_backend_t backends[1]{};ggml_backend_event_t events[1][1]{};int cur_copy=0;',
            'bool host_copy(){return ggml_backend_webgpu_cpy_tensor_async(&cpu,&gpu,&src,&dst);}',
        ):
            with self.subTest(old=old):
                self.assertIn('// unique replacement //', harness.replace_once(code, old, '// unique replacement //'))

    def test_replace_once_rejects_ambiguous_and_semantic_changes(self):
        with self.assertRaisesRegex(ValueError, 'needs review'):
            harness.replace_once('int a = 1; int a=1;', 'int a=1;', 'unused')
        with self.assertRaisesRegex(ValueError, 'needs review'):
            harness.replace_once('int a = 2;', 'int a=1;', 'unused')
        with self.assertRaisesRegex(ValueError, 'main entrypoint changed'):
            harness.before_main('int not_main() {}')


if __name__ == '__main__':
    unittest.main()
