"""Optional host snapshots: actual registry/provider/scheduler bodies, CPU only."""
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

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'tests/webgpu-host-input-snapshot'
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
import prepare_webgpu_host_input_snapshot as snapshot
import prepare_webgpu_tensor_copy as tensor_copy

spec = importlib.util.spec_from_file_location('host_snapshot_harness', HERE / 'emit_harness.py')
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)


class HostInputSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / source.SOURCE_PATH).is_file():
            raise unittest.SkipTest('Initialize upstream or set LCB_TEST_LLAMA_SOURCE')
        cls.identity = json.loads((HERE / 'inputs.json').read_text())
        for name, expected in cls.identity['upstream_sha256'].items():
            if tensor_copy.digest(cls.upstream / name) != expected:
                raise ValueError('Host snapshot fixture needs semantic review: ' + name)

    def prepare(self, directory):
        return source.prepare(self.upstream, directory)

    def test_reviewed_stage_protocol_identity_and_pristine_upstream(self):
        self.assertEqual(self.identity['contract_inputs'], snapshot.CONTRACT_INPUTS)
        self.assertEqual(self.identity['module_sha256'], tensor_copy.digest(ROOT / 'scripts/prepare_webgpu_host_input_snapshot.py'))
        self.assertEqual(self.identity['inherited_harness_sha256'], tensor_copy.digest(ROOT / 'tests/webgpu-async-tensor-copy/mock-harness.cpp'))
        self.assertEqual(self.identity['patch_sha256'], snapshot.PATCH_SHA256)
        self.assertEqual(self.identity['input_sha256'], snapshot.INPUT_SHA256)
        self.assertEqual(self.identity['output_sha256'], snapshot.OUTPUT_SHA256)
        self.assertEqual(self.identity['backend_output_sha256'], snapshot.BACKEND_OUTPUT_SHA256)
        self.assertEqual(self.identity['header_output_sha256'], snapshot.HEADER_OUTPUT_SHA256)
        before = {name: tensor_copy.digest(self.upstream / name) for name in self.identity['upstream_sha256']}
        with tempfile.TemporaryDirectory(prefix='host-snapshot-identity-') as temporary:
            directory = Path(temporary)
            target = self.prepare(directory)
            self.assertEqual(tensor_copy.digest(target), source.FINAL_OUTPUTS[False]['ggml-webgpu.cpp'])
            self.assertEqual(tensor_copy.digest(directory / 'ggml-backend.cpp'), snapshot.BACKEND_OUTPUT_SHA256)
            self.assertEqual(tensor_copy.digest(directory / 'ggml-backend-host-snapshot.h'), snapshot.HEADER_OUTPUT_SHA256)
            self.assertEqual(tensor_copy.digest(ROOT / source.PATCH_DIRECTORY / snapshot.PATCH_NAME), snapshot.PATCH_SHA256)
            provider = target.read_text()
            self.assertRegex(provider, r'/\* \.get_proc_address\s+= \*/\s*ggml_backend_webgpu_reg_get_proc_address,')
            contract = (directory / 'ggml-backend-host-snapshot.h').read_text()
            self.assertIn('ggml_backend_copy_host_input_snapshot_v1', contract)
            self.assertIn('false: unsupported, with no synchronization, enqueue, or tensor mutation.', contract)
        self.assertEqual(before, {name: tensor_copy.digest(self.upstream / name) for name in before})

    def test_module_guards_reject_contract_patch_input_output_and_existing_source_drift(self):
        with tempfile.TemporaryDirectory(prefix='host-snapshot-drift-') as temporary:
            root = Path(temporary)
            with patch.object(source.moe_tight_tile_bound, 'prepare'), patch.object(source.matmul_vec4_layout, 'prepare'), patch.object(snapshot, 'prepare', return_value={}), \
                    patch.object(source.ssm_conv_silu, 'prepare', return_value={}), \
                    patch.object(source.moe_group_skip, 'prepare', return_value={}), \
                patch.object(source.router_tail, 'prepare', return_value={}), \
                    patch.object(source.host_encode_assembly, 'prepare'), \
                    patch.object(source.f32_small_m, 'prepare'), \
                    patch.object(source.moe_gather_logical32, 'prepare', return_value={}), \
                    patch.object(source, 'OUTPUT_SHA256', snapshot.INPUT_SHA256), \
                    patch.object(source, 'SSM_HEADER_OUTPUT_SHA256', source.ssm_conv_silu.INPUTS['ggml-webgpu-shader-lib.hpp']):
                baseline = self.prepare(root / 'base')
            self.assertEqual(tensor_copy.digest(baseline), snapshot.INPUT_SHA256)
            reviewed = root / 'upstream'
            for name in snapshot.CONTRACT_INPUTS:
                destination = reviewed / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.upstream / name, destination)
            patches = root / 'patches'; patches.mkdir()
            patch_file = patches / snapshot.PATCH_NAME
            original_patch = ROOT / source.PATCH_DIRECTORY / snapshot.PATCH_NAME
            shutil.copyfile(original_patch, patch_file)

            def stage(name):
                work = root / name
                target = work / source.SOURCE_PATH
                target.parent.mkdir(parents=True)
                shutil.copyfile(baseline, target)
                return work, target

            for index, name in enumerate(snapshot.CONTRACT_INPUTS):
                dependency = reviewed / name
                original = dependency.read_bytes()
                dependency.write_bytes(original + b'\n')
                work, target = stage('contract-' + str(index))
                with self.subTest(path=name), self.assertRaisesRegex(ValueError, 'contract changed'):
                    snapshot.prepare(work, reviewed, patches, source.patch_environment(work))
                self.assertEqual(target.read_bytes(), baseline.read_bytes())
                self.assertFalse((work / snapshot.BACKEND_SOURCE_PATH).exists())
                dependency.write_bytes(original)
            work, target = stage('input'); target.write_bytes(b'changed input\n')
            with self.assertRaisesRegex(ValueError, 'composition input changed'):
                snapshot.prepare(work, reviewed, patches, source.patch_environment(work))
            work, target = stage('patch'); patch_file.write_bytes(patch_file.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'patch identity changed'):
                snapshot.prepare(work, reviewed, patches, source.patch_environment(work))
            self.assertEqual(target.read_bytes(), baseline.read_bytes())
            shutil.copyfile(original_patch, patch_file)
            for index, path in enumerate((snapshot.BACKEND_SOURCE_PATH, snapshot.HEADER_PATH)):
                work, target = stage('existing-' + str(index))
                existing = work / path; existing.parent.mkdir(parents=True, exist_ok=True); existing.write_bytes(b'unreviewed\n')
                with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'unreviewed source'):
                    snapshot.prepare(work, reviewed, patches, source.patch_environment(work))
                self.assertEqual(existing.read_bytes(), b'unreviewed\n')
                self.assertEqual(target.read_bytes(), baseline.read_bytes())
            new_upstream_header = reviewed / snapshot.HEADER_PATH
            new_upstream_header.write_text('upstream now provides this contract\n')
            work, _ = stage('upstream-extension')
            with self.assertRaisesRegex(ValueError, 'unreviewed source'):
                snapshot.prepare(work, reviewed, patches, source.patch_environment(work))
            new_upstream_header.unlink()
            for field in ('OUTPUT_SHA256', 'BACKEND_OUTPUT_SHA256', 'HEADER_OUTPUT_SHA256'):
                work, _ = stage(field)
                with self.subTest(field=field), patch.object(snapshot, field, '0' * 64), self.assertRaisesRegex(ValueError, 'output identity'):
                    snapshot.prepare(work, reviewed, patches, source.patch_environment(work))
            work, _ = stage('success')
            extra = snapshot.prepare(work, reviewed, patches, source.patch_environment(work))
            self.assertEqual(set(extra), {'ggml-backend.cpp', 'ggml-backend-host-snapshot.h'})
            with self.assertRaisesRegex(ValueError, 'composition input changed'):
                snapshot.prepare(work, reviewed, patches, source.patch_environment(work))

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
            with patch.object(source.moe_tight_tile_bound, 'prepare'), patch.object(source.matmul_vec4_layout, 'prepare'), patch.object(snapshot, 'prepare', return_value={}), \
                    patch.object(source.ssm_conv_silu, 'prepare', return_value={}), \
                    patch.object(source.moe_group_skip, 'prepare', return_value={}), \
                patch.object(source.router_tail, 'prepare', return_value={}), \
                    patch.object(source.host_encode_assembly, 'prepare'), \
                    patch.object(source.f32_small_m, 'prepare'), \
                    patch.object(source.moe_gather_logical32, 'prepare', return_value={}), \
                    patch.object(source, 'OUTPUT_SHA256', snapshot.INPUT_SHA256), \
                    patch.object(source, 'SSM_HEADER_OUTPUT_SHA256', source.ssm_conv_silu.INPUTS['ggml-webgpu-shader-lib.hpp']):
                before = self.prepare(directory / 'before').read_text()
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

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_complete_prepared_ggml_base_and_webgpu_cpp_syntax_with_pinned_dawn(self):
        dawn = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parent / '.tools/emdawnwebgpu_pkg'))
        if not dawn.is_dir():
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE for the reviewed local Dawn package')
        tensor_copy.verify_dawn(dawn)
        with tempfile.TemporaryDirectory(prefix='host-snapshot-syntax-') as temporary:
            directory = Path(temporary)
            target = self.prepare(directory)
            includes = [directory, self.upstream / 'ggml/include', self.upstream / 'ggml/src']
            base = ['c++', '-std=c++20', '-fsyntax-only', '-DGGML_SCHED_MAX_COPIES=4']
            base += ['-I' + str(path) for path in includes]
            subprocess.run(base + [str(directory / 'ggml-backend.cpp')], check=True)
            upstream_webgpu = self.upstream / 'ggml/src/ggml-webgpu'
            subprocess.run([sys.executable, str(upstream_webgpu / 'wgsl-shaders/embed_wgsl.py'), '--input_dir',
                str(upstream_webgpu / 'wgsl-shaders'), '--output_file', str(directory / 'ggml-wgsl-shaders.hpp')], check=True)
            command = base + ['-D__wasi__', '-I' + str(upstream_webgpu),
                '-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include')]
            for macros in (['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'],
                           ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS', '-DGGML_WEBGPU_GPU_PROFILE', '-DGGML_WEBGPU_CPU_PROFILE']):
                subprocess.run(command + macros + [str(target)], check=True)


if __name__ == '__main__':
    unittest.main()
