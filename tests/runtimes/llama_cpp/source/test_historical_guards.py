"""Frozen historical guards, distinct from current-upstream build contracts."""
import shutil
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from build_tools.project import runtime_root, runtime_test_root
from tests.support import frozen_webgpu_source as reference_source
from tests.support import frozen_webgpu_source as legacy_source
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy as copy
from tests.support.historical_webgpu import historical_source, FIXTURE

ROOT = runtime_root('llama-cpp')


class HistoricalComposerGuards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = cls.enterClassContext(historical_source())
        cls.patch_root = cls.upstream / "patches"

    def test_patch_result_and_overlap_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); patches = root / 'patches'; patches.mkdir()
            for name in [copy.PATCH_NAME, reference_source.PARAM_PATCH_NAME, reference_source.LOADER_PATCH_NAME, reference_source.SSM_PATCH_NAME, reference_source.hc.PATCH_NAME, reference_source.q8_short.PATCH_NAME, reference_source.moe_sum.PATCH_NAME]:
                shutil.copyfile(self.patch_root / name, patches / name)
            (patches / reference_source.PARAM_PATCH_NAME).write_bytes(b'wrong')
            with self.assertRaisesRegex(ValueError, 'patch identity'):
                legacy_source.prepare(self.upstream, root / 'out', patch_root=patches)
            self.assertFalse((root / 'out').exists())
            with patch.object(legacy_source, 'OUTPUT_SHA256', '0' * 64), self.assertRaisesRegex(ValueError, 'combined'):
                legacy_source.prepare(self.upstream, root / 'out', patch_root=self.patch_root)
        with self.assertRaisesRegex(ValueError, 'outside'):
            legacy_source.prepare(self.upstream, self.upstream / 'out', patch_root=self.patch_root)

    def test_loader_inputs_patch_and_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in reference_source.LOADER_INPUTS:
                inputs = dict(reference_source.LOADER_INPUTS); inputs[relative] = '0' * 64
                with patch.object(legacy_source, 'LOADER_INPUTS', inputs), self.assertRaisesRegex(ValueError, 'semantic review'):
                    legacy_source.prepare(self.upstream, root / 'out', patch_root=self.patch_root)
                self.assertFalse((root / 'out').exists())
            with patch.object(legacy_source, 'LOADER_PATCH_SHA256', '0' * 64), self.assertRaisesRegex(ValueError, 'patch identity'):
                legacy_source.prepare(self.upstream, root / 'out', patch_root=self.patch_root)
            self.assertFalse((root / 'out').exists())
            with patch.object(legacy_source, 'LOADER_OUTPUT_SHA256', '0' * 64), self.assertRaisesRegex(ValueError, 'loader source identity'):
                legacy_source.prepare(self.upstream, root / 'out', patch_root=self.patch_root)
            self.assertFalse((root / 'out/llama-model-loader.cpp').exists())

    def test_ssm_shader_patch_and_header_fail_closed_before_publishing(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'out'
            for name, message in [('SSM_SHADER_SHA256', 'upstream shader'),
                                  ('SSM_PATCH_SHA256', 'patch identity'),
                                  ('SSM_HEADER_OUTPUT_SHA256', 'header identity')]:
                with self.subTest(guard=name), patch.object(legacy_source, name, '0' * 64):
                    with self.assertRaisesRegex(ValueError, message):
                        legacy_source.prepare(self.upstream, output, patch_root=self.patch_root)
                self.assertFalse((output / 'ggml-webgpu.cpp').exists())
                self.assertFalse((output / 'llama-model-loader.cpp').exists())
                self.assertFalse((output / 'ggml-webgpu-shader-lib.hpp').exists())

    def test_ordered_readback_identity_guards_fail_before_publishing(self):
        ordered = reference_source.ordered_readback
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('PATCH_SHA256', 'INPUT_SHA256', 'OUTPUT_SHA256'):
                output = Path(tmp) / name
                with self.subTest(guard=name), patch.object(ordered, name, '0' * 64):
                    with self.assertRaisesRegex(ValueError, 'Ordered readback|ordered readback'):
                        legacy_source.prepare(self.upstream, output, patch_root=self.patch_root)
                self.assertFalse((output / 'ggml-webgpu.cpp').exists())
                self.assertFalse((output / 'llama-model-loader.cpp').exists())
            for relative in ordered.CONTRACT_INPUTS:
                output = Path(tmp) / ('contract-' + str(list(ordered.CONTRACT_INPUTS).index(relative)))
                inputs = {**ordered.CONTRACT_INPUTS, relative: '0' * 64}
                with self.subTest(contract=relative), patch.object(ordered, 'CONTRACT_INPUTS', inputs):
                    with self.assertRaisesRegex(ValueError, 'Ordered readback contract'):
                        legacy_source.prepare(self.upstream, output, patch_root=self.patch_root)
                self.assertFalse((output / 'ggml-webgpu.cpp').exists())

    def test_sampler_guards_and_configure_dependencies(self):
        sampler = reference_source.sampler_single_sync
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = [('DEPENDENCIES', (*sampler.DEPENDENCIES, 'src/missing-sampler-contract.h')),
                     ('PATCH_SHA256', '0' * 64)]
            for name, value in cases:
                output = root / name
                with self.subTest(guard=name), patch.object(sampler, name, value):
                    with self.assertRaises(ValueError):
                        legacy_source.prepare(self.upstream, output, patch_root=self.patch_root)
                self.assertFalse((output / 'llama-sampler.cpp').exists())
                self.assertFalse((output / 'ggml-webgpu.cpp').exists())

    def test_historical_fusion_and_subgroup_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for module in (reference_source.scale_unary, reference_source.q8_subgroup, reference_source.rms_scale, reference_source.iq3s_subgroup, reference_source.ssm_conv_silu, reference_source.moe_group_skip):
                for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS'):
                    values = getattr(module, field)
                    for index, relative in enumerate(values):
                        output = root / (module.PATCH_NAME + field + str(index))
                        with self.subTest(stage=module.PATCH_NAME, field=field, path=relative), \
                                patch.object(module, field, {**values, relative: '0' * 64}):
                            with self.assertRaises(ValueError):
                                legacy_source.prepare(self.upstream, output, patch_root=self.patch_root)
                        for name in ('ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp', module.HEADER_NAME):
                            self.assertFalse((output / name).exists())
                for field in ('PATCH_SHA256', 'HEADER_SHA256'):
                    output = root / (module.PATCH_NAME + field)
                    with self.subTest(stage=module.PATCH_NAME, field=field), patch.object(module, field, '0' * 64):
                        with self.assertRaises(ValueError):
                            legacy_source.prepare(self.upstream, output, patch_root=self.patch_root)
                    self.assertFalse((output / 'ggml-webgpu.cpp').exists())

    def test_historical_async_copy_guards(self):
        module = reference_source.async_tensor_copy
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in module.CONTRACT_INPUTS:
                values = {**module.CONTRACT_INPUTS, relative: '0' * 64}
                output = root / ('input-' + str(list(values).index(relative)))
                with self.subTest(path=relative), patch.object(module, 'CONTRACT_INPUTS', values):
                    with self.assertRaises(ValueError):
                        legacy_source.prepare(self.upstream, output, patch_root=self.patch_root)
                self.assertFalse((output / 'ggml-webgpu.cpp').exists())
            for field in ('INPUT_SHA256', 'OUTPUT_SHA256', 'PATCH_SHA256'):
                output = root / field
                with self.subTest(field=field), patch.object(module, field, '0' * 64):
                    with self.assertRaises(ValueError):
                        legacy_source.prepare(self.upstream, output, patch_root=self.patch_root)
                self.assertFalse((output / 'ggml-webgpu.cpp').exists())

    def test_host_snapshot_guards_and_configure_dependencies(self):
        module = reference_source.host_input_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for relative in module.CONTRACT_INPUTS:
                values = {**module.CONTRACT_INPUTS, relative: '0' * 64}
                output = root / ('input-' + str(list(values).index(relative)))
                with self.subTest(path=relative), patch.object(module, 'CONTRACT_INPUTS', values):
                    with self.assertRaises(ValueError):
                        legacy_source.prepare(self.upstream, output, patch_root=self.patch_root)
                for name in ('ggml-webgpu.cpp', 'ggml-backend.cpp', 'ggml-backend-host-snapshot.h'):
                    self.assertFalse((output / name).exists())
            for field in ('INPUT_SHA256', 'OUTPUT_SHA256', 'PATCH_SHA256',
                          'BACKEND_OUTPUT_SHA256', 'HEADER_OUTPUT_SHA256'):
                output = root / field
                with self.subTest(field=field), patch.object(module, field, '0' * 64):
                    with self.assertRaises(ValueError):
                        legacy_source.prepare(self.upstream, output, patch_root=self.patch_root)
                for name in ('ggml-webgpu.cpp', 'ggml-backend.cpp', 'ggml-backend-host-snapshot.h'):
                    self.assertFalse((output / name).exists())


    def test_historical_arena_and_graph_regressions(self):
        command = [sys.executable, str(FIXTURE / 'checks/webgpu-param-upload-batching/check_uploads.py'), str(self.upstream), '--patch', str(self.patch_root / 'ggml-webgpu-batch-param-uploads.patch')]
        if os.environ.get('LCB_TEST_DAWN_PACKAGE'):
            command += ['--dawn-package', os.environ['LCB_TEST_DAWN_PACKAGE']]
        subprocess.run(command, check=True)

    def test_default_patch_selection_uses_the_same_frozen_input_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            explicit = legacy_source.prepare(self.upstream, root / 'explicit', patch_root=self.patch_root)
            # An implicit live runtime root would now use the updated, incompatible patches.
            implicit = legacy_source.prepare(self.upstream, root / 'implicit')
            self.assertEqual(explicit.read_bytes(), implicit.read_bytes())
