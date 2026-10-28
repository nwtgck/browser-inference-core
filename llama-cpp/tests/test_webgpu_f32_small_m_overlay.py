"""Source-bound F32 small-M geometry; no GPU or model-speed assertions."""
import importlib.util
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
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'tests')]
import prepare_webgpu_source as source
import prepare_webgpu_f32_small_m as small
import prepare_webgpu_tensor_copy as copy

HERE = ROOT / 'tests/webgpu-f32-small-m'
spec = importlib.util.spec_from_file_location('f32_small_m_factory', HERE / 'emit_factory.py')
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)


class F32SmallMOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temporary = tempfile.TemporaryDirectory(prefix='f32-small-m-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        with patch.object(source.matmul_vec4_layout, 'prepare'), patch.object(small, 'prepare'), patch.object(source.moe_gather_logical32, 'prepare', return_value={}), \
                patch.object(source, 'SSM_HEADER_OUTPUT_SHA256', small.INPUT_SHA256):
            cls.baseline = source.prepare(cls.upstream, cls.directory / 'baseline').parent
        with patch.object(source.matmul_vec4_layout, 'prepare'), patch.object(source.moe_gather_logical32, 'prepare', return_value={}), \
                patch.object(source, 'SSM_HEADER_OUTPUT_SHA256', small.OUTPUT_SHA256):
            cls.f32_stage = source.prepare(cls.upstream, cls.directory / 'f32-stage').parent
        cls.normal = source.prepare(cls.upstream, cls.directory / 'normal').parent
        cls.profile = source.prepare(cls.upstream, cls.directory / 'profile', gpu_profile=True).parent

    def test_only_the_shared_library_changes_and_all_canonical_inputs_remain_exact(self):
        self.assertEqual(copy.digest(self.baseline / small.HEADER_NAME), small.INPUT_SHA256)
        self.assertEqual(copy.digest(self.f32_stage / small.HEADER_NAME), small.OUTPUT_SHA256)
        self.assertEqual(copy.digest(self.normal / small.HEADER_NAME), source.FINAL_OUTPUTS[False]['ggml-webgpu-shader-lib.hpp'])
        self.assertEqual(copy.digest(self.profile / small.HEADER_NAME), source.FINAL_OUTPUTS[True][small.HEADER_NAME])
        self.assertEqual({p.name for p in self.baseline.iterdir()}, {p.name for p in self.f32_stage.iterdir()})
        for path in self.baseline.iterdir():
            if path.name != small.HEADER_NAME:
                self.assertEqual(path.read_bytes(), (self.f32_stage / path.name).read_bytes(), path.name)
        self.assertEqual(copy.digest(self.normal / 'ggml-webgpu.cpp'), source.FINAL_OUTPUTS[False]['ggml-webgpu.cpp'])
        self.assertEqual(copy.digest(self.profile / 'ggml-webgpu.cpp'), source.FINAL_OUTPUTS[True]['ggml-webgpu.cpp'])
        for name, expected in small.CONTRACT_INPUTS.items():
            self.assertEqual(copy.digest(self.upstream / name), expected)
        self.assertFalse((self.normal / 'ggml-wgsl-shaders.hpp').exists())
        # This exact existing CPP controls N<=4 matvec/MMVQ, offsets, all sixteen
        # parameters, physical bindings and the production grid calculation.
        backend = (self.normal / 'ggml-webgpu.cpp').read_text()
        self.assertIn('bool use_mat_vec = (dst->ne[1] <= 4);', backend)

    def test_upstream_contract_input_patch_and_output_drift_fail_closed(self):
        for name in small.CONTRACT_INPUTS:
            with self.subTest(contract=name), patch.dict(small.CONTRACT_INPUTS, {name: '0' * 64}):
                with self.assertRaisesRegex(ValueError, 'contract changed'):
                    source.prepare(self.upstream, self.directory / 'drift')
        for field, error in [('INPUT_SHA256', 'composition input'), ('PATCH_SHA256', 'patch identity'), ('OUTPUT_SHA256', 'output identity')]:
            with self.subTest(field=field), patch.object(small, field, '0' * 64):
                with self.assertRaisesRegex(ValueError, error):
                    source.prepare(self.upstream, self.directory / 'drift')

    def test_all_inputs_trigger_existing_build_reconfiguration_and_updater_checks(self):
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_f32_small_m.py', cmake)
        self.assertIn(small.PATCH_NAME, cmake)
        for name in small.CONTRACT_INPUTS:
            self.assertIn('${LCB_LLAMA_SOURCE}/' + name, cmake)
        updater = (ROOT / 'scripts/update_llama_cpp.py').read_text()
        self.assertIn('webgpu_source.prepare(root / \'vendor/llama.cpp\'', updater)

    def test_actual_patch_rejects_byte_drift_and_duplicate_application(self):
        with tempfile.TemporaryDirectory(dir=self.directory) as temporary:
            work = Path(temporary)
            target = work / small.HEADER_PATH
            target.parent.mkdir(parents=True)
            shutil.copyfile(self.baseline / small.HEADER_NAME, target)
            patches = work / 'patches'
            patches.mkdir()
            actual = ROOT / source.PATCH_DIRECTORY / small.PATCH_NAME
            changed = patches / small.PATCH_NAME
            changed.write_bytes(actual.read_bytes() + b'\n')
            with self.assertRaisesRegex(ValueError, 'patch identity'):
                small.prepare(work, self.upstream, patches, source.patch_environment(work))
            self.assertEqual(copy.digest(target), small.INPUT_SHA256)
            small.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))
            with self.assertRaisesRegex(ValueError, 'composition input'):
                small.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))
            self.assertEqual(copy.digest(target), small.OUTPUT_SHA256)

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_actual_factory_guard_geometry_cache_both_orders_and_dispatch(self):
        executable = factory.build_host(self.upstream, self.baseline, self.normal, self.directory / 'host-normal')
        result = json.loads(subprocess.check_output([str(executable)], text=True))
        self.assertGreaterEqual(result['casesPerOrder'], 250)
        self.assertEqual(result['orders'], 2)
        self.assertGreater(result['checks'], 1000)

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_profile_uses_identical_actual_factory_shaders_and_decisions(self):
        normal = factory.build_host(self.upstream, self.baseline, self.normal, self.directory / 'host-normal-identity')
        profiled = factory.build_host(self.upstream, self.baseline, self.profile, self.directory / 'host-profile')
        normal_shaders = self.directory / 'normal-shaders'
        profile_shaders = self.directory / 'profile-shaders'
        subprocess.run([str(normal), str(normal_shaders)], check=True, capture_output=True)
        subprocess.run([str(profiled), str(profile_shaders)], check=True, capture_output=True)
        self.assertEqual({p.name for p in normal_shaders.iterdir()}, {p.name for p in profile_shaders.iterdir()})
        for path in normal_shaders.iterdir():
            self.assertEqual(path.read_bytes(), (profile_shaders / path.name).read_bytes())


if __name__ == '__main__':
    unittest.main()
