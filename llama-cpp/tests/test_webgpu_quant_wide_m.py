"""Guarded Q4 wide-M mode, actual factories/cache and unchanged fallback."""
import copy
import hashlib
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
import prepare_webgpu_quant_wide_m as wide
from test_webgpu_bind_group_cache import definition
import test_webgpu_quant_half_pairs as pair_tests

HERE = ROOT / 'tests/webgpu-quant-wide-m'
spec = importlib.util.spec_from_file_location('wide_m_factory', ROOT / 'tests/webgpu-f32-small-m/emit_factory.py')
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


class QuantWideM(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='quant-wide-m-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.before, cls.final = {}, {}
        for profile in (False, True):
            with patch.object(wide, 'prepare'):
                cls.before[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile).parent
            cls.final[profile] = source.prepare(cls.upstream, cls.directory / f'final-{profile}', gpu_profile=profile).parent

    def test_exact_inverse_and_all_non_matrix_paths_stay_unchanged(self):
        for profile in (False, True):
            old, new = self.before[profile], self.final[profile]
            self.assertEqual({p.name for p in old.iterdir()}, {p.name for p in new.iterdir()})
            self.assertEqual({p.name for p in old.iterdir() if p.read_bytes() != (new / p.name).read_bytes()},
                             {'ggml-webgpu-shader-lib.hpp'})
            for name, expected in wide.INPUTS[profile].items():
                self.assertEqual(sha(old / name), expected)
                self.assertEqual(sha(new / name), wide.OUTPUTS[profile][name])
            self.assertEqual(sha(new / source.quant_half_pairs.HEADER_NAME), wide.EMBEDDED_HEADER_SHA256)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); target = work / wide.BASE / 'ggml-webgpu-shader-lib.hpp'; target.parent.mkdir(parents=True)
                shutil.copy2(new / target.name, target)
                subprocess.run(['git', 'apply', '--no-index', '-R', str(ROOT / source.PATCH_DIRECTORY / wide.PATCH_NAME)],
                               cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                self.assertEqual(target.read_bytes(), (old / target.name).read_bytes())
            for marker in ('    webgpu_pipeline get_mul_mat_vec_pipeline(', '    webgpu_pipeline get_mul_mat_id_pipeline(',
                           '    webgpu_pipeline get_mul_mat_id_vec_pipeline('):
                self.assertEqual(definition((old / 'ggml-webgpu-shader-lib.hpp').read_text(), marker),
                                 definition((new / 'ggml-webgpu-shader-lib.hpp').read_text(), marker))

    def test_contract_input_patch_embedding_output_and_atomic_drift(self):
        for profile in (False, True):
            mutations = [('CONTRACT_INPUTS', key) for key in wide.CONTRACT_INPUTS]
            mutations += [(field, key) for field in ('INPUTS', 'OUTPUTS') for key in getattr(wide, field)[profile]]
            mutations += [('PATCH_SHA256', None), ('EMBEDDED_HEADER_SHA256', None)]
            for field, key in mutations:
                with self.subTest(profile=profile, field=field, key=key), tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp); base = work / wide.BASE; base.mkdir(parents=True)
                    for name in wide.INPUTS[profile]: shutil.copy2(self.before[profile] / name, base / name)
                    shutil.copy2(self.before[profile] / source.quant_half_pairs.HEADER_NAME,
                                 work / source.quant_half_pairs.HEADER_NAME)
                    value = copy.deepcopy(getattr(wide, field))
                    if key is None: value = '0' * 64
                    elif field in ('INPUTS', 'OUTPUTS'): value[profile][key] = '0' * 64
                    else: value[key] = '0' * 64
                    with patch.object(wide, field, value), self.assertRaises(ValueError):
                        wide.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY,
                                     source.patch_environment(work), gpu_profile=profile)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); base = work / wide.BASE; base.mkdir(parents=True)
                for name in wide.INPUTS[profile]: shutil.copy2(self.final[profile] / name, base / name)
                with self.assertRaisesRegex(ValueError, 'composition input'):
                    wide.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY,
                                 source.patch_environment(work), gpu_profile=profile)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source.prepare(self.upstream, root, gpu_profile=True)
            before = {p.name: p.read_bytes() for p in root.iterdir()}
            with patch.object(wide, 'EMBEDDED_HEADER_SHA256', '0' * 64), self.assertRaises(ValueError):
                source.prepare(self.upstream, root)
            self.assertEqual(before, {p.name: p.read_bytes() for p in root.iterdir()})

    def test_final_recipe_dependencies_and_idempotence(self):
        self.assertEqual(wide.INPUTS, source.direct_mapped_read.OUTPUTS)
        self.assertEqual(wide.OUTPUTS, source.FINAL_OUTPUTS)
        self.assertEqual(wide.CONTRACT_INPUTS, source.quant_half_pairs.CONTRACT_INPUTS)
        self.assertEqual(wide.EMBEDDED_HEADER_SHA256, source.quant_half_pairs.HEADER_SHA256)
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_quant_wide_m.py', cmake)
        self.assertIn(wide.PATCH_NAME, cmake)
        self.assertEqual(sha(ROOT / source.PATCH_DIRECTORY / wide.PATCH_NAME), wide.PATCH_SHA256)
        for profile in (False, True):
            final = self.final[profile]; stamps = {p.name: p.stat().st_mtime_ns for p in final.iterdir()}
            source.prepare(self.upstream, final, gpu_profile=profile)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in final.iterdir()})

    def test_wide_pair_ownership_and_storage(self):
        for dtype in ('Q4_0', 'Q4_K'):
            writes = [0] * 1024
            for thread in range(64):
                if dtype == 'Q4_0':
                    for i in range(thread * 16, 2048, 64 * 16):
                        block, offset = divmod(i, 32); start = block * 32 + (offset // 16) * 8
                        for j in (0, 4):
                            for k in (0, 2):
                                for half in (0, 16): writes[(start + j + k + half) // 2] += 1
                else:
                    for i in range(thread * 4, 2048, 64 * 4):
                        writes[i // 2] += 1; writes[i // 2 + 1] += 1
            self.assertEqual(writes, [1] * 1024)
        x = [0] * 512
        for thread in range(64):
            for i in range(thread * 2, 1024, 128): x[i // 2] += 1
        self.assertEqual(x, [1] * 512)
        self.assertEqual(6144, (2048 + 1024) // 2 * 4)
        self.assertEqual(32, 8 * 4)

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_actual_normal_profile_modes_cache_grid_and_frozen_wgsl(self):
        results = []
        all_shaders = []
        for profile in (False, True):
            output = self.directory / f'host-{profile}'
            executable = factory.build_host(self.upstream, self.before[profile], self.final[profile], output,
                                            cases_path=HERE / 'cases.cpp')
            result = json.loads(subprocess.check_output([str(executable), str(output / 'shaders')], text=True))
            self.assertEqual(result['status'], 'passed'); self.assertGreater(result['checks'], 5000)
            results.append(result)
            all_shaders.append({p.name: p.read_bytes() for p in (output / 'shaders').iterdir()})
        self.assertEqual(results[0], results[1]); self.assertEqual(all_shaders[0], all_shaders[1])
        expected = json.loads((HERE / 'shader-identities.json').read_text())
        self.assertEqual({name: hashlib.sha256(data).hexdigest() for name, data in all_shaders[0].items()}, expected)
        # The four-row kernels remain the existing public recipe. Eight-row
        # identities are source-bound targets, without embedding private paths.
        for dtype, narrow, large in [('q4_0', 'd3f92e81e39f966c0dfca317f6e715510916b673790c6c8029c91723117a9d4d',
                                     '34bc8ff7885593338aa50317d0e5577a124cf9bf26c5f7c31a00da56d7af142e'),
                                    ('q4_K', '129e15fcc66b0e011d08b6a078035430ae71be29d386165f613ce4307a2301a9',
                                     'c15bedade8a40763557c3b83fd556a84833fb90e6509e21dd32aa3c4ccb1810e')]:
            stem = 'mul_mat_reg_tile_' + dtype + '_f32_half_pairs'
            self.assertEqual(expected[stem + '-candidate.wgsl'], narrow)
            self.assertEqual(expected[stem + '_m8-candidate.wgsl'], large)

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_unreviewed_base_geometry_keeps_the_original_factory(self):
        executable = factory.build_host(self.upstream, self.before[False], self.final[False],
                                        self.directory / 'unreviewed-geometry', cases_path=HERE / 'cases.cpp',
                                        constant_overrides={'WEBGPU_MUL_MAT_TILE_M': 8})
        self.assertEqual(json.loads(subprocess.check_output([str(executable)], text=True))['status'], 'passed')

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_complete_normal_profile_translation_units(self):
        # Reuse the pinned-header compiler gate with these final source paths.
        pair_tests.QuantHalfPairs.test_complete_normal_profile_translation_units_with_pinned_dawn(self)


if __name__ == '__main__': unittest.main()
