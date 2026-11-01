"""Guarded ordinary Q4 subgroup selection; source/native tests, no model speed claim."""
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
import prepare_webgpu_q4_subgroup as q4
import prepare_webgpu_source as source
from webgpu_prefix_context import without_shared_word_load
import test_webgpu_quant_half_pairs as pair_tests
from matvec_subgroup_test_support import q4_subgroup_helpers
from test_webgpu_bind_group_cache import definition

HERE = ROOT / 'tests/webgpu-q4-subgroup'
spec = importlib.util.spec_from_file_location('q4_subgroup_factory', HERE / 'emit_factory.py')
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


class Q4Subgroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shared_word_prefix = without_shared_word_load(source)
        shared_word_prefix.__enter__()
        cls.addClassCleanup(shared_word_prefix.__exit__, None, None, None)
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='q4-subgroup-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.before, cls.final = {}, {}
        for profile in (False, True):
            with patch.object(q4, 'prepare'):
                cls.before[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}',
                                                     gpu_profile=profile).parent
            cls.final[profile] = source.prepare(cls.upstream, cls.directory / f'final-{profile}',
                                                gpu_profile=profile).parent

    def test_exact_inverse_preserves_cpp_all_embeddings_keys_and_other_factories(self):
        for profile in (False, True):
            old, new = self.before[profile], self.final[profile]
            self.assertEqual({p.name for p in old.iterdir()}, {p.name for p in new.iterdir()})
            self.assertEqual({p.name for p in old.iterdir() if p.read_bytes() != (new / p.name).read_bytes()},
                             {'ggml-webgpu-shader-lib.hpp'})
            for name, expected in q4.INPUTS[profile].items():
                self.assertEqual(sha(old / name), expected)
                self.assertEqual(sha(new / name), q4.OUTPUTS[profile][name])
            self.assertEqual(sha(new / q4.EMBEDDED_HEADER_NAME), q4.EMBEDDED_HEADER_SHA256)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                target = work / q4.BASE / 'ggml-webgpu-shader-lib.hpp'
                target.parent.mkdir(parents=True)
                shutil.copy2(new / target.name, target)
                subprocess.run(['git', 'apply', '--no-index', '-R',
                                str(ROOT / source.PATCH_DIRECTORY / q4.PATCH_NAME)],
                               cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                self.assertEqual(target.read_bytes(), (old / target.name).read_bytes())
            a, b = (old / 'ggml-webgpu-shader-lib.hpp').read_text(), (new / 'ggml-webgpu-shader-lib.hpp').read_text()
            for marker in ('inline uint32_t ggml_webgpu_q8_matvec_subgroup_size(',
                           'struct ggml_webgpu_mul_mat_vec_pipeline_key {',
                           'struct ggml_webgpu_mul_mat_vec_pipeline_key_hash {',
                           '    webgpu_pipeline get_mul_mat_fast_pipeline(',
                           '    webgpu_pipeline get_mul_mat_id_pipeline(',
                           '    webgpu_pipeline get_mul_mat_id_vec_pipeline('):
                self.assertEqual(definition(a, marker), definition(b, marker))

    def test_contract_patch_embedding_input_output_and_atomic_failure(self):
        for profile in (False, True):
            changes = [('CONTRACT_INPUTS', key) for key in q4.CONTRACT_INPUTS]
            changes += [(name, key) for name in ('INPUTS', 'OUTPUTS') for key in getattr(q4, name)[profile]]
            changes += [('PATCH_SHA256', None), ('EMBEDDED_HEADER_SHA256', None)]
            for field, key in changes:
                with self.subTest(profile=profile, field=field, key=key), tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp)
                    base = work / q4.BASE
                    base.mkdir(parents=True)
                    for name in q4.INPUTS[profile]:
                        shutil.copy2(self.before[profile] / name, base / name)
                    shutil.copy2(self.before[profile] / q4.EMBEDDED_HEADER_NAME, work / q4.EMBEDDED_HEADER_NAME)
                    value = copy.deepcopy(getattr(q4, field))
                    if key is None:
                        value = '0' * 64
                    elif field in ('INPUTS', 'OUTPUTS'):
                        value[profile][key] = '0' * 64
                    else:
                        value[key] = '0' * 64
                    with patch.object(q4, field, value), self.assertRaises(ValueError):
                        q4.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY,
                                   source.patch_environment(work), gpu_profile=profile)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                base = work / q4.BASE
                base.mkdir(parents=True)
                for name in q4.INPUTS[profile]:
                    shutil.copy2(self.final[profile] / name, base / name)
                with self.assertRaisesRegex(ValueError, 'composition input'):
                    q4.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY,
                               source.patch_environment(work), gpu_profile=profile)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            source.prepare(self.upstream, output, gpu_profile=True)
            previous = {p.name: p.read_bytes() for p in output.iterdir()}
            with patch.object(q4, 'EMBEDDED_HEADER_SHA256', '0' * 64), self.assertRaises(ValueError):
                source.prepare(self.upstream, output)
            self.assertEqual(previous, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_final_recipe_idempotence_and_historical_extraction_seam(self):
        self.assertEqual(q4.INPUTS, source.quant_wide_m.OUTPUTS)
        self.assertEqual(q4.OUTPUTS, source.aligned_word_load.INPUTS)
        self.assertEqual(q4.CONTRACT_INPUTS, source.q8_subgroup.CONTRACT_INPUTS)
        self.assertEqual(q4.EMBEDDED_HEADER_SHA256, source.q8_subgroup.HEADER_SHA256)
        self.assertEqual(sha(ROOT / source.PATCH_DIRECTORY / q4.PATCH_NAME), q4.PATCH_SHA256)
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_q4_subgroup.py', cmake)
        self.assertIn(q4.PATCH_NAME, cmake)
        self.assertEqual(q4_subgroup_helpers('historical factory without subgroup selection'), '')
        for text in ('ggml_webgpu_q4_matvec_subgroup_size(context, false, false)',
                     'inline bool ggml_webgpu_q4_matvec_subgroup_eligible() {}'):
            with self.assertRaises(ValueError):
                q4_subgroup_helpers(text)
        for profile in (False, True):
            final = self.final[profile]
            self.assertTrue(q4_subgroup_helpers((final / 'ggml-webgpu-shader-lib.hpp').read_text()))
            stamps = {p.name: p.stat().st_mtime_ns for p in final.iterdir()}
            source.prepare(self.upstream, final, gpu_profile=profile)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in final.iterdir()})

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_actual_normal_profile_factory_cache_grid_fallback_and_frozen_arithmetic(self):
        expected = json.loads((HERE / 'shader-identities.json').read_text())
        results = []
        all_shaders = []
        for profile in (False, True):
            output = self.directory / f'host-{profile}'
            exe = factory.build(self.upstream, self.before[profile], self.final[profile], output, profile=profile)
            result = subprocess.run([str(exe), str(output / 'shaders')], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            value = json.loads(result.stdout)
            self.assertEqual(value['status'], 'passed')
            self.assertGreater(value['checks'], 700)
            emitted = {p.name: p.read_bytes() for p in (output / 'shaders').iterdir()}
            self.assertEqual({name: hashlib.sha256(data).hexdigest() for name, data in emitted.items()}, expected)
            results.append(value)
            all_shaders.append(emitted)
        self.assertEqual(results[0], results[1])
        self.assertEqual(all_shaders[0], all_shaders[1])

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_complete_normal_profile_translation_units_with_pinned_headers(self):
        pair_tests.QuantHalfPairs.test_complete_normal_profile_translation_units_with_pinned_dawn(self)


if __name__ == '__main__':
    unittest.main()
