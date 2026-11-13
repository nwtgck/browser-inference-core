"""Actual Q4 factory specialization, source contracts and shared-pair ownership."""
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

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path[:0] = [str(ROOT.parents[1]), str(runtime_test_root('llama-cpp'))]
# These tests isolate an accepted historical patch prefix.
from tests.support import legacy_webgpu_source as source
from tests.runtimes.llama_cpp.webgpu_prefix_context import without_shared_word_load
from tests.support.llama_cpp.webgpu_stages import quant_half_pairs as pairs
from tests.runtimes.llama_cpp.test_webgpu_bind_group_cache import definition

HERE = runtime_test_root('llama-cpp') / 'webgpu-quant-half-pairs'
spec = importlib.util.spec_from_file_location('half_pair_factory', runtime_test_root('llama-cpp') / 'webgpu-f32-small-m/emit_factory.py')
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


from tests.support import legacy_webgpu_source as legacy_source

class QuantHalfPairs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        shared_word_prefix = without_shared_word_load(source)
        shared_word_prefix.__enter__()
        cls.addClassCleanup(shared_word_prefix.__exit__, None, None, None)
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='quant-half-pairs-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        q4_tail = patch.object(legacy_source.q4_subgroup, 'prepare')
        q4_tail.start()
        cls.addClassCleanup(q4_tail.stop)
        wide = patch.object(legacy_source.quant_wide_m, 'prepare')
        wide.start()
        cls.addClassCleanup(wide.stop)
        tail = patch.object(legacy_source.direct_mapped_read, 'prepare')
        tail.start()
        cls.addClassCleanup(tail.stop)
        cls.before, cls.final, cls.derived = {}, {}, {}
        for profile in (False, True):
            with patch.object(pairs, 'prepare', return_value={}):
                cls.before[profile] = legacy_source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile).parent
            original = pairs.prepare
            def capture(work, upstream, patches, env, *, gpu_profile):
                result = original(work, upstream, patches, env, gpu_profile=gpu_profile)
                cls.derived[profile] = cls.directory / f'derived-{profile}'
                for name in pairs.DERIVED_SHA256:
                    target = cls.derived[profile] / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(work / pairs.BASE / name, target)
                return result
            with patch.object(pairs, 'prepare', side_effect=capture):
                cls.final[profile] = legacy_source.prepare(cls.upstream, cls.directory / f'final-{profile}', gpu_profile=profile).parent

    def test_exact_inverse_retains_all_prefix_files_and_canonical_templates(self):
        for profile in (False, True):
            old, new = self.before[profile], self.final[profile]
            self.assertEqual({p.name for p in new.iterdir()} - {p.name for p in old.iterdir()}, {pairs.HEADER_NAME})
            self.assertEqual({p.name for p in old.iterdir() if p.read_bytes() != (new / p.name).read_bytes()},
                             {'ggml-webgpu-shader-lib.hpp'})
            self.assertEqual(sha(new / pairs.HEADER_NAME), pairs.HEADER_SHA256)
            for name, expected in pairs.INPUTS[profile].items():
                self.assertEqual(sha(old / name), expected)
                self.assertEqual(sha(new / name), pairs.OUTPUTS[profile][name])
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); base = work / pairs.BASE; base.mkdir(parents=True)
                shutil.copy2(new / 'ggml-webgpu-shader-lib.hpp', base / 'ggml-webgpu-shader-lib.hpp')
                for name, expected in pairs.DERIVED_SHA256.items():
                    p = base / name; p.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(self.derived[profile] / name, p)
                    self.assertEqual(sha(p), expected)
                subprocess.run(['git', 'apply', '--no-index', '-R', str(ROOT / source.PATCH_DIRECTORY / pairs.PATCH_NAME)],
                               cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                self.assertEqual((base / 'ggml-webgpu-shader-lib.hpp').read_bytes(), (old / 'ggml-webgpu-shader-lib.hpp').read_bytes())
                for target, original in pairs.TEMPLATE_COPIES.items():
                    self.assertEqual((base / 'wgsl-shaders' / target).read_bytes(),
                                     (self.upstream / pairs.BASE / 'wgsl-shaders' / original).read_bytes())
            # All other factories and the whole host TU retain their exact bytes.
            for marker in ('    webgpu_pipeline get_mul_mat_id_pipeline(', '    webgpu_pipeline get_mul_mat_id_vec_pipeline(',
                           '    webgpu_pipeline get_mul_mat_vec_pipeline('):
                self.assertEqual(definition((old / 'ggml-webgpu-shader-lib.hpp').read_text(), marker),
                                 definition((new / 'ggml-webgpu-shader-lib.hpp').read_text(), marker))

    def test_inputs_drift_duplicate_and_atomic_failure(self):
        for profile in (False, True):
            mutations = [('CONTRACT_INPUTS', key) for key in pairs.CONTRACT_INPUTS]
            mutations += [(name, key) for name in ('INPUTS', 'OUTPUTS') for key in getattr(pairs, name)[profile]]
            mutations += [('DERIVED_SHA256', key) for key in pairs.DERIVED_SHA256]
            mutations += [('PATCH_SHA256', None), ('HEADER_SHA256', None)]
            for field, key in mutations:
                with self.subTest(profile=profile, field=field, key=key), tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp); base = work / pairs.BASE; base.mkdir(parents=True)
                    for name in pairs.INPUTS[profile]: shutil.copy2(self.before[profile] / name, base / name)
                    value = copy.deepcopy(getattr(pairs, field))
                    if key is None: value = '0' * 64
                    elif field in ('INPUTS', 'OUTPUTS'): value[profile][key] = '0' * 64
                    else: value[key] = '0' * 64
                    with patch.object(pairs, field, value), self.assertRaises(ValueError):
                        pairs.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work), gpu_profile=profile)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); base = work / pairs.BASE; base.mkdir(parents=True)
                for name in pairs.INPUTS[profile]: shutil.copy2(self.final[profile] / name, base / name)
                with self.assertRaisesRegex(ValueError, 'composition input'):
                    pairs.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work), gpu_profile=profile)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp); legacy_source.prepare(self.upstream, directory, gpu_profile=True)
            before = {p.name: p.read_bytes() for p in directory.iterdir()}
            with patch.object(pairs, 'HEADER_SHA256', '0' * 64), self.assertRaises(ValueError):
                legacy_source.prepare(self.upstream, directory)
            self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})

    def test_stage_inventory_and_idempotent_publication(self):
        self.assertEqual(sha(ROOT / source.PATCH_DIRECTORY / pairs.PATCH_NAME), pairs.PATCH_SHA256)
        self.assertEqual(pairs.INPUTS, source.q6_short.OUTPUTS)
        self.assertEqual(pairs.OUTPUTS, source.direct_mapped_read.INPUTS)
        for profile in (False, True):
            final = self.final[profile]; stamps = {p.name: p.stat().st_mtime_ns for p in final.iterdir()}
            source.prepare(self.upstream, final, gpu_profile=profile)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in final.iterdir()})
        self.assertFalse((self.final[False] / 'ggml-wgsl-shaders.hpp').exists())

    def test_pair_ownership_k_order_and_unchanged_global_loaders(self):
        # Independently enumerate complete shared pairs, including the boundary
        # between the two 2048-byte tiles. No new global vector load is used.
        for dtype in ('Q4_0', 'Q4_K'):
            writes = [0] * 512
            for thread in range(64):
                if dtype == 'Q4_0':
                    for i in range(thread * 16, 1024, 64 * 16):
                        block, offset = divmod(i, 32)
                        start = block * 32 + (offset // 16) * 8
                        for j in (0, 4):
                            for k in (0, 2):
                                for half in (0, 16): writes[(start + j + k + half) // 2] += 1
                else:
                    for i in range(thread * 4, 1024, 64 * 4):
                        writes[i // 2] += 1; writes[i // 2 + 1] += 1
            self.assertEqual(writes, [1] * 512)
        x_writes = [0] * 512
        for thread in range(64):
            for i in range(thread * 2, 1024, 128): x_writes[i // 2] += 1
        self.assertEqual(x_writes, [1] * 512)
        self.assertEqual([k + component for k in range(0, 32, 2) for component in (0, 1)], list(range(32)))
        raw = (self.derived[False] / 'wgsl-shaders/mul_mat_reg_tile_half_pairs.wgsl').read_text()
        old = (self.upstream / pairs.BASE / 'wgsl-shaders/mul_mat_reg_tile.wgsl').read_text()
        self.assertEqual(raw.count('workgroupBarrier();'), 2)
        self.assertEqual(raw[:raw.index('fn get_local_n(')], old[:old.index('fn get_local_n(')].replace(
            '#include "mul_mat_decls.tmpl"', '#include "mul_mat_half_pairs_decls.tmpl"'))
        self.assertEqual(raw[raw.index('    let dst_batch_offset ='):], old[old.index('    let dst_batch_offset ='):])
        canonical = (self.upstream / pairs.BASE / 'wgsl-shaders/mul_mat_decls.tmpl').read_text()
        derived = (self.derived[False] / 'wgsl-shaders/mul_mat_half_pairs_decls.tmpl').read_text()
        # Every src0 loader (including Q4_K half expressions and packed byte
        # reads) remains verbatim; only the shared store helper is replaced.
        self.assertIn(definition(canonical[canonical.index('// k-quants'):], 'fn init_shmem_src0('), derived)
        self.assertIn(canonical[canonical.index('// legacy-quants'):canonical.index('#endif // legacy-quants')], derived)
        self.assertEqual(4096, (1024 + 1024) // 2 * 4)

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_actual_normal_profile_factory_cache_fallback_and_grid(self):
        records = []
        shaders = []
        for profile in (False, True):
            output = self.directory / f'host-{profile}'
            executable = factory.build_host(self.upstream, self.before[profile], self.final[profile], output,
                                            cases_path=HERE / 'cases.cpp')
            emitted = output / 'shaders'
            result = json.loads(subprocess.check_output([str(executable), str(emitted)], text=True))
            self.assertEqual(result['status'], 'passed'); self.assertGreater(result['checks'], 1000)
            records.append(result)
            shaders.append({p.name: p.read_bytes() for p in emitted.iterdir()})
        self.assertEqual(records[0], records[1]); self.assertEqual(shaders[0], shaders[1])
        expected = json.loads((HERE / 'shader-identities.json').read_text())
        self.assertEqual({name: hashlib.sha256(data).hexdigest() for name, data in shaders[0].items()}, expected)

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_unreviewed_geometry_retains_actual_canonical_factory(self):
        for name, value in [('WEBGPU_MUL_MAT_TILE_M', 8), ('WEBGPU_MUL_MAT_TILE_N', 8),
                            ('WEBGPU_MUL_MAT_WG_SIZE_M', 4), ('WEBGPU_MUL_MAT_REG_TILE_K_QUANT', 64)]:
            with self.subTest(name=name):
                executable = factory.build_host(self.upstream, self.before[False], self.final[False],
                                                self.directory / name, cases_path=HERE / 'cases.cpp',
                                                constant_overrides={name: value})
                result = json.loads(subprocess.check_output([str(executable)], text=True))
                self.assertEqual(result['status'], 'passed')

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_complete_normal_profile_translation_units_with_pinned_dawn(self):
        dawn = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parents[1] / '.tools/emdawnwebgpu_pkg'))
        if not dawn.is_dir(): self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        identity = json.loads((runtime_test_root('llama-cpp') / 'webgpu-scale-unary/inputs.json').read_text())
        for name, expected in identity['dawn_header_sha256'].items(): self.assertEqual(sha(dawn / name), expected)
        shaders = self.upstream / pairs.BASE / 'wgsl-shaders'
        subprocess.run([sys.executable, str(shaders / 'embed_wgsl.py'), '--input_dir', str(shaders),
                        '--output_file', str(self.directory / 'ggml-wgsl-shaders.hpp')], check=True, capture_output=True)
        for profile in (False, True):
            includes = [self.final[profile], self.directory, self.upstream / pairs.BASE,
                        self.upstream / 'ggml/include', self.upstream / 'ggml/src']
            cmd = ['c++', '-std=c++20', '-fsyntax-only', '-D__wasi__', '-DGGML_SCHED_MAX_COPIES=4',
                   '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'] + ['-I' + str(p) for p in includes]
            cmd += ['-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include')]
            if profile: cmd += ['-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE']
            subprocess.run(cmd + [str(self.final[profile] / 'ggml-webgpu.cpp')], check=True, capture_output=True)


if __name__ == '__main__': unittest.main()
