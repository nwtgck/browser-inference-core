"""Exact host capacity only; source/allocator checks do not claim GPU speed."""
import copy as deepcopy_module
import importlib.util
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
import prepare_webgpu_moe_tight_tile_bound as bound
import prepare_webgpu_source as source
import prepare_webgpu_tensor_copy as copy

spec = importlib.util.spec_from_file_location('moe_bound_emitter', ROOT / 'tests/webgpu-moe-tight-tile-bound/emit_host.py')
emitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(emitter)


def prepare_prefix(*args, **kwargs):
    # This fixture isolates the historical host-bound stage.
    with patch.object(source, 'replace_iq3s_row_split_embedding'):
        return source.prepare(*args, **kwargs)


class MoeTightTileBound(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temporary = tempfile.TemporaryDirectory(prefix='moe-tight-tile-bound-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.before, cls.final = {}, {}
        for profile in (False, True):
            with patch.object(bound, 'prepare'):
                cls.before[profile] = prepare_prefix(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile).parent
            cls.final[profile] = prepare_prefix(cls.upstream, cls.directory / f'after-{profile}', gpu_profile=profile).parent
        cls.library = None

    def test_only_capacity_helper_changes_and_inverse_restores_complete_sources(self):
        before = {name: copy.digest(self.upstream / name) for name in bound.CONTRACT_INPUTS}
        for profile in (False, True):
            old, final = self.before[profile], self.final[profile]
            self.assertEqual({p.name for p in old.iterdir()}, {p.name for p in final.iterdir()})
            self.assertEqual({p.name for p in old.iterdir() if p.read_bytes() != (final / p.name).read_bytes()}, {'ggml-webgpu.cpp'})
            text = (final / 'ggml-webgpu.cpp').read_text()
            expected = (old / 'ggml-webgpu.cpp').read_text()
            self.assertEqual(text.replace(
                '    const uint32_t max_active_experts = std::min(n_expert, n_slots);\n'
                '    return max_active_experts + (n_slots - max_active_experts) / (WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N);',
                '    return CEIL_DIV(n_slots, WEBGPU_MUL_MAT_TILE_N * WEBGPU_MUL_MAT_WG_SIZE_N) + std::min(n_expert, n_slots);'), expected)
            self.assertEqual(copy.digest(old / 'ggml-webgpu.cpp'), bound.INPUTS[profile]['ggml-webgpu.cpp'])
            self.assertEqual(copy.digest(final / 'ggml-webgpu.cpp'), bound.OUTPUTS[profile]['ggml-webgpu.cpp'])
            with tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                (work / bound.BASE).mkdir(parents=True)
                shutil.copyfile(final / 'ggml-webgpu.cpp', work / bound.BASE / 'ggml-webgpu.cpp')
                subprocess.run(['git', 'apply', '--no-index', '--reverse', str(ROOT / source.PATCH_DIRECTORY / bound.PATCH_NAME)],
                               cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                self.assertEqual((work / bound.BASE / 'ggml-webgpu.cpp').read_bytes(), (old / 'ggml-webgpu.cpp').read_bytes())
        self.assertEqual(before, {name: copy.digest(self.upstream / name) for name in before})

    def test_inputs_patch_outputs_and_reapplication_fail_closed(self):
        for profile in (False, True):
            cases = [('CONTRACT_INPUTS', key) for key in bound.CONTRACT_INPUTS]
            cases += [(field, key) for field in ('INPUTS', 'OUTPUTS') for key in getattr(bound, field)[profile]]
            cases += [('PATCH_SHA256', None)]
            for field, key in cases:
                with self.subTest(profile=profile, field=field, key=key), tempfile.TemporaryDirectory() as temporary:
                    work = Path(temporary)
                    (work / bound.BASE).mkdir(parents=True)
                    for name in bound.INPUTS[profile]:
                        shutil.copyfile(self.before[profile] / name, work / bound.BASE / name)
                    changed = deepcopy_module.deepcopy(getattr(bound, field))
                    if field == 'CONTRACT_INPUTS':
                        changed[key] = '0' * 64
                    elif key:
                        changed[profile][key] = '0' * 64
                    else:
                        changed = '0' * 64
                    with patch.object(bound, field, changed), self.assertRaises(ValueError):
                        bound.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work), gpu_profile=profile)
            with tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                (work / bound.BASE).mkdir(parents=True)
                for name in bound.INPUTS[profile]:
                    shutil.copyfile(self.final[profile] / name, work / bound.BASE / name)
                with self.assertRaisesRegex(ValueError, 'composition input'):
                    bound.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work), gpu_profile=profile)

    def test_failed_prepare_preserves_existing_outputs_and_repeat_preserves_mtimes(self):
        for profile in (False, True):
            output = self.final[profile]
            saved = {p.name: p.read_bytes() for p in output.iterdir()}
            stamps = {p.name: p.stat().st_mtime_ns for p in output.iterdir()}
            with patch.object(bound, 'PATCH_SHA256', '0' * 64), self.assertRaisesRegex(ValueError, 'patch identity'):
                prepare_prefix(self.upstream, output, gpu_profile=profile)
            self.assertEqual(saved, {p.name: p.read_bytes() for p in output.iterdir()})
            prepare_prefix(self.upstream, output, gpu_profile=profile)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in output.iterdir()})

    def test_final_and_historical_prefix_contracts_and_build_inputs_are_distinct(self):
        self.assertEqual(bound.INPUTS, source.matmul_vec4_layout.OUTPUTS)
        self.assertNotEqual(source.FINAL_OUTPUTS, source.matmul_vec4_layout.OUTPUTS)
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_moe_tight_tile_bound.py', cmake)
        self.assertIn(bound.PATCH_NAME, cmake)
        for name in bound.CONTRACT_INPUTS:
            self.assertIn('${LCB_LLAMA_SOURCE}/' + name, cmake)
        for profile in (False, True):
            for name, expected in source.FINAL_OUTPUTS[profile].items():
                self.assertEqual(copy.digest(self.final[profile] / name), expected)
            hpp = (self.final[profile] / 'ggml-webgpu-shader-lib.hpp').read_text()
            self.assertIn('#define WEBGPU_MUL_MAT_TILE_N           4', hpp)
            self.assertIn('#define WEBGPU_MUL_MAT_WG_SIZE_N        8', hpp)

    @classmethod
    def native_library(cls):
        if cls.library:
            return cls.library
        configured = os.environ.get('LCB_TEST_GGML_BASE_LIBRARY')
        if configured:
            cls.library = Path(configured).resolve()
            return cls.library
        build = cls.directory / 'native-build'
        subprocess.run(['cmake', '-S', str(cls.upstream), '-B', str(build), '-DBUILD_SHARED_LIBS=OFF',
                        '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF', '-DGGML_CCACHE=OFF', '-DLLAMA_BUILD_TESTS=OFF',
                        '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_TOOLS=OFF', '-DLLAMA_BUILD_COMMON=OFF', '-DLLAMA_CURL=OFF'],
                       check=True, capture_output=True)
        subprocess.run(['cmake', '--build', str(build), '--target', 'ggml-base', '-j', '2'], check=True, capture_output=True)
        cls.library = build / 'ggml/src/libggml-base.a'
        return cls.library

    @unittest.skipUnless(shutil.which('c++'), 'Native compiler required')
    def test_actual_allocator_dispatcher_integer_extremes_binding_capacity_and_guards(self):
        results = []
        for profile in (False, True):
            executable = emitter.build(self.upstream, self.before[profile], self.final[profile],
                                       self.directory / f'host-{profile}', self.native_library(), profile=profile)
            result = subprocess.check_output([str(executable)], text=True)
            self.assertIn('real ggml allocator / recorded dispatcher / capacity and CPU guard checks passed', result)
            results.append(result)
        self.assertEqual(results[0], results[1])
        print(results[0].strip())

    @unittest.skipUnless(shutil.which('c++'), 'Native compiler required')
    def test_complete_normal_and_profile_cpp_syntax_with_pinned_dawn_headers(self):
        dawn = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parent / '.tools/emdawnwebgpu_pkg'))
        if not dawn.is_dir():
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        copy.verify_dawn(dawn)
        with tempfile.TemporaryDirectory() as temporary:
            work = Path(temporary)
            shaders = self.upstream / bound.BASE / 'wgsl-shaders'
            subprocess.run([sys.executable, str(shaders / 'embed_wgsl.py'), '--input_dir', str(shaders),
                            '--output_file', str(work / 'ggml-wgsl-shaders.hpp')], check=True)
            for profile in (False, True):
                flags = ['-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE'] if profile else []
                includes = [self.final[profile], work, self.upstream / bound.BASE, self.upstream / 'ggml/include', self.upstream / 'ggml/src']
                subprocess.run(['c++', '-std=c++20', '-fsyntax-only', '-D__wasi__', '-DGGML_SCHED_MAX_COPIES=4',
                                '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS', *flags, *['-I' + str(p) for p in includes],
                                '-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include'),
                                str(self.final[profile] / 'ggml-webgpu.cpp')], check=True)


if __name__ == '__main__':
    unittest.main()
