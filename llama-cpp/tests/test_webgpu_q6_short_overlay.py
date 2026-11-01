"""Source-bound ordinary Q6_K short workgroups; no target-device speed claim."""
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
import prepare_webgpu_q6_short as q6
import prepare_webgpu_source as source

spec = importlib.util.spec_from_file_location('q6_short_factory', ROOT / 'tests/webgpu-q6-short/emit_factory.py')
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()


class Q6ShortOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='q6-short-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        q4_tail = patch.object(source.q4_subgroup, 'prepare')
        q4_tail.start()
        cls.addClassCleanup(q4_tail.stop)
        wide = patch.object(source.quant_wide_m, 'prepare')
        wide.start()
        cls.addClassCleanup(wide.stop)
        direct = patch.object(source.direct_mapped_read, 'prepare')
        direct.start()
        cls.addClassCleanup(direct.stop)
        tail = patch.object(source.quant_half_pairs, 'prepare', return_value={})
        tail.start()
        cls.addClassCleanup(tail.stop)
        cls.before, cls.final = {}, {}
        for profile in (False, True):
            with patch.object(q6, 'prepare'):
                cls.before[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile).parent
            cls.final[profile] = source.prepare(cls.upstream, cls.directory / f'final-{profile}', gpu_profile=profile).parent

    def test_only_header_changes_and_full_inverse_preserves_every_other_file(self):
        for profile in (False, True):
            old, final = self.before[profile], self.final[profile]
            self.assertEqual({p.name for p in old.iterdir()}, {p.name for p in final.iterdir()})
            changed = {p.name for p in old.iterdir() if p.read_bytes() != (final / p.name).read_bytes()}
            self.assertEqual(changed, {'ggml-webgpu-shader-lib.hpp'})
            for name in q6.INPUTS[profile]:
                self.assertEqual(sha(old / name), q6.INPUTS[profile][name])
                self.assertEqual(sha(final / name), q6.OUTPUTS[profile][name])
                self.assertEqual(q6.OUTPUTS[profile][name], source.quant_half_pairs.INPUTS[profile][name])
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); target = work / q6.BASE; target.mkdir(parents=True)
                shutil.copy2(final / 'ggml-webgpu-shader-lib.hpp', target / 'ggml-webgpu-shader-lib.hpp')
                subprocess.run(['git', 'apply', '--no-index', '-R', str(ROOT / source.PATCH_DIRECTORY / q6.PATCH_NAME)],
                               cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                self.assertEqual((target / 'ggml-webgpu-shader-lib.hpp').read_bytes(), (old / 'ggml-webgpu-shader-lib.hpp').read_bytes())
            # The indexed factory itself is outside the appended patch.
            marker = '    webgpu_pipeline get_mul_mat_id_vec_pipeline('
            self.assertEqual(factory.definition((old / 'ggml-webgpu-shader-lib.hpp').read_text(), marker),
                             factory.definition((final / 'ggml-webgpu-shader-lib.hpp').read_text(), marker))

    def test_contract_patch_input_output_duplicate_and_atomic_drift(self):
        for profile in (False, True):
            cases = [('CONTRACT_INPUTS', key) for key in q6.CONTRACT_INPUTS]
            cases += [(name, key) for name in ('INPUTS', 'OUTPUTS') for key in getattr(q6, name)[profile]]
            cases += [('PATCH_SHA256', None)]
            for name, key in cases:
                with self.subTest(profile=profile, name=name, key=key), tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp); target = work / q6.BASE; target.mkdir(parents=True)
                    for filename in q6.INPUTS[profile]:
                        shutil.copy2(self.before[profile] / filename, target / filename)
                    changed = copy.deepcopy(getattr(q6, name))
                    if name == 'PATCH_SHA256': changed = '0' * 64
                    elif name == 'CONTRACT_INPUTS': changed[key] = '0' * 64
                    else: changed[profile][key] = '0' * 64
                    with patch.object(q6, name, changed), self.assertRaises(ValueError):
                        q6.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work), gpu_profile=profile)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); target = work / q6.BASE; target.mkdir(parents=True)
                for name in q6.INPUTS[profile]: shutil.copy2(self.final[profile] / name, target / name)
                with self.assertRaisesRegex(ValueError, 'composition input'):
                    q6.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work), gpu_profile=profile)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp); source.prepare(self.upstream, output, gpu_profile=True)
            saved = {p.name: p.read_bytes() for p in output.iterdir()}
            with patch.object(q6, 'PATCH_SHA256', '0' * 64), self.assertRaises(ValueError):
                source.prepare(self.upstream, output)
            self.assertEqual(saved, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_stage_inventory_contracts_and_idempotence(self):
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_q6_short.py', cmake)
        self.assertIn(q6.PATCH_NAME, cmake)
        self.assertEqual(sha(ROOT / source.PATCH_DIRECTORY / q6.PATCH_NAME), q6.PATCH_SHA256)
        self.assertEqual(q6.INPUTS[False], {name: source.f32_matvec_epilogue.OUTPUTS[False][name] for name in q6.INPUTS[False]})
        for profile in (False, True):
            final = self.final[profile]
            stamps = {p.name: p.stat().st_mtime_ns for p in final.iterdir()}
            source.prepare(self.upstream, final, gpu_profile=profile)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in final.iterdir()})

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_actual_normal_profile_selector_factory_cache_and_canonical_shaders(self):
        results = []
        expected = json.loads((ROOT / 'tests/webgpu-q6-short/shader-identities.json').read_text())
        for profile in (False, True):
            output = self.directory / f'host-{profile}'
            exe = factory.build(self.upstream, self.before[profile], self.final[profile], output, profile=profile)
            result = subprocess.run([str(exe), str(output / 'shaders')], check=True, capture_output=True, text=True)
            data = json.loads(result.stdout)
            self.assertGreater(data['checks'], 6000)
            self.assertEqual(data['status'], 'passed')
            self.assertEqual({p.name: sha(p) for p in (output / 'shaders').iterdir()}, expected)
            results.append(data)
        self.assertEqual(results[0], results[1])

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_complete_normal_profile_source_with_pinned_dawn_headers(self):
        dawn = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parent / '.tools/emdawnwebgpu_pkg'))
        if not dawn.is_dir(): self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        identity = json.loads((ROOT / 'tests/webgpu-scale-unary/inputs.json').read_text())
        for name, expected in identity['dawn_header_sha256'].items():
            self.assertEqual(sha(dawn / name), expected)
        shaders = self.upstream / q6.BASE / 'wgsl-shaders'
        subprocess.run([sys.executable, str(shaders / 'embed_wgsl.py'), '--input_dir', str(shaders),
                        '--output_file', str(self.directory / 'ggml-wgsl-shaders.hpp')], check=True, capture_output=True)
        for profile in (False, True):
            includes = [self.final[profile], self.directory, self.upstream / q6.BASE,
                        self.upstream / 'ggml/include', self.upstream / 'ggml/src']
            command = ['c++', '-std=c++20', '-fsyntax-only', '-D__wasi__', '-DGGML_SCHED_MAX_COPIES=4',
                       '-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'] + ['-I' + str(p) for p in includes]
            command += ['-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include')]
            if profile: command += ['-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE']
            subprocess.run(command + [str(self.final[profile] / 'ggml-webgpu.cpp')], check=True, capture_output=True)


if __name__ == '__main__': unittest.main()
