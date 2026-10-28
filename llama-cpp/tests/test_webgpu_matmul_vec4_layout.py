"""Source-bound vec4 layout admission; no GPU/model/performance claim."""
import copy as deepcopy_module
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
import prepare_webgpu_matmul_vec4_layout as layout
import prepare_webgpu_source as source
import prepare_webgpu_tensor_copy as copy

spec = importlib.util.spec_from_file_location('matmul_vec4_emitter', ROOT / 'tests/webgpu-matmul-vec4/emit_host.py')
emitter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(emitter)


class MatmulVec4Layout(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='matmul-vec4-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.baseline, cls.final = {}, {}
        for profile in (False, True):
            with patch.object(source.moe_tight_tile_bound, 'prepare'), patch.object(layout, 'prepare'):
                cls.baseline[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile).parent
            with patch.object(source.moe_tight_tile_bound, 'prepare'):
                cls.final[profile] = source.prepare(cls.upstream, cls.directory / f'after-{profile}', gpu_profile=profile).parent

    def test_only_caller_and_factory_change_and_inverse_restores_full_prefix(self):
        before = {name: copy.digest(self.upstream / name) for name in layout.CONTRACT_INPUTS}
        for profile in (False, True):
            old, final = self.baseline[profile], self.final[profile]
            self.assertEqual(set(p.name for p in old.iterdir()), set(p.name for p in final.iterdir()))
            changed = {p.name for p in old.iterdir() if p.read_bytes() != (final / p.name).read_bytes()}
            self.assertEqual(changed, set(layout.OUTPUTS[profile]))
            for name in changed:
                self.assertEqual(copy.digest(old / name), layout.INPUTS[profile][name])
                self.assertEqual(copy.digest(final / name), layout.OUTPUTS[profile][name])
            with tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                (work / layout.BASE).mkdir(parents=True)
                for name in changed:
                    shutil.copyfile(final / name, work / layout.BASE / name)
                subprocess.run(['git', 'apply', '--no-index', '-R',
                                str(ROOT / source.PATCH_DIRECTORY / layout.PATCHES[profile][0])],
                               cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                for name in changed:
                    self.assertEqual((work / layout.BASE / name).read_bytes(), (old / name).read_bytes())
        self.assertEqual(before, {name: copy.digest(self.upstream / name) for name in before})

    def test_contract_patch_input_output_and_duplicate_drift_fail_closed(self):
        for profile in (False, True):
            cases = [('CONTRACT_INPUTS', key) for key in layout.CONTRACT_INPUTS]
            cases += [(name, key) for name in ('INPUTS', 'OUTPUTS') for key in getattr(layout, name)[profile]]
            cases += [('PATCHES', None)]
            for name, key in cases:
                with self.subTest(profile=profile, field=name, key=key), tempfile.TemporaryDirectory() as temporary:
                    work = Path(temporary)
                    (work / layout.BASE).mkdir(parents=True)
                    for filename in layout.INPUTS[profile]:
                        shutil.copyfile(self.baseline[profile] / filename, work / layout.BASE / filename)
                    changed = deepcopy_module.deepcopy(getattr(layout, name))
                    if name == 'PATCHES':
                        changed[profile] = (changed[profile][0], '0' * 64)
                    elif name == 'CONTRACT_INPUTS':
                        changed[key] = '0' * 64
                    else:
                        changed[profile][key] = '0' * 64
                    with patch.object(layout, name, changed), self.assertRaises(ValueError):
                        layout.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY,
                                       source.patch_environment(work), gpu_profile=profile)
            with tempfile.TemporaryDirectory() as temporary:
                work = Path(temporary)
                (work / layout.BASE).mkdir(parents=True)
                for filename in layout.INPUTS[profile]:
                    shutil.copyfile(self.final[profile] / filename, work / layout.BASE / filename)
                with self.assertRaisesRegex(ValueError, 'composition input'):
                    layout.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY,
                                   source.patch_environment(work), gpu_profile=profile)

    def test_failed_normal_stage_preserves_existing_profile_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            source.prepare(self.upstream, output, gpu_profile=True)
            saved = {p.name: p.read_bytes() for p in output.iterdir()}
            with patch.object(layout, 'PATCHES', {**layout.PATCHES, False: (layout.PATCHES[False][0], '0' * 64)}):
                with self.assertRaisesRegex(ValueError, 'patch identity'):
                    source.prepare(self.upstream, output)
            self.assertEqual(saved, {p.name: p.read_bytes() for p in output.iterdir()})

    def test_build_and_historical_prefix_contracts_are_explicit(self):
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_matmul_vec4_layout.py', cmake)
        for name, digest in layout.PATCHES.values():
            self.assertIn(name, cmake)
            self.assertEqual(copy.digest(ROOT / source.PATCH_DIRECTORY / name), digest)
        for name in layout.CONTRACT_INPUTS:
            self.assertIn('${LCB_LLAMA_SOURCE}/' + name, cmake)
        self.assertEqual(source.OUTPUT_SHA256, layout.INPUTS[False]['ggml-webgpu.cpp'])
        self.assertEqual(source.q8_diagnostic.OUTPUTS, layout.INPUTS[True])
        for profile in (False, True):
            hpp = (self.final[profile] / 'ggml-webgpu-shader-lib.hpp').read_text()
            self.assertEqual(hpp.count('vec4_aligned && context.src0->ne[0] % 4'), 4)
            self.assertNotIn('bool vec4_aligned =', hpp)

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_complete_actual_callers_factories_cache_parameters_and_profile_agree(self):
        reports = {}
        for profile in (False, True):
            output = self.directory / f'host-{profile}'
            executable = emitter.build(self.upstream, self.baseline[profile], self.final[profile], output, profile=profile)
            report = json.loads(subprocess.check_output([str(executable), str(output / 'shaders')], text=True))
            self.assertGreater(report['cases'], 5000)
            self.assertGreater(report['checks'], 30000)
            self.assertGreater(report['newScalarFallbacks'], 100)
            self.assertGreater(report['unusedStride'], 0)
            self.assertGreater(report['mergedBindings'], 0)
            self.assertGreater(report['mmvq'], 0)
            self.assertGreater(report['nonfloat'], 0)
            self.assertGreater(report['subgroupMatrix'], 0)
            self.assertTrue({22, 23, 24, 25, 26}.issubset({row['scenario'] for row in report['rows']}))
            reports[profile] = report
        self.assertEqual(reports[False], reports[True])
        old = self.directory / 'host-False/shaders'
        final = self.directory / 'host-True/shaders'
        self.assertEqual({p.name for p in old.iterdir()}, {p.name for p in final.iterdir()})
        for shader in old.iterdir():
            self.assertEqual(shader.read_bytes(), (final / shader.name).read_bytes())


if __name__ == '__main__':
    unittest.main()
