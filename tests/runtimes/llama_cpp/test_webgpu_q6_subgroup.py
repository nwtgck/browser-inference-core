"""Owned Q6 admission history and independently configured current optional recording."""
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.support.historical_webgpu import historical_source

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path[:0] = [str(ROOT.parents[1]), str(runtime_test_root('llama-cpp'))]
# These tests isolate an accepted historical patch prefix.
from tests.support import frozen_webgpu_source as source
from tests.support.llama_cpp.frozen_webgpu_stages import q6_subgroup as q6
from tests.support.llama_cpp.frozen_prefix_context import without_q6_subgroup
from tests.runtimes.llama_cpp import test_webgpu_quant_half_pairs as pairs
from tests.support import frozen_webgpu_source as legacy_source

class HistoricalQ6Subgroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = cls.enterClassContext(historical_source())
        cls.patch_root = cls.upstream / 'patches'
        cls.temp = tempfile.TemporaryDirectory(prefix='q6-subgroup-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.before, cls.final = {}, {}
        for profile in (False, True):
            with without_q6_subgroup(source):
                cls.before[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile, patch_root=cls.patch_root).parent
            cls.final[profile] = source.prepare(cls.upstream, cls.directory / f'final-{profile}', gpu_profile=profile, patch_root=cls.patch_root).parent

    def seed(self, work, profile, directory):
        target = work / q6.BASE
        target.mkdir(parents=True)
        for name in [*q6.INPUTS[profile], q6.EMBEDDED_HEADER_NAME]:
            shutil.copyfile(directory / name, target / name)

    def test_exact_inverse_all_nonheader_bytes_and_final_identity(self):
        for profile in (False, True):
            before, final = self.before[profile], self.final[profile]
            self.assertEqual({p.name for p in before.iterdir()}, {p.name for p in final.iterdir()})
            self.assertEqual({p.name for p in before.iterdir() if p.read_bytes() != (final / p.name).read_bytes()},
                             {'ggml-webgpu-shader-lib.hpp'})
            for name in q6.INPUTS[profile]:
                self.assertEqual(source.tensor_copy.digest(before / name), q6.INPUTS[profile][name])
                self.assertEqual(source.tensor_copy.digest(final / name), q6.OUTPUTS[profile][name])
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); self.seed(work, profile, final)
                subprocess.run(['git', 'apply', '--no-index', '-R', str(self.patch_root / q6.PATCH_NAME)],
                               cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                self.assertEqual((work / q6.BASE / 'ggml-webgpu-shader-lib.hpp').read_bytes(),
                                 (before / 'ggml-webgpu-shader-lib.hpp').read_bytes())
            stamps = {p.name: p.stat().st_mtime_ns for p in final.iterdir()}
            source.prepare(self.upstream, final, gpu_profile=profile, patch_root=self.patch_root)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in final.iterdir()})
        self.assertEqual(q6.INPUTS, source.subgroup_size_control.OUTPUTS)
        self.assertEqual(q6.OUTPUTS, source.FINAL_OUTPUTS)
        self.assertEqual((self.before[False] / q6.EMBEDDED_HEADER_NAME).read_bytes(),
                         (self.final[False] / q6.EMBEDDED_HEADER_NAME).read_bytes())

    def test_drift_duplicate_and_atomic_profile_to_normal_failure(self):
        for profile in (False, True):
            cases = [('CONTRACT_INPUTS', key) for key in q6.CONTRACT_INPUTS]
            cases += [(field, key) for field in ('INPUTS', 'OUTPUTS') for key in getattr(q6, field)[profile]]
            cases += [('PATCH_SHA256', None)]
            for field, key in cases:
                with self.subTest(profile=profile, field=field, key=key), tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp); self.seed(work, profile, self.before[profile])
                    value = copy.deepcopy(getattr(q6, field))
                    if key is None: value = '0' * 64
                    elif field in ('INPUTS', 'OUTPUTS'): value[profile][key] = '0' * 64
                    else: value[key] = '0' * 64
                    with patch.object(q6, field, value), self.assertRaises(ValueError):
                        q6.prepare(work, self.upstream, self.patch_root, source.patch_environment(work), gpu_profile=profile)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); self.seed(work, profile, self.final[profile])
                with self.assertRaisesRegex(ValueError, 'composition input'):
                    q6.prepare(work, self.upstream, self.patch_root, source.patch_environment(work), gpu_profile=profile)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp); legacy_source.prepare(self.upstream, output, gpu_profile=True, patch_root=self.patch_root)
            previous = {p.name: p.read_bytes() for p in output.iterdir()}
            with patch.object(q6, 'PATCH_SHA256', '0' * 64), self.assertRaises(ValueError):
                legacy_source.prepare(self.upstream, output, patch_root=self.patch_root)
            self.assertEqual(previous, {p.name: p.read_bytes() for p in output.iterdir()})

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_complete_normal_profile_translation_units(self):
        pairs.QuantHalfPairs.test_complete_normal_profile_translation_units_with_pinned_dawn(self)


class CurrentQ6Subgroup(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='current-q6-subgroup-')
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    @unittest.skipUnless(shutil.which('c++'), 'native compiler required')
    def test_complete_normal_profile_factories_cache_source_and_grid(self):
        from tests.support.llama_cpp.current_webgpu import require_recording_inputs
        require_recording_inputs(self, clang_ast=True)
        # Exercise the current full header with the existing pinned SDK/C API
        # recorder, including Q6 rejection memo context changes and Q4/Q8 controls.
        result = subprocess.run([sys.executable, str(runtime_test_root('llama-cpp') / 'webgpu-subgroup-size-control/run_contracts.py'),
                                 '--output', str(self.directory / 'actual-factories')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for name in ('q6_K-fixed-sg16.wgsl', 'q6_K-fixed-sg32.wgsl', 'q6_K-requested.wgsl', 'q6_K-c8.wgsl'):
            base = self.directory / 'actual-factories'
            self.assertEqual((base / 'normal/emissions' / name).read_bytes(),
                             (base / 'profile/emissions' / name).read_bytes())


if __name__ == '__main__': unittest.main()
