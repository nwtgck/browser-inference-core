"""Checked final composition without invoking a browser, runtime or GPU."""
import copy
import os
import io
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_source as source
import prepare_webgpu_subgroup_size_control as optional
sys.path.insert(0, str(ROOT / 'tests'))
from webgpu_prefix_context import without_subgroup_size_control, without_q6_subgroup
from webgpu_embedding_test_support import embedding_sha256, legacy_fixed_sources


class SubgroupSizeControl(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temp = tempfile.TemporaryDirectory(prefix='subgroup-size-control-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        q6_prefix = without_q6_subgroup(source)
        q6_prefix.__enter__()
        cls.addClassCleanup(q6_prefix.__exit__, None, None, None)
        cls.before, cls.final = {}, {}
        for profile in (False, True):
            with without_subgroup_size_control(source):
                cls.before[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile).parent
            cls.final[profile] = source.prepare(cls.upstream, cls.directory / f'final-{profile}', gpu_profile=profile).parent

    def seed(self, work, profile):
        target = work / optional.BASE
        target.mkdir(parents=True)
        for name in optional.INPUTS[profile]:
            shutil.copyfile(self.before[profile] / name, target / name)
        headers = {}
        for name in optional.SHARED_INPUTS:
            path = target / name if name == 'ggml-webgpu-batch-readback.hpp' else work / name
            shutil.copyfile(self.before[profile] / name, path)
            headers[name] = path
        return headers

    def prepare(self, work, profile, headers):
        return optional.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY,
                                source.patch_environment(work), gpu_profile=profile, headers=headers)

    def test_exact_inverse_and_unchanged_other_sources(self):
        for profile in (False, True):
            before, final = self.before[profile], self.final[profile]
            old = {p.name: p.read_bytes() for p in before.iterdir()}
            new = {p.name: p.read_bytes() for p in final.iterdir()}
            self.assertEqual(new.keys() - old.keys(), {optional.HELPER_NAME})
            self.assertEqual(old.keys() - new.keys(), set())
            self.assertEqual({name for name in old if old[name] != new[name]},
                             {*optional.INPUTS[profile], *optional.SHARED_INPUTS})
            for name, expected in {**optional.OUTPUTS[profile], **optional.GENERATED_OUTPUTS}.items():
                self.assertEqual(source.tensor_copy.digest(final / name), expected)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                for name in [*optional.INPUTS[profile], *optional.GENERATED_OUTPUTS]:
                    path = work / optional.BASE / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(new[name])
                for patch_name, _ in (optional.PATCHES[profile], optional.SHARED_PATCH):
                    subprocess.run(['git', 'apply', '--no-index', '-R', str(ROOT / source.PATCH_DIRECTORY / patch_name)],
                                   cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                for name in [*optional.INPUTS[profile], *optional.SHARED_INPUTS]:
                    self.assertEqual((work / optional.BASE / name).read_bytes(), old[name])
                self.assertFalse((work / optional.BASE / optional.HELPER_NAME).exists())
        for name in optional.GENERATED_OUTPUTS:
            self.assertEqual((self.final[False] / name).read_bytes(), (self.final[True] / name).read_bytes())

    def test_stage_drift_and_reapplication_fail_closed(self):
        for profile in (False, True):
            cases = [('INPUTS', name) for name in optional.INPUTS[profile]]
            cases += [('SHARED_INPUTS', name) for name in optional.SHARED_INPUTS]
            cases += [('GENERATED_OUTPUTS', optional.HELPER_NAME), ('PATCHES', None), ('SHARED_PATCH', None)]
            for field, name in cases:
                with self.subTest(profile=profile, field=field, name=name), tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp); headers = self.seed(work, profile)
                    value = copy.deepcopy(getattr(optional, field))
                    if field == 'PATCHES': value[profile] = (value[profile][0], '0' * 64)
                    elif field == 'SHARED_PATCH': value = (value[0], '0' * 64)
                    elif field == 'INPUTS': value[profile][name] = '0' * 64
                    else: value[name] = '0' * 64
                    with patch.object(optional, field, value), self.assertRaises(ValueError):
                        self.prepare(work, profile, headers)
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp); headers = self.seed(work, profile)
                self.prepare(work, profile, headers)
                with self.assertRaisesRegex(ValueError, 'composition input'):
                    self.prepare(work, profile, headers)

    def test_isolation_and_existing_helper_rejected(self):
        for path in [self.upstream, self.upstream / 'output', self.upstream.parent]:
            with self.assertRaisesRegex(ValueError, 'outside the upstream'):
                self.prepare(path, False, {})
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp); headers = self.seed(work, False)
            (work / optional.BASE / optional.HELPER_NAME).write_text('unreviewed')
            with self.assertRaisesRegex(ValueError, 'unreviewed source'):
                self.prepare(work, False, headers)
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp); headers = self.seed(work, False)
            path = work / optional.BASE / 'ggml-webgpu.cpp'
            path.unlink(); path.symlink_to(self.before[False] / path.name)
            with self.assertRaisesRegex(ValueError, 'inside the isolated'):
                self.prepare(work, False, headers)

    def test_final_contract_and_idempotence(self):
        self.assertEqual(optional.INPUTS, source.checkpoint_readback.OUTPUTS)
        self.assertEqual(optional.OUTPUTS, source.FINAL_OUTPUTS)
        self.assertEqual(optional.REVIEWED_DAWN_HEADERS, source.tensor_copy.DAWN_HEADERS)
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_subgroup_size_control.py', cmake)
        for name, expected in [*optional.PATCHES.values(), optional.SHARED_PATCH]:
            self.assertIn(name, cmake)
            self.assertEqual(source.tensor_copy.digest(ROOT / source.PATCH_DIRECTORY / name), expected)
        for profile in (False, True):
            final = self.final[profile]
            stamps = {p.name: p.stat().st_mtime_ns for p in final.iterdir()}
            source.prepare(self.upstream, final, gpu_profile=profile)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in final.iterdir()})

    def test_current_entry_uses_final_embedding_and_checked_legacy_inverse(self):
        for profile in (False, True):
            final, before = self.final[profile], self.before[profile]
            self.assertEqual(embedding_sha256(final, optional.EMBEDDED_HEADER_NAME, 'unreviewed'),
                             optional.GENERATED_OUTPUTS[optional.EMBEDDED_HEADER_NAME])
            legacy = legacy_fixed_sources(final)
            for name in ['ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp']:
                self.assertEqual(legacy[name], (before / name).read_text())

    def test_wrapper_reuses_upstream_chunk_and_reserve_contract(self):
        path = self.final[False] / optional.EMBEDDED_HEADER_NAME
        text = path.read_text()
        parts = [m.group(2) for m in re.finditer(r'R"([A-Za-z0-9_]*)\((.*?)\)\1"', text, re.S)]
        logical = ''.join(parts)
        self.assertTrue(parts and all(len(part) <= 60000 for part in parts))
        self.assertEqual(int(re.search(r'tmp.reserve\((\d+)\)', text).group(1)), len(logical))
        emitter = runpy.run_path(str(self.upstream / optional.BASE / 'wgsl-shaders/embed_wgsl.py'))
        output = io.StringIO(); output.write('// Auto-generated shader embedding\n#include <string>\n\n')
        emitter['write_shader']('q8_subgroup_matvec', logical, None, output, str(path.parent))
        self.assertEqual(output.getvalue(), text)
        inverse = logical.replace('#ifdef REQUESTED_SUBGROUP_SIZE\nenable subgroup_size_control;\n#endif\n', '')
        inverse = inverse.replace('#ifdef REQUESTED_SUBGROUP_SIZE\n@subgroup_size(REQUESTED_SUBGROUP_SIZE)\n#endif\n', '')
        before = (self.before[False] / optional.EMBEDDED_HEADER_NAME).read_text()
        old_logical = ''.join(m.group(2) for m in re.finditer(r'R"([A-Za-z0-9_]*)\((.*?)\)\1"', before, re.S))
        self.assertEqual(inverse, old_logical)


if __name__ == '__main__': unittest.main()
