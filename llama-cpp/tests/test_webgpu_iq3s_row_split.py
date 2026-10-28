"""Append only the checked four-row IQ3 embedding; inspect actual factories."""
import copy as copy_module
import hashlib
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
import prepare_webgpu_iq3s_field_loads as fields
import prepare_webgpu_iq3s_row_split as rows
import prepare_webgpu_source as source
import prepare_webgpu_tensor_copy as copy

spec = importlib.util.spec_from_file_location('iq3_rows_factory', ROOT / 'tests/webgpu-iq3s-subgroup/emit_factory.py')
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)


def block(text, marker):
    start = text.index(marker)
    position = text.index('{', start) + 1
    depth = 1
    while depth:
        depth += (text[position] == '{') - (text[position] == '}')
        position += 1
    return text[start:position]


class Iq3sRowSplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temporary = tempfile.TemporaryDirectory(prefix='iq3s-row-split-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        cls.before, cls.final, cls.stage_inputs = {}, {}, {}
        for profile in (False, True):
            captured = cls.directory / f'inputs-{profile}'
            def capture(work, source, patch_root, env, headers, *, gpu_profile):
                del source, patch_root, env
                if gpu_profile != profile:
                    raise AssertionError('Wrong profile passed to appended stage')
                (captured / rows.BASE).mkdir(parents=True)
                for name in rows.INPUTS[profile]:
                    shutil.copyfile(work / rows.BASE / name, captured / rows.BASE / name)
                shutil.copytree(work / 'iq3s-field-loads-source', captured / 'iq3s-field-loads-source')
                shutil.copyfile(headers[rows.HEADER_NAME], captured / rows.HEADER_NAME)
            with patch.object(source, 'replace_iq3s_row_split_embedding', side_effect=capture):
                cls.before[profile] = source.prepare(cls.upstream, cls.directory / f'before-{profile}', gpu_profile=profile).parent
            cls.stage_inputs[profile] = captured
            cls.final[profile] = source.prepare(cls.upstream, cls.directory / f'final-{profile}', gpu_profile=profile).parent

    def stage(self, directory, profile):
        shutil.copytree(self.stage_inputs[profile], directory)
        return rows.prepare(directory, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(directory),
                            previous_header=directory / rows.HEADER_NAME, gpu_profile=profile)

    def test_only_final_iq3_header_changes_and_history_stays_pinned(self):
        self.assertEqual(rows.PRIOR_HEADER_SHA256, fields.HEADER_SHA256)
        self.assertEqual(rows.ACCUMULATOR_INPUT_SHA256, fields.ACCUMULATOR_OUTPUT_SHA256)
        self.assertEqual(rows.ROWS_PER_SUBGROUP, 4)
        self.assertNotEqual(rows.HEADER_SHA256, fields.HEADER_SHA256)
        for profile in (False, True):
            old, new = self.before[profile], self.final[profile]
            self.assertEqual({p.name for p in old.iterdir()}, {p.name for p in new.iterdir()})
            self.assertEqual({p.name for p in old.iterdir() if p.read_bytes() != (new / p.name).read_bytes()}, {rows.HEADER_NAME})
            self.assertEqual(copy.digest(old / rows.HEADER_NAME), fields.HEADER_SHA256)
            self.assertEqual(copy.digest(new / rows.HEADER_NAME), rows.HEADER_SHA256)
            stamps = {p.name: p.stat().st_mtime_ns for p in new.iterdir()}
            source.prepare(self.upstream, new, gpu_profile=profile)
            self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in new.iterdir()})

    def test_complete_inverse_preserves_other_types_main_and_float_body(self):
        canonical = {name: copy.digest(self.upstream / name) for name in rows.CONTRACT_INPUTS}
        for profile in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp) / 'work'
                result = self.stage(work, profile)
                old = (work / 'iq3s-field-loads-source' / rows.ACCUMULATOR_PATH).read_text()
                new = (work / 'iq3s-row-split-source' / rows.ACCUMULATOR_PATH).read_text()
                branch = old[old.index('#ifdef MUL_ACC_IQ3_S\n'):old.index('#ifdef MUL_ACC_IQ4_NL\n')]
                original = block(branch, 'for (var row = 0u; row < OUTPUTS_PER_WG; row++)')
                body = original[original.index('{') + 1:-1]
                split = 'for (var row = 0u; row < 2; row++) {' + body + '}\n        ' + 'for (var row = 2u; row < OUTPUTS_PER_WG; row++) {' + body + '}'
                self.assertEqual(new.replace(split, original), old)
                self.assertEqual(new.count(split), 1)
                for name in (rows.SHADER_NAME, 'common_decls.tmpl'):
                    self.assertEqual((work / 'iq3s-row-split-source' / rows.BASE / 'wgsl-shaders' / name).read_bytes(),
                                     (work / 'iq3s-field-loads-source' / rows.BASE / 'wgsl-shaders' / name).read_bytes())
                self.assertEqual(copy.digest(result), rows.HEADER_SHA256)
                self.assertEqual(copy.digest(work / rows.HEADER_NAME), rows.PRIOR_HEADER_SHA256)
                for name, expected in rows.INPUTS[profile].items():
                    self.assertEqual(copy.digest(work / rows.BASE / name), expected)
        self.assertEqual(canonical, {name: copy.digest(self.upstream / name) for name in canonical})

    def test_contract_prefix_derived_inputs_patch_and_output_fail_closed(self):
        for profile in (False, True):
            cases = [('CONTRACT_INPUTS', key) for key in rows.CONTRACT_INPUTS]
            cases += [('INPUTS', key) for key in rows.INPUTS[profile]]
            cases += [(field, None) for field in ('PRIOR_HEADER_SHA256', 'PATCH_SHA256', 'SHADER_SHA256',
                       'ACCUMULATOR_INPUT_SHA256', 'ACCUMULATOR_OUTPUT_SHA256', 'HEADER_SHA256')]
            for field, key in cases:
                with self.subTest(profile=profile, field=field, key=key), tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp) / 'work'; shutil.copytree(self.stage_inputs[profile], work)
                    previous = (work / rows.HEADER_NAME).read_bytes()
                    changed = copy_module.deepcopy(getattr(rows, field))
                    if field == 'CONTRACT_INPUTS':
                        changed[key] = '0' * 64
                    elif field == 'INPUTS':
                        changed[profile][key] = '0' * 64
                    else:
                        changed = '0' * 64
                    with patch.object(rows, field, changed), self.assertRaises(ValueError):
                        rows.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work),
                                     previous_header=work / rows.HEADER_NAME, gpu_profile=profile)
                    self.assertEqual((work / rows.HEADER_NAME).read_bytes(), previous)
        with patch.object(rows, 'prepare') as prepare, self.assertRaises(KeyError):
            source.replace_iq3s_row_split_embedding(Path('unused'), self.upstream, ROOT / source.PATCH_DIRECTORY, {}, {}, gpu_profile=False)
        prepare.assert_not_called()

    def test_actual_factory_four_rows_sg16_32_64_and_unchanged_fallback(self):
        # These are processed shader identities, independent of labels or cache order.
        expected = {16: '75b1ae4c3f115d68aed07b3a6d01688114085a81fec7b0052f581691cdf8658c',
                    32: 'c1a1caffce956490fd1294ba301fcf190a76a26bcb5d3cbe87718aa9000b328b',
                    64: '8a088145ac612cb6ab0cf24ec29cde878453d8943a1deb1706a3daa3bad54c9e'}
        outputs = {}
        for profile in (False, True):
            for final in (False, True):
                prepared = self.final[profile] if final else self.before[profile]
                root = self.directory / f'factory-{profile}-{final}'
                executable = factory.build_host(self.upstream, prepared, root)
                subprocess.run([str(executable), str(root / 'shaders')], check=True, capture_output=True)
                outputs[profile, final] = {p.name: p.read_text() for p in (root / 'shaders').glob('*.wgsl')}
                for sg in (16, 32, 64):
                    text = outputs[profile, final][f'iq3s-independent-sg{sg}.wgsl']
                    self.assertIn('var acc: array<array<f32, 4>, 1>;', text)
                    self.assertIn('-> array<array<f32, 4>, 1>', text)
                    if not final:
                        continue
                    original = outputs[profile, False][f'iq3s-independent-sg{sg}.wgsl']
                    loop = block(original, 'for (var row = 0u; row < 4; row++)')
                    body = loop[loop.index('{') + 1:-1]
                    split = 'for (var row = 0u; row < 2; row++) {' + body + '}\n        ' + 'for (var row = 2u; row < 4; row++) {' + body + '}'
                    self.assertEqual(text.replace(split, loop), original)
                    self.assertEqual(text.count(split), 1)
                    self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), expected[sg])
                    for k in (2048, 2560, 4096):
                        for tid in range(sg):
                            blocks = range(tid // 16, k // 256, sg // 16)
                            for m, base in ((257, 256), (640, 636), (640, 639)):
                                old = [(block, row) for block in blocks for row in range(4) if base + row < m]
                                split_visits = [(block, row) for block in blocks for lo, hi in ((0, 2), (2, 4)) for row in range(lo, hi) if base + row < m]
                                self.assertEqual(split_visits, old)
                if final:
                    self.assertEqual(outputs[profile, final]['iq3s-baseline.wgsl'], outputs[profile, False]['iq3s-baseline.wgsl'])
        self.assertEqual(outputs[False, True], outputs[True, True])

    def test_build_dependencies_and_final_factory_contract_are_explicit(self):
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_iq3s_row_split.py', cmake)
        self.assertIn(rows.PATCH_NAME, cmake)
        for name in rows.CONTRACT_INPUTS:
            self.assertIn('${LCB_LLAMA_SOURCE}/' + name, cmake)
        for prepared in self.final.values():
            header = (prepared / 'ggml-webgpu-shader-lib.hpp').read_text()
            for macro in ('FLOAT', 'LEGACY_Q', 'K_Q'):
                self.assertRegex(header, rf'#define WEBGPU_MUL_MAT_VEC_{macro}_OUTPUTS_PER_WG\s+4\b')
            self.assertIn('defines.push_back(std::string("OUTPUTS_PER_WG=") + std::to_string(outputs_per_wg));', header)


if __name__ == '__main__':
    unittest.main()
