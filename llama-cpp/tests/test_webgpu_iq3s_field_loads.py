"""The appended IQ3 field stage changes one dedicated embedding only."""
import hashlib
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_iq3s_field_loads as fields
import prepare_webgpu_iq3s_subgroup as original
import prepare_webgpu_source as source
from webgpu_prefix_context import without_final_matvec_stages
import prepare_webgpu_tensor_copy as copy


class Iq3sFieldLoads(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / fields.ACCUMULATOR_PATH).is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def prior_stage(self, work):
        # Prepare the original three emitter inputs without changing any host
        # stage. Its accepted patch and original output pin remain independent.
        original_patch = ROOT / source.PATCH_DIRECTORY / original.PATCH_NAME
        self.assertEqual(copy.digest(original_patch), original.PATCH_SHA256)
        text = original_patch.read_text().split('+++ b/' + str(original.BASE / 'wgsl-shaders' / original.SHADER_NAME), 1)[1]
        text = text.split('\ndiff --git ', 1)[0]
        main = ''.join(line[1:] + '\n' for line in text.splitlines() if line.startswith('+') and not line.startswith('+++'))
        self.assertEqual(hashlib.sha256(main.encode()).hexdigest(), fields.SHADER_SHA256)
        input_dir = work / 'original-inputs'
        input_dir.mkdir(parents=True)
        (input_dir / original.SHADER_NAME).write_text(main)
        for name in ('common_decls.tmpl', 'mul_mat_vec_acc.tmpl'):
            shutil.copyfile(self.upstream / original.BASE / 'wgsl-shaders' / name, input_dir / name)
        main_path = work / original.BASE / 'wgsl-shaders' / original.SHADER_NAME
        main_path.parent.mkdir(parents=True)
        main_path.write_text(main)
        previous = work / original.HEADER_NAME
        subprocess.run([sys.executable, str(self.upstream / original.BASE / 'wgsl-shaders/embed_wgsl.py'),
                        '--input_dir', str(input_dir), '--output_file', str(previous)], check=True, capture_output=True)
        self.assertEqual(copy.digest(previous), original.HEADER_SHA256)
        return previous

    def test_only_iq3_accumulator_fields_change_and_original_stage_survives(self):
        before = {name: copy.digest(self.upstream / name) for name in fields.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            previous = self.prior_stage(work)
            old_header = previous.read_bytes()
            sentinels = {name: b'Unchanged host/factory source' for name in ('ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp')}
            for name, data in sentinels.items():
                (work / name).write_bytes(data)
            result = fields.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work),
                                    previous_header=previous)
            self.assertEqual(result.name, original.HEADER_NAME)
            self.assertNotEqual(result, previous)
            self.assertEqual(copy.digest(result), fields.HEADER_SHA256)
            self.assertEqual(previous.read_bytes(), old_header)
            for name, data in sentinels.items():
                self.assertEqual((work / name).read_bytes(), data)
            self.assertFalse((work / 'ggml-wgsl-shaders.hpp').exists())
            patched = (work / 'iq3s-field-loads-source' / fields.ACCUMULATOR_PATH).read_text()
            canonical = (self.upstream / fields.ACCUMULATOR_PATH).read_text()
            before_iq3, original_rest = canonical.split('#ifdef MUL_ACC_IQ3_S\n', 1)
            original_iq3, after_iq3 = original_rest.split('#ifdef MUL_ACC_IQ4_NL\n', 1)
            new_before, new_rest = patched.split('#ifdef MUL_ACC_IQ3_S\n', 1)
            new_iq3, new_after = new_rest.split('#ifdef MUL_ACC_IQ4_NL\n', 1)
            self.assertEqual((new_before, new_after), (before_iq3, after_iq3))
            arithmetic = re.compile(r'^.*(?:let d =|let db =|let b1 =|let b2 =|let s1 =|let s2 =|row_sum \+=|acc\[col\]\[row\] \+=).*$', re.M)
            self.assertEqual(arithmetic.findall(new_iq3), arithmetic.findall(original_iq3))
            self.assertIn('let selected_qs = load_u32_at_src0(', new_iq3)
            self.assertIn('let sign_pair = load_u16_at_src0(', new_iq3)
            private_inputs = result.parent.parent / 'iq3s-field-loads-source' / original.BASE / 'wgsl-shaders'
            self.assertEqual((private_inputs / original.SHADER_NAME).read_bytes(),
                             (work / original.BASE / 'wgsl-shaders' / original.SHADER_NAME).read_bytes())
            self.assertEqual((private_inputs / 'common_decls.tmpl').read_bytes(),
                             (self.upstream / original.BASE / 'wgsl-shaders/common_decls.tmpl').read_bytes())
        self.assertEqual(before, {name: copy.digest(self.upstream / name) for name in before})

    def test_contract_prior_main_patch_and_outputs_fail_closed(self):
        cases = [(name, key) for name in ('CONTRACT_INPUTS',) for key in getattr(fields, name)]
        cases += [(name, None) for name in ('PRIOR_HEADER_SHA256', 'SHADER_SHA256', 'PATCH_SHA256',
                                          'ACCUMULATOR_OUTPUT_SHA256', 'HEADER_SHA256')]
        for name, key in cases:
            with self.subTest(field=name, path=key), tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                previous = self.prior_stage(work)
                before = previous.read_bytes()
                replacement = {**getattr(fields, name), key: '0' * 64} if key else '0' * 64
                with patch.object(fields, name, replacement), self.assertRaises(ValueError):
                    fields.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work),
                                   previous_header=previous)
                self.assertEqual(previous.read_bytes(), before)

    def test_normal_profile_change_one_header_and_preserve_shared_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for profile in (False, True):
                with self.subTest(profile=profile):
                    with without_final_matvec_stages(source), patch.object(fields, 'prepare', side_effect=lambda work, source, patch_root, env, *, previous_header: previous_header):
                        baseline = source.prepare(self.upstream, root / f'before-{profile}', gpu_profile=profile).parent
                    with without_final_matvec_stages(source):
                        candidate = source.prepare(self.upstream, root / f'after-{profile}', gpu_profile=profile).parent
                    old = {p.name: p.read_bytes() for p in baseline.iterdir() if p.is_file()}
                    new = {p.name: p.read_bytes() for p in candidate.iterdir() if p.is_file()}
                    self.assertEqual(set(old), set(new))
                    self.assertEqual([name for name in old if old[name] != new[name]], [fields.HEADER_NAME])
                    self.assertEqual(copy.digest(baseline / fields.HEADER_NAME), original.HEADER_SHA256)
                    self.assertEqual(copy.digest(candidate / fields.HEADER_NAME), fields.HEADER_SHA256)

    def test_build_dependencies_and_original_pin_are_explicit(self):
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_iq3s_field_loads.py', cmake)
        self.assertIn(fields.PATCH_NAME, cmake)
        for name in fields.CONTRACT_INPUTS:
            self.assertIn(name, cmake)
        self.assertEqual(fields.PRIOR_HEADER_SHA256, original.HEADER_SHA256)
        self.assertNotEqual(fields.HEADER_SHA256, original.HEADER_SHA256)
        with patch.object(fields, 'prepare') as prepare, self.assertRaises(KeyError):
            source.replace_iq3s_field_embedding(Path('unused'), self.upstream, ROOT / source.PATCH_DIRECTORY, {}, {})
        prepare.assert_not_called()


if __name__ == '__main__':
    unittest.main()
