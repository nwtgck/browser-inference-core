"""IQ3_S indexed source contracts; GPU/model validation remains separately opt-in."""
import importlib.util
import hashlib
import json
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
HERE = ROOT / 'tests/webgpu-iq3s-subgroup'
sys.path.insert(0, str(ROOT / 'scripts'))
import prepare_webgpu_iq3s_subgroup as iq3
import prepare_webgpu_source as source
import prepare_webgpu_tensor_copy as copy

spec = importlib.util.spec_from_file_location('iq3s_factory', HERE / 'emit_factory.py')
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)


class Iq3sSubgroup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / iq3.BASE / 'ggml-webgpu-shader-lib.hpp').is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def stage(self, destination):
        base = destination / 'base'
        with patch.object(source.host_input_snapshot, 'prepare', return_value={}), \
                patch.object(source.ssm_conv_silu, 'prepare', return_value={}), \
                patch.object(source, 'OUTPUT_SHA256', source.host_input_snapshot.INPUT_SHA256), \
                patch.object(source, 'SSM_HEADER_OUTPUT_SHA256', source.ssm_conv_silu.INPUTS['ggml-webgpu-shader-lib.hpp']):
            factory.prepare_base(self.upstream, base)
        (destination / iq3.BASE).mkdir(parents=True)
        for name in iq3.INPUTS:
            shutil.copyfile(base / name, destination / iq3.BASE / name)
        if copy.digest(destination / iq3.BASE / 'ggml-webgpu-shader-lib.hpp') == iq3.OUTPUTS['ggml-webgpu-shader-lib.hpp']:
            subprocess.run(['git', 'apply', '--reverse', '--no-index',
                            '--include=' + str(iq3.BASE / 'ggml-webgpu.cpp'),
                            '--include=' + str(iq3.BASE / 'ggml-webgpu-shader-lib.hpp'),
                            str((ROOT / source.PATCH_DIRECTORY / iq3.PATCH_NAME).resolve())],
                           cwd=destination, env=source.patch_environment(destination), check=True, capture_output=True)
        return destination

    def test_preparation_reuses_exact_decoder_and_preserves_ordinary_and_matrix_factories(self):
        before = {name: copy.digest(self.upstream / name) for name in iq3.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory() as temporary:
            work = self.stage(Path(temporary))
            header = work / iq3.BASE / 'ggml-webgpu-shader-lib.hpp'
            old = header.read_text()
            extra = iq3.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))
            self.assertEqual(copy.digest(header), iq3.OUTPUTS[header.name])
            self.assertEqual(copy.digest(extra[iq3.HEADER_NAME]), iq3.HEADER_SHA256)
            self.assertFalse((work / 'ggml-wgsl-shaders.hpp').exists())
            expanded = ''.join(re.findall(r'R"wgsl\((.*?)\)wgsl"', extra[iq3.HEADER_NAME].read_text(), re.DOTALL))
            for name in ('common_decls.tmpl', 'mul_mat_vec_acc.tmpl'):
                self.assertIn((self.upstream / iq3.BASE / 'wgsl-shaders' / name).read_text(), expanded)
            new = header.read_text()
            for start, end in [('    webgpu_pipeline get_mul_mat_vec_pipeline(', '    webgpu_pipeline get_mul_mat_fast_pipeline('),
                               ('    webgpu_pipeline get_mul_mat_id_pipeline(', '    webgpu_pipeline get_mul_mat_id_vec_pipeline(')]:
                self.assertEqual(old[old.index(start):old.index(end)], new[new.index(start):new.index(end)])
            shader = (work / iq3.BASE / 'wgsl-shaders' / iq3.SHADER_NAME).read_text()
            self.assertLess(shader.index('if (wg_linear >='), shader.index('ids[params.offset_ids + slot]'))
            self.assertNotIn('var<workgroup>', shader)
            self.assertNotIn('workgroupBarrier', shader)
        self.assertEqual(before, {name: copy.digest(self.upstream / name) for name in before})

    def test_all_contract_patch_composition_and_output_guards_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = self.stage(Path(temporary) / 'stage')
            for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS'):
                for name in getattr(iq3, field):
                    with self.subTest(field=field, name=name), tempfile.TemporaryDirectory() as tmp:
                        work = Path(tmp)
                        shutil.copytree(base / iq3.BASE, work / iq3.BASE)
                        with patch.object(iq3, field, {**getattr(iq3, field), name: '0' * 64}), self.assertRaises(ValueError):
                            iq3.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))
            for field in ('PATCH_SHA256', 'HEADER_SHA256'):
                with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                    work = Path(tmp)
                    shutil.copytree(base / iq3.BASE, work / iq3.BASE)
                    with patch.object(iq3, field, '0' * 64), self.assertRaises(ValueError):
                        iq3.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))

    def test_partial_subgroups_cover_every_logical_lane_and_only_owned_rows(self):
        for width in (16, 32, 64):
            layouts = [[width] * (128 // width)]
            for active in (1, 3, 7, width - 1):
                if active > width:
                    continue
                layout, remaining = [], 128
                while remaining:
                    count = min(active, remaining)
                    layout.append(count)
                    remaining -= count
                layouts.append(layout)
            # Different subgroups may also have different, non-power-of-two counts.
            layout, remaining = [], 128
            while remaining:
                count = min((1, 3, 7, width - 1)[len(layout) % 4], width, remaining)
                layout.append(count)
                remaining -= count
            layouts.append(layout)
            expected = 128 // width
            for layout in layouts:
                with self.subTest(width=width, active=layout):
                    self.assertEqual(sum(layout), 128)
                    if len(layout) == expected:
                        self.assertEqual(layout, [width] * expected)
                    row_visits = [0] * (expected * 4)
                    for group, active in enumerate(layout):
                        # This mirrors the noncollective owner guard. Extra groups
                        # reach reductions, but perform no source reads or stores.
                        if group >= expected:
                            continue
                        logical = [tid for lane in range(active) for tid in range(lane, width, active)]
                        self.assertEqual(sorted(logical), list(range(width)))
                        for row in range(4):
                            row_visits[group * 4 + row] += 1
                    self.assertEqual(row_visits, [1] * (expected * 4))

    def test_runner_rejects_invalid_commands_before_preparation(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / 'not-created'
            for command in ('[]', 'null', '{}', '"node"', '[""]', '["node", 7]', '[invalid'):
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, str(HERE / 'check_gpu.py'), '--llama-source', str(out),
                        '--work', str(out), '--webgpu-module', str(out), '--subgroup-size', '16',
                        '--node-command-json', command], capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('--node-command-json must', result.stderr)
                    self.assertFalse(out.exists())

    @unittest.skipUnless(shutil.which('cc'), 'C compiler needed')
    def test_deterministic_synthetic_fixtures_use_actual_cpu_dequantization(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            subprocess.run([sys.executable, str(HERE / 'make_fixtures.py'), '--llama-source', str(self.upstream),
                            '--work', str(out)], check=True, capture_output=True)
            metadata = json.loads((out / 'fixture-source.json').read_text())
            self.assertIn('Unmodified pinned dequantize_row_iq3_s', metadata['cpuOracle'])
            self.assertEqual(metadata['sourceSha256']['ggml/src/ggml-common.h'],
                             iq3.CONTRACT_INPUTS['ggml/src/ggml-common.h'])
            rows = json.loads((out / 'fixtures/manifest.json').read_text())
            self.assertEqual(len(rows), 8)
            self.assertEqual(sum(row['m'] * row['slots'] for row in rows), 29510)
            for row in rows:
                self.assertEqual(len(row['params']), 13)
                self.assertEqual((out / 'fixtures' / row['referenceFile']).stat().st_size, row['m'] * row['slots'] * 16)
                for key in ('weights', 'input', 'ids', 'reference'):
                    self.assertEqual(hashlib.sha256((out / 'fixtures' / row[key + 'File']).read_bytes()).hexdigest(), row['hashes'][key])

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler needed')
    def test_actual_selector_cache_factory_full_dispatch_and_legacy_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            factory.prepare(self.upstream, out / 'prepared')
            exe = factory.build_host(self.upstream, out / 'prepared', out / 'host')
            result = subprocess.run([str(exe), str(out / 'shaders')], check=True, capture_output=True, text=True)
            self.assertIn('actual IQ3_S selector/cache/factory/dispatch checks passed', result.stdout)
            self.assertEqual(len(list((out / 'shaders').glob('*.wgsl'))), 4)
            for sg in (16, 32, 64):
                text = (out / 'shaders' / f'iq3s-independent-sg{sg}.wgsl').read_text()
                self.assertIn('@workgroup_size(128)', text)
                self.assertIn('if (num_subgroups == expected_subgroups)', text)
                self.assertIn('acc = accumulate_vec_dot(subgroup_invocation_id,', text)
                self.assertIn('let active_count = subgroupAdd(1u)', text)
                self.assertIn('virtual_tid += active_count', text)
                self.assertIn('acc[col][row] += partial[col][row]', text)
                self.assertIn('if (owns_rows && subgroup_invocation_id == 0u', text)
                self.assertEqual(text.count('return;'), 1)
                self.assertLess(text.index('return;'), text.index('subgroupAdd(1u)'))
                self.assertNotIn('#include', text)
                self.assertNotIn('var<workgroup>', text)
            print(result.stdout.strip())

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler needed')
    def test_complete_cpp_with_pinned_dawn_headers_and_profile_modes(self):
        dawn = Path(os.environ.get('LCB_TEST_DAWN_PACKAGE', ROOT.parent / '.tools/emdawnwebgpu_pkg'))
        if not dawn.is_dir():
            self.skipTest('Set LCB_TEST_DAWN_PACKAGE')
        copy.verify_dawn(dawn)
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            factory.prepare(self.upstream, out / 'prepared')
            shader_dir = self.upstream / iq3.BASE / 'wgsl-shaders'
            subprocess.run([sys.executable, str(shader_dir / 'embed_wgsl.py'), '--input_dir', str(shader_dir),
                            '--output_file', str(out / 'ggml-wgsl-shaders.hpp')], check=True)
            includes = [out / 'prepared', out, self.upstream / iq3.BASE, self.upstream / 'ggml/include', self.upstream / 'ggml/src']
            command = ['c++', '-std=c++20', '-fsyntax-only', '-D__wasi__', '-DGGML_SCHED_MAX_COPIES=4']
            command += ['-I' + str(path) for path in includes]
            command += ['-isystem', str(dawn / 'webgpu/include'), '-isystem', str(dawn / 'webgpu_cpp/include')]
            for flags in (['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'],
                          ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS', '-DGGML_WEBGPU_CPU_PROFILE', '-DGGML_WEBGPU_GPU_PROFILE']):
                subprocess.run(command + flags + [str(out / 'prepared/ggml-webgpu.cpp')], check=True)


if __name__ == '__main__':
    unittest.main()
