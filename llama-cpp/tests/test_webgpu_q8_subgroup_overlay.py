"""Q8 subgroup source contracts; GPU and real-model execution remain opt-in."""
import importlib.util
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
import prepare_webgpu_q8_subgroup as q8
import prepare_webgpu_source as source
from webgpu_prefix_context import without_final_matvec_stages
import prepare_webgpu_tensor_copy as copy

spec = importlib.util.spec_from_file_location('q8_subgroup_factory', ROOT / 'tests/webgpu-q8-subgroup/emit_factory.py')
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)


class Q8SubgroupOverlay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        if not (cls.upstream / q8.BASE / 'ggml-webgpu-shader-lib.hpp').is_file():
            raise unittest.SkipTest('Set LCB_TEST_LLAMA_SOURCE')

    def test_preparation_reuses_upstream_accumulator_and_preserves_upstream(self):
        before = {name: copy.digest(self.upstream / name) for name in q8.CONTRACT_INPUTS}
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            factory.prepare(self.upstream, out)
            self.assertEqual(copy.digest(out / 'ggml-webgpu-shader-lib.hpp'), source.FINAL_OUTPUTS[False]['ggml-webgpu-shader-lib.hpp'])
            self.assertEqual(copy.digest(out / q8.HEADER_NAME), q8.HEADER_SHA256)
            self.assertFalse((out / 'ggml-wgsl-shaders.hpp').exists())
            embedded = (out / q8.HEADER_NAME).read_text()
            expanded = ''.join(re.findall(r'R"wgsl\((.*?)\)wgsl"', embedded, re.DOTALL))
            for name in ('common_decls.tmpl', 'mul_mat_vec_acc.tmpl'):
                self.assertIn((self.upstream / q8.BASE / 'wgsl-shaders' / name).read_text(), expanded)
            self.assertNotIn('#include "mul_mat_vec_acc.tmpl"', embedded)
        self.assertEqual(before, {name: copy.digest(self.upstream / name) for name in before})

    def test_stage_fails_closed_on_contract_patch_composition_and_output_drift(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch.object(source.rms_scale, 'prepare', return_value={}), \
                    patch.object(source.async_tensor_copy, 'prepare'), \
                    patch.object(source.iq3s_subgroup, 'prepare', return_value={}), \
                    patch.object(source, 'replace_iq3s_field_embedding'), without_final_matvec_stages(source), patch.object(source.moe_tight_tile_bound, 'prepare'), patch.object(source.matmul_vec4_layout, 'prepare'), \
                    patch.object(source.host_input_snapshot, 'prepare', return_value={}), \
                    patch.object(source.ssm_conv_silu, 'prepare', return_value={}), \
                    patch.object(source.moe_group_skip, 'prepare', return_value={}), \
                patch.object(source.router_tail, 'prepare', return_value={}), \
                    patch.object(source.host_encode_assembly, 'prepare'), \
                    patch.object(source.f32_small_m, 'prepare'), \
                    patch.object(source.moe_gather_logical32, 'prepare', return_value={}), \
                    patch.object(source, 'OUTPUT_SHA256', q8.OUTPUTS['ggml-webgpu.cpp']), \
                    patch.object(source, 'SSM_HEADER_OUTPUT_SHA256', q8.OUTPUTS['ggml-webgpu-shader-lib.hpp']):
                source.prepare(self.upstream, root / 'base')
            for field in ('CONTRACT_INPUTS', 'INPUTS', 'OUTPUTS'):
                for name in getattr(q8, field):
                    with self.subTest(field=field, path=name), tempfile.TemporaryDirectory(dir=root) as tmp:
                        work = Path(tmp)
                        (work / q8.BASE).mkdir(parents=True)
                        for target in q8.INPUTS:
                            shutil.copyfile(root / 'base' / target, work / q8.BASE / target)
                        # When integrated, source.prepare has already applied this
                        # stage; use its source-bound base to test the stage alone.
                        if copy.digest(work / q8.BASE / 'ggml-webgpu-shader-lib.hpp') != q8.INPUTS['ggml-webgpu-shader-lib.hpp']:
                            header = work / q8.BASE / 'ggml-webgpu-shader-lib.hpp'
                            subprocess.run(['git', 'apply', '--reverse', '--no-index', '--include=' + str(q8.BASE / header.name),
                                            str((ROOT / source.PATCH_DIRECTORY / q8.PATCH_NAME).resolve())],
                                           cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                            self.assertEqual(copy.digest(header), q8.INPUTS[header.name])
                        with patch.object(q8, field, {**getattr(q8, field), name: '0' * 64}):
                            with self.assertRaises(ValueError):
                                q8.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))
            for field in ('PATCH_SHA256', 'HEADER_SHA256'):
                with self.subTest(field=field), tempfile.TemporaryDirectory(dir=root) as tmp:
                    work = Path(tmp)
                    (work / q8.BASE).mkdir(parents=True)
                    for target in q8.INPUTS:
                        shutil.copyfile(root / 'base' / target, work / q8.BASE / target)
                    if copy.digest(work / q8.BASE / 'ggml-webgpu-shader-lib.hpp') != q8.INPUTS['ggml-webgpu-shader-lib.hpp']:
                        subprocess.run(['git', 'apply', '--reverse', '--no-index', '--include=' + str(q8.BASE / 'ggml-webgpu-shader-lib.hpp'),
                                        str((ROOT / source.PATCH_DIRECTORY / q8.PATCH_NAME).resolve())],
                                       cwd=work, env=source.patch_environment(work), check=True, capture_output=True)
                    with patch.object(q8, field, '0' * 64), self.assertRaises(ValueError):
                        q8.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))

    def test_partial_subgroups_cover_every_logical_lane_and_only_owned_rows(self):
        for width in (4, 8, 16, 32, 64):
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

    @unittest.skipUnless(shutil.which('node'), 'Node needed')
    def test_probe_accepts_legal_partial_geometry_and_rejects_invalid_partitions(self):
        helper = (ROOT / 'tests/webgpu-q8-subgroup/subgroup-geometry.mjs').as_uri()
        code = """
import { checkSubgroupGeometry } from 'HELPER';
function fixture(width, counts) {
    const words = [];
    for (let group = 0; group < counts.length; group++)
        for (let lane = 0; lane < counts[group]; lane++)
            words.push(width, counts.length, group, lane, counts[group]);
    return new Uint32Array(words);
}
for (const width of [4, 8, 16, 32, 64]) {
    const full = fixture(width, Array(128 / width).fill(width));
    if (!checkSubgroupGeometry({ geometry: full, advertisedWidth: width }).fullPacked)
        throw new Error('Full geometry misclassified');
    const partial = fixture(width, Array(128).fill(1));
    if (checkSubgroupGeometry({ geometry: partial, advertisedWidth: width }).fullPacked)
        throw new Error('Legal partial geometry rejected');
    for (const [index, value] of [[0, width * 2], [6, 127], [8, 0], [9, 1]]) {
        const invalid = Uint32Array.from(full); invalid[index] = value;
        let rejected = false;
        try { checkSubgroupGeometry({ geometry: invalid, advertisedWidth: width }); }
        catch { rejected = true; }
        if (!rejected) throw new Error('Invalid geometry accepted');
    }
}
for (const width of [32, 64]) {
    const actual = checkSubgroupGeometry({ geometry: fixture(width, Array(8).fill(16)), advertisedWidth: width });
    if (actual.fullPacked || actual.numSubgroups !== 8 || actual.activeCounts.some(count => count !== 16))
        throw new Error('Partial fixed-width geometry misclassified');
}
""".replace('HELPER', helper)
        subprocess.run(['node', '--input-type=module', '-e', code], check=True)

    def test_checker_rejects_invalid_node_commands_before_preparation(self):
        script = ROOT / 'tests/webgpu-q8-subgroup/check_gpu.py'
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not-created'
            for command in ('[]', 'null', '{}', '"node"', '[""]', '["node", 7]', '[invalid'):
                with self.subTest(command=command):
                    result = subprocess.run([sys.executable, str(script), '--llama-source', str(out), '--work', str(out),
                                             '--webgpu-module', str(out), '--subgroup-size', '4', '--node-command-json', command],
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn('--node-command-json must', result.stderr)
                    self.assertFalse(out.exists())

    @unittest.skipUnless(shutil.which('g++'), 'C++ compiler needed')
    def test_actual_selector_cache_factory_dispatch_and_shader_includes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            factory.prepare(self.upstream, out / 'prepared')
            exe = factory.build_host(self.upstream, out / 'prepared', out / 'host')
            result = subprocess.run([str(exe), str(out / 'shaders')], check=True, capture_output=True, text=True)
            self.assertIn('actual subgroup selector/cache/factory/dispatch checks passed', result.stdout)
            self.assertEqual(len(list((out / 'shaders').glob('*.wgsl'))), 6)
            for sg in (4, 8, 16, 32, 64):
                text = (out / 'shaders' / f'q8-independent-sg{sg}.wgsl').read_text()
                self.assertIn('@workgroup_size(128)', text)
                self.assertIn('if (num_subgroups == expected_subgroups)', text)
                self.assertIn('acc = accumulate_vec_dot(subgroup_invocation_id,', text)
                self.assertIn('let active_count = subgroupAdd(1u)', text)
                self.assertIn('virtual_tid += active_count', text)
                self.assertIn('acc[col][row] += partial[col][row]', text)
                self.assertIn('if (owns_rows && subgroup_invocation_id == 0u', text)
                self.assertEqual(text.count('return;'), 1)
                self.assertLess(text.index('return;'), text.index('subgroupAdd(1u)'))
                self.assertNotIn('workgroupBarrier', text)
                self.assertNotIn('#include', text)
            print(result.stdout.strip())


if __name__ == '__main__':
    unittest.main()
