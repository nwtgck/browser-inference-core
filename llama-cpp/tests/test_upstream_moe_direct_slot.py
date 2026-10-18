"""Retirement guards for the upstream selected-slot kernel, not GPU execution."""
import os
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class UpstreamSelectedSlot(unittest.TestCase):
    def setUp(self):
        source = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp'))
        directory = source / 'ggml/src/ggml-webgpu'
        if not (directory / 'wgsl-shaders/mul_mat_id_vec.wgsl').is_file():
            self.skipTest('The pinned upstream source is required')
        self.shader = (directory / 'wgsl-shaders/mul_mat_id_vec.wgsl').read_text().split('fn main(', 1)[1]
        self.host = (directory / 'ggml-webgpu.cpp').read_text().split(
            'static webgpu_encoded_op ggml_webgpu_mul_mat_id_vec(', 1)[1].split(
            'static uint32_t ggml_webgpu_mul_mat_id_max_tiles(', 1)[0]

    def test_upstream_uses_selected_slots_not_the_old_expert_scan(self):
        compact = re.sub(r'\s+', '', self.shader)
        for expression in (
            'letoutput_groups:u32=(params.m+OUTPUTS_PER_WG-1u)/OUTPUTS_PER_WG;',
            'letwg_linear=wg_id.y*num_wg.x+wg_id.x;',
            'letslot=wg_linear/output_groups;',
            'letown_expert=ids[params.offset_ids+slot];',
            'letrow_base=(wg_linear%output_groups)*OUTPUTS_PER_WG;',
            'letsrc0_batch_offset=params.offset_src0+own_expert*params.stride_02;',
            'letsrc1_idx_base=params.offset_src1+(slot%params.b_ne1)*params.stride_11;',
            'letdst_idx_base=params.offset_dst+slot*params.m+row_base;',
        ):
            self.assertIn(expression, compact)
        # Upstream may leave unused workgroup declarations outside main; neither
        # scan array may participate in the actual selected-slot kernel.
        self.assertNotIn('gathered_count_ids', self.shader)
        self.assertNotIn('gathered_expert_used', self.shader)

    def test_rectangular_tail_returns_before_indexing_or_synchronizing(self):
        guard = re.search(r'if\s*\(wg_linear\s*>=\s*output_groups\s*\*\s*params.n_expert_used\)\s*{\s*return;\s*}', self.shader)
        self.assertIsNotNone(guard)
        self.assertLess(guard.end(), self.shader.index('ids[params.offset_ids + slot]'))
        self.assertLess(guard.end(), self.shader.index('workgroupBarrier()'))
        self.assertIn('row_base + row < params.m', self.shader)
        self.assertIn('if (output_row < params.m)', self.shader)

    def test_host_dispatch_and_uniforms_match_shader_contract(self):
        compact = re.sub(r'\s+', '', self.host)
        for expression in (
            'param_n_expert_used=(uint32_t)dst->ne[1];',
            'output_groups=CEIL_DIV(dst->ne[0],decisions->outputs_per_wg);',
            'total_wg=output_groups*param_n_expert_used;',
            'compute_2d_workgroups(total_wg,max_wg_per_dim,wg_x,wg_y);',
            '(uint32_t)src1->ne[1],',
        ):
            self.assertIn(expression, compact)

    def test_address_mapping_matches_expert_scan_for_broadcasts_offsets_and_tails(self):
        # Independent address enumeration, coupled to the actual host/shader
        # expressions by the tests above. This does not execute WGSL reductions.
        for experts, selected in ((1, 1), (8, 1), (8, 3), (8, 8), (64, 7)):
            chosen = list(reversed(range(experts)))[:selected]
            for rows in (1, 3, 7, 17):
                for width in (1, 4, 8):
                    for broadcast in sorted({1, selected}):
                        with self.subTest(experts=experts, selected=selected, rows=rows, width=width, broadcast=broadcast):
                            reference = []
                            for expert in range(experts):
                                for slot, chosen_expert in enumerate(chosen):
                                    if expert == chosen_expert:
                                        for row in range(rows):
                                            reference.append((5 + expert * 1024 + row * 32,
                                                              11 + (slot % broadcast) * 64,
                                                              23 + slot * rows + row))
                            groups = (rows + width - 1) // width
                            total = groups * selected
                            # Force a rectangular tail and more than one y row.
                            nx = min(total, 5)
                            ny = (total + nx - 1) // nx
                            ids = [999, 998] + chosen
                            actual = []
                            for y in range(ny):
                                for x in range(nx):
                                    linear = y * nx + x
                                    if linear >= total:
                                        continue
                                    slot, group = divmod(linear, groups)
                                    expert = ids[2 + slot]
                                    for lane in range(width):
                                        row = group * width + lane
                                        if row < rows:
                                            actual.append((5 + expert * 1024 + row * 32,
                                                           11 + (slot % broadcast) * 64,
                                                           23 + slot * rows + row))
                            self.assertEqual(sorted(actual), sorted(reference))
                            self.assertEqual(len(actual), selected * rows)
                            self.assertEqual(len({entry[2] for entry in actual}), len(actual))


if __name__ == '__main__':
    unittest.main()
