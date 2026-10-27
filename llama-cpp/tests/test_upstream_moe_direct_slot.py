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
            'letsrc0_batch_offset=params.offset_src0+select(own_expert,0u,skipped)*params.stride_02;',
            'letsrc1_idx_base=params.offset_src1+(slot%params.b_ne1)*params.stride_11;',
            'letdst_idx_base=params.offset_dst+slot*params.m+row_base;',
        ):
            self.assertIn(expression, compact)
        # Upstream may leave unused workgroup declarations outside main; neither
        # scan array may participate in the actual selected-slot kernel.
        self.assertNotIn('gathered_count_ids', self.shader)
        self.assertNotIn('gathered_expert_used', self.shader)

    def test_skipped_slots_use_neutral_addresses_and_zero_accumulators(self):
        compact = re.sub(r'\s+|//[^\n]*|/\*.*?\*/', '', self.shader, flags=re.S)
        self.assertIn('letskipped=i32(own_expert)==-1;', compact)
        self.assertIn('letsrc0_batch_offset=params.offset_src0+select(own_expert,0u,skipped)*params.stride_02;', compact)
        # WGSL function-scope variables without an initializer are zero-valued.
        # The sole weight-reading helper call must be inside the non-skipped
        # branch, leaving zero accumulators for both reduction implementations.
        self.assertIn(
            'varacc:array<array<f32,OUTPUTS_PER_WG>,NUM_COLS>;'
            'if(!skipped){acc=accumulate_vec_dot(thread_id,row_base,src0_batch_offset,src1_idx_base);}',
            compact)
        self.assertEqual(compact.count('accumulate_vec_dot('), 1)
        self.assertNotIn('src0[', compact)
        self.assertNotIn('src1[', compact)
        self.assertEqual(len(re.findall(r'\bacc\s*(?:\[[^\]]*\])*\s*=(?!=)', self.shader)), 1)

    def test_skipped_slots_reach_the_same_reductions_and_barriers(self):
        code = re.sub(r'//[^\n]*|/\*.*?\*/', '', self.shader, flags=re.S)
        compact = re.sub(r'\s+', '', code)
        # Skipped state may only select a neutral weight offset and guard the
        # accumulation. It cannot guard a barrier, a reduction, or an output.
        self.assertEqual(len(re.findall(r'\bskipped\b', code)), 3)
        tail = re.search(r'if\s*\(wg_linear\s*>=\s*output_groups\s*\*\s*params.n_expert_used\)\s*{\s*return;\s*}', code)
        self.assertIsNotNone(tail)
        self.assertNotRegex(code[:tail.start()] + code[tail.end():], r'\breturn\b')
        non_skipped = re.search(r'if\s*\(!skipped\)\s*{([^{}]*)}', code)
        self.assertIsNotNone(non_skipped)
        self.assertNotRegex(non_skipped.group(1), r'\b(?:workgroupBarrier|storageBarrier|subgroupAdd|return)\b')
        self.assertLess(non_skipped.end(), code.index('workgroupBarrier()'))
        accumulation = compact.index('if(!skipped){acc=accumulate_vec_dot(')
        reduction = compact.index('#ifdefUSE_SUBGROUP_REDUCTION', accumulation)
        self.assertLess(accumulation, reduction)
        for expression in (
            'letsubgroup_total=subgroupAdd(acc[0][row]);',
            'partial_sums[partial_index(row,thread_id)]=acc[0][row];',
            'dst[dst_idx_base+row]=row_total;',
            'dst[dst_idx_base+thread_id]=partial_sums[partial_index(thread_id,0)];',
        ):
            self.assertIn(expression, compact)
        self.assertEqual(compact.count('workgroupBarrier();'), 3)

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

    def test_skipped_slots_preserve_broadcast_offsets_and_output_coverage(self):
        # A skipped selected slot still owns every output row. Its sentinel is
        # stored as u32 in WGSL, but cannot address expert 0xffffffff or read
        # weights; valid selected slots keep the original address mapping.
        for chosen in ((-1,), (-1, -1, -1), (-1, 7, 2), (7, -1, 2), (7, 2, -1)):
            for rows in (1, 3, 7, 17):
                for width in (1, 4, 8):
                    for broadcast in sorted({1, len(chosen)}):
                        with self.subTest(chosen=chosen, rows=rows, width=width, broadcast=broadcast):
                            groups = (rows + width - 1) // width
                            total = groups * len(chosen)
                            nx = min(total, 5)
                            ny = (total + nx - 1) // nx
                            ids = [999, 998] + [expert & 0xffffffff for expert in chosen]
                            actual, weight_reads, neutral_offsets = {}, [], []
                            for y in range(ny):
                                for x in range(nx):
                                    linear = y * nx + x
                                    if linear >= total:
                                        continue
                                    slot, group = divmod(linear, groups)
                                    expert = ids[2 + slot]
                                    signed_expert = expert if expert < 0x80000000 else expert - 0x100000000
                                    skipped = signed_expert == -1
                                    weight_base = 5 + (0 if skipped else expert) * 1024
                                    if skipped:
                                        neutral_offsets.append(weight_base)
                                    for lane in range(width):
                                        row = group * width + lane
                                        if row >= rows:
                                            continue
                                        destination = 23 + slot * rows + row
                                        self.assertNotIn(destination, actual)
                                        accumulator = 0.0
                                        if not skipped:
                                            addresses = (weight_base + row * 32, 11 + (slot % broadcast) * 64)
                                            weight_reads.append((destination, *addresses))
                                            accumulator = float(addresses[0] + addresses[1])
                                        actual[destination] = accumulator
                            expected = {
                                23 + slot * rows + row: 0.0 if expert == -1 else
                                float(5 + expert * 1024 + row * 32 + 11 + (slot % broadcast) * 64)
                                for slot, expert in enumerate(chosen) for row in range(rows)
                            }
                            self.assertEqual(actual, expected)
                            self.assertEqual(set(neutral_offsets), {5})
                            self.assertEqual(len(weight_reads), sum(expert != -1 for expert in chosen) * rows)
                            self.assertTrue(all(5 <= weight < 5 + 8 * 1024 for _, weight, _ in weight_reads))


if __name__ == '__main__':
    unittest.main()
