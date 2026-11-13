"""Source-bound gather ownership/factory checks; no GPU or speed assertions."""
import hashlib
import importlib.util
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

from build_tools.project import runtime_root, runtime_test_root
ROOT = runtime_root('llama-cpp')
sys.path[:0] = [str(ROOT.parents[1])]
# These tests isolate an accepted historical patch prefix.
from tests.support import legacy_webgpu_source as source
from tests.runtimes.llama_cpp.webgpu_prefix_context import without_final_matvec_stages
from tests.support.llama_cpp.webgpu_stages import moe_gather_logical32 as gather
spec = importlib.util.spec_from_file_location('logical32_actual_factory', runtime_test_root('llama-cpp') / 'webgpu-moe-group-skip/emit_factory.py')
factory = importlib.util.module_from_spec(spec)
spec.loader.exec_module(factory)


from tests.support import legacy_webgpu_source as legacy_source

class MoeGatherLogical32(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temporary = tempfile.TemporaryDirectory(prefix='moe-gather-logical32-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        with without_final_matvec_stages(legacy_source), patch.object(legacy_source.moe_tight_tile_bound, 'prepare'), patch.object(legacy_source.matmul_vec4_layout, 'prepare'), patch.object(gather, 'prepare', return_value={}), patch.object(legacy_source, 'SSM_HEADER_OUTPUT_SHA256', gather.INPUT_SHA256):
            cls.baseline = legacy_source.prepare(cls.upstream, cls.directory / 'baseline').parent
        # This suite isolates the gather stage; the final layout stage has its
        # own full caller/factory tests, and does not change gather WGSL.
        with without_final_matvec_stages(legacy_source), patch.object(legacy_source.moe_tight_tile_bound, 'prepare'), patch.object(legacy_source.matmul_vec4_layout, 'prepare'):
            cls.normal = legacy_source.prepare(cls.upstream, cls.directory / 'normal').parent
            cls.profile = legacy_source.prepare(cls.upstream, cls.directory / 'profile', gpu_profile=True).parent
        cls.factories = {}

    def actual_factory(self, mode):
        if mode not in self.factories:
            self.factories[mode] = factory.build(self.upstream, getattr(self, mode), self.directory / ('factory-' + mode), profile=mode == 'profile')
        return self.factories[mode]

    def test_only_zeroing_and_two_library_links_change(self):
        original = (self.upstream / gather.BASE / 'wgsl-shaders/mul_mat_id_gather.wgsl').read_text()
        header = (self.normal / gather.HEADER_NAME).read_text()
        chunks = re.findall(r'R"(wgsl(?:_x)*)\((.*?)\)\1"', header, re.S)
        logical = ''.join(text for _, text in chunks)
        self.assertEqual(hashlib.sha256(logical.encode()).hexdigest(), gather.SHADER_SHA256)
        marker = '    // the matmul writes only the rows that an expert owns'
        self.assertEqual(original[:original.index(marker)], logical[:logical.index('    let zero_lanes =')])
        self.assertEqual(logical.count('workgroupBarrier()'), original.count('workgroupBarrier()'))
        for operation in ('atomicAdd(', 'atomicLoad(', 'atomicStore(', 'var<workgroup>'):
            self.assertEqual(logical.count(operation), original.count(operation))
        self.assertEqual(hashlib.sha256(header.encode()).hexdigest(), gather.HEADER_SHA256)
        for directory, expected in ((self.normal, gather.INPUT_SHA256),
                                    (self.profile, 'e205d9778ae3fc8ee392c037884660df444c3cef8f19936a5e67b7a0b312a13e')):
            text = (directory / 'ggml-webgpu-shader-lib.hpp').read_text()
            before = text.replace('#include "' + gather.HEADER_NAME + '"\n', '').replace(
                'preprocessor.preprocess(wgsl_mul_mat_id_gather_logical32, defines)',
                'preprocessor.preprocess(wgsl_mul_mat_id_gather, defines)')
            self.assertEqual(hashlib.sha256(before.encode()).hexdigest(), expected)
        self.assertEqual({p.name for p in self.normal.iterdir()} - {p.name for p in self.baseline.iterdir()}, {gather.HEADER_NAME})
        for path in self.baseline.iterdir():
            if path.name != 'ggml-webgpu-shader-lib.hpp':
                self.assertEqual(path.read_bytes(), (self.normal / path.name).read_bytes(), path.name)
        self.assertEqual(source.tensor_copy.digest(self.normal / 'ggml-webgpu.cpp'), source.matmul_vec4_layout.INPUTS[False]['ggml-webgpu.cpp'])
        self.assertEqual(source.tensor_copy.digest(self.profile / 'ggml-webgpu.cpp'), source.matmul_vec4_layout.INPUTS[True]['ggml-webgpu.cpp'])
        self.assertFalse((self.normal / 'ggml-wgsl-shaders.hpp').exists())

    def test_pins_and_existing_outputs_fail_closed(self):
        before = {p.name: p.read_bytes() for p in self.normal.iterdir()}
        for name in gather.CONTRACT_INPUTS:
            with self.subTest(contract=name), patch.dict(gather.CONTRACT_INPUTS, {name: '0' * 64}), self.assertRaisesRegex(ValueError, 'contract changed'):
                legacy_source.prepare(self.upstream, self.normal)
            self.assertEqual(before, {p.name: p.read_bytes() for p in self.normal.iterdir()})
        for field in ('INPUT_SHA256', 'PATCH_SHA256', 'OUTPUT_SHA256', 'SHADER_SHA256', 'HEADER_SHA256'):
            with self.subTest(field=field), patch.object(gather, field, '0' * 64), self.assertRaises(ValueError):
                legacy_source.prepare(self.upstream, self.normal)
            self.assertEqual(before, {p.name: p.read_bytes() for p in self.normal.iterdir()})

    def test_unknown_specializations_and_reapplication_are_rejected(self):
        for location in ('upstream', 'work', 'work-root'):
            for name in (gather.HEADER_NAME, 'wgsl-shaders/' + gather.SHADER_NAME):
                if location == 'work-root' and name != gather.HEADER_NAME:
                    continue
                with self.subTest(location=location, name=name), tempfile.TemporaryDirectory(dir=self.directory) as temporary:
                    base = Path(temporary)
                    upstream, work = base / 'upstream', base / 'work'
                    for relative in gather.CONTRACT_INPUTS:
                        dst = upstream / relative
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(self.upstream / relative, dst)
                    library = work / gather.BASE / 'ggml-webgpu-shader-lib.hpp'
                    library.parent.mkdir(parents=True)
                    shutil.copyfile(self.baseline / library.name, library)
                    target = work / name if location == 'work-root' else (upstream if location == 'upstream' else work) / gather.BASE / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text('unknown specialization')
                    with self.assertRaisesRegex(ValueError, 'unreviewed'):
                        gather.prepare(work, upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))
        with tempfile.TemporaryDirectory(dir=self.directory) as temporary:
            work = Path(temporary)
            library = work / gather.BASE / 'ggml-webgpu-shader-lib.hpp'
            library.parent.mkdir(parents=True)
            shutil.copyfile(self.baseline / library.name, library)
            gather.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))
            with self.assertRaisesRegex(ValueError, 'composition input'):
                gather.prepare(work, self.upstream, ROOT / source.PATCH_DIRECTORY, source.patch_environment(work))

    def test_reprepare_repairs_embedding_without_unnecessary_rewrites(self):
        stamps = {p.name: p.stat().st_mtime_ns for p in self.normal.iterdir()}
        with without_final_matvec_stages(legacy_source), patch.object(legacy_source.moe_tight_tile_bound, 'prepare'), patch.object(legacy_source.matmul_vec4_layout, 'prepare'):
            legacy_source.prepare(self.upstream, self.normal)
        self.assertEqual(stamps, {p.name: p.stat().st_mtime_ns for p in self.normal.iterdir()})
        (self.normal / gather.HEADER_NAME).write_text('stale generated header')
        with without_final_matvec_stages(legacy_source), patch.object(legacy_source.moe_tight_tile_bound, 'prepare'), patch.object(legacy_source.matmul_vec4_layout, 'prepare'):
            legacy_source.prepare(self.upstream, self.normal)
        self.assertEqual(source.tensor_copy.digest(self.normal / gather.HEADER_NAME), gather.HEADER_SHA256)
        for name, expected in gather.CONTRACT_INPUTS.items():
            self.assertEqual(source.tensor_copy.digest(self.upstream / name), expected)

    def test_logical_owner_is_unique_for_partial_groups_and_sparse_skips(self):
        cases = 0
        for wg in (1, 17, 31, 32, 33, 63, 64, 255, 256, 1024):
            lanes = min(32, wg)
            groups = wg // lanes
            self.assertTrue(1 <= lanes <= wg and 1 <= groups <= wg)
            for slots in (0, 1, 3, 9, 33, 129):
                for rows in (0, 1, 5, 17, 33, 65):
                    for period in (1, 13, slots + 1):
                        writes = bytearray(slots * rows)
                        skipped = {slot for slot in range(slots) if slot % period == 0}
                        for thread in range(wg):
                            group, lane = divmod(thread, lanes)
                            if group >= groups:
                                continue
                            for slot in range(group, slots, groups):
                                if slot in skipped:
                                    for row in range(lane, rows, lanes):
                                        writes[slot * rows + row] += 1
                        self.assertEqual(writes, bytearray(int(slot in skipped) for slot in range(slots) for _ in range(rows)))
                        cases += 1
        self.assertEqual(cases, 1080)

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_gather_factory_cache_capacity_and_profile_identity(self):
        shaders = self.directory / 'shaders'
        shaders.mkdir(exist_ok=True)
        for wg in (1, 17, 31, 33, 256, 1024):
            for index, (kind, k, m, experts, used, tokens) in enumerate(((0, 32, 16384, 1, 1, 2), (1, 256, 33, 11, 4, 5), (2, 256, 640, 512, 10, 128), (0, 32, 1, 4096, 1, 2))):
                block = 32 if kind == 1 else 256
                row = k if kind == 0 else k // block
                vectorized = kind == 0 and k % 4 == 0 and m % 4 == 0
                a_offset, x_offset, dst_offset = (4, 4, 8) if vectorized else (1, 3, 7)
                row_stride, input_stride = (row + 4, k + 4) if vectorized else (row + 1, k + 5)
                fields = [kind, k, m, experts, used, tokens, used, a_offset, x_offset, dst_offset, 2, row_stride, m * row_stride + 4, input_stride, used * input_stride + 4, used + 2]
                outputs = []
                for mode in ('normal', 'profile'):
                    name = f'{mode}-{wg}-{index}'
                    command = [str(self.actual_factory(mode)), str(shaders), name, *map(str, fields), 'f32', str(wg)]
                    metadata = json.loads(subprocess.check_output(command, text=True))
                    self.assertEqual(metadata['gatherWorkgroupSize'], wg)
                    self.assertEqual(metadata['gatherParams'], [2, dst_offset, experts, used, tokens, used + 2, m])
                    actual = (shaders / (name + '-gather.wgsl')).read_bytes()
                    baseline = (shaders / (name + '-gather-baseline.wgsl')).read_bytes()
                    self.assertNotEqual(actual, baseline)
                    self.assertIn(b'let zero_lanes = min(32u,', actual)
                    self.assertEqual(actual.count(b'var<workgroup>'), 1)
                    self.assertEqual(actual.count(b'workgroupBarrier()'), 3)
                    if experts == 4096:
                        self.assertIn(b'array<atomic<u32>, 4096u>', actual)
                    outputs.append((actual, (shaders / (name + '-candidate.wgsl')).read_bytes()))
                self.assertEqual(outputs[0], outputs[1])

    def test_build_inputs_and_opt_in_checker_record_both_real_gathers(self):
        checker = (runtime_test_root('llama-cpp') / 'webgpu-moe-group-skip/check_gpu.mjs').read_text()
        self.assertIn("['gather-baseline','gather']", checker)
        self.assertIn('gatherVariants.push', checker)
        self.assertIn('Prepared/canonical gather output byte mismatch', checker)


if __name__ == '__main__':
    unittest.main()
