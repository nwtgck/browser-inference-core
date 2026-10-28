"""Keep the failing predecessor and validate the real profile query control flow."""
import hashlib
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
import prepare_webgpu_source as source
import prepare_webgpu_profile_query_budget as budget
import prepare_webgpu_q8_diagnostic as diagnostic
from test_webgpu_host_encode_assembly import extracted_model

HERE = ROOT / 'tests/webgpu-profile-query-budget'
BASELINE_PROFILE_SHA256 = 'eb38bb6fba491b03ef3cfc685f05982ad580c4a587082c0f288b88efa1558aac'
CAPACITY_ASSERT = 'GGML_ASSERT(webgpu_ctx->param_arena.slot_count <= WEBGPU_MAX_PROFILE_QUERY_COUNT / 2);'
PREFIX = '''#include <array>
#include <cstdint>
#include <iostream>
#include <vector>
static void event(char, uintptr_t = 0, uint32_t = 0, uint32_t = 0, uint32_t = 0) {}
static std::vector<uint32_t> collected_queries;
'''


def counter_model(actual, header):
    # The graph, build, arena and cache are actual source. Only Encode and the
    # blocking collector's device work are CPU callbacks in this model.
    model = extracted_model(actual, header)
    before = '    std::vector<webgpu_dispatch_desc> dispatches;\n};'
    if model.count(before) != 1:
        raise ValueError('Counter model graph changed')
    model = model.replace(before, '    std::vector<webgpu_dispatch_desc> dispatches;\n    std::vector<size_t> dispatch_counts{};\n};')
    before = '    return ggml_backend_webgpu_build_multi(ctx, {graph->dispatches[index]});'
    if model.count(before) != 1:
        raise ValueError('Counter model Encode changed')
    model = model.replace(before, '''    const size_t count = graph->dispatch_counts.empty() ? 1 : graph->dispatch_counts[index];
    std::vector<webgpu_dispatch_desc> node_dispatches(count, graph->dispatches[index]);
    return ggml_backend_webgpu_build_multi(ctx, node_dispatches);''')
    before = 'static void ggml_backend_webgpu_collect_profile_results(webgpu_context &, const std::vector<std::string> &, uint32_t &) {}'
    if model.count(before) != 1:
        raise ValueError('Counter model profile collector changed')
    model = model.replace(before, '''static void ggml_backend_webgpu_collect_profile_results(webgpu_context & ctx, const std::vector<std::string> & names, uint32_t &) {
    if (names.empty()) return;
    assert(ctx->profile_timestamp_query_count == 2 * names.size());
    assert(ctx->profile_timestamp_query_count <= WEBGPU_MAX_PROFILE_QUERY_COUNT);
    collected_queries.push_back(ctx->profile_timestamp_query_count);
    ctx->global_ctx->stats->drain();
}''')
    guard = CAPACITY_ASSERT if CAPACITY_ASSERT in actual else '(void) webgpu_ctx;'
    model += '\nstatic void validate_extracted_profile_capacity(webgpu_context & webgpu_ctx) { ' + guard + ' }\n'
    return model


class ProfileQueryBudget(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.upstream = Path(os.environ.get('LCB_TEST_LLAMA_SOURCE', ROOT / 'vendor/llama.cpp')).resolve()
        cls.temporary = tempfile.TemporaryDirectory(prefix='profile-query-budget-')
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.directory = Path(cls.temporary.name)
        with patch.object(source, 'replace_iq3s_row_split_embedding'), patch.object(source.moe_tight_tile_bound, 'prepare'), patch.object(source.matmul_vec4_layout, 'prepare'):
            cls.normal = source.prepare(cls.upstream, cls.directory / 'normal')
        # The predecessor remains pinned and is actually executed by the test;
        # do not mutate its query counter or replace its assertion.
        old_inputs = {**diagnostic.INPUTS, 'ggml-webgpu.cpp': budget.INPUT_SHA256}
        old_outputs = {**diagnostic.OUTPUTS, 'ggml-webgpu.cpp': BASELINE_PROFILE_SHA256}
        with patch.object(source, 'replace_iq3s_row_split_embedding'), patch.object(source.moe_tight_tile_bound, 'prepare'), patch.object(source.matmul_vec4_layout, 'prepare'), patch.object(budget, 'prepare'), patch.object(diagnostic, 'INPUTS', old_inputs), \
                patch.object(diagnostic, 'OUTPUTS', old_outputs):
            cls.baseline = source.prepare(cls.upstream, cls.directory / 'baseline', gpu_profile=True)
        with patch.object(source, 'replace_iq3s_row_split_embedding'), patch.object(source.moe_tight_tile_bound, 'prepare'), patch.object(source.matmul_vec4_layout, 'prepare'):
            cls.profile = source.prepare(cls.upstream, cls.directory / 'profile', gpu_profile=True)

    def test_normal_is_exact_and_does_not_enter_the_profile_stage(self):
        self.assertEqual(source.tensor_copy.digest(self.normal), budget.INPUT_SHA256)
        self.assertEqual(source.OUTPUT_SHA256, budget.INPUT_SHA256)
        self.assertNotIn(CAPACITY_ASSERT, self.normal.read_text())
        self.assertIn(CAPACITY_ASSERT, self.profile.read_text())
        for name in ('ggml-webgpu-shader-lib.hpp', 'ggml-backend.cpp', 'llama-model-loader.cpp', 'llama-sampler.cpp'):
            self.assertEqual((self.profile.parent / name).read_bytes(), (self.baseline.parent / name).read_bytes())
        with patch.object(source, 'replace_iq3s_row_split_embedding'), patch.object(source.moe_tight_tile_bound, 'prepare'), patch.object(source.matmul_vec4_layout, 'prepare'), patch.object(budget, 'prepare', side_effect=AssertionError('profile stage entered normal preparation')):
            normal = source.prepare(self.upstream, self.directory / 'normal-again')
        before = {p.name: p.read_bytes() for p in self.normal.parent.iterdir() if p.is_file()}
        after = {p.name: p.read_bytes() for p in normal.parent.iterdir() if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(diagnostic.INPUTS['ggml-webgpu.cpp'], budget.OUTPUT_SHA256)
        self.assertEqual(source.tensor_copy.digest(self.profile), diagnostic.OUTPUTS['ggml-webgpu.cpp'])
        cmake = (ROOT / 'cmake/WebgpuSourceOverlay.cmake').read_text()
        self.assertIn('scripts/prepare_webgpu_profile_query_budget.py', cmake)
        self.assertIn(budget.PATCH_NAME, cmake)

    def test_profile_change_is_only_lookahead_and_initial_capacity_guard(self):
        actual = self.profile.read_text()
        actual = actual.replace(
            '// Include multi-dispatch overshoot: every accepted batch fits in the arena.',
            '// flush before the next batch can overflow the QuerySet')
        actual = actual.replace('ctx->profile_timestamp_query_count + 2 * ctx->param_arena.slot_count >=',
                                'ctx->profile_timestamp_query_count + 2 * ctx->global_ctx->command_submit_batch_size >=')
        actual = actual.replace('''    // A fresh query set must fit every arena slot, including batch overshoot.
    // This bound also keeps the query lookahead arithmetic within uint32_t.
    ''' + CAPACITY_ASSERT + '\n', '')
        self.assertEqual(actual, self.baseline.read_text())
        self.assertEqual(source.tensor_copy.digest(self.baseline), BASELINE_PROFILE_SHA256)

    def test_profile_guards_preserve_existing_outputs_and_leave_normal_usable(self):
        before = {p.name: p.read_bytes() for p in self.profile.parent.iterdir() if p.is_file()}
        for field in ('INPUT_SHA256', 'OUTPUT_SHA256', 'PATCH_SHA256'):
            with self.subTest(field=field), patch.object(budget, field, '0' * 64):
                with self.assertRaises(ValueError):
                    source.prepare(self.upstream, self.profile.parent, gpu_profile=True)
                normal = source.prepare(self.upstream, self.directory / ('normal-' + field))
                self.assertEqual(source.tensor_copy.digest(normal), source.FINAL_OUTPUTS[False]['ggml-webgpu.cpp'])
            self.assertEqual(before, {p.name: p.read_bytes() for p in self.profile.parent.iterdir() if p.is_file()})
        for relative in budget.CONTRACT_INPUTS:
            with self.subTest(contract=relative), patch.object(budget, 'CONTRACT_INPUTS', {**budget.CONTRACT_INPUTS, relative: '0' * 64}):
                with self.assertRaisesRegex(ValueError, 'Profile query budget contract'):
                    source.prepare(self.upstream, self.directory / ('bad-' + Path(relative).name), gpu_profile=True)

    def test_actual_patch_drift_prefix_and_duplicate_application_fail_closed(self):
        work = self.directory / 'prefix'
        target = work / source.SOURCE_PATH
        target.parent.mkdir(parents=True)
        target.write_bytes(self.normal.read_bytes())
        patches = self.directory / 'patches'
        patches.mkdir()
        patch_file = patches / budget.PATCH_NAME
        original = (ROOT / source.PATCH_DIRECTORY / budget.PATCH_NAME).read_bytes()
        patch_file.write_bytes(original + b'\n')
        with self.assertRaisesRegex(ValueError, 'patch identity changed'):
            budget.prepare(work, self.upstream, patches, source.patch_environment(work))
        self.assertEqual(target.read_bytes(), self.normal.read_bytes())
        patch_file.write_bytes(original)
        budget.prepare(work, self.upstream, patches, source.patch_environment(work))
        self.assertEqual(source.tensor_copy.digest(target), budget.OUTPUT_SHA256)
        with self.assertRaisesRegex(ValueError, 'composition input changed'):
            budget.prepare(work, self.upstream, patches, source.patch_environment(work))

    @unittest.skipUnless(shutil.which('c++'), 'C++ compiler required')
    def test_actual_counterexample_just_fit_overshoot_and_capacity_rejection(self):
        cases = (HERE / 'counter-cases.cpp').read_text()
        for batching in (False, True):
            results = {}
            for label, prepared, profile, safe in (
                    ('normal', self.normal, False, False), ('baseline', self.baseline, True, False),
                    ('prepared', self.profile, True, True)):
                actual = prepared.read_text()
                header = (prepared.parent / 'ggml-webgpu-shader-lib.hpp').read_text()
                code = PREFIX + counter_model(actual, header) + cases
                name = 'counter-' + label + ('-batch' if batching else '-individual')
                cpp, binary = self.directory / (name + '.cpp'), self.directory / name
                cpp.write_text(code)
                macros = ['-DGGML_WEBGPU_BATCH_PARAM_UPLOADS'] if batching else []
                if profile:
                    macros += ['-DGGML_WEBGPU_GPU_PROFILE']
                if safe:
                    macros += ['-DEXPECT_SAFE_QUERY_BUDGET']
                command = ['c++', '-std=c++17', '-O3', '-Wall', '-Wextra', '-Werror', '-Wno-unused-function',
                           *macros, str(cpp), '-o', str(binary)]
                build = subprocess.run(command, text=True, capture_output=True)
                self.assertEqual(build.returncode, 0, build.stderr)
                run = subprocess.run([str(binary)], text=True, capture_output=True)
                self.assertEqual(run.returncode, 0, run.stderr)
                report = json.loads(run.stdout)
                self.assertEqual(report['status'], 'passed')
                self.assertEqual(len(report['cases']), 14)
                results[label] = {case['name']: case for case in report['cases']}
            for name in ('first2049-dispatches', 'minimal187nodes-max74slots', 'small-width518nodes'):
                self.assertEqual(results['baseline'][name]['failureKind'], 'query')
                self.assertFalse(results['normal'][name]['failed'])
                self.assertFalse(results['prepared'][name]['failed'])
            self.assertEqual(results['baseline']['minimal187nodes-max74slots']['arenaUsedAtReturn'], 74)
            self.assertEqual(results['baseline']['first2049-dispatches']['queryCounterAtReturn'], 4096)
            self.assertEqual(results['prepared']['minimal187nodes-max74slots']['collectedQueries'], [3960, 148])
            for label in results:
                self.assertFalse(results[label]['exact2048-dispatches']['failed'])
                self.assertEqual(results[label]['existing75-arena-rejection']['failureKind'], 'arena')


if __name__ == '__main__':
    unittest.main()
