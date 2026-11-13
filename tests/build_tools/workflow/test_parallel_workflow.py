"""Guard this workflow's dependency graph and cache trust boundaries without YAML dependencies."""
from pathlib import Path
import re
import unittest

from build_tools.project import REPOSITORY_ROOT as ROOT
BUILD = (ROOT/'.github/workflows/build.yml').read_text()
from tests.support.workflow import jobs, step_id


def dependencies(body):
    found = re.search(r'^    needs: \[([^]]*)\]', body, re.M)
    return set(part.strip() for part in found[1].split(',')) if found else set()


class ParallelWorkflow(unittest.TestCase):
    def test_only_final_publication_joins_llama_and_image(self):
        graph = jobs(BUILD)
        self.assertEqual(set(graph), {'test','qsa-native','compile','build','image-native','image-compile','image-build','publish'})
        self.assertEqual(dependencies(graph['compile']), set())
        self.assertEqual(dependencies(graph['qsa-native']), set())
        self.assertEqual(dependencies(graph['image-compile']), set())
        self.assertEqual(dependencies(graph['image-native']), set())
        self.assertEqual(dependencies(graph['build']), {'test','qsa-native','compile'})
        self.assertEqual(dependencies(graph['image-build']), {'image-native','image-compile'})
        self.assertEqual(dependencies(graph['publish']), {'build','image-build'})
        self.assertNotIn('max-parallel:', re.sub(r'#.*', '', BUILD))

    def test_qsa_parity_is_a_required_gate_with_failure_logs(self):
        graph = jobs(BUILD)
        job = graph['qsa-native']
        execution = job.split('- name: Compare separate reference and optimized native builds', 1)[1].split('- name: Upload native comparison traces and logs', 1)[0]
        self.assertIn('../../tests/runtimes/llama_cpp/qsa-selection/run.py', execution)
        self.assertIn('--jobs 2', execution)
        self.assertNotIn('continue-on-error:', execution)
        self.assertNotIn('if:', execution)
        self.assertNotIn('|| true', execution)
        self.assertIn('qsa-native', dependencies(graph['build']))
        upload = job.split('- name: Upload native comparison traces and logs', 1)[1].split('- name:', 1)[0]
        self.assertIn('if: always()', upload)
        self.assertIn('path: runtimes/llama-cpp/build/qsa-regression/qsa-results/', upload)
        self.assertNotIn('path: runtimes/llama-cpp/build/\n', upload)
        self.assertNotIn('browser-cache', job)

    def test_image_package_has_no_llama_source_or_artifact_dependency(self):
        graph = jobs(BUILD)
        for name in ('image-native', 'image-compile', 'image-build'):
            self.assertNotIn('llama-cpp/', graph[name])
            self.assertNotIn('profile-build-', graph[name])
        self.assertIn('pattern: image-build-*', graph['image-build'])
        self.assertIn('build/toolchain-licenses/emscripten', graph['image-build'])
        self.assertIn('extra+=(--include-toolchain-notices)', graph['image-compile'])
        self.assertIn('"$PROFILE" == "webgpu-wasm32-jspi"', graph['image-compile'])
        self.assertIn('"$VARIANT" == "browser"', graph['image-compile'])

    def test_cache_restore_does_not_skip_verification_or_configuration(self):
        for name in ('compile', 'image-compile'):
            block = jobs(BUILD)[name]
            self.assertEqual(len(re.findall(r'uses: actions/cache/restore@[^\s]+', block)), 3)
            self.assertEqual(block.count('restore-keys:'), 1)
            self.assertIn('restore-keys: ${{ steps.plan.outputs.cc-prefix }}', block)
            self.assertLess(block.index('id: dawn'), block.index('--phase sdk.prepare'))
            self.assertLess(block.index('--phase sdk.prepare'), block.index('id: em'))
            verification = block.split('- name: Verify and prepare local compiler inputs', 1)[1].split('      - ', 1)[0]
            self.assertNotIn('if:', verification)
            self.assertIn('--fresh --profile', block)

    def test_only_trusted_default_branch_pushes_invoke_and_save_caches(self):
        conditions = ["github.event_name == 'push'", "github.event.repository.default_branch", "github.actor != 'dependabot[bot]'", "success()"]
        for name in ('compile', 'image-compile'):
            block = jobs(BUILD)[name]
            self.assertEqual(len(re.findall(r'uses: actions/cache/save@[^\s]+', block)), 3)
            # Guard both the live-data preflight and the actual remote save.
            for kind in ('dawn', 'em', 'cc'):
                for prefix in ('check-', 'save-'):
                    step = step_id(block, prefix + kind)
                    for condition in conditions:
                        self.assertIn(condition, step)
        for name in ('publish', 'build', 'image-build'):
            self.assertNotIn('cache/save', jobs(BUILD)[name])
        self.assertNotIn('pull_request_target:', BUILD)
        self.assertNotIn('workflow_run:', BUILD)
        self.assertNotIn('./.github/actions/', BUILD)

    def test_artifact_steps_do_not_repeat_hidden_file_options(self):
        for step in re.split(r'^      - ', BUILD, flags=re.M):
            self.assertLessEqual(step.count('include-hidden-files:'), 1)

    def test_callback_test_hook_is_test_variant_only_and_incoming_options_are_supported(self):
        cmake = (ROOT/'runtimes/stable-diffusion-cpp/CMakeLists.txt').read_text()
        incoming = re.search(r'"-sINCOMING_MODULE_JS_API=([^\n]+)', cmake)[1]
        self.assertNotIn('onLog', incoming); self.assertNotIn('onProgress', incoming)
        self.assertIn('if(SDCB_VARIANT STREQUAL "test")', cmake)
        self.assertIn('target_sources(core PRIVATE "${SDCB_TEST_ROOT}/wasm-probes.cpp")', cmake)
        self.assertNotIn('--pre-js', cmake)
        smoke = (ROOT/'tests/runtimes/stable_diffusion_cpp/browser-smoke.mjs').read_text()
        for check in ('new Worker(', 'worker.terminate()', 'mountReadOnlyFile', 'sd_set_log_callback(0n, 0n)', 'module._sdc_test_callbacks()', '120000'):
            self.assertIn(check, smoke)
        self.assertNotIn('realModelInference = true', smoke)


if __name__ == '__main__': unittest.main()
