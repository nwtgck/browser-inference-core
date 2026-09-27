"""Guard this workflow's dependency graph and cache trust boundaries without YAML dependencies."""
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
BUILD = (ROOT/'.github/workflows/build.yml').read_text()
RESTORE = (ROOT/'.github/actions/restore-browser-cache/action.yml').read_text()
SAVE = (ROOT/'.github/actions/save-browser-cache/action.yml').read_text()


def jobs(text):
    body = text.split('\njobs:\n', 1)[1]
    names = list(re.finditer(r'^  ([a-z][a-z-]*):\s*$', body, re.M))
    return {m[1]: body[m.end():names[i+1].start() if i+1 < len(names) else len(body)] for i,m in enumerate(names)}


def dependencies(body):
    found = re.search(r'^    needs: \[([^]]*)\]', body, re.M)
    return set(part.strip() for part in found[1].split(',')) if found else set()


class ParallelWorkflow(unittest.TestCase):
    def test_only_final_publication_joins_llama_and_image(self):
        graph = jobs(BUILD)
        self.assertEqual(set(graph), {'test','compile','build','image-native','image-compile','image-build','publish'})
        self.assertEqual(dependencies(graph['compile']), set())
        self.assertEqual(dependencies(graph['image-compile']), set())
        self.assertEqual(dependencies(graph['image-native']), set())
        self.assertEqual(dependencies(graph['build']), {'test','compile'})
        self.assertEqual(dependencies(graph['image-build']), {'image-native','image-compile'})
        self.assertEqual(dependencies(graph['publish']), {'build','image-build'})
        self.assertNotIn('max-parallel:', re.sub(r'#.*', '', BUILD))

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
        self.assertEqual(RESTORE.count('uses: actions/cache/restore@v4'), 3)
        self.assertEqual(RESTORE.count('restore-keys:'), 1)
        self.assertIn('restore-keys: ${{ steps.plan.outputs.cc-prefix }}', RESTORE)
        self.assertNotIn('cache-hit ==', RESTORE)
        self.assertLess(RESTORE.index('id: dawn'), RESTORE.index('scripts/setup_toolchain.py'))
        self.assertLess(RESTORE.index('scripts/setup_toolchain.py'), RESTORE.index('id: em'))
        self.assertIn('--fresh --profile', jobs(BUILD)['compile'])
        self.assertIn('--fresh --profile', jobs(BUILD)['image-compile'])

    def test_only_trusted_default_branch_pushes_invoke_and_save_caches(self):
        conditions = ["github.event_name == 'push'", "github.event.repository.default_branch", "github.actor != 'dependabot[bot]'"]
        self.assertEqual(SAVE.count('uses: actions/cache/save@v4'), 3)
        # Each of three save steps AND its live-data preflight is trust-gated.
        for condition in conditions: self.assertEqual(SAVE.count(condition), 6)
        for name in ('compile','image-compile'):
            body = jobs(BUILD)[name].split('- name: Save reusable intermediates', 1)[1]
            for condition in conditions: self.assertIn(condition, body)
            self.assertIn('success()', body)
        for name in ('publish','build','image-build'):
            self.assertNotIn('browser-cache', jobs(BUILD)[name])
        self.assertNotIn('cache/save', RESTORE)
        self.assertNotIn('pull_request_target:', BUILD)
        self.assertNotIn('workflow_run:', BUILD)

    def test_artifact_steps_do_not_repeat_hidden_file_options(self):
        for step in re.split(r'^      - ', BUILD, flags=re.M):
            self.assertLessEqual(step.count('include-hidden-files:'), 1)

    def test_callback_test_hook_is_test_variant_only_and_incoming_options_are_supported(self):
        cmake = (ROOT/'stable-diffusion-cpp/CMakeLists.txt').read_text()
        incoming = re.search(r'"-sINCOMING_MODULE_JS_API=([^\n]+)', cmake)[1]
        self.assertNotIn('onLog', incoming); self.assertNotIn('onProgress', incoming)
        self.assertIn('if(SDCB_VARIANT STREQUAL "test")', cmake)
        self.assertIn('target_sources(core PRIVATE tests/wasm-probes.cpp)', cmake)
        self.assertNotIn('--pre-js', cmake)
        smoke = (ROOT/'stable-diffusion-cpp/tests/browser-smoke.mjs').read_text()
        for check in ('new Worker(', 'worker.terminate()', 'mountReadOnlyFile', 'sd_set_log_callback(0n, 0n)', 'module._sdc_test_callbacks()', '120000'):
            self.assertIn(check, smoke)
        self.assertNotIn('realModelInference = true', smoke)


if __name__ == '__main__': unittest.main()
