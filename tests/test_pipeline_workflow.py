"""Pipeline contract checks without a YAML dependency or remote runner."""
import json
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (ROOT / '.github/workflows/build.yml').read_text()
JOBS = dict(re.findall(r'^  ([a-z-]+):\n(.*?)(?=^  [a-z-]+:\n|\Z)', WORKFLOW.split('\njobs:\n', 1)[1], re.M | re.S))


class PipelineWorkflow(unittest.TestCase):
    def test_checkout_keeps_recursive_notices_for_selected_runtime(self):
        for name in ('test', 'compile', 'build', 'image-native', 'image-compile', 'image-build'):
            block = JOBS[name]
            self.assertIn('submodules: false', block)
            if name.startswith('image'):
                command = re.search(r'git submodule update[^\n]+', block).group()
                self.assertIn('--recursive --depth=1 -- ', command)
                self.assertTrue(command.endswith(
                    'stable-diffusion-cpp/vendor/stable-diffusion.cpp stable-diffusion-cpp/vendor/ggml-webgpu-source'))
            else:
                self.assertNotIn('git submodule update', block)
                source = 'upstream-stable' if name == 'test' else '"$SOURCE_ID"'
                self.assertIn('python3 "$GITHUB_WORKSPACE/llama-cpp/scripts/checkout_source.py" --source ' + source, block)
            self.assertIn('ref: ${{ env.LCB_SOURCE_COMMIT }}', block)
            self.assertIn('persist-credentials: false', block)

    def test_each_package_has_one_deferred_assembly_then_mandatory_smoke_finalizer(self):
        for job, runtime in [('build', 'llama-cpp'), ('image-build', 'stable-diffusion-cpp')]:
            block = JOBS[job]
            self.assertEqual(block.count('scripts/package_runtime.py'), 1)
            self.assertEqual(block.count('--defer-npm-pack'), 1)
            self.assertEqual(block.count('scripts/validate_runtime_package.py'), 1)
            self.assertNotIn('record_browser_validation.py', block)
            final = f'python3 ../scripts/validate_runtime_package.py --runtime {runtime} --package dist/package'
            self.assertIn(final, block)
            before_upload = block[:block.index('- uses: actions/upload-artifact@v4')]
            self.assertIn(final, before_upload)
            self.assertIn('restore_source_package.py', before_upload)
            self.assertLess(block.index(final), block.index('name: '+('image-runtime-package' if job=='image-build' else 'llama-runtime-package')))
            final_step = before_upload.split('- name: Test and finalize the same runtime snapshot', 1)[1]
            self.assertNotIn('continue-on-error', final_step)
            fresh_guard = "if: ${{ !matrix.reuse }}" if job=='build' else "if: needs.plan.outputs.reuse-image != 'true'"
            self.assertIn(fresh_guard, final_step)
            self.assertIn('Revalidate and restore', final_step)

    def test_existing_node_and_browser_checks_remain_explicit(self):
        source = (ROOT / 'scripts/validate_runtime_package.py').read_text()
        self.assertIn("'tests/asyncify-rewind.mjs'", source)
        self.assertIn("'tests/browser-smoke.mjs'", source)
        self.assertIn("'test.generate_fixture'", source)
        self.assertIn('validator.validate(package)', source)
        self.assertIn('check_npm_pack=False', source)
        for job in ('build', 'image-build'):
            self.assertIn('playwright@1.58.2', JOBS[job])
            self.assertIn('install --with-deps chromium', JOBS[job])

    def test_sdk_activation_cannot_silently_remove_planned_cache(self):
        for name in ('compile', 'image-compile'):
            block = JOBS[name]
            positions = [block.index(text) for text in (
                'planned_em_cache="$EM_CACHE"', 'source ../.tools/emsdk/emsdk_env.sh',
                'export EM_CACHE="$planned_em_cache"', 'pipeline_metrics.py" environment', '--phase build.total')]
            self.assertEqual(positions, sorted(positions))
            self.assertIn('scripts/build.py --fresh --profile "$PROFILE" --variant "$VARIANT"', block)

    def test_metrics_uploads_are_separate_bounded_and_best_effort(self):
        self.assertIn("options: [basic, 'off']", WORKFLOW)
        for name, block in JOBS.items():
            start = block.index('- name: Upload optional pipeline timings')
            step = block[start:]
            self.assertIn("if: always() && env.BIC_METRICS_MODE == 'basic'", step)
            self.assertIn('continue-on-error: true', step)
            self.assertIn('path: ${{ runner.temp }}/bic-pipeline-metrics/', step)
            self.assertIn('pipeline-metrics-${{ github.run_attempt }}-' + name, step)
            self.assertIn('retention-days: 14', step)
            self.assertNotIn('include-hidden-files: true', step)

    def test_synthetic_unit_tests_do_not_pollute_pipeline_metrics(self):
        for phase in ('test.llama_python', 'test.publication_python', 'test.image_python'):
            self.assertIn('--phase ' + phase + ' -- env BIC_METRICS_MODE=off python3 -m unittest', WORKFLOW)

    def test_observation_does_not_weaken_cache_save_conditions(self):
        for name in ('compile', 'image-compile'):
            step = JOBS[name].split('- name: Save reusable intermediates', 1)[1].split('- name:', 1)[0]
            self.assertIn("success() && github.event_name == 'push'", step)
            self.assertIn('github.event.repository.default_branch', step)
            self.assertIn("github.actor != 'dependabot[bot]'", step)
        action = (ROOT / '.github/actions/restore-browser-cache/action.yml').read_text()
        self.assertIn('steps.em.outputs.cache-matched-key', action)
        self.assertIn('steps.cc.outputs.cache-matched-key', action)
        self.assertIn('Observation only', (ROOT / 'scripts/pipeline_metrics.py').read_text())

    def test_verification_cannot_opt_out_of_npm(self):
        for runtime in ('llama-cpp', 'stable-diffusion-cpp'):
            result = subprocess.run(['python3', str(ROOT / runtime / 'scripts/package_runtime.py'),
                                     '--verify-only', '--defer-npm-pack'], capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn(b'--verify-only always includes npm packing', result.stderr)

    def test_profile_counts_and_job_dependencies_remain(self):
        self.assertIn('needs: [plan, test, compile]', JOBS['build'])
        self.assertIn('needs: [plan, image-native, image-compile]', JOBS['image-build'])
        self.assertIn('needs: [assemble]', JOBS['publish'])
        self.assertEqual(len(json.loads((ROOT / 'llama-cpp/config/profiles.json').read_text())), 5)
        self.assertEqual(len(json.loads((ROOT / 'stable-diffusion-cpp/config/profiles.json').read_text())), 3)
        self.assertNotIn('max-parallel:', '\n'.join(line for line in WORKFLOW.splitlines() if not line.strip().startswith('#')))

    def test_cache_recovery_is_mandatory_after_best_effort_restore(self):
        action = (ROOT / '.github/actions/restore-browser-cache/action.yml').read_text()
        for kind in ('dawn', 'em', 'cc'):
            restore = action.split('      id: ' + kind + '\n', 1)[1].split('    - ', 1)[0]
            self.assertIn("if: inputs.mode != 'cold'", restore)
            self.assertIn('continue-on-error: true', restore)
            guard = action.split('      id: admit-' + kind + '\n', 1)[1].split('    - ', 1)[0]
            self.assertNotIn('continue-on-error:', guard); self.assertNotIn('      if:', guard)
            self.assertIn('steps.' + kind + '.outcome', guard)
            self.assertIn('steps.' + kind + '.outputs.cache-matched-key', guard)
            self.assertIn('finish --kind ' + kind, guard)
        positions = [action.index(text) for text in (
            '--phase sdk.install', ' park-em --requested', '      id: em\n',
            '      id: admit-em\n', '      id: cc\n', '      id: admit-cc\n', 'ccache --zero-stats')]
        self.assertEqual(positions, sorted(positions))

    def test_cold_runs_do_not_restore_or_save_shared_caches(self):
        self.assertIn('options: [enabled, cold]', WORKFLOW)
        self.assertIn("BIC_CACHE_MODE: ${{ github.event.inputs.cache_mode || 'enabled' }}", WORKFLOW)
        for name in ('compile', 'image-compile'):
            block = JOBS[name]
            self.assertEqual(block.count('mode: ${{ env.BIC_CACHE_MODE }}'), 2)
            self.assertIn("env.BIC_CACHE_MODE != 'cold'", block)
            for kind in ('dawn', 'em', 'cc'):
                self.assertIn(kind + '-save-allowed: ${{ steps.cache.outputs.' + kind + '-save-allowed }}', block)
        action = (ROOT / '.github/actions/save-browser-cache/action.yml').read_text()
        for kind in ('dawn', 'em', 'cc'):
            save = action.split('      id: save-' + kind + '\n', 1)[1].split('    - ', 1)[0]
            self.assertIn("inputs.mode != 'cold'", save)
            self.assertIn("inputs." + kind + "-save-allowed == 'true'", save)
            self.assertIn("steps.check-" + kind + ".outputs.allowed == 'true'", save)
            self.assertIn("github.event_name == 'push'", save)
            self.assertIn('github.event.repository.default_branch', save)
            self.assertIn('continue-on-error: true', save)

    def test_save_outcomes_do_not_claim_server_persistence(self):
        action = (ROOT / '.github/actions/save-browser-cache/action.yml').read_text()
        self.assertIn('Observe cache save outcomes', action)
        self.assertIn('steps.save-em.outcome', action)
        source = (ROOT / 'scripts/pipeline_metrics.py').read_text()
        self.assertIn('action-completed-not-server-proof', source)


class SourceShardPathContract(unittest.TestCase):
    def test_public_assembler_reads_flat_downloads_not_local_nightly_build_tree(self):
        text=JOBS['build']
        self.assertIn('python3 scripts/package_runtime.py --source "$SOURCE_ID" --build-root build --defer-npm-pack',text)
        self.assertIn('path: llama-cpp/build/',text)
        self.assertIn('pattern: profile-build-${{ matrix.source }}-*',text)


if __name__ == '__main__': unittest.main()
