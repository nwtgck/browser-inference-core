"""Guard the security and event boundaries of the three automation workflows."""
from pathlib import Path
import json
import os
import re
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WorkflowBoundaries(unittest.TestCase):
    def setUp(self):
        folder = ROOT.parent / '.github/workflows'
        self.build = (folder / 'build.yml').read_text()
        self.update = (folder / 'update-llama-cpp.yml').read_text()
        self.report = (folder / 'runtime-comments.yml').read_text()

    def test_existing_push_publish_and_configured_matrix_remain(self):
        import sys
        sys.path.insert(0,str(ROOT/'scripts'))
        from source_config import matrix
        entries=matrix(ROOT)
        self.assertEqual({e['profile'] for e in entries if e['source']=='upstream-stable'},
                         set(json.loads((ROOT/'config/profiles.json').read_text())))
        self.assertEqual(len([e for e in entries if e['source']=='upstream-nightly']),4)
        self.assertIn('matrix: ${{ fromJSON(needs.plan.outputs.compile-matrix) }}',self.build)
        self.assertIn('needs: [plan, test, compile]', self.build)
        self.assertIn('fail-fast: false', self.build)
        self.assertIn('  pull_request:', self.build)
        self.assertIn('types: [opened, reopened, synchronize]', self.build)
        self.assertIn('Push + synchronize duplicates are intentional', self.build)
        self.assertNotIn('concurrency:', self.build)

    def test_matrix_has_no_artificial_parallelism_cap(self):
        for workflow in (ROOT.parent / '.github/workflows').glob('*.yml'):
            with self.subTest(workflow=workflow.name):
                self.assertNotRegex(workflow.read_text(), r'(?m)^\s*max-parallel\s*:')
        self.assertIn('# Intentionally omit max-parallel:', self.build)
        self.assertIn('reduce build wait time', self.build)
        self.assertIn('Do not add a workflow-level parallelism cap', self.build)
        distribution = (ROOT / 'docs/distribution.md').read_text()
        self.assertIn('intentionally omits `max-parallel`', distribution)
        self.assertNotIn('up to four concurrent builds', distribution)

    def test_all_jobs_check_out_the_same_immutable_head_not_the_merge_commit(self):
        self.assertIn('LCB_SOURCE_COMMIT: ${{ github.event.pull_request.head.sha || github.sha }}', self.build)
        self.assertEqual(self.build.count('ref: ${{ env.LCB_SOURCE_COMMIT }}'), 9)
        self.assertEqual(self.build.count('persist-credentials: false'), 9)
        self.assertNotIn('expected_source', self.build)
        self.assertIn('test "$(git rev-parse HEAD)" = "$LCB_SOURCE_COMMIT"', self.build)
        self.assertIn("source != os.environ['LCB_SOURCE_COMMIT']", self.build)
        self.assertLess(self.build.index('Verify package source before publication'), self.build.index('scripts/publish_artifacts.py'))

    def test_pr_publication_is_limited_to_same_repository_heads(self):
        publish = self.build.split('  publish:', 1)[1]
        self.assertIn('github.event.pull_request.head.repo.full_name == github.repository', publish)
        self.assertIn("github.actor != 'dependabot[bot]'", publish)
        self.assertEqual(self.build.count('contents: write'), 1)
        self.assertNotIn('pull_request_target:', self.build)
        self.assertNotIn('actions: write', self.build)
        self.assertNotIn('pull-requests: write', self.build)

    def test_artifact_branches_do_not_reenter_source_builds(self):
        self.assertEqual(self.build.count("github.head_ref != 'artifacts'"), 2)
        self.assertEqual(self.build.count("github.ref_name != 'artifacts'"), 2)
        self.assertEqual(self.build.count('    branches-ignore:'), 2)
        # The two independent entry jobs reject artifact refs. Downstream jobs
        # depend on the admitted plan rather than duplicating this policy.
        blocks=dict(re.findall(r'^  ([a-z-]+):\n(.*?)(?=^  [a-z-]+:\n|\Z)',self.build.split('\njobs:\n',1)[1],re.M|re.S))
        for name in ('plan','test'):
            self.assertIn("github.ref_name != 'artifacts'",blocks[name])
        for name in ('compile','image-compile','image-native'):
            self.assertIn('needs: [plan]',blocks[name])
        self.assertIn("needs.plan.result == 'success'",blocks['assemble'])


    def test_lock_resolution_is_after_publication_and_report_is_attempt_bound(self):
        self.assertLess(self.build.index('scripts/publish_artifacts.py'), self.build.index('scripts/consumer_metadata.py'))
        self.assertIn('consumer-update-${{ github.run_attempt }}', self.build)
        self.assertIn('steps.publish.outputs.commit', self.build)
        self.assertIn('consumer reporting failed', self.build)

    def test_actual_prepublication_guard_rejects_wrong_or_missing_manifest_identity(self):
        # Exercise the actual workflow snippet, not a second implementation of
        # the guard. No package publication or network command is executed.
        match = re.search(r'      - name: Verify package source before publication\n        run: \|\n((?:          .*\n)+)', self.build)
        self.assertIsNotNone(match)
        command = textwrap.dedent(match.group(1))
        for manifest, expected, passed in (
            ({'sourceCommit': 'a' * 40}, 'a' * 40, True),
            ({'sourceCommit': 'b' * 40}, 'a' * 40, False),
            ({}, 'a' * 40, False),
            ({'sourceCommit': 'a' * 40}, '', False),
        ):
            with self.subTest(manifest=manifest, expected=expected), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'dist/package/manifest.json'
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps(manifest))
                original = path.read_bytes()
                env = {**os.environ, 'LCB_SOURCE_COMMIT': expected}
                result = subprocess.run(['bash', '-e', '-c', command], cwd=tmp, env=env,
                                        text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode == 0, passed, result.stderr)
                self.assertEqual(path.read_bytes(), original)

    def test_updater_write_permission_overrides_only_the_update_job(self):
        # Scope matters: merely finding "contents: write" somewhere in a
        # workflow would not prove that the branch-pushing job receives it.
        defaults, jobs = self.update.split('\njobs:\n', 1)
        self.assertRegex(defaults, r'(?m)^permissions:\n  contents: read\n')
        self.assertNotRegex(defaults, r'(?m)^  [\w-]+: write$')
        update_job = jobs.split('  update:\n', 1)[1]
        self.assertRegex(update_job, r'(?m)^    permissions:\n      contents: write\n    steps:')
        self.assertEqual(self.update.count('\n    permissions:\n'), 1)
        self.assertNotRegex(self.update, r'(?m)^\s+(?:actions|pull-requests): write$')
        self.assertIn('This is not a cap:', self.update)
        self.assertIn('contents: read alone cannot push', self.update)

    def test_reporter_has_read_only_content_and_actions_with_comment_writes(self):
        defaults, jobs = self.report.split('\njobs:\n', 1)
        self.assertRegex(defaults, r'(?m)^permissions:\n  contents: read\n  actions: read\n  pull-requests: write\n')
        self.assertNotRegex(jobs, r'(?m)^\s+permissions:')

    def test_updater_inputs_are_data_not_shell_interpolation(self):
        for item in ['options: [latest, custom]', 'allow_non_fast_forward:', 'contents: write']:
            self.assertIn(item, self.update)
        run = self.update.split('        run: |', 1)[1].split('      - uses:', 1)[0]
        self.assertNotIn('${{', run)
        self.assertIn('--custom-ref "$CUSTOM_REF"', run)
        self.assertNotIn('secrets.', self.update)
        self.assertNotIn('npm ', self.update)
        self.assertNotIn('actions: write', self.update)
        self.assertNotIn('pull-requests: write', self.update)
        updater = (ROOT / 'scripts/update_llama_cpp.py').read_text()
        self.assertNotIn('api.create_pull_request', updater)
        self.assertNotIn('api.iter_pull_requests', updater)
        self.assertNotIn('api.dispatch_build', updater)
        self.assertNotIn('expected_source', updater)
        self.assertIn('Open pull request form', updater)

    def test_privileged_reporter_always_checks_out_the_default_branch(self):
        self.assertIn('ref: ${{ github.event.repository.default_branch }}', self.report)
        self.assertIn('persist-credentials: false', self.report)
        self.assertNotIn('ref: ${{ github.event.pull_request.head', self.report)
        self.assertNotIn('submodules:', self.report)
        self.assertNotIn('contents: write', self.report)
        self.assertNotIn('npm install', self.report)
        self.assertNotIn('actions: write', self.report)
        self.assertIn('head.repo.full_name == github.repository', self.report)

    def test_reporter_covers_late_prs_reruns_and_manual_repair(self):
        self.assertIn('types: [opened, reopened, synchronize]', self.report)
        self.assertIn('types: [requested, in_progress, completed]', self.report)
        self.assertIn('workflow_dispatch:', self.report)
        self.assertIn('cancel-in-progress: false', self.report)

    def test_generated_consumer_knowledge_is_not_part_of_the_package_builder(self):
        generator = (ROOT / 'scripts/consumer_metadata.py').read_text()
        self.assertIn('Intentional policy exception', generator)
        for relative in ['scripts/build.py', 'scripts/package_runtime.py', 'scripts/stage_ci_build.py', 'CMakeLists.txt']:
            code = (ROOT / relative).read_text().lower()
            self.assertNotIn('naidan', code)
            self.assertNotIn('consumer_knowledge', code)
        self.assertNotIn('must not normally', generator.split('def consumer_knowledge()', 1)[1].split('return {', 1)[1])


if __name__ == '__main__':
    unittest.main()
