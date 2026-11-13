"""Execute literal reporter YAML with local gh/curl/Docker stand-ins.

These checks cover command ordering, bounded pages and credentials at handoffs;
they do not exercise live GitHub writes, signed storage or Docker isolation.
"""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from build_tools.project import REPOSITORY_ROOT
from build_tools.publication import comments
from tests.build_tools.publication.test_comments import pr, run, archive_for, bot_comment, payload, REPO, A, B
from tests.support.workflow import jobs, steps, shell_body, docker_standin


@unittest.skipUnless(shutil.which('bash') and shutil.which('jq'), 'bash and jq required')
class ReporterWorkflow(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='bicore-comment-workflow-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / 'bin'; self.bin.mkdir()
        docker_standin(self.bin)
        script = (REPOSITORY_ROOT / 'tests/support/comment_service_fixture.py').read_text()
        for name in ('gh', 'curl'):
            path = self.bin / name; path.write_text('#!' + sys.executable + '\n' + script); path.chmod(0o755)
        (self.root / 'build_tools').symlink_to(REPOSITORY_ROOT / 'build_tools', target_is_directory=True)
        self.config = {'pulls': [pr()], 'runs': [run()]}
        self.archive = archive_for(run())
        self.env = {**os.environ, 'PATH': str(self.bin) + os.pathsep + os.environ['PATH'],
                    'BIC_TEST_CONTAINER_PATH': os.environ['PATH'], 'GITHUB_REPOSITORY': REPO,
                    'GITHUB_WORKSPACE': str(self.root), 'RUNNER_TEMP': str(self.root),
                    'REPORT_STATE': str(self.root / 'state'), 'COMMENTS_FIXTURE': str(self.root),
                    'BIC_OFFLINE_IMAGE': 'local-test-only-no-isolation'}
        self.job = jobs((REPOSITORY_ROOT / '.github/workflows/runtime-comments.yml').read_text())['report']

    def execute(self):
        (self.root / 'scenario.json').write_text(json.dumps(self.config))
        (self.root / 'report.zip').write_bytes(self.archive)
        for step in steps(self.job):
            if '        run: |\n' not in step or step.startswith('name: Acquire offline automation image'): continue
            env = {**self.env}
            # Match the real workflow: only explicit env declarations supply a
            # credential, and the signed-storage step declares none.
            env.pop('GH_TOKEN', None)
            if '          GH_TOKEN: ' in step: env['GH_TOKEN'] = 'fixture-token'
            result = subprocess.run(['bash', '-euo', 'pipefail', '-c', shell_body(step)],
                                    cwd=self.root, env=env, text=True, capture_output=True)
            if result.returncode: return result
        return result

    def recorded(self, name):
        path = self.root / name
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def assert_success(self):
        result = self.execute()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return self.recorded('writes.jsonl')

    def test_successful_attempt_flows_through_credential_separated_download_and_comment(self):
        writes = self.assert_success()
        self.assertEqual(len(writes), 1)
        self.assertEqual(writes[0]['method'], 'POST')
        self.assertEqual(writes[0]['endpoint'], f'repos/{REPO}/issues/7/comments')
        self.assertIn('succeeded for this PR head', writes[0]['body'])
        self.assertIn('npm install', writes[0]['body'])
        curl = [call for call in self.recorded('service-calls.jsonl') if call['tool'] == 'curl']
        self.assertEqual([call['hasToken'] for call in curl], [True, False])

    def test_successful_direct_archive_response_needs_no_storage_request(self):
        self.config['directArchive'] = True
        self.assertEqual(len(self.assert_success()), 1)
        self.assertEqual(len([call for call in self.recorded('service-calls.jsonl') if call['tool'] == 'curl']), 1)

    def test_only_an_owned_bot_comment_is_updated(self):
        self.config['comments'] = [bot_comment(comments.MARKER + '\nOld')]
        writes = self.assert_success()
        self.assertEqual(writes[0]['method'], 'PATCH')
        self.assertEqual(writes[0]['endpoint'], f'repos/{REPO}/issues/comments/42')

    def test_fork_pr_produces_no_authenticated_followup(self):
        self.config['pulls'] = [pr(repository='attacker/core')]
        self.assertEqual(self.assert_success(), [])
        calls = self.recorded('service-calls.jsonl')
        self.assertEqual(len(calls), 1)
        self.assertIn(f'repos/{REPO}/pulls', calls[0]['args'])

    def test_fresh_head_or_rerun_change_suppresses_the_write(self):
        self.config['currentRuns'] = [run(attempt=2, status='queued', conclusion=None)]
        self.assertEqual(self.assert_success(), [])
        self.assertFalse(json.loads((self.root / 'state/7/write.json').read_text())['write'])

    def test_newest_rerun_on_a_later_page_is_selected(self):
        values = [{**run(id=200 + i), 'run_started_at': '2026-09-22T10:00:00Z'} for i in range(100)]
        newest = {**run(id=100, attempt=3), 'run_started_at': '2026-09-22T12:00:00Z'}
        self.config['runs'] = values + [newest]
        self.archive = archive_for(newest)
        writes = self.assert_success()
        self.assertIn('/actions/runs/100', writes[0]['body'])
        self.assertIn('(attempt 3)', writes[0]['body'])
        calls = self.recorded('service-calls.jsonl')
        history = [call for call in calls if any('/actions/workflows/build.yml/runs' in arg for arg in call['args'])]
        self.assertEqual(len(history), 4)
        self.assertEqual(sum('page=2' in call['args'] for call in history), 2)

    def test_excessive_or_incomplete_build_history_never_writes(self):
        self.config['totalCount'] = 1001
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.recorded('writes.jsonl'), [])

    def test_truncated_history_never_claims_the_latest_run(self):
        for total in (2, 101):
            with self.subTest(total=total):
                self.config['totalCount'] = total
                result = self.execute()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Incomplete workflow run history', result.stderr)
                self.assertEqual(self.recorded('writes.jsonl'), [])

    def test_truncated_final_refresh_cannot_overwrite_a_comment(self):
        self.config['currentRuns'] = []
        self.config['currentTotalCount'] = 1
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Incomplete workflow run history', result.stderr)
        self.assertEqual(self.recorded('writes.jsonl'), [])

    def test_archive_listing_failure_preserves_prior_report_with_an_error(self):
        self.config['failArtifactListing'] = True
        self.config['comments'] = [bot_comment(comments.comment_body(pr(), run(), payload()))]
        writes = self.assert_success()
        self.assertIn('Previous successful publication', writes[0]['body'])
        self.assertIn('Report retrieval error', writes[0]['body'])
        self.assertNotIn('succeeded for this PR head', writes[0]['body'])
        self.assertFalse(any(call['tool'] == 'curl' for call in self.recorded('service-calls.jsonl')))

    def test_unsafe_redirect_is_not_followed(self):
        self.config['redirect'] = 'http://storage.invalid/untrusted'
        writes = self.assert_success()
        self.assertIn('Unexpected artifact download redirect', writes[0]['body'])
        self.assertEqual(len([call for call in self.recorded('service-calls.jsonl') if call['tool'] == 'curl']), 1)

    def test_mutation_failure_is_not_retried(self):
        self.config['failWrite'] = True
        result = self.execute()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.recorded('writes.jsonl')), 1)

    def test_every_reconciled_pr_gets_a_fresh_head_and_run_read(self):
        self.config['pulls'] = [pr(number=7), pr(number=8)]
        writes = self.assert_success()
        self.assertEqual({item['endpoint'] for item in writes},
                         {f'repos/{REPO}/issues/{number}/comments' for number in (7, 8)})
        calls = self.recorded('service-calls.jsonl')
        for number in (7, 8):
            self.assertTrue(any(f'repos/{REPO}/pulls/{number}' in call['args'] for call in calls))


if __name__ == '__main__': unittest.main()
