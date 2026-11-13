"""Literal updater acquisition shell against local Git and gh/Docker stand-ins."""
import json
from pathlib import Path
import sys
import unittest

from build_tools.project import REPOSITORY_ROOT
from build_tools.runtimes.llama_cpp import update
from tests.support.updater import UpdaterFixture


def release(name, *, prerelease=False, draft=False):
    return {'tag_name': name, 'draft': draft, 'prerelease': prerelease,
            'published_at': '2026-09-22T00:00:00Z',
            'html_url': 'https://github.com/ggml-org/llama.cpp/releases/tag/' + name}


class UpdaterWorkflow(UpdaterFixture):
    def setUp(self):
        super().setUp()
        code = (REPOSITORY_ROOT / 'tests/support/update_service_fixture.py').read_text()
        path = self.bin / 'gh'; path.write_text('#!' + sys.executable + '\n' + code); path.chmod(0o755)
        self.service = {'relation': 'ahead', 'latest': release('v1.0.0'), 'releases': []}
        self.save_service()

    def save_service(self):
        (self.home / 'service.json').write_text(json.dumps(self.service))

    def execute(self, name, **env):
        result = self.run_step(name, **env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def request(self, target='custom', ref='refs/tags/v1.0.0'):
        self.execute('Validate the source and update request', TARGET=target, CUSTOM_REF=ref,
                     BASE='develop', ALLOW_NON_FAST_FORWARD='false')

    def selected(self):
        self.execute('Select the requested upstream ref')
        return update.load(self.state, 'selection.json')['ref']

    def fetch(self, selected):
        return self.run_step('Fetch the exact upstream and comparison (online)',
                             SELECTED_REF=selected, PREVIOUS_COMMIT=self.old)

    def api_calls(self):
        path = self.home / 'api-calls.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []

    def test_annotated_tag_resolves_to_its_commit_without_checking_out_the_candidate(self):
        self.git('tag', '-a', 'v1.0.0', '-m', 'annotated release fixture', self.new, cwd=self.upstream)
        self.request()
        result = self.fetch(self.selected())
        self.assertEqual(result.returncode, 0, result.stderr)
        resolved = update.load(self.state, 'resolved.json')
        self.assertEqual(resolved, {'ref': 'refs/tags/v1.0.0', 'commit': self.new})
        self.assertEqual(self.git('rev-parse', 'HEAD', cwd=self.root / 'vendor/llama.cpp'), self.old)
        self.execute('Plan the source-bound candidate branch')
        plan = update.load(self.state, 'plan.json')
        self.assertEqual(plan['baseCommit'], self.base)
        self.assertEqual(plan['target']['commit'], self.new)
        self.assertEqual(plan['status'], 'planned')

    def test_ambiguous_tag_and_branch_require_an_explicit_prefix(self):
        self.git('tag', 'shared', self.new, cwd=self.upstream)
        self.git('branch', 'shared', self.old, cwd=self.upstream)
        self.request(ref='shared')
        result = self.fetch(self.selected())
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('missing or ambiguous', result.stderr)
        self.assertEqual(self.api_calls(), [])
        self.assertFalse((self.state / 'resolved.json').exists())
        result = self.fetch('refs/heads/shared')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(update.load(self.state, 'resolved.json')['commit'], self.old)

    def test_full_commit_is_fetched_and_checked_exactly(self):
        self.request(ref=self.new)
        result = self.fetch(self.selected())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(update.load(self.state, 'resolved.json')['commit'], self.new)
        self.assertEqual(len(self.api_calls()), 1)

    def test_non_commit_tag_cannot_become_a_candidate(self):
        blob = self.git('rev-parse', self.new + ':new-feature', cwd=self.upstream)
        self.git('tag', 'not-a-commit', blob, cwd=self.upstream)
        self.request(ref='refs/tags/not-a-commit')
        result = self.fetch(self.selected())
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.state / 'resolved.json').exists())
        self.assertEqual(self.api_calls(), [])

    def test_released_nightly_search_reads_later_pages_and_stops_at_the_first_match(self):
        self.service['releases'] = [release('v1.0.' + str(i)) for i in range(100)] + [
            release('b100', prerelease=True, draft=True), release('b99', prerelease=True)]
        self.save_service()
        self.request(target='latest-unstable', ref='')
        self.execute('Acquire the requested release metadata (online)', TARGET='latest-unstable')
        self.assertEqual(self.selected(), 'refs/tags/b99')
        self.assertEqual(len(self.api_calls()), 2)
        self.assertIn('page=2', self.api_calls()[-1])

    def test_stable_read_failure_never_falls_back_to_a_list_or_master(self):
        self.service['fail'] = True; self.save_service()
        self.request(target='latest', ref='')
        result = self.run_step('Acquire the requested release metadata (online)', TARGET='latest')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(self.api_calls()), 1)
        self.assertIn('repos/ggml-org/llama.cpp/releases/latest', self.api_calls()[0])
        self.execute('Render the manual pull request handoff', PUSH_OUTCOME='skipped')
        self.assertEqual(update.load(self.state, 'report.json')['status'], 'failed')
        self.assertNotIn('[Open pull request form]', (self.home / 'summary').read_text())


if __name__ == '__main__': unittest.main()
