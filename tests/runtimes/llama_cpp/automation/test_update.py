"""Offline selection/preflight plus real local Git workflow handoff regressions."""
import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlparse

from build_tools.github import full_sha, repository_name
from build_tools.runtimes.llama_cpp import update
from tests.support.updater import UpdaterFixture

A, B, C = 'a' * 40, 'b' * 40, 'c' * 40


def release(name, prerelease=False, draft=False):
    return {'tag_name': name, 'draft': draft, 'prerelease': prerelease,
            'published_at': '2026-09-22T00:00:00Z', 'html_url': 'https://github.com/ggml-org/llama.cpp/releases/tag/' + name}


class ReleaseSelection(unittest.TestCase):
    def test_stable_uses_release_status_and_exact_tag(self):
        result = update.select({'requested': 'latest'}, release('v1.2.3'))
        self.assertEqual(result['ref'], 'refs/tags/v1.2.3')
        self.assertFalse(result['release']['prerelease'])

    def test_stable_never_falls_back_to_nightly_or_master(self):
        for value in [release('b999'), release('v1.2.3', True), release('v1.2.3', draft=True),
                      release('v1.2.3-rc1'), release('v01.2.3'), release('v1.2.3extra'), None,
                      {**release('v1.2.3'), 'published_at': None}]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                update.select({'requested': 'latest'}, value)

    def test_unstable_uses_published_nightly_not_draft_or_other_tags(self):
        values = [release('b100', True, True), release('v2.0.0-rc1', True), release('v1.2.3'),
                  release('b99', True), release('b98', True)]
        result = update.select({'requested': 'latest-unstable'}, values)
        self.assertEqual(result['ref'], 'refs/tags/b99')

    def test_no_nightly_is_an_error(self):
        with self.assertRaises(ValueError):
            update.select({'requested': 'latest-unstable'}, [release('v1.0.0')])

    def test_custom_full_commit_and_explicit_master_remain_explicit(self):
        for ref in (A, 'master', 'refs/heads/master', 'refs/tags/v1.0.0'):
            with self.subTest(ref=ref):
                self.assertEqual(update.select({'requested': 'custom', 'customRef': ref})['ref'], ref)

    def test_input_injection_and_revision_expressions_are_rejected(self):
        for value in ['', '-x', 'HEAD~1', 'x^{}', 'x\ny', 'https://evil/x', 'x;id', 'x..y', 'x@{1}', 'a.lock']:
            with self.subTest(value=value), self.assertRaises(ValueError): update.checked_ref(value)

    def test_identifiers_and_branch_deduplication(self):
        with self.assertRaises(ValueError): full_sha('abc')
        with self.assertRaises(ValueError): repository_name('owner/repo?token=secret')
        self.assertEqual(update.candidate_branch('develop', A, B), update.candidate_branch('develop', A, B))
        self.assertNotEqual(update.candidate_branch('develop', A, B), update.candidate_branch('main', A, B))
        self.assertNotEqual(update.candidate_branch('develop', A, B), update.candidate_branch('develop', B, B))


class ErrorReports(unittest.TestCase):
    def errors(self):
        return [subprocess.CalledProcessError(1, ['git', 'apply'], stderr=value)
                for value in (b'bad utf8 \xff\n', ' text diagnostic\n', b'', '', None,
                              ('é' * 5000).encode('utf-8'))] + [OSError('fixture error')]

    def test_preflight_errors_are_bounded_text_for_json_and_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            for error in self.errors():
                with self.subTest(stderr=getattr(error, 'stderr', None)), \
                     patch.object(update, 'prepare', side_effect=error):
                    result = update.overlay_preflight(Path(temporary))
                    self.assertEqual(result['status'], 'failed')
                    self.assertIsInstance(result['error'], str)
                    self.assertLessEqual(len(result['error']), 4000)
                    self.assertEqual(json.loads(json.dumps(result)), result)
                    self.assertIn('```json', update.render_summary({'preflight': result}))
                    if getattr(error, 'stderr', None) == b'bad utf8 \xff\n':
                        self.assertEqual(result['error'], 'bad utf8 \ufffd')
                    if not getattr(error, 'stderr', None):
                        self.assertEqual(result['error'], str(error))

    def test_main_writes_failed_json_reports_for_bytes_text_and_empty_stderr(self):
        with tempfile.TemporaryDirectory() as temporary:
            state = Path(temporary)
            for error in self.errors():
                args = ['updater', 'request', '--state', str(state), '--base', 'develop']
                with self.subTest(stderr=getattr(error, 'stderr', None)), \
                     patch.object(update, 'request', side_effect=error), \
                     patch.object(sys, 'argv', args), \
                     patch.dict(os.environ, {'GITHUB_REPOSITORY': 'example/core'}), \
                     self.assertRaises(type(error)):
                    update.main()
                report = update.load(state, 'report.json')
                self.assertEqual(report['status'], 'failed')
                self.assertEqual(report['error'], update.error_details(error))
                self.assertIsInstance(report['error'], str)
                self.assertLessEqual(len(report['error']), 4000)
                output = io.StringIO()
                with patch.object(sys, 'argv', ['updater', 'report', '--state', str(state)]), \
                     patch.dict(os.environ, {'GITHUB_STEP_SUMMARY': ''}), contextlib.redirect_stdout(output):
                    update.main()
                self.assertIn('```json', output.getvalue())


class LocalGitProposal(UpdaterFixture):
    def test_preflight_fails_when_required_webgpu_overlay_is_incompatible(self):
        for message in ('WebGPU source changed', 'SSM convolution needs semantic review: upstream shader changed'):
            with self.subTest(message=message):
                self.webgpu_prepare.side_effect = ValueError(message)
                result = update.overlay_preflight(self.root)
                self.assertEqual(result['status'], 'failed')
                self.assertIn(message, result['error'])
                self.assertEqual(self.webgpu_prepare.call_args.kwargs['patch_root'], self.root / update.PATCH_DIRECTORY)

    def test_preflight_covers_both_qsa_source_variants_and_propagates_drift(self):
        result = update.overlay_preflight(self.root)
        self.assertEqual(result['status'], 'passed')
        self.qsa_preflight.assert_called_once()
        self.assertEqual(self.qsa_preflight.call_args.args[0], self.root / 'vendor/llama.cpp')
        self.qsa_preflight.side_effect = ValueError('QSA selection needs upstream semantic review')
        result = update.overlay_preflight(self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('QSA selection', result['error'])

    def test_preflight_also_fails_closed_on_audio_patch_conflicts(self):
        audio = self.root / 'vendor/llama.cpp/tools/mtmd/mtmd-subcontext.cpp'
        audio.write_text('upstream audio changed\n')
        result = update.overlay_preflight(self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('mtmd-subcontext.cpp', result['error'])
        self.assertEqual(audio.read_text(), 'upstream audio changed\n')

    def test_preflight_covers_the_target_roots_model_graph_and_propagates_drift(self):
        result = update.overlay_preflight(self.root)
        self.assertEqual(result['status'], 'passed')
        self.model_graph_prepare.assert_called_once()
        self.assertEqual(self.model_graph_prepare.call_args.args[0], self.root / 'vendor/llama.cpp')
        self.assertEqual(self.model_graph_prepare.call_args.kwargs['runtime'], self.root)
        self.model_graph_prepare.side_effect = ValueError('Independent vocabulary patch identity changed')
        result = update.overlay_preflight(self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('Independent vocabulary', result['error'])

    def test_preflight_checks_the_target_roots_threadpool_patch(self):
        path = self.root / update.PATCH_DIRECTORY / 'mtmd-non-pthread-threadpool.patch'
        path.write_bytes(path.read_bytes() + b'\n')
        result = update.overlay_preflight(self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('mtmd-non-pthread-threadpool.patch', result['error'])
        self.assertNotIn('preflight', [call.args[0] for call in self.webgpu_prepare.call_args_list])

    def test_unrelated_tts_changes_do_not_expand_preflight_dependencies(self):
        # These internal files were dependencies of the removed optional overlay.
        # They must not gate a compatibility preflight for vision/reference audio.
        vendor = self.root / 'vendor/llama.cpp'
        for name in ('mtmd-helper-gen.cpp', 'mtmd-helper.h', 'models/models.h', 'models/qwen3tts-gen.cpp'):
            path = vendor / 'tools/mtmd' / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('Unrelated upstream generation refactor\n')
        result = update.overlay_preflight(self.root)
        self.assertEqual(result['status'], 'passed')
        self.assertEqual((vendor / 'tools/mtmd/mtmd-helper-gen.cpp').read_text(),
                         'Unrelated upstream generation refactor\n')

    def test_real_git_changes_exactly_two_pins_and_stops_after_push(self):
        result = self.propose()
        self.assertEqual(result['status'], 'branch-created')
        self.assertEqual(result['preflight']['status'], 'passed')
        self.assertEqual(set(self.git('diff', '--no-relative', '--name-only', self.base, result['sourceCommit'], cwd=self.root).splitlines()), {'runtimes/llama-cpp/' + path for path in update.PINS})
        self.assertEqual(json.loads((self.root / 'config/toolchain.json').read_text())['otherPin'], 'unchanged')
        self.assertEqual((self.root / 'vendor/llama.cpp/tools/mtmd/mtmd-model.cpp').read_text(), 'before\noriginal\nafter\n')
        remote = self.git('ls-remote', str(self.remote), 'refs/heads/' + result['branch'], cwd=self.home)
        self.assertEqual(remote.split()[0], result['sourceCommit'])
        query = parse_qs(urlparse(result['createPullRequestUrl']).query)
        self.assertEqual(query['quick_pull'], ['1'])
        self.assertEqual(query['title'], [result['pullRequestTitle']])
        self.assertEqual(query['body'], [result['pullRequestBody']])
        self.assertIn('develop...' + result['branch'], unquote(urlparse(result['compareUrl']).path))
        self.assertNotIn('pullRequest', result)
        self.assertNotIn('pullRequestUrl', result)

    def test_no_op_creates_no_branch_or_remote_operations(self):
        self.target['commit'] = self.old
        self.assertEqual(self.propose()['status'], 'unchanged')
        self.assertEqual(self.calls(), [])
        self.assertEqual(self.git('ls-remote', '--heads', str(self.remote), cwd=self.home), '')

    def test_dirty_checkout_is_rejected(self):
        (self.root / 'untracked').write_text('not an update input')
        with self.assertRaisesRegex(ValueError, 'clean'):
            self.propose()

    def test_mismatched_committed_pins_are_rejected(self):
        (self.root / 'config/toolchain.json').write_text(json.dumps({'llamaCommit': self.new}))
        self.git('add', '.', cwd=self.root)
        self.git('commit', '-qm', 'Mismatched pins', cwd=self.root)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            self.propose()

    def test_rollback_requires_explicit_opt_in(self):
        self.relation = 'behind'
        with self.assertRaisesRegex(ValueError, 'allow_non_fast_forward'):
            self.propose()
        self.assertEqual(self.propose(allow_non_fast_forward=True)['status'], 'branch-created')

    def test_conflicting_overlay_preserves_candidate_without_creating_a_pr(self):
        (self.root / 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-webgpu-bf16.patch').write_text('not a patch\n')
        self.git('add', '.', cwd=self.root)
        self.git('commit', '-qm', 'Broken patch fixture', cwd=self.root)
        result = self.propose()
        self.assertEqual(result['status'], 'branch-created-needs-overlay-repair')
        self.assertTrue(self.git('ls-remote', str(self.remote), 'refs/heads/' + result['branch'], cwd=self.home))
        self.assertNotIn('draft', result)
        self.assertIn('patch', result['preflight']['error'])
        summary = update.render_summary(result)
        self.assertIn('draft status is not a build gate', summary)
        self.assertIn('Overlay preflight failed', summary)

    def test_rerun_preserves_manual_repair_commit(self):
        first = self.propose()
        self.start_repair(first)
        (self.root / 'manual-repair').write_text('human changes\n')
        self.git('add', '.', cwd=self.root)
        self.git('commit', '-qm', 'Human repair', cwd=self.root)
        repair = self.git('rev-parse', 'HEAD', cwd=self.root)
        self.git('push', str(self.remote), 'HEAD:refs/heads/' + first['branch'], cwd=self.root)
        self.restore_base()
        call_count = len(self.calls())
        second = self.propose()
        self.assertEqual(second['sourceCommit'], repair)
        self.assertEqual(second['status'], 'branch-exists')
        self.assertNotIn('push', [call[0] for call in self.calls()[call_count:]])
        self.assertIn('does not re-run its CI', update.render_summary(second))
        self.assertEqual(self.git('rev-parse', 'HEAD', cwd=self.root), self.base)

    def test_existing_branch_with_different_pins_is_not_overwritten(self):
        first = self.propose()
        self.start_repair(first)
        config = self.root / 'config/toolchain.json'
        config.write_text(json.dumps({'llamaCommit': A}))
        self.git('add', '.', cwd=self.root)
        self.git('commit', '-qm', 'Different human target', cwd=self.root)
        self.git('push', str(self.remote), 'HEAD:refs/heads/' + first['branch'], cwd=self.root)
        self.restore_base()
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            self.propose()

    def test_reused_branch_validates_the_fetched_tip_when_it_moves_during_fetch(self):
        first = self.propose()
        self.start_repair(first)
        config = self.root / 'config/toolchain.json'
        config.write_text(json.dumps({'llamaCommit': A}))
        self.git('add', '.', cwd=self.root)
        self.git('commit', '-qm', 'Human switches target during fetch', cwd=self.root)
        changed = self.git('rev-parse', 'HEAD', cwd=self.root)
        self.restore_base()
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite|disagree'):
            self.propose(advance=changed)
        self.assertEqual(self.git('rev-parse', 'HEAD', cwd=self.root), self.base)
        remote_head = self.git('ls-remote', str(self.remote), 'refs/heads/' + first['branch'], cwd=self.home)
        self.assertEqual(remote_head.split()[0], changed)


    def test_reused_branch_reports_a_human_repair_fetched_after_the_remote_probe(self):
        first = self.propose()
        self.start_repair(first)
        (self.root / 'manual-repair').write_text('human repair during fetch\n')
        self.git('add', '.', cwd=self.root)
        self.git('commit', '-qm', 'Human repairs during fetch', cwd=self.root)
        repair = self.git('rev-parse', 'HEAD', cwd=self.root)
        self.restore_base()
        start = len(self.calls())
        result = self.propose(advance=repair)
        self.assertNotIn('push', [call[0] for call in self.calls()[start:]])
        self.assertEqual(result['sourceCommit'], repair)
        self.assertEqual(result['status'], 'branch-exists')


    def test_existing_branch_needs_no_pr_read_or_write_permission(self):
        first = self.propose()
        self.restore_base()
        second = self.propose()
        self.assertEqual(second['status'], 'branch-exists')
        self.assertEqual(second['sourceCommit'], first['sourceCommit'])
        self.assertEqual(second['branch'], first['branch'])
        self.assertEqual(second['preflight']['status'], 'not-repeated')
        self.assertFalse(any('pulls' in ' '.join(call) or 'dispatches' in ' '.join(call) for call in self.calls()))

    def test_push_failure_retains_partial_identity_without_claiming_pr_creation(self):
        with self.assertRaisesRegex(RuntimeError, 'Branch write denied'):
            self.propose(deny_push=True)
        progress = update.load(self.state, 'report.json')
        self.assertEqual(progress['target']['commit'], self.new)
        self.assertIn('branchUrl', progress)
        self.assertIn('sourceCommit', progress)
        self.assertNotIn('pullRequest', progress)
        result = self.run_step('Render the manual pull request handoff', PUSH_OUTCOME='failure')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(update.load(self.state, 'report.json')['status'], 'failed')
        self.assertNotIn('[Open pull request form]', (self.home / 'summary').read_text())



    def test_unrelated_staged_change_outside_runtime_cannot_enter_candidate(self):
        def stage_other(_):
            path = self.repository / 'unrelated.txt'; path.write_text('not an updater input')
            self.git('add', str(path), cwd=self.repository)
        with self.assertRaisesRegex(ValueError, 'only the submodule'):
            self.propose(before_candidate=stage_other)
        self.assertEqual(self.git('ls-remote', '--heads', str(self.remote), cwd=self.home), '')

    def test_parent_head_change_during_preflight_cannot_retarget_candidate(self):
        def move(*args, **kwargs):
            self.git('commit', '--allow-empty', '-qm', 'Unexpected source move', cwd=self.root)
        self.webgpu_prepare.side_effect = move
        with self.assertRaisesRegex(ValueError, 'HEAD changed'):
            self.propose()
        self.assertEqual(self.git('ls-remote', '--heads', str(self.remote), cwd=self.home), '')

    def test_same_path_index_race_cannot_change_the_toolchain_payload(self):
        def change(*args, **kwargs):
            path = self.root / 'config/toolchain.json'
            value = json.loads(path.read_text()); value['otherPin'] = 'unexpected'
            path.write_text(json.dumps(value))
            self.git('add', 'config/toolchain.json', cwd=self.root)
        self.webgpu_prepare.side_effect = change
        with self.assertRaisesRegex(ValueError, 'Candidate pins changed'):
            self.propose()
        self.assertEqual(self.git('ls-remote', '--heads', str(self.remote), cwd=self.home), '')

    def test_candidate_parent_is_the_original_plan_identity(self):
        result = self.propose()
        parent = self.git('rev-parse', result['sourceCommit'] + '^', cwd=self.root)
        self.assertEqual(parent, self.base)
        self.assertEqual(self.git('rev-parse', 'HEAD', cwd=self.root), self.base)

class BrowserHandoff(unittest.TestCase):
    def result(self):
        return {'base': 'feature/base', 'branch': 'automation/llama-cpp/candidate',
                'branchUrl': 'https://github.com/example/core/tree/automation/llama-cpp/candidate',
                'previousCommit': A, 'sourceCommit': C, 'upstreamRelation': 'ahead',
                'target': {'requested': 'custom', 'ref': 'refs/heads/topic/new', 'commit': B},
                'preflight': {'status': 'passed', 'scope': 'patch application only'},
                'status': 'branch-created'}

    def test_form_link_is_encoded_and_not_an_api_operation(self):
        result = self.result()
        links = update.pull_request_links('example/core', result)
        url = urlparse(links['createPullRequestUrl'])
        self.assertEqual(url.netloc, 'github.com')
        self.assertEqual(unquote(url.path), '/example/core/compare/feature/base...automation/llama-cpp/candidate')
        query = parse_qs(url.query)
        self.assertEqual(query['quick_pull'], ['1'])
        self.assertEqual(query['title'], ['chore: update llama.cpp to refs/heads/topic/new'])
        self.assertIn(A, query['body'][0])
        self.assertIn(B, query['body'][0])
        self.assertNotIn('draft', query)

    def test_long_title_does_not_create_an_unusable_prefilled_url(self):
        result = self.result()
        result['target']['ref'] = 'refs/tags/' + 'x' * 8000
        links = update.pull_request_links('example/core', result)
        self.assertLess(len(links['createPullRequestUrl']), 7000)
        self.assertGreater(len(links['pullRequestBody']), 8000)
        self.assertEqual(parse_qs(urlparse(links['createPullRequestUrl']).query), {'quick_pull': ['1']})

    def test_diagnostics_stay_in_report_not_in_query_parameters(self):
        result = self.result()
        result['preflight'] = {'status': 'failed', 'scope': 'patch application only', 'error': 'diagnostic ' * 1000}
        links = update.pull_request_links('example/core', result)
        self.assertNotIn('diagnostic+diagnostic', links['createPullRequestUrl'])
        summary = update.render_summary({**result, **links})
        self.assertIn('diagnostic diagnostic', summary)
        self.assertIn('[Open pull request form]', summary)
        self.assertIn('The links do not create a PR', summary)

    def test_no_op_summary_has_no_broken_pr_links(self):
        summary = update.render_summary({'status': 'unchanged'})
        self.assertNotIn('Open pull request form', summary)
        self.assertIn('unchanged', summary)


class FailureReporting(UpdaterFixture):
    def test_invalid_request_still_produces_a_browser_visible_report(self):
        args = ['updater', 'request', '--root', str(self.root), '--state', str(self.state),
                '--base', 'develop', '--target', 'custom', '--custom-ref', 'a.lock']
        with patch.object(sys, 'argv', args), patch.dict(os.environ, {'GITHUB_REPOSITORY': 'example/core'}):
            with self.assertRaises(ValueError): update.main()
        report = update.load(self.state, 'report.json')
        self.assertEqual(report['status'], 'failed')
        self.assertIn('Invalid named Git ref', report['error'])
        result = self.run_step('Render the manual pull request handoff', PUSH_OUTCOME='skipped')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Invalid named Git ref', (self.home / 'summary').read_text())

    def test_custom_ref_for_a_release_channel_fails_before_acquisition(self):
        with self.assertRaisesRegex(ValueError, 'only valid'):
            update.request(self.root, 'example/core', 'develop', 'latest', 'master')

    def test_artifact_or_candidate_base_is_rejected(self):
        for base in ('artifacts', 'artifacts/test', 'automation/llama-cpp/prior'):
            with self.subTest(base=base), self.assertRaisesRegex(ValueError, 'source base'):
                update.request(self.root, 'example/core', base, 'latest')

    def test_changed_upstream_selection_cannot_replace_captured_request(self):
        requested = update.request(self.root, 'example/core', 'develop', 'latest')
        with self.assertRaisesRegex(ValueError, 'differs from selection'):
            update.plan(self.root, requested, {'ref': 'refs/tags/v1.0.0'},
                        {'ref': 'refs/heads/master', 'commit': self.new}, {'status': 'ahead'})


if __name__ == '__main__': unittest.main()
