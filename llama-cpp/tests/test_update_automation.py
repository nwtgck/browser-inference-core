"""Offline release selection and real local-Git branch handoff tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from github_api import ApiError, full_sha, repository_name
import update_llama_cpp as update

A, B, C = 'a' * 40, 'b' * 40, 'c' * 40


class ReleaseApi:
    def __init__(self, releases=None, latest=None, refs=None, tags=None):
        self.releases = releases or []
        self.latest = latest
        self.refs = refs or {}
        self.tags = tags or {}
        self.calls = []

    def iter_releases(self, repository):
        self.calls.append(('iter_releases', repository))
        yield from self.releases

    def get_latest_release(self, repository):
        self.calls.append(('get_latest_release', repository))
        if self.latest is None:
            raise ApiError(404, 'No stable release')
        return self.latest

    def get_ref(self, repository, ref):
        self.calls.append(('get_ref', repository, ref))
        if ref not in self.refs:
            raise ApiError(404, 'Missing ref')
        return {'object': self.refs[ref]}

    def get_tag(self, repository, tag_sha):
        self.calls.append(('get_tag', repository, tag_sha))
        return {'object': self.tags[tag_sha]}

    def get_commit(self, repository, commit):
        self.calls.append(('get_commit', repository, commit))
        return {'sha': commit}


def release(name, prerelease=False, draft=False):
    return {'tag_name': name, 'draft': draft, 'prerelease': prerelease,
            'published_at': '2026-09-22T00:00:00Z', 'html_url': 'https://github.com/ggml-org/llama.cpp/releases/tag/' + name}


class ReleaseSelection(unittest.TestCase):
    def test_stable_uses_release_status_and_exact_tag(self):
        api = ReleaseApi(latest=release('v1.2.3'), refs={'tags/v1.2.3': {'type': 'commit', 'sha': A}})
        result = update.resolve_target(api, 'latest')
        self.assertEqual(result['commit'], A)
        self.assertEqual(result['ref'], 'refs/tags/v1.2.3')
        self.assertFalse(result['release']['prerelease'])

    def test_stable_never_falls_back_to_nightly_or_master(self):
        for value in [release('b999'), release('v1.2.3', True), release('v1.2.3', draft=True),
                      release('v1.2.3-rc1'), release('v01.2.3'), release('v1.2.3extra')]:
            with self.subTest(value=value):
                api = ReleaseApi(latest=value)
                with self.assertRaises(ValueError):
                    update.resolve_target(api, 'latest')
                self.assertEqual(api.calls, [('get_latest_release', update.UPSTREAM)])
        with self.assertRaises(ApiError):
            update.resolve_target(ReleaseApi(), 'latest')

    def test_unstable_uses_published_nightly_not_draft_or_other_tags(self):
        values = [release('b100', True, True), release('v2.0.0-rc1', True), release('v1.2.3'),
                  release('b99', True), release('b98', True)]
        api = ReleaseApi(releases=values, refs={'tags/b99': {'type': 'commit', 'sha': B}})
        result = update.resolve_target(api, 'latest-unstable')
        self.assertEqual(result['ref'], 'refs/tags/b99')
        self.assertNotIn(('get_ref', update.UPSTREAM, 'heads/master'), api.calls)

    def test_no_nightly_is_an_error(self):
        with self.assertRaises(ValueError):
            update.resolve_target(ReleaseApi(releases=[release('v1.0.0')]), 'latest-unstable')

    def test_annotated_tags_are_peeled(self):
        api = ReleaseApi(refs={'tags/v1.0.0': {'type': 'tag', 'sha': B}},
                         tags={B: {'type': 'commit', 'sha': A}})
        self.assertEqual(update.resolve_named_ref(api, 'refs/tags/v1.0.0'), A)

    def test_tag_loop_or_blob_is_rejected(self):
        for obj in ({'type': 'blob', 'sha': B}, {'type': 'tag', 'sha': B}):
            api = ReleaseApi(refs={'tags/loop': obj}, tags={B: obj})
            with self.assertRaises(ValueError):
                update.resolve_named_ref(api, 'refs/tags/loop')

    def test_custom_full_commit_and_explicit_master(self):
        self.assertEqual(update.resolve_target(ReleaseApi(), 'custom', A)['commit'], A)
        api = ReleaseApi(refs={'heads/master': {'type': 'commit', 'sha': B}})
        self.assertEqual(update.resolve_target(api, 'custom', 'master')['commit'], B)

    def test_ambiguous_ref_requires_qualification(self):
        api = ReleaseApi(refs={key: {'type': 'commit', 'sha': A} for key in ['heads/shared', 'tags/shared']})
        with self.assertRaises(ValueError):
            update.resolve_named_ref(api, 'shared')
        self.assertEqual(update.resolve_named_ref(api, 'refs/heads/shared'), A)

    def test_input_injection_and_revision_expressions_are_rejected(self):
        for value in ['', '-x', 'HEAD~1', 'x^{}', 'x\ny', 'https://evil/x', 'x;id', 'x..y', 'x@{1}', 'a.lock']:
            with self.subTest(value=value), self.assertRaises((ValueError, subprocess.CalledProcessError)):
                update.checked_ref(value)
        with self.assertRaises(ValueError):
            update.resolve_target(ReleaseApi(), 'latest', 'master')

    def test_api_errors_are_not_treated_as_missing_tags(self):
        api = ReleaseApi()
        api.get_ref = lambda *args: (_ for _ in ()).throw(ApiError(403, 'Rate limited'))
        with self.assertRaises(ApiError):
            update.resolve_named_ref(api, 'master')

    def test_identifiers_and_branch_deduplication(self):
        with self.assertRaises(ValueError):
            full_sha('abc')
        with self.assertRaises(ValueError):
            repository_name('owner/repo?token=secret')
        self.assertEqual(update.candidate_branch('develop', A, B), update.candidate_branch('develop', A, B))
        self.assertNotEqual(update.candidate_branch('develop', A, B), update.candidate_branch('main', A, B))
        self.assertNotEqual(update.candidate_branch('develop', A, B), update.candidate_branch('develop', B, B))


class ProposalApi:
    # Intentionally no PR or workflow operations: the branch updater only needs
    # upstream reads. Any accidental orchestration call fails these tests.
    def __init__(self):
        self.relation = 'ahead'
        self.calls = []

    def compare_commits(self, repository, base, head):
        assert repository == update.UPSTREAM
        self.calls.append(('compare_commits', repository, base, head))
        return {'status': self.relation}


class LocalGitProposal(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.root, self.upstream, self.remote = [self.home / name for name in ('source', 'upstream', 'remote.git')]
        for directory in (self.root, self.upstream):
            directory.mkdir()
            self.git('init', '-q', cwd=directory)
            self.git('config', 'user.name', 'Fixture', cwd=directory)
            self.git('config', 'user.email', 'fixture@example.invalid', cwd=directory)
        clip = self.upstream / 'tools/mtmd/clip.cpp'
        clip.parent.mkdir(parents=True)
        clip.write_text('before\noriginal\nafter\n')
        (clip.parent / 'mtmd-audio.cpp').write_text('before\noriginal\nafter\n')
        self.git('add', '.', cwd=self.upstream)
        self.git('commit', '-qm', 'Upstream A', cwd=self.upstream)
        self.old = self.git('rev-parse', 'HEAD', cwd=self.upstream)
        (self.upstream / 'new-feature').write_text('B\n')
        self.git('add', '.', cwd=self.upstream)
        self.git('commit', '-qm', 'Upstream B', cwd=self.upstream)
        self.new = self.git('rev-parse', 'HEAD', cwd=self.upstream)
        self.git('-c', 'protocol.file.allow=always', 'submodule', 'add', str(self.upstream), 'vendor/llama.cpp', cwd=self.root)
        self.git('checkout', '--detach', self.old, cwd=self.root / 'vendor/llama.cpp')
        (self.root / 'config').mkdir()
        (self.root / 'config/toolchain.json').write_text(json.dumps({'llamaCommit': self.old, 'otherPin': 'unchanged'}, indent=2) + '\n')
        (self.root / 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval').mkdir()
        (self.root / 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-webgpu-bf16.patch').write_text('--- a/clip.cpp\n+++ b/clip.cpp\n@@ -1,3 +1,3 @@\n before\n-original\n+patched\n after\n')
        (self.root / 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-audio-single-thread.patch').write_text((self.root / 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval/mtmd-webgpu-bf16.patch').read_text().replace('clip.cpp', 'mtmd-audio.cpp'))
        self.git('add', '.', cwd=self.root)
        self.git('commit', '-qm', 'Core base', cwd=self.root)
        self.base = self.git('rev-parse', 'HEAD', cwd=self.root)
        self.git('init', '--bare', '-q', str(self.remote), cwd=self.home)
        self.api = ProposalApi()
        self.target = {'commit': self.new, 'ref': 'refs/tags/v1.0.0', 'requested': 'latest'}
        original = update.git
        def local_git(*args, **kwargs):
            replacements = {'https://github.com/example/core.git': str(self.remote),
                            'https://github.com/ggml-org/llama.cpp.git': str(self.upstream)}
            return original(*(replacements.get(arg, arg) for arg in args), **kwargs)
        self.addCleanup(patch.stopall)
        patch.object(update, 'git', side_effect=local_git).start()
        patch.dict(os.environ, {'GH_TOKEN': 'fixture-token', 'GIT_ALLOW_PROTOCOL': 'file'}).start()

    @staticmethod
    def git(*args, cwd):
        return subprocess.check_output(['git', *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()

    def propose(self, **kwargs):
        return update.propose(self.root, self.api, 'example/core', 'develop', self.target, **kwargs)

    def restore_base(self):
        self.git('checkout', '--detach', self.base, cwd=self.root)
        self.git('submodule', 'update', '--init', cwd=self.root)

    def test_preflight_also_fails_closed_on_audio_patch_conflicts(self):
        audio = self.root / 'vendor/llama.cpp/tools/mtmd/mtmd-audio.cpp'
        audio.write_text('upstream audio changed\n')
        result = update.overlay_preflight(self.root)
        self.assertEqual(result['status'], 'failed')
        self.assertIn('mtmd-audio.cpp', result['error'])
        self.assertEqual(audio.read_text(), 'upstream audio changed\n')

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
        self.assertEqual(set(self.git('diff', '--name-only', self.base, 'HEAD', cwd=self.root).splitlines()), update.PINS)
        self.assertEqual(json.loads((self.root / 'config/toolchain.json').read_text())['otherPin'], 'unchanged')
        self.assertEqual((self.root / 'vendor/llama.cpp/tools/mtmd/clip.cpp').read_text(), 'before\noriginal\nafter\n')
        remote = self.git('ls-remote', str(self.remote), 'refs/heads/' + result['branch'], cwd=self.home)
        self.assertEqual(remote.split()[0], result['sourceCommit'])
        self.assertEqual(self.api.calls, [('compare_commits', update.UPSTREAM, self.old, self.new)])
        query = parse_qs(urlparse(result['createPullRequestUrl']).query)
        self.assertEqual(query['quick_pull'], ['1'])
        self.assertEqual(query['title'], [result['pullRequestTitle']])
        self.assertEqual(query['body'], [result['pullRequestBody']])
        self.assertIn('develop...' + result['branch'], unquote(urlparse(result['compareUrl']).path))
        self.assertNotIn('pullRequest', result)
        self.assertNotIn('pullRequestUrl', result)

    def test_nightly_changes_only_its_gitlink_and_pin_file(self):
        import hashlib
        self.git('-c','protocol.file.allow=always','submodule','add',str(self.upstream),'vendor/llama.cpp-nightly',cwd=self.root)
        self.git('checkout','--detach',self.old,cwd=self.root/'vendor/llama.cpp-nightly')
        pin=self.root/'sources/upstream-nightly/pin.json';pin.parent.mkdir(parents=True)
        pin.write_text(json.dumps({'llamaCommit':self.old,'kept':True}))
        items=[]
        for component,file,unit in [('vision','mtmd-webgpu-bf16.patch','clip.cpp'),('audio','mtmd-audio-single-thread.patch','mtmd-audio.cpp')]:
            relative='upstream-patches-only-as-a-last-resort-with-explicit-user-approval/'+file
            items.append({'component':component,'file':relative,'translationUnit':unit,'sha256':hashlib.sha256((self.root/relative).read_bytes()).hexdigest()})
        plan=pin.parent/'patches.json';plan.write_text(json.dumps({'formatVersion':1,'patches':items}))
        self.git('add','.',cwd=self.root);self.git('commit','-qm','Separate nightly source',cwd=self.root)
        base=self.git('rev-parse','HEAD',cwd=self.root);old_toolchain=(self.root/'config/toolchain.json').read_bytes()
        descriptor={'id':'upstream-nightly','pinFile':'sources/upstream-nightly/pin.json','vendorPath':'vendor/llama.cpp-nightly','repository':update.UPSTREAM,'patchSeries':'sources/upstream-nightly/patches.json'}
        result=self.propose(descriptor=descriptor)
        self.assertEqual(result['preflight']['status'],'passed')
        self.assertEqual(set(self.git('diff','--name-only',base,'HEAD',cwd=self.root).splitlines()),{'vendor/llama.cpp-nightly','sources/upstream-nightly/pin.json'})
        self.assertEqual((self.root/'config/toolchain.json').read_bytes(),old_toolchain)
        self.assertEqual(self.git('rev-parse','HEAD',cwd=self.root/'vendor/llama.cpp'),self.old)
        self.assertEqual(json.loads(pin.read_text())['llamaCommit'],self.new)

    def test_no_op_creates_no_branch_or_remote_operations(self):
        self.target['commit'] = self.old
        self.assertEqual(self.propose()['status'], 'unchanged')
        self.assertEqual(self.api.calls, [])
        self.assertEqual(self.git('ls-remote', '--heads', str(self.remote), cwd=self.home), '')

    def test_dirty_checkout_is_rejected(self):
        (self.root / 'untracked').write_text('not an update input')
        with self.assertRaisesRegex(ValueError, 'clean'):
            self.propose()

    def test_mismatched_committed_pins_are_rejected(self):
        (self.root / 'config/toolchain.json').write_text(json.dumps({'llamaCommit': self.new}))
        self.git('add', '.', cwd=self.root)
        self.git('commit', '-qm', 'Mismatched pins', cwd=self.root)
        with self.assertRaisesRegex(ValueError, 'Source gitlink/pin mismatch') as caught:
            self.propose()
        message = str(caught.exception)
        self.assertIn(self.new, message)
        self.assertIn(self.old, message)
        self.assertIn('config/toolchain.json', message)
        self.assertIn('actual 160000 commit', message)

    def test_rollback_requires_explicit_opt_in(self):
        self.api.relation = 'behind'
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
        (self.root / 'manual-repair').write_text('human changes\n')
        self.git('add', '.', cwd=self.root)
        self.git('commit', '-qm', 'Human repair', cwd=self.root)
        repair = self.git('rev-parse', 'HEAD', cwd=self.root)
        self.git('push', str(self.remote), 'HEAD:refs/heads/' + first['branch'], cwd=self.root)
        self.restore_base()
        call_count = update.git.call_count
        second = self.propose()
        self.assertEqual(second['sourceCommit'], repair)
        self.assertEqual(second['status'], 'branch-exists')
        self.assertNotIn('push', [call.args[0] for call in update.git.call_args_list[call_count:]])
        self.assertEqual(len(self.api.calls), 2)
        self.assertIn('does not re-run its CI', update.render_summary(second))
        self.assertEqual(self.git('rev-parse', 'HEAD', cwd=self.root), self.base)

    def test_existing_branch_with_different_pins_is_not_overwritten(self):
        first = self.propose()
        config = self.root / 'config/toolchain.json'
        config.write_text(json.dumps({'llamaCommit': A}))
        self.git('add', '.', cwd=self.root)
        self.git('commit', '-qm', 'Different human target', cwd=self.root)
        self.git('push', str(self.remote), 'HEAD:refs/heads/' + first['branch'], cwd=self.root)
        self.restore_base()
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            self.propose()

    def test_reused_branch_validates_the_fetched_tip_when_it_moves_during_fetch(self):
        # ls-remote and fetch are separate reads. A human can advance the branch
        # between them; the fetched commit is the snapshot available for review.
        first = self.propose()
        config = self.root / 'config/toolchain.json'
        config.write_text(json.dumps({'llamaCommit': A}))
        self.git('add', '.', cwd=self.root)
        self.git('commit', '-qm', 'Human switches target during fetch', cwd=self.root)
        changed = self.git('rev-parse', 'HEAD', cwd=self.root)
        self.restore_base()
        local_git = update.git.side_effect

        def advance_before_fetch(*args, **kwargs):
            if args[0] == 'fetch' and 'https://github.com/example/core.git' in args:
                self.git('push', str(self.remote), changed + ':refs/heads/' + first['branch'], cwd=self.root)
            return local_git(*args, **kwargs)

        with patch.object(update, 'git', side_effect=advance_before_fetch):
            with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
                self.propose()
        self.assertEqual(self.git('rev-parse', 'HEAD', cwd=self.root), self.base)
        remote_head = self.git('ls-remote', str(self.remote), 'refs/heads/' + first['branch'], cwd=self.home)
        self.assertEqual(remote_head.split()[0], changed)

    def test_reused_branch_reports_a_human_repair_fetched_after_the_remote_probe(self):
        first = self.propose()
        (self.root / 'manual-repair').write_text('human repair during fetch\n')
        self.git('add', '.', cwd=self.root)
        self.git('commit', '-qm', 'Human repairs during fetch', cwd=self.root)
        repair = self.git('rev-parse', 'HEAD', cwd=self.root)
        self.restore_base()
        local_git = update.git.side_effect

        def advance_before_fetch(*args, **kwargs):
            if args[0] == 'fetch' and 'https://github.com/example/core.git' in args:
                self.git('push', str(self.remote), repair + ':refs/heads/' + first['branch'], cwd=self.root)
            return local_git(*args, **kwargs)

        with patch.object(update, 'git', side_effect=advance_before_fetch) as calls:
            result = self.propose()
            self.assertNotIn('push', [call.args[0] for call in calls.call_args_list])
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
        self.assertTrue(all(call[0] == 'compare_commits' for call in self.api.calls))

    def test_push_failure_retains_partial_identity_without_claiming_pr_creation(self):
        local_git = update.git.side_effect
        def fail_push(*args, **kwargs):
            if args[0] == 'push':
                raise RuntimeError('Branch write denied')
            return local_git(*args, **kwargs)
        progress = {}
        with patch.object(update, 'git', side_effect=fail_push):
            with self.assertRaisesRegex(RuntimeError, 'Branch write denied'):
                self.propose(progress=progress)
        self.assertEqual(progress['target']['commit'], self.new)
        self.assertIn('branchUrl', progress)
        self.assertIn('sourceCommit', progress)
        self.assertNotIn('createPullRequestUrl', progress)
        self.assertNotIn('pullRequest', progress)


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


class FailureReporting(unittest.TestCase):
    def test_resolution_failure_still_produces_a_browser_visible_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / 'summary.md'
            with patch.object(update, 'ROOT', root), \
                 patch.object(update, 'resolve_target', side_effect=ApiError(404, 'No stable release')), \
                 patch.object(update, 'get_source', return_value={'updateTarget':'latest','repository':update.UPSTREAM}), \
                 patch.object(sys, 'argv', ['update_llama_cpp.py', '--base', 'develop']), \
                 patch.dict(os.environ, {'GITHUB_STEP_SUMMARY': str(summary)}), patch('builtins.print'):
                with self.assertRaises(SystemExit) as failure:
                    update.main()
            self.assertEqual(failure.exception.code, 1)
            report = json.loads((root / 'build/upstream-update/report.json').read_text())
            self.assertEqual(report['status'], 'failed')
            self.assertIn('No stable release', summary.read_text())


if __name__ == '__main__':
    unittest.main()
