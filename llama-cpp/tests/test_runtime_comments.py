"""Exercise stale-event, rerun, late-PR and untrusted-report handling offline."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import runtime_comments as comments
from github_api import GitHub

A, B, C = 'a' * 40, 'b' * 40, 'c' * 40
REPO = 'example/core'


def pr(head=A, number=7, repository=REPO):
    return {'number': number, 'state': 'open', 'head': {'sha': head, 'ref': 'feature/update', 'repo': {'full_name': repository}}}


def run(head=A, id=123, attempt=1, status='completed', conclusion='success', repository=REPO, event='push'):
    return {'id': id, 'run_attempt': attempt, 'head_sha': head, 'head_branch': 'feature/update',
            'head_repository': {'full_name': repository}, 'event': event, 'status': status, 'conclusion': conclusion,
            'html_url': f'https://github.com/{REPO}/actions/runs/{id}'}


def payload(head=A):
    return f'## Runtime artifact published\n\nSource: {head}\n\nnpm install github:{REPO}#{C}\n\n<details>\n<summary>Metadata</summary>\n\n```yaml\nfixture: true\n```\n\n</details>\n'


_DEFAULT_ENVELOPE = object()


def archive_for(workflow_run, changes=None, extras=None, body=None, envelope=_DEFAULT_ENVELOPE,
                yaml_body='fixture: true\n'):
    body = body if body is not None else payload(workflow_run['head_sha'])
    encoded = body.encode('utf-8')
    fields = {'schemaVersion': 1, 'repository': REPO, 'sourceCommit': workflow_run['head_sha'],
              'artifactCommit': C, 'runId': workflow_run['id'], 'runAttempt': workflow_run['run_attempt'],
              'markdownSha256': hashlib.sha256(encoded).hexdigest()}
    fields.update(changes or {})
    if envelope is _DEFAULT_ENVELOPE:
        envelope = fields
    output = io.BytesIO()
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('report.json', json.dumps(envelope))
        archive.writestr('consumer-update.md', encoded)
        archive.writestr('consumer-update.yaml', yaml_body)
        for name, data in (extras or {}).items():
            archive.writestr(name, data)
    return output.getvalue()


def bot_comment(body, id=42):
    return {'id': id, 'user': {'login': 'github-actions[bot]', 'type': 'Bot'}, 'body': body}


class FakeApi:
    def __init__(self, pull_requests=None, runs=None, existing=None):
        self.pull_requests = pull_requests or [pr()]
        self.runs = runs or []
        self.comments = existing or []
        self.writes = []
        self.calls = []
        self.fresh_head = None
        self.fresh_run = None
        self.run_reads = 0
        self.expired = False
        self.missing_report = False

    def iter_pull_requests(self, repository, *, state):
        self.calls.append(('iter_pull_requests', repository, state))
        return iter(copy.deepcopy(self.pull_requests))

    def iter_pull_request_comments(self, repository, number):
        self.calls.append(('iter_pull_request_comments', repository, number))
        return iter(copy.deepcopy(self.comments))

    def iter_build_runs(self, repository, *, branch, head_sha):
        self.calls.append(('iter_build_runs', repository, branch, head_sha))
        self.run_reads += 1
        if self.run_reads > 1 and self.fresh_run:
            return iter(copy.deepcopy(self.fresh_run))
        return iter(copy.deepcopy(self.runs))

    def iter_run_artifacts(self, repository, run_id, *, name):
        self.calls.append(('iter_run_artifacts', repository, run_id, name))
        return iter([] if self.missing_report else [
            {'id': 91, 'name': name, 'expired': self.expired},
            {'id': 90, 'name': 'consumer-update-0', 'expired': False},
        ])

    def download_artifact_archive(self, repository, artifact_id, *, max_bytes):
        self.calls.append(('download_artifact_archive', repository, artifact_id, max_bytes))
        assert artifact_id == 91 and max_bytes == comments.MAX_ARCHIVE
        return self.report

    def get_pull_request(self, repository, number):
        self.calls.append(('get_pull_request', repository, number))
        value = copy.deepcopy(next(p for p in self.pull_requests if p['number'] == number))
        if self.fresh_head:
            value['head']['sha'] = self.fresh_head
        return value

    def update_pull_request_comment(self, repository, comment_id, body):
        self.calls.append(('update_pull_request_comment', repository, comment_id))
        self.writes.append(('update_pull_request_comment', comment_id, {'body': body}))
        return {'id': comment_id}

    def create_pull_request_comment(self, repository, number, body):
        self.calls.append(('create_pull_request_comment', repository, number))
        self.writes.append(('create_pull_request_comment', number, {'body': body}))
        return {'id': 42}


class ReportDecoder(unittest.TestCase):
    def test_valid_bounded_archive(self):
        value = run()
        self.assertEqual(comments.decode_report(archive_for(value), REPO, value), payload())

    def test_large_download_only_yaml_is_not_read_by_the_notifier(self):
        value = run()
        content = archive_for(value, yaml_body='fixture: ' + 'x' * 1000000 + '\n')
        self.assertLess(len(content), comments.MAX_ARCHIVE)
        original_read = zipfile.ZipFile.read
        def read(archive, name, *args, **kwargs):
            self.assertNotEqual(name, 'consumer-update.yaml')
            return original_read(archive, name, *args, **kwargs)
        with patch.object(zipfile.ZipFile, 'read', read):
            self.assertEqual(comments.decode_report(content, REPO, value), payload())

    def test_identity_mismatch_is_rejected(self):
        for change in [{'repository': 'fork/core'}, {'sourceCommit': B}, {'runId': 124},
                       {'runAttempt': 2}, {'schemaVersion': 2}, {'artifactCommit': 'moving-branch'},
                       {'markdownSha256': '0' * 64}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                comments.decode_report(archive_for(run(), change), REPO, run())

    def test_non_object_envelope_is_a_validation_error(self):
        for invalid in (None, [], 'invalid', 1):
            with self.subTest(envelope=invalid), self.assertRaisesRegex(ValueError, 'object'):
                comments.decode_report(archive_for(run(), envelope=invalid), REPO, run())

    def test_envelope_fields_require_their_declared_types(self):
        # Python otherwise treats True == 1 and 123.0 == 123 as equal.
        changes = [{'schemaVersion': True}, {'schemaVersion': 1.0},
                   {'runId': 123.0}, {'runId': '123'}, {'runAttempt': True},
                   {'runAttempt': 1.0}, {'artifactCommit': None},
                   {'sourceCommit': []}, {'markdownSha256': 123}]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(ValueError):
                comments.decode_report(archive_for(run(), change), REPO, run())

    def test_path_traversal_and_extra_files_are_rejected_without_extraction(self):
        for name in ['../evil.py', '/tmp/evil', 'workflow.sh']:
            with self.assertRaises(ValueError):
                comments.decode_report(archive_for(run(), extras={name: 'not executed'}), REPO, run())

    def test_duplicate_files_are_rejected(self):
        content = io.BytesIO(archive_for(run()))
        with zipfile.ZipFile(content, 'a') as archive:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                archive.writestr('consumer-update.md', 'duplicate')
        with self.assertRaises(ValueError):
            comments.decode_report(content.getvalue(), REPO, run())

    def test_zip_bomb_and_oversized_body_are_rejected(self):
        with self.assertRaises(ValueError):
            comments.decode_report(archive_for(run(), body='x' * 1000000), REPO, run())
        with self.assertRaises(ValueError):
            comments.decode_report(archive_for(run(), changes={'padding': 'x' * 1000000}), REPO, run())
        with self.assertRaises(ValueError):
            comments.decode_report(b'0' * (comments.MAX_ARCHIVE + 1), REPO, run())

    def test_symlink_and_delimiter_injection_are_rejected(self):
        content = io.BytesIO()
        with zipfile.ZipFile(content, 'w') as archive:
            entry = zipfile.ZipInfo('consumer-update.md')
            entry.create_system = 3
            entry.external_attr = 0o120777 << 16
            archive.writestr(entry, 'report.json')
            archive.writestr('report.json', '{}')
            archive.writestr('consumer-update.yaml', 'test: true')
        with self.assertRaises(ValueError):
            comments.decode_report(content.getvalue(), REPO, run())
        with self.assertRaises(ValueError):
            comments.decode_report(archive_for(run(), body=payload() + comments.START), REPO, run())


class CommentReconciliation(unittest.TestCase):
    def reconcile(self, api, workflow=None):
        workflow = workflow or (api.runs[0] if api.runs else run())
        api.report = archive_for(workflow)
        return comments.reconcile(api, REPO)

    def test_pr_opened_after_build_gets_its_existing_result(self):
        api = FakeApi(runs=[run()])
        self.assertEqual(self.reconcile(api), 1)
        body = api.writes[0][2]['body']
        self.assertIn('succeeded for this PR head', body)
        self.assertIn('npm install', body)
        self.assertIn('<details>', body)

    def test_existing_owned_comment_is_edited_not_appended(self):
        pending = comments.comment_body(pr(), run(status='in_progress'), None)
        api = FakeApi(runs=[run()], existing=[bot_comment(pending)])
        self.reconcile(api)
        self.assertEqual(api.writes[0][0], 'update_pull_request_comment')
        self.assertEqual(api.writes[0][1], 42)

    def test_identical_report_causes_no_write(self):
        body = comments.comment_body(pr(), run(), payload())
        api = FakeApi(runs=[run()], existing=[bot_comment(body)])
        self.assertEqual(self.reconcile(api), 0)

    def test_user_marker_spoof_does_not_allow_editing_their_comment(self):
        other = bot_comment(comments.MARKER + '\nUser text')
        other['user'] = {'login': 'someone', 'type': 'User'}
        api = FakeApi(runs=[run()], existing=[other])
        self.reconcile(api)
        self.assertEqual(api.writes[0][0], 'create_pull_request_comment')

    def test_new_head_marks_previous_success_as_previous_not_current(self):
        old = comments.comment_body(pr(A), run(A), payload(A))
        api = FakeApi(pull_requests=[pr(B)], runs=[run(B, id=124, status='in_progress', conclusion=None)], existing=[bot_comment(old)])
        self.reconcile(api)
        body = api.writes[0][2]['body']
        self.assertIn('Current PR head:** `' + B, body)
        self.assertIn('not confirmation of the current head/run', body)
        self.assertIn(A, body)
        self.assertEqual(body.count(comments.START), 1)
        self.assertNotIn('succeeded for this PR head', body)

    def test_out_of_order_old_completion_does_not_replace_new_running_head(self):
        api = FakeApi(pull_requests=[pr(B)], runs=[run(A), run(B, id=124, status='queued', conclusion=None)])
        self.reconcile(api)
        body = api.writes[0][2]['body']
        self.assertIn('**queued**', body)
        self.assertNotIn('npm install', body)

    def test_rerun_attempt_has_its_own_report_identity(self):
        api = FakeApi(runs=[run(attempt=2)])
        self.reconcile(api)
        self.assertIn(('iter_run_artifacts', REPO, 123, 'consumer-update-2'), api.calls)
        self.assertIn('(attempt 2)', api.writes[0][2]['body'])

    def test_older_attempt_archive_is_not_accepted_for_new_rerun(self):
        api = FakeApi(runs=[run(attempt=2)])
        self.reconcile(api, workflow=run(attempt=1))
        self.assertIn('unavailable', api.writes[0][2]['body'])
        self.assertNotIn('npm install', api.writes[0][2]['body'])

    def test_latest_run_search_includes_later_pages(self):
        def response(method, path):
            page = int(parse_qs(urlparse(path).query)['page'][0])
            entries = [{**run(id=200 + i), 'run_started_at': '2026-09-22T10:00:00Z'} for i in range(100)]
            if page == 2:
                entries = [{**run(id=100, attempt=3), 'run_started_at': '2026-09-22T12:00:00Z'}]
            return {'workflow_runs': entries, 'total_count': 101}
        api = GitHub(token='fixture-token')
        with patch.object(api, '_request', side_effect=response) as request:
            self.assertEqual(comments.latest_run(api, REPO, pr())['id'], 100)
        self.assertEqual(request.call_count, 2)

    def test_incomplete_filtered_build_history_is_rejected(self):
        api = GitHub(token='fixture-token')
        with patch.object(api, '_request', return_value={'workflow_runs': [], 'total_count': 1001}):
            with self.assertRaisesRegex(ValueError, 'Too many matching'):
                comments.latest_run(api, REPO, pr())

    def test_rerun_of_an_older_run_id_can_be_the_newest_attempt(self):
        old_rerun = {**run(id=120, attempt=2), 'run_started_at': '2026-09-22T12:00:00Z'}
        later_id = {**run(id=123), 'run_started_at': '2026-09-22T10:00:00Z'}
        api = FakeApi(runs=[later_id, old_rerun])
        self.reconcile(api, workflow=old_rerun)
        self.assertIn('(attempt 2)', api.writes[0][2]['body'])
        self.assertIn('/actions/runs/120', api.writes[0][2]['body'])

    def test_fork_and_non_source_event_runs_are_ignored(self):
        for wrong in [run(repository='attacker/core'), {**run(), 'event': 'pull_request_target'}]:
            api = FakeApi(runs=[wrong])
            self.reconcile(api)
            self.assertIn('No matching runtime build', api.writes[0][2]['body'])

    def test_human_opened_pr_build_produces_the_runtime_comment(self):
        api = FakeApi(runs=[run(event='pull_request')])
        self.assertEqual(self.reconcile(api), 1)
        body = api.writes[0][2]['body']
        self.assertIn('succeeded for this PR head', body)
        self.assertIn('npm install', body)
        self.assertIn('```yaml', body)

    def test_push_and_synchronize_runs_are_both_eligible_without_deduplication(self):
        for event in ('push', 'pull_request'):
            with self.subTest(latest=event):
                earlier = run(id=120, event='pull_request' if event == 'push' else 'push')
                later = run(id=123, event=event)
                api = FakeApi(runs=[earlier, later])
                self.reconcile(api, workflow=later)
                self.assertIn('/actions/runs/123', api.writes[0][2]['body'])
                self.assertIn('succeeded for this PR head', api.writes[0][2]['body'])

    def test_pr_run_report_cannot_substitute_a_synthetic_merge_commit(self):
        workflow = run(event='pull_request')
        with self.assertRaisesRegex(ValueError, 'identity'):
            comments.decode_report(archive_for(workflow, {'sourceCommit': B}), REPO, workflow)

    def test_no_run_status_explains_human_pr_trigger_not_bot_dispatch(self):
        api = FakeApi(runs=[])
        self.reconcile(api)
        body = api.writes[0][2]['body']
        self.assertIn('Human PR creation normally starts it', body)
        self.assertNotIn('explicit build dispatch', body)
        self.assertNotIn('succeeded for this PR head', body)

    def test_pr_run_for_an_old_head_is_not_reported_as_current(self):
        api = FakeApi(pull_requests=[pr(B)], runs=[run(A, event='pull_request')])
        self.reconcile(api)
        self.assertIn('No matching runtime build', api.writes[0][2]['body'])
        self.assertNotIn('npm install', api.writes[0][2]['body'])

    def test_fork_pr_is_not_processed(self):
        api = FakeApi(pull_requests=[pr(repository='fork/core')], runs=[run()])
        self.assertEqual(self.reconcile(api), 0)
        self.assertEqual(len(api.calls), 1)

    def test_head_race_skips_a_stale_write(self):
        api = FakeApi(runs=[run()]); api.fresh_head = B
        self.assertEqual(self.reconcile(api), 0)

    def test_run_race_skips_a_stale_write(self):
        api = FakeApi(runs=[run()]); api.fresh_run = [run(id=124, status='queued', conclusion=None)]
        self.assertEqual(self.reconcile(api), 0)

    def test_expired_or_missing_metadata_is_explicit(self):
        for key in ['expired', 'missing_report']:
            api = FakeApi(runs=[run()]); setattr(api, key, True)
            self.reconcile(api)
            self.assertIn('unavailable or expired', api.writes[0][2]['body'])

    def test_failure_preserves_previous_payload_without_claiming_new_success(self):
        old = comments.comment_body(pr(), run(), payload())
        api = FakeApi(runs=[run(id=124, conclusion='failure')], existing=[bot_comment(old)])
        self.reconcile(api)
        body = api.writes[0][2]['body']
        self.assertIn('**failure**', body)
        self.assertIn('not confirmation of the current head/run', body)
        self.assertIn('npm install', body)

    def test_repeated_pending_states_do_not_nest_previous_sections(self):
        successful = comments.comment_body(pr(), run(), payload())
        pending = comments.comment_body(pr(B), run(B, status='queued'), None, successful)
        again = comments.comment_body(pr(B), run(B, status='in_progress'), None, pending)
        self.assertEqual(again.count('Previous successful publication'), 1)
        self.assertEqual(again.count(comments.START), 1)

    def test_malformed_report_does_not_abort_other_prs_or_erase_previous_metadata(self):
        previous = comments.comment_body(pr(), run(), payload())
        next_run = run(head=B, id=124)
        api = FakeApi(pull_requests=[pr(number=7), pr(head=B, number=8)],
                      runs=[run(), next_run], existing=[bot_comment(previous)])
        reports = [archive_for(run(), envelope=[]), archive_for(next_run)]
        with patch.object(api, 'download_artifact_archive', side_effect=reports):
            self.assertEqual(comments.reconcile(api, REPO), 2)
        unavailable = api.writes[0][2]['body']
        self.assertIn('unavailable', unavailable)
        self.assertIn('Report retrieval error:', unavailable)
        self.assertIn('not confirmation of the current head/run', unavailable)
        self.assertNotIn('succeeded for this PR head', unavailable)
        recovered = api.writes[1][2]['body']
        self.assertIn('succeeded for this PR head', recovered)
        self.assertIn(B, recovered)
        self.assertNotIn('Report retrieval error:', recovered)

    def test_reconciliation_handles_more_than_the_triggering_pr(self):
        api = FakeApi(pull_requests=[pr(number=7), pr(number=8)], runs=[run()])
        self.assertEqual(self.reconcile(api), 2)
        self.assertEqual(len(api.writes), 2)


if __name__ == '__main__':
    unittest.main()
