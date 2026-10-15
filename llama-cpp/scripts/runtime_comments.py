#!/usr/bin/env python3
"""Reconcile runtime comments using trusted default-branch code and bounded report data."""
from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import zipfile

from github_api import GitHub, full_sha, repository_name

MARKER = '<!-- lcb-runtime-artifact-comment:v1 -->'
START = '<!-- lcb-published-runtime:start -->'
END = '<!-- lcb-published-runtime:end -->'
BOT = 'github-actions[bot]'
REPORT_FILES = {'report.json', 'consumer-update.md', 'consumer-update.yaml'}
MAX_ARCHIVE = 256 * 1024


def decode_report(content: bytes, repository: str, run: dict) -> str:
    if len(content) > MAX_ARCHIVE:
        raise ValueError('Report archive exceeds the size limit')
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        entries = archive.infolist()
        if len(entries) != len(REPORT_FILES) or {item.filename for item in entries} != REPORT_FILES:
            raise ValueError('Unexpected or duplicate files in report archive')
        # YAML is download-only metadata: never read or decompress it here. Its
        # size is independent of the bounded JSON/Markdown actually consumed by
        # the notifier; the downloaded archive still has a total byte limit.
        if any(stat.S_ISLNK(item.external_attr >> 16)
               or (item.filename != 'consumer-update.yaml' and item.file_size > 60000)
               for item in entries):
            raise ValueError('Unsafe report archive entry')
        envelope = json.loads(archive.read('report.json'))
        if not isinstance(envelope, dict):
            raise ValueError('Report envelope must be an object')
        # This JSON is workflow output, not trusted Python input. Strict integer
        # types prevent True/1 and 123.0/123 from passing the identity checks.
        # Validation failures stay local to this PR via reconcile's error path.
        for key in ('schemaVersion', 'runId', 'runAttempt'):
            if type(envelope.get(key)) is not int or envelope[key] <= 0:
                raise ValueError(f'Report {key} must be a positive integer')
        source = full_sha(envelope.get('sourceCommit'))
        commit = full_sha(envelope.get('artifactCommit'))
        raw = archive.read('consumer-update.md')
        if (envelope.get('schemaVersion') != 1 or envelope.get('repository') != repository
                or envelope.get('runId') != run['id'] or envelope.get('runAttempt') != run['run_attempt']
                or source != run['head_sha']):
            raise ValueError('Report identity does not match the source workflow run')
        if hashlib.sha256(raw).hexdigest() != envelope.get('markdownSha256'):
            raise ValueError('Report Markdown digest mismatch')
        text = raw.decode('utf-8')
        if len(raw) > 55000 or START in text or END in text or MARKER in text:
            raise ValueError('Unexpected report delimiters or length')
        if f'github:{repository}#{commit}' not in text or run['head_sha'] not in text:
            raise ValueError('Report body does not identify its source and artifact')
        return text


def same_repo_pr(pr: dict, repository: str) -> bool:
    return (pr['state'] == 'open' and (pr['head'].get('repo') or {}).get('full_name') == repository
            and pr['head']['ref'] != 'artifacts' and not pr['head']['ref'].startswith('artifacts/'))


def latest_run(api: GitHub, repository: str, pr: dict) -> dict | None:
    runs = [run for run in api.iter_build_runs(repository, branch=pr['head']['ref'], head_sha=full_sha(pr['head']['sha']))
            if run['head_sha'] == pr['head']['sha'] and run['head_branch'] == pr['head']['ref']
            and (run.get('head_repository') or {}).get('full_name') == repository
            and run['event'] in ('push', 'pull_request', 'workflow_dispatch')]
    # Push and pull_request runs are both allowed, even for the same commit.
    # They build the exact head, not GitHub's synthetic merge commit.
    # A rerun keeps its run ID; start time distinguishes a newly requested rerun
    # from a newer-ID run that actually started earlier. Completion time is not
    # used, because an old slow build must not win by finishing last.
    return max(runs, key=lambda run: (run.get('run_started_at') or run.get('created_at') or '',
                                      run['id'], run['run_attempt'])) if runs else None


def run_stamp(run: dict | None):
    return None if run is None else (run['id'], run['run_attempt'], run['status'], run.get('conclusion'))


def owned_comment(comments: list[dict]) -> dict | None:
    return next((comment for comment in comments if comment['user']['login'] == BOT
                 and comment['user'].get('type') == 'Bot' and comment.get('body', '').startswith(MARKER)), None)


def previous_payload(body: str) -> str | None:
    if body.count(START) != 1 or body.count(END) != 1:
        return None
    start, end = body.index(START) + len(START), body.index(END)
    return body[start:end].strip() if end > start else None


def comment_body(pr: dict, run: dict | None, payload: str | None, previous: str = '', error: str | None = None) -> str:
    head = full_sha(pr['head']['sha'])
    header = f'{MARKER}\n## Runtime publication status\n\n**Current PR head:** `{head}`\n\n'
    if run is None:
        status = ('No matching runtime build is recorded for this head yet. Human PR creation normally starts it; '
                  'workflow approval, merge conflicts, or a missing workflow can prevent a run from starting.')
    elif run['status'] != 'completed':
        status = f'The latest runtime build/report run is **{run["status"]}**. No new complete consumer report is available yet.'
    elif run['conclusion'] == 'success' and payload is not None:
        status = 'The runtime build and consumer report succeeded for this PR head.'
    elif run['conclusion'] == 'success':
        status = 'The runtime build succeeded, but its consumer report is unavailable or expired.'
    else:
        status = (f'The latest runtime build/report run concluded **{run["conclusion"]}**. '
                  'Publication may have completed before a reporting failure; this is not a new complete consumer report.')
    header += status + '\n'
    if run:
        header += f'\n[Build/report run]({run["html_url"]}) (attempt {run["run_attempt"]})\n'
    if error:
        header += '\nReport retrieval error: ' + error.replace('`', "'").replace('\n', ' ')[:500] + '\n'
    if payload is not None:
        header += '\n' + START + '\n' + payload.strip() + '\n' + END + '\n'
    else:
        old = previous_payload(previous)
        if old:
            header += ('\n<details>\n<summary>Previous successful publication — not confirmation of the current head/run</summary>\n\n'
                       + START + '\n' + old + '\n' + END + '\n\n</details>\n')
    if len(header.encode('utf-8')) > 62000:
        raise ValueError('PR comment size limit exceeded')
    return header


def reconcile(api: GitHub, repository: str) -> int:
    repository_name(repository)
    changed = 0
    # The workflow serializes reconciliation repository-wide. Every invocation
    # scans open same-repository PRs, so coalesced pending events cannot strand a
    # different PR. PRs opened after a successful push build are covered as well.
    for pr in api.iter_pull_requests(repository, state='open'):
        if not same_repo_pr(pr, repository):
            continue
        run = latest_run(api, repository, pr)
        comments = list(api.iter_pull_request_comments(repository, pr['number']))
        previous = owned_comment(comments)
        payload = None
        error = None
        if run and run['status'] == 'completed' and run['conclusion'] == 'success':
            try:
                name = f'consumer-update-{run["run_attempt"]}'
                artifacts = api.iter_run_artifacts(repository, run['id'], name=name)
                matches = [item for item in artifacts if item['name'] == name and not item['expired']]
                if len(matches) != 1:
                    raise ValueError('No unique unexpired report artifact for this attempt')
                content = api.download_artifact_archive(repository, matches[0]['id'], max_bytes=MAX_ARCHIVE)
                payload = decode_report(content, repository, run)
            except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as failure:
                error = str(failure)
        # Event payloads and run completion order are not authoritative. Refresh
        # both PR head and newest run before writing, including rerun attempts.
        current = api.get_pull_request(repository, pr['number'])
        if not same_repo_pr(current, repository) or current['head']['sha'] != pr['head']['sha']:
            continue
        if run_stamp(latest_run(api, repository, current)) != run_stamp(run):
            continue
        body = comment_body(current, run, payload, previous['body'] if previous else '', error)
        if previous and previous['body'] == body:
            continue
        if previous:
            api.update_pull_request_comment(repository, previous['id'], body)
        else:
            api.create_pull_request_comment(repository, pr['number'], body)
        changed += 1
    return changed


def main() -> None:
    changed = reconcile(GitHub(), repository_name(os.environ['GITHUB_REPOSITORY']))
    print(f'Reconciled runtime publication comments: {changed} changed.')


if __name__ == '__main__':
    main()
