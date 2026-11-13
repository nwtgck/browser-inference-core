#!/usr/bin/env python3
"""Reconcile runtime comments using trusted default-branch code and bounded report data."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import stat
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from build_tools.github import full_sha, repository_name, positive_id

MARKER = '<!-- lcb-runtime-artifact-comment:v1 -->'
START = '<!-- lcb-published-runtime:start -->'
END = '<!-- lcb-published-runtime:end -->'
BOT = 'github-actions[bot]'
REPORT_FILES = {'report.json', 'consumer-update.md', 'consumer-update.yaml'}
MAX_ARCHIVE = 256 * 1024


def validate_pr(pr: dict, repository: str) -> dict:
    repository_name(repository)
    positive_id(pr['number'])
    full_sha(pr['head']['sha'])
    ref = pr['head']['ref']
    if not isinstance(ref, str) or not ref or any(char in ref for char in '\n\r\0\t'):
        raise ValueError('Unsafe pull-request head ref')
    return pr


def validate_run(run: dict, repository: str) -> dict:
    run_id = positive_id(run['id'])
    positive_id(run['run_attempt'])
    full_sha(run['head_sha'])
    if run.get('html_url') != f'https://github.com/{repository_name(repository)}/actions/runs/{run_id}':
        raise ValueError('Unexpected source workflow run URL')
    if run['status'] not in ('queued', 'in_progress', 'completed', 'waiting', 'requested', 'pending'):
        raise ValueError('Unexpected source workflow status')
    if run.get('conclusion') not in (None, 'success', 'failure', 'cancelled', 'skipped', 'timed_out',
                                    'action_required', 'neutral', 'stale', 'startup_failure'):
        raise ValueError('Unexpected source workflow conclusion')
    return run


def decode_report(content: bytes, repository: str, run: dict) -> str:
    repository_name(repository)
    validate_run(run, repository)
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


def latest_run(runs: list[dict], repository: str, pr: dict) -> dict | None:
    runs = [validate_run(run, repository) for run in runs
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
                 and comment['user'].get('type') == 'Bot' and isinstance(comment.get('body'), str) and comment['body'].startswith(MARKER)), None)


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


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def pull_numbers(state: Path) -> list[int]:
    values = read_json(state / 'requests.json')
    if not isinstance(values, list) or len(set(values)) != len(values):
        raise ValueError('Expected distinct pull-request identifiers')
    return [positive_id(value) for value in values]


def plan_pulls(state: Path, repository: str) -> None:
    repository_name(repository)
    pulls = read_json(state / 'pulls.json')
    if not isinstance(pulls, list): raise ValueError('Expected an open pull-request array')
    numbers = []
    for pr in pulls:
        if not same_repo_pr(pr, repository): continue
        validate_pr(pr, repository)
        number = pr['number']
        if number in numbers: raise ValueError('Duplicate pull request in downloaded pages')
        numbers.append(number)
        write_json(state / str(number) / 'pr.json', pr)
    write_json(state / 'requests.json', numbers)


def checked_runs(path: Path) -> list[dict]:
    values = read_json(path)
    if not isinstance(values, list) or len(values) > 1000:
        raise ValueError('Incomplete or oversized build history')
    identities = [(positive_id(run['id']), positive_id(run['run_attempt'])) for run in values]
    if len(set(identities)) != len(identities):
        raise ValueError('Duplicate workflow run in downloaded pages')
    return values


def plan_runs(state: Path, repository: str) -> None:
    for number in pull_numbers(state):
        directory = state / str(number)
        pr = validate_pr(read_json(directory / 'pr.json'), repository)
        run = latest_run(checked_runs(directory / 'runs.json'), repository, pr)
        write_json(directory / 'plan.json', {'run': run, 'download': bool(run and run['status'] == 'completed'
                                                                         and run['conclusion'] == 'success')})


def record_error(directory: Path, error: Exception | str) -> None:
    write_json(directory / 'error.json', {'error': str(error)[:1000]})


def select_artifacts(state: Path) -> None:
    for number in pull_numbers(state):
        directory = state / str(number)
        plan = read_json(directory / 'plan.json')
        if not plan['download'] or (directory / 'error.json').exists(): continue
        try:
            run = plan['run']
            values = read_json(directory / 'artifacts.json')
            if not isinstance(values, list): raise ValueError('Expected a report artifact array')
            name = f'consumer-update-{positive_id(run["run_attempt"])}'
            matches = [item for item in values if item.get('name') == name and item.get('expired') is False]
            if len(matches) != 1: raise ValueError('No unique unexpired report artifact for this attempt')
            artifact = positive_id(matches[0]['id'])
            write_json(directory / 'archive.json', {'artifactId': artifact})
        except (OSError, ValueError, KeyError, TypeError) as error:
            record_error(directory, error)


def archive_urls(state: Path) -> None:
    for number in pull_numbers(state):
        directory = state / str(number)
        if not (directory / 'archive.json').exists() or (directory / 'error.json').exists(): continue
        try:
            status = (directory / 'archive-status').read_text().strip()
            if status == '200':
                raw = (directory / 'archive-initial').read_bytes()
                if len(raw) > MAX_ARCHIVE: raise ValueError('Report archive exceeds the size limit')
                (directory / 'report.zip').write_bytes(raw)
                continue
            if status != '302': raise ValueError('Unexpected artifact API response: ' + status)
            headers = (directory / 'archive-headers').read_text()
            locations = [line.split(':', 1)[1].strip() for line in headers.splitlines()
                         if line.lower().startswith('location:')]
            if len(locations) != 1: raise ValueError('Missing or ambiguous artifact redirect')
            url = locations[0]
            parsed = urlparse(url)
            if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
                    or any(char in url for char in '\n\r\0') or parsed.port not in (None, 443)):
                raise ValueError('Unexpected artifact download redirect')
            write_json(directory / 'download.json', {'url': url})
        except (OSError, ValueError, KeyError) as error:
            record_error(directory, error)


def write_plan(state: Path, repository: str, number: int) -> dict:
    """Refresh trust, PR head and newest run immediately before one workflow write."""
    positive_id(number)
    directory = state / str(number)
    original = validate_pr(read_json(directory / 'pr.json'), repository)
    current = validate_pr(read_json(directory / 'current-pr.json'), repository)
    selected = read_json(directory / 'plan.json')['run']
    latest = latest_run(checked_runs(directory / 'current-runs.json'), repository, current)
    if (not same_repo_pr(current, repository) or current['head']['sha'] != original['head']['sha']
            or current['head']['ref'] != original['head']['ref'] or run_stamp(latest) != run_stamp(selected)):
        result = {'write': False, 'reason': 'PR head, trust or latest workflow run changed'}
    else:
        comments = read_json(directory / 'current-comments.json')
        if not isinstance(comments, list): raise ValueError('Expected a conversation-comment array')
        previous = owned_comment(comments)
        if previous: positive_id(previous['id'])
        payload, error = None, None
        if selected and selected['status'] == 'completed' and selected['conclusion'] == 'success':
            try:
                if (directory / 'error.json').exists():
                    raise ValueError(read_json(directory / 'error.json')['error'])
                payload = decode_report((directory / 'report.zip').read_bytes(), repository, selected)
            except (OSError, ValueError, RuntimeError, KeyError, zipfile.BadZipFile) as failure:
                error = str(failure)
        body = comment_body(current, selected, payload, previous['body'] if previous else '', error)
        if previous and previous['body'] == body:
            result = {'write': False, 'reason': 'Comment already matches current status'}
        else:
            write_json(directory / 'body.json', {'body': body})
            result = {'write': True, 'commentId': previous['id'] if previous else None}
    write_json(directory / 'write.json', result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('plan-pulls', 'plan-runs', 'select-artifacts', 'archive-urls', 'write-plan'))
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--number', type=int)
    args = parser.parse_args()
    repository = repository_name(os.environ['GITHUB_REPOSITORY'])
    if args.command == 'plan-pulls': plan_pulls(args.state, repository)
    elif args.command == 'plan-runs': plan_runs(args.state, repository)
    elif args.command == 'select-artifacts': select_artifacts(args.state)
    elif args.command == 'archive-urls': archive_urls(args.state)
    else: write_plan(args.state, repository, positive_id(args.number))


if __name__ == '__main__': main()
