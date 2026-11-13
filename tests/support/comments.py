"""Feed fake service snapshots through the real offline comment stages.

This models data handoffs only. Separate workflow tests execute the literal shell
with local command stand-ins; neither suite contacts GitHub or claims Docker.
"""
from pathlib import Path
import tempfile

from build_tools.publication import comments


def reconcile(api, repository):
    with tempfile.TemporaryDirectory(prefix='bicore-comment-fixture-') as tmp:
        state = Path(tmp)
        comments.write_json(state / 'pulls.json', list(api.iter_pull_requests(repository, state='open')))
        comments.plan_pulls(state, repository)
        for number in comments.pull_numbers(state):
            directory = state / str(number)
            pr = comments.read_json(directory / 'pr.json')
            runs = list(api.iter_build_runs(repository, branch=pr['head']['ref'], head_sha=pr['head']['sha']))
            comments.write_json(directory / 'runs.json', runs)
        comments.plan_runs(state, repository)
        for number in comments.pull_numbers(state):
            directory = state / str(number)
            plan = comments.read_json(directory / 'plan.json')
            if plan['download']:
                run = plan['run']
                values = list(api.iter_run_artifacts(repository, run['id'], name=f'consumer-update-{run["run_attempt"]}'))
                comments.write_json(directory / 'artifacts.json', values)
        comments.select_artifacts(state)
        for number in comments.pull_numbers(state):
            directory = state / str(number)
            if (directory / 'archive.json').exists():
                try:
                    artifact = comments.read_json(directory / 'archive.json')['artifactId']
                    raw = api.download_artifact_archive(repository, artifact, max_bytes=comments.MAX_ARCHIVE)
                    (directory / 'report.zip').write_bytes(raw)
                except (OSError, ValueError, RuntimeError) as error:
                    comments.record_error(directory, error)
        changed = 0
        for number in comments.pull_numbers(state):
            directory = state / str(number)
            pr = api.get_pull_request(repository, number)
            comments.write_json(directory / 'current-pr.json', pr)
            comments.write_json(directory / 'current-runs.json', list(api.iter_build_runs(
                repository, branch=pr['head']['ref'], head_sha=pr['head']['sha'])))
            comments.write_json(directory / 'current-comments.json', list(api.iter_pull_request_comments(repository, number)))
            result = comments.write_plan(state, repository, number)
            if not result['write']: continue
            body = comments.read_json(directory / 'body.json')['body']
            if result['commentId'] is None: api.create_pull_request_comment(repository, number, body)
            else: api.update_pull_request_comment(repository, result['commentId'], body)
            changed += 1
        return changed
