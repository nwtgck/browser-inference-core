#!/usr/bin/env python3
"""Prepare and push an upstream-update branch; pull-request creation stays manual."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
from urllib.parse import quote, urlencode

from github_api import ApiError, GitHub, full_sha, git, git_auth_env, repository_name
from prepare_mtmd import PATCH_DIRECTORY, prepare
import prepare_webgpu_source as webgpu_source

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = 'ggml-org/llama.cpp'
STABLE = re.compile(r'v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z')
NIGHTLY = re.compile(r'b[0-9]+\Z')
PINS = {'vendor/llama.cpp', 'config/toolchain.json'}

def repository_path(root: Path, relative: str) -> str:
    prefix = git('rev-parse', '--show-prefix', cwd=root).stdout.strip()
    return prefix + relative



def checked_ref(value: str) -> str:
    # Deliberately exclude revision expressions, options, URLs and abbreviated
    # hashes. Named refs and full commits have unambiguous, auditable identities.
    if not value or value.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
        raise ValueError('Custom ref must be a tag, branch, or full commit SHA')
    subprocess.run(['git', 'check-ref-format', 'refs/tags/' + value], check=True, capture_output=True)
    return value


def peel(api: GitHub, obj: dict) -> str:
    for _ in range(10):
        if obj['type'] == 'commit':
            return full_sha(obj['sha'])
        if obj['type'] != 'tag':
            break
        obj = api.get_tag(UPSTREAM, full_sha(obj['sha']))['object']
    raise ValueError('Upstream ref does not resolve to a commit')


def resolve_named_ref(api: GitHub, ref: str) -> str:
    checked_ref(ref)
    if re.fullmatch(r'[0-9a-f]{40}', ref):
        result = api.get_commit(UPSTREAM, ref)
        if result['sha'] != ref:
            raise ValueError('Commit resolution mismatch')
        return ref
    candidates = [ref[5:]] if ref.startswith(('refs/tags/', 'refs/heads/')) else ['tags/' + ref, 'heads/' + ref]
    objects = []
    for candidate in candidates:
        try:
            objects.append(api.get_ref(UPSTREAM, candidate)['object'])
        except ApiError as error:
            if error.status != 404:
                raise
    if len(objects) != 1:
        raise ValueError('Ref is missing or ambiguous; use refs/tags/... or refs/heads/...')
    return peel(api, objects[0])


def resolve_target(api: GitHub, target: str, custom_ref: str = '') -> dict:
    release = None
    if target == 'custom':
        ref = checked_ref(custom_ref)
    else:
        if custom_ref:
            raise ValueError('custom_ref is only valid with target=custom')
        if target == 'latest':
            release = api.get_latest_release(UPSTREAM)
            # A tag pattern alone cannot establish release status. Conversely,
            # older bNNNN releases must never silently become the stable channel.
            if release['draft'] or release['prerelease'] or not STABLE.fullmatch(release['tag_name']):
                raise ValueError('Latest release is not a published stable vX.Y.Z release; no fallback')
        elif target == 'latest-unstable':
            # First published bNNNN pre-release in GitHub's release-list order.
            # This is the released nightly channel, never an unbuilt master tip.
            release = next((item for item in api.iter_releases(UPSTREAM)
                            if not item['draft'] and item['prerelease']
                            and item.get('published_at') and NIGHTLY.fullmatch(item['tag_name'])), None)
            if release is None:
                raise ValueError('No published bNNNN pre-release was found; no fallback')
        else:
            raise ValueError('Unknown update target')
        ref = 'refs/tags/' + release['tag_name']
    result = {'requested': target, 'ref': ref, 'commit': resolve_named_ref(api, ref)}
    if release:
        result['release'] = {key: release[key] for key in ('tag_name', 'published_at', 'html_url', 'prerelease')}
    return result


def read_pins(root: Path) -> tuple[str, dict]:
    toolchain = json.loads((root / 'config/toolchain.json').read_text())
    commit = full_sha(toolchain['llamaCommit'])
    entry = git('ls-tree', 'HEAD', '--', 'vendor/llama.cpp', cwd=root).stdout.split()
    if len(entry) != 4 or entry[:3] != ['160000', 'commit', commit]:
        raise ValueError('Submodule gitlink and toolchain.json disagree')
    return commit, toolchain


def change_pins(root: Path, commit: str) -> None:
    _, toolchain = read_pins(root)
    src = root / 'vendor/llama.cpp'
    git('fetch', '--no-tags', '--depth=1', 'https://github.com/' + UPSTREAM + '.git', full_sha(commit), cwd=src)
    git('checkout', '--detach', commit, cwd=src)
    git('submodule', 'update', '--init', '--recursive', cwd=src)
    toolchain['llamaCommit'] = commit
    (root / 'config/toolchain.json').write_text(json.dumps(toolchain, indent=2) + '\n')
    git('add', '--', *sorted(PINS), cwd=root)
    changed = set(git('diff', '--cached', '--name-only', '--relative', cwd=root).stdout.splitlines())
    if changed != PINS:
        raise ValueError('Updater may change only the submodule gitlink and llamaCommit')


def overlay_preflight(root: Path) -> dict:
    try:
        with tempfile.TemporaryDirectory(prefix='lcb-update-overlay-') as tmp:
            prepare(root / 'vendor/llama.cpp', Path(tmp) / 'vision', root / f'{PATCH_DIRECTORY}/mtmd-webgpu-bf16.patch', capture_output=True)
            prepare(root / 'vendor/llama.cpp', Path(tmp) / 'audio', root / f'{PATCH_DIRECTORY}/mtmd-audio-single-thread.patch',
                    capture_output=True, filename='mtmd-audio.cpp')
            webgpu_source.prepare(root / 'vendor/llama.cpp', Path(tmp) / 'webgpu',
                                  patch_root=root / PATCH_DIRECTORY)
        return {'status': 'passed', 'scope': 'patch application only; not compilation or inference'}
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        # A failed overlay still leaves a candidate branch for human repair.
        # No automatic PR, build dispatch, patch fuzz, or patch deletion.
        details = error.stderr if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
        return {'status': 'failed', 'scope': 'patch application only', 'error': details.strip()[:4000]}


def candidate_branch(base: str, source: str, target: str) -> str:
    # Base identity avoids collisions across source branches; including the base
    # commit avoids ever rebasing or force-pushing over manual repair commits.
    identity = hashlib.sha256(base.encode()).hexdigest()[:12]
    return f'automation/llama-cpp/{identity}-{full_sha(source)[:12]}-{full_sha(target)[:12]}'


def pull_request_links(repository: str, result: dict) -> dict:
    """Describe a browser handoff, without reading or mutating pull requests."""
    base, branch = result['base'], result['branch']
    target, preflight = result['target'], result['preflight']
    title = 'chore: update llama.cpp to ' + target['ref'].removeprefix('refs/tags/')
    body = ('## Upstream update\n\n'
            f'- Requested channel: `{target["requested"]}`\n'
            f'- Resolved ref: `{target["ref"]}`\n'
            f'- Previous llama.cpp: `{result["previousCommit"]}`\n'
            f'- Proposed llama.cpp: `{target["commit"]}`\n'
            f'- Upstream relation: `{result["upstreamRelation"]}`\n'
            f'- Overlay preflight: **{preflight["status"]}** ({preflight["scope"]})\n\n'
            f'[Upstream comparison](https://github.com/{UPSTREAM}/compare/{result["previousCommit"]}...{target["commit"]})\n\n'
            'The automated commit changes only the submodule gitlink and '
            '`config/toolchain.json`. A reused branch may also contain manual repairs. '
            'Preflight is not build or inference validation. Full preflight diagnostics '
            'are in the updater run summary and upstream-update-report artifact.\n\n'
            'Opening this PR starts the ordinary runtime workflow. A successful '
            'same-repository build publishes before merge; its runtime comment '
            'contains the installation command and consumer metadata.\n')
    compare = f'https://github.com/{repository_name(repository)}/compare/{quote(base, safe="")}...{quote(branch, safe="")}'
    url = compare + '?' + urlencode({'quick_pull': '1', 'title': title, 'body': body})
    # A long ref must not turn the browser handoff into an unusable URL. The
    # complete suggested title/body remain available in the summary/report.
    if len(url) > 7000:
        url = compare + '?quick_pull=1'
    return {'compareUrl': compare, 'createPullRequestUrl': url,
            'pullRequestTitle': title, 'pullRequestBody': body}


def propose(root: Path, api: GitHub, repository: str, base: str, target: dict,
            allow_non_fast_forward: bool = False, *, progress: dict | None = None) -> dict:
    repository_name(repository)
    checked_ref(base)
    if base == 'artifacts' or base.startswith(('artifacts/', 'automation/llama-cpp/')):
        raise ValueError('Choose a source base branch, not artifacts or an updater branch')
    if git('status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=none', cwd=root).stdout:
        raise ValueError('Updater requires a clean source checkout')
    source = full_sha(git('rev-parse', 'HEAD', cwd=root).stdout.strip())
    previous, _ = read_pins(root)
    result = progress if progress is not None else {}
    result.update({'base': base, 'baseCommit': source, 'previousCommit': previous, 'target': target})
    if previous == target['commit']:
        return {**result, 'status': 'unchanged'}
    comparison = api.compare_commits(UPSTREAM, previous, target['commit'])
    if comparison['status'] not in ('ahead', 'identical') and not allow_non_fast_forward:
        raise ValueError('Target is older or divergent; allow_non_fast_forward is required for this change')
    result['upstreamRelation'] = comparison['status']
    branch = candidate_branch(base, source, target['commit'])
    result.update({'branch': branch, 'branchUrl': f'https://github.com/{repository}/tree/{branch}'})
    remote = 'https://github.com/' + repository + '.git'
    existing = git('ls-remote', '--exit-code', '--heads', remote, 'refs/heads/' + branch, cwd=root, check=False, env=git_auth_env(os.environ['GH_TOKEN']))
    if existing.returncode not in (0, 2):
        raise RuntimeError('Cannot inspect the remote update branch')
    if existing.stdout:
        git('fetch', '--no-tags', '--depth=1', remote, 'refs/heads/' + branch, cwd=root,
            env=git_auth_env(os.environ['GH_TOKEN']))
        # ls-remote is only an existence probe. A human may advance the branch
        # before fetch, so validate and report the fetched snapshot, not the
        # potentially stale probe SHA (which might not even be fetched).
        head = full_sha(git('rev-parse', '--verify', 'FETCH_HEAD', cwd=root).stdout.strip())
        # Only validate its pins. Any human fixes on a previous attempt survive.
        pins = json.loads(git('show', head + ':' + repository_path(root, 'config/toolchain.json'), cwd=root).stdout)
        link = git('ls-tree', head, '--', 'vendor/llama.cpp', cwd=root).stdout.split()
        if pins.get('llamaCommit') != target['commit'] or len(link) != 4 or link[:3] != ['160000', 'commit', target['commit']]:
            raise ValueError('Existing updater branch has different pins; refusing to overwrite it')
        preflight = {'status': 'not-repeated', 'scope': 'existing branch preserved; full build is authoritative'}
        status = 'branch-exists'
    else:
        change_pins(root, target['commit'])
        preflight = overlay_preflight(root)
        result['preflight'] = preflight
        git('-c', 'user.name=github-actions[bot]', '-c',
            'user.email=41898282+github-actions[bot]@users.noreply.github.com',
            'commit', '-m', 'chore: update llama.cpp to ' + target['ref'].removeprefix('refs/tags/'), cwd=root)
        head = full_sha(git('rev-parse', 'HEAD', cwd=root).stdout.strip())
        result['sourceCommit'] = head
        # No force-push, including races between simultaneous updater runs.
        git('push', remote, head + ':refs/heads/' + branch, cwd=root,
            env=git_auth_env(os.environ['GH_TOKEN']))
        status = 'branch-created-needs-overlay-repair' if preflight['status'] == 'failed' else 'branch-created'
    result.update({'branch': branch, 'sourceCommit': head, 'preflight': preflight})
    result.update(pull_request_links(repository, result))
    # GITHUB_TOKEN pushes do not launch push workflows. Automation deliberately
    # ends here: a human opens the PR, whose pull_request event starts the build.
    # Reusing a branch is a no-write operation, even when it already has a PR.
    return {**result, 'status': status}


def render_summary(result: dict) -> str:
    text = '## llama.cpp update branch\n\n'
    if result.get('createPullRequestUrl'):
        text += ('[Open pull request form](' + result['createPullRequestUrl'] + ') · '
                 '[Compare branches](' + result['compareUrl'] + ') · '
                 '[Update branch](' + result['branchUrl'] + ')\n\n'
                 '**The updater ends at branch push.** The links do not create a PR. '
                 'Submitting the PR in GitHub starts the ordinary runtime workflow; '
                 'the bot-authenticated push is not replayed as a human push.\n\n')
        if result['preflight']['status'] == 'failed':
            text += ('**Overlay preflight failed.** The candidate branch is preserved '
                     'for manual repair. Opening a PR, including a draft PR, can '
                     'still start the build; draft status is not a build gate.\n\n')
        elif result['status'] == 'branch-exists':
            text += ('**Existing branch preserved without a push.** No build was requested. '
                     'An already-open PR is unchanged; merely re-running this updater '
                     'does not re-run its CI.\n\n')
    return text + '<details>\n<summary>Update details and suggested PR text</summary>\n\n```json\n' + json.dumps(result, indent=2) + '\n```\n\n</details>\n'


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['latest', 'latest-unstable', 'custom'], default='latest')
    parser.add_argument('--custom-ref', default='')
    parser.add_argument('--base', required=True)
    parser.add_argument('--allow-non-fast-forward', action='store_true')
    args = parser.parse_args()
    api = GitHub()
    result = {'requested': args.target, 'customRef': args.custom_ref, 'base': args.base}
    failed = False
    try:
        target = resolve_target(api, args.target, args.custom_ref)
        result = propose(ROOT, api, os.environ['GITHUB_REPOSITORY'], args.base, target,
                         args.allow_non_fast_forward, progress=result)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        # A pushed branch can survive a later reporting failure. Its identity
        # remains visible and a rerun reuses it without replacing human changes.
        details = error.stderr if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
        result.update({'status': 'failed', 'error': details.strip()[:4000]})
        failed = True
    # Report output stays outside the source index and is never a build input.
    folder = ROOT / 'build/upstream-update'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    text = render_summary(result)
    print(text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as output:
            output.write(text)
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
