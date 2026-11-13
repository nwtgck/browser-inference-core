"""Offline upstream selection, candidate preparation and manual PR handoff.

Release/API/Git remote acquisition and the final push are explicit workflow steps.
All Git operations here use the shared network-disabled local source helper.
"""
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

from build_tools.github import full_sha, repository_name
from build_tools.project import REPOSITORY_ROOT
from build_tools.source.checkout import git
from build_tools.runtimes.llama_cpp.sources.mtmd import PATCH_DIRECTORY, prepare
from build_tools.runtimes.llama_cpp import source as webgpu_source
from build_tools.runtimes.llama_cpp.sources import qsa_selection as qsa
from build_tools.runtimes.llama_cpp.sources import model_graph_chat as model_graph

ROOT = REPOSITORY_ROOT / 'runtimes/llama-cpp'
UPSTREAM = 'ggml-org/llama.cpp'
STABLE = re.compile(r'v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\Z')
NIGHTLY = re.compile(r'b[0-9]+\Z')
PINS = {'vendor/llama.cpp', 'config/toolchain.json'}


def checked_ref(value: str) -> str:
    if not value or value.startswith('-') or not re.fullmatch(r'[A-Za-z0-9_./-]+', value):
        raise ValueError('Custom ref must be a tag, branch, or full commit SHA')
    result = subprocess.run(['git', 'check-ref-format', 'refs/tags/' + value],
                            check=False, capture_output=True, text=True)
    if result.returncode:
        raise ValueError('Invalid named Git ref: ' + value)
    return value


def repository_path(root: Path, relative: str) -> str:
    return git(root, 'rev-parse', '--show-prefix').strip() + relative


def read_pins(root: Path, revision: str = 'HEAD') -> tuple[str, dict]:
    if revision == 'HEAD':
        toolchain = json.loads((root / 'config/toolchain.json').read_text())
    else:
        full_sha(revision)
        toolchain = json.loads(git(root, 'show', revision + ':' + repository_path(root, 'config/toolchain.json')))
    commit = full_sha(toolchain['llamaCommit'])
    entry = git(root, 'ls-tree', revision, '--', 'vendor/llama.cpp').split()
    if len(entry) != 4 or entry[:3] != ['160000', 'commit', commit]:
        raise ValueError('Submodule gitlink and toolchain.json disagree')
    return commit, toolchain


def assert_source(root: Path, source: str, *, clean: bool = False) -> None:
    if git(root, 'rev-parse', 'HEAD').strip() != full_sha(source):
        raise ValueError('Source HEAD changed after update planning')
    if clean and git(root, 'status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=none'):
        raise ValueError('Updater requires a clean source checkout')


def request(root: Path, repository: str, base: str, target: str, custom_ref: str = '',
            allow_non_fast_forward: bool = False) -> dict:
    repository_name(repository)
    checked_ref(base)
    if base == 'artifacts' or base.startswith(('artifacts/', 'automation/llama-cpp/')):
        raise ValueError('Choose a source base branch, not artifacts or an updater branch')
    if target not in ('latest', 'latest-unstable', 'custom'):
        raise ValueError('Unknown update target')
    if target == 'custom': checked_ref(custom_ref)
    elif custom_ref: raise ValueError('custom_ref is only valid with target=custom')
    source = full_sha(git(root, 'rev-parse', 'HEAD').strip())
    assert_source(root, source, clean=True)
    previous, toolchain = read_pins(root)
    return {'repository': repository, 'base': base, 'baseCommit': source,
            'previousCommit': previous, 'requested': target, 'customRef': custom_ref,
            'allowNonFastForward': bool(allow_non_fast_forward), 'toolchain': toolchain}


def select(requested: dict, releases: object = None) -> dict:
    target = requested['requested']
    release = None
    if target == 'custom':
        ref = checked_ref(requested['customRef'])
    elif target == 'latest':
        release = releases
        if (not isinstance(release, dict) or release.get('draft') is not False
                or release.get('prerelease') is not False
                or not release.get('published_at') or not STABLE.fullmatch(release.get('tag_name', ''))):
            raise ValueError('Latest release is not a published stable vX.Y.Z release; no fallback')
        ref = 'refs/tags/' + release['tag_name']
    elif target == 'latest-unstable':
        if not isinstance(releases, list):
            raise ValueError('Expected the published release list')
        release = next((item for item in releases if isinstance(item, dict)
                        and item.get('draft') is False and item.get('prerelease') is True
                        and item.get('published_at') and NIGHTLY.fullmatch(item.get('tag_name', ''))), None)
        if release is None:
            raise ValueError('No published bNNNN pre-release was found; no fallback')
        ref = 'refs/tags/' + release['tag_name']
    else:
        raise ValueError('Unknown update target')
    result = {'requested': target, 'ref': ref}
    if release:
        result['release'] = {key: release[key] for key in ('tag_name', 'published_at', 'html_url', 'prerelease')}
    return result


def plan(root: Path, requested: dict, selected: dict, resolved: dict,
         comparison: dict | None = None) -> dict:
    assert_source(root, requested['baseCommit'], clean=True)
    commit = full_sha(resolved['commit'])
    if resolved['ref'] != selected['ref']:
        raise ValueError('Resolved upstream ref differs from selection')
    target = {**selected, 'commit': commit}
    result = {**requested, 'target': target}
    if requested['previousCommit'] == commit:
        return {**result, 'status': 'unchanged'}
    relation = (comparison or {}).get('status')
    if relation not in ('ahead', 'identical', 'behind', 'diverged'):
        raise ValueError('Unknown upstream comparison status')
    if relation not in ('ahead', 'identical') and not requested['allowNonFastForward']:
        raise ValueError('Target is older or divergent; allow_non_fast_forward is required for this change')
    branch = candidate_branch(requested['base'], requested['baseCommit'], commit)
    return {**result, 'status': 'planned', 'upstreamRelation': relation, 'branch': branch,
            'branchUrl': f'https://github.com/{requested["repository"]}/tree/{branch}'}


def error_details(error: BaseException) -> str:
    details = error.stderr if isinstance(error, subprocess.CalledProcessError) and error.stderr else str(error)
    if isinstance(details, bytes):
        details = details.decode('utf-8', errors='replace')
    return details.strip()[:4000]


def overlay_preflight(root: Path) -> dict:
    try:
        with tempfile.TemporaryDirectory(prefix='lcb-update-overlay-') as tmp:
            for mode in ('cpu', 'webgpu'):
                prepare(root / 'vendor/llama.cpp', Path(tmp) / mode,
                        mode=mode, patch_root=root / PATCH_DIRECTORY)
            webgpu_source.prepare(root / 'vendor/llama.cpp', Path(tmp) / 'webgpu',
                                  patch_root=root / PATCH_DIRECTORY)
            qsa.preflight(root / 'vendor/llama.cpp', Path(tmp) / 'qsa')
            model_graph.prepare(root / 'vendor/llama.cpp', Path(tmp) / 'model-graph', runtime=root)
        return {'status': 'passed', 'scope': 'patch application only; not compilation or inference'}
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        # A failed overlay still leaves a candidate branch for human repair.
        # No automatic PR, build dispatch, patch fuzz, or patch deletion.
        return {'status': 'failed', 'scope': 'patch application only', 'error': error_details(error)}


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


def checkout(root: Path, result: dict, existing: dict) -> None:
    if result['status'] == 'unchanged' or existing.get('commit') is not None:
        return
    assert_source(root, result['baseCommit'], clean=True)
    commit = full_sha(result['target']['commit'])
    # The workflow fetched this exact object already. Never acquire source here.
    if git(root / 'vendor/llama.cpp', 'rev-parse', '--verify', commit + '^{commit}').strip() != commit:
        raise ValueError('Selected upstream commit is not locally available')
    git(root / 'vendor/llama.cpp', 'checkout', '--detach', commit)


def candidate(root: Path, result: dict, existing: dict) -> dict:
    if result['status'] == 'unchanged': return result
    assert_source(root, result['baseCommit'])
    target = full_sha(result['target']['commit'])
    head = existing.get('commit')
    if head is not None:
        head = full_sha(head)
        try:
            previous, _ = read_pins(root, head)
        except ValueError as error:
            raise ValueError('Existing updater branch has inconsistent pins; refusing to overwrite it') from error
        if previous != target:
            raise ValueError('Existing updater branch has different pins; refusing to overwrite it')
        preflight = {'status': 'not-repeated', 'scope': 'existing branch preserved; full build is authoritative'}
        status = 'branch-exists'
    else:
        # Only the fetched and detached upstream checkout may differ here.
        if git(root / 'vendor/llama.cpp', 'rev-parse', 'HEAD').strip() != target:
            raise ValueError('Local upstream checkout differs from the selected target')
        if json.loads((root / 'config/toolchain.json').read_text()) != result['toolchain']:
            raise ValueError('Toolchain pins changed after update planning')
        toolchain = {**result['toolchain'], 'llamaCommit': target}
        (root / 'config/toolchain.json').write_text(json.dumps(toolchain, indent=2) + '\n')
        git(root, 'add', '--', *sorted(PINS))
        expected = {repository_path(root, path) for path in PINS}
        # A multi-runtime repository must check the whole index, not --relative:
        # unrelated changes outside this runtime must never enter the commit.
        changed = set(git(root, 'diff', '--cached', '--name-only', '--no-relative').splitlines())
        if changed != expected:
            raise ValueError('Updater may change only the submodule gitlink and llamaCommit')
        if git(root, 'diff', '--name-only', '--no-relative'):
            raise ValueError('Unstaged source changes appeared during candidate preparation')
        preflight = overlay_preflight(root)
        assert_source(root, result['baseCommit'])
        if set(git(root, 'diff', '--cached', '--name-only', '--no-relative').splitlines()) != expected:
            raise ValueError('Source index changed during overlay preflight')
        staged_toolchain = json.loads(git(root, 'show', ':' + repository_path(root, 'config/toolchain.json')))
        staged_link = git(root, 'ls-files', '--stage', '--', 'vendor/llama.cpp').split()
        if (staged_toolchain != toolchain or len(staged_link) != 4
                or staged_link[:3] != ['160000', target, '0']):
            raise ValueError('Candidate pins changed during overlay preflight')
        if git(root, 'diff', '--name-only', '--no-relative'):
            raise ValueError('Unstaged source changes appeared during overlay preflight')
        tree = full_sha(git(root, 'write-tree').strip())
        head = full_sha(git(root, '-c', 'user.name=github-actions[bot]', '-c',
             'user.email=41898282+github-actions[bot]@users.noreply.github.com',
             'commit-tree', tree, '-p', result['baseCommit'], '-m',
             'chore: update llama.cpp to ' + result['target']['ref'].removeprefix('refs/tags/')).strip())
        status = 'candidate-prepared'
    updated = {**result, 'sourceCommit': head, 'preflight': preflight, 'status': status}
    updated.update(pull_request_links(result['repository'], updated))
    return updated


def render_summary(result: dict) -> str:
    text = '## llama.cpp update branch\n\n'
    if (result.get('createPullRequestUrl') and result.get('status') in
            ('branch-exists', 'branch-created', 'branch-created-needs-overlay-repair')):
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


def save(state: Path, name: str, value: dict) -> None:
    state.mkdir(parents=True, exist_ok=True)
    (state / name).write_text(json.dumps(value, indent=2) + '\n')


def load(state: Path, name: str) -> dict:
    return json.loads((state / name).read_text())


def output(values: dict) -> None:
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
            for key, value in values.items():
                value = str(value)
                if any(char in value for char in '\r\n\0'):
                    raise ValueError('Unsafe workflow output')
                out.write(key + '=' + value + '\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('request', 'select', 'plan', 'checkout', 'candidate', 'report'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--state', type=Path, default=ROOT / 'build/upstream-update')
    parser.add_argument('--target', choices=('latest', 'latest-unstable', 'custom'), default='latest')
    parser.add_argument('--custom-ref', default='')
    parser.add_argument('--base', default='')
    parser.add_argument('--allow-non-fast-forward', action='store_true')
    parser.add_argument('--push-outcome', default='')
    args = parser.parse_args()
    result = {'requested': args.target, 'customRef': args.custom_ref, 'base': args.base}
    if (args.state / 'report.json').is_file(): result = load(args.state, 'report.json')
    try:
        if args.command == 'request':
            result = request(args.root, os.environ['GITHUB_REPOSITORY'], args.base, args.target,
                             args.custom_ref, args.allow_non_fast_forward)
            save(args.state, 'request.json', result)
            output({'previous-commit': result['previousCommit']})
        elif args.command == 'select':
            selection = select(load(args.state, 'request.json'),
                               load(args.state, 'releases.json') if (args.state / 'releases.json').exists() else None)
            save(args.state, 'selection.json', selection)
            output({'selected-ref': selection['ref']})
        elif args.command == 'plan':
            result = plan(args.root, load(args.state, 'request.json'), load(args.state, 'selection.json'),
                          load(args.state, 'resolved.json'),
                          load(args.state, 'comparison.json') if (args.state / 'comparison.json').exists() else None)
            save(args.state, 'plan.json', result)
            output({'changed': str(result['status'] != 'unchanged').lower(), 'branch': result.get('branch', '')})
        elif args.command == 'checkout':
            checkout(args.root, load(args.state, 'plan.json'), load(args.state, 'existing.json'))
        elif args.command == 'candidate':
            result = candidate(args.root, load(args.state, 'plan.json'), load(args.state, 'existing.json'))
            output({'needs-push': str(result['status'] == 'candidate-prepared').lower(),
                    'commit': result.get('sourceCommit', ''), 'branch': result.get('branch', '')})
        elif args.command == 'report':
            if result.get('status') == 'candidate-prepared':
                if args.push_outcome == 'success':
                    result['status'] = ('branch-created-needs-overlay-repair' if result['preflight']['status'] == 'failed'
                                        else 'branch-created')
                else:
                    result.update({'status': 'failed', 'error': 'Candidate push did not complete successfully; rerun to inspect any surviving branch.'})
            elif result.get('status') not in ('unchanged', 'branch-exists', 'failed'):
                result.update({'status': 'failed', 'error': 'Workflow acquisition or planning did not finish; inspect the failed step.'})
        save(args.state, 'report.json', result)
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.CalledProcessError) as error:
        details = error_details(error)
        result.update({'status': 'failed', 'error': str(details).strip()[:4000]})
        save(args.state, 'report.json', result)
        raise
    if args.command == 'report':
        text = render_summary(result)
        print(text)
        if os.environ.get('GITHUB_STEP_SUMMARY'):
            with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as out: out.write(text)


if __name__ == '__main__': main()
