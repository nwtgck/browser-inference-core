#!/usr/bin/env python3
"""Diagnose source pins, or explicitly stage missing gitlinks after a file-only patch.

Default: read-only checks of HEAD. --stage-missing is a local migration command,
not an Actions fallback. It never fetches, checks out, updates pins, overwrites an
existing index entry, commits, or weakens the build planner's HEAD checks.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'llama-cpp/scripts'))
from source_config import get_source, load_sources, patch_series
from source_git import check_gitlink, committed_entry, git, repository_location


def index_entries(repository: Path, path: str) -> list[dict]:
    records = git(repository, 'ls-files', '--stage', '-z', '--', path).stdout
    result = []
    for row in records.split('\0'):
        if not row:
            continue
        fields, actual_path = row.split('\t', 1)
        mode, commit, stage = fields.split()
        result.append({'mode': mode, 'commit': commit, 'stage': stage, 'path': actual_path})
    return result


def check_registration(repository: Path, path: str, source: dict) -> None:
    """Bind the registry to the staged .gitmodules file, not a local URL override."""
    modules = git(repository, 'config', '--blob', ':0:.gitmodules', '--null',
                  '--get-regexp', r'^submodule\..*\.path$', check=False)
    if modules.returncode not in (0, 1):
        raise ValueError('Cannot read staged .gitmodules: ' + modules.stderr.strip())
    matches = []
    for row in modules.stdout.split('\0'):
        if row:
            key, value = row.split('\n', 1)
            if value == path:
                matches.append(key.removesuffix('.path'))
    if len(matches) != 1:
        raise ValueError(f'{path}: require exactly one staged .gitmodules registration')
    url = git(repository, 'config', '--blob', ':0:.gitmodules', '--get-all',
              matches[0] + '.url', check=False)
    expected = 'https://github.com/' + source['repository']
    if url.returncode or url.stdout.strip() not in (expected, expected + '.git'):
        raise ValueError(f'{path}: .gitmodules URL does not match {source["repository"]}')


def check_existing_directory(repository: Path, path: str, source: dict) -> None:
    directory = repository
    for part in path.split('/'):
        directory /= part
        if directory.is_symlink():
            raise ValueError('Linked source directory: ' + path)
    if not directory.exists():
        return
    if not directory.is_dir():
        raise ValueError('Source path is not a directory: ' + path)
    if not any(directory.iterdir()):
        return
    # Refuse a populated plain directory or a checkout of another revision.
    top = git(directory, 'rev-parse', '--show-toplevel', check=False)
    if top.returncode or Path(top.stdout.strip()).resolve() != directory.resolve():
        raise ValueError('Nonempty source directory is not its own Git checkout: ' + path)
    if git(directory, 'rev-parse', 'HEAD').stdout.strip() != source['commit']:
        raise ValueError('Populated source checkout differs from its pin: ' + path)
    if git(directory, 'status', '--porcelain', '--untracked-files=all').stdout:
        raise ValueError('Dirty source checkout: ' + path)


def stage_missing(root: Path, names: list[str]) -> list[dict]:
    """Preflight every input before one atomic Git-index update."""
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        raise ValueError('Gitlink registration is local-only; commit the fix before Actions runs')
    sources = [get_source(root, name) for name in names]
    repository, _ = repository_location(root, sources[0]['vendorPath'])
    prefix = root.resolve().relative_to(repository)
    tracked_inputs = ['.gitmodules', str(prefix / 'config/sources.json')]
    tracked_inputs.extend(str(prefix / source['pinFile']) for source in sources)
    diff = git(repository, 'diff', '--exit-code', '--', *tracked_inputs, check=False)
    if diff.returncode:
        raise ValueError('Stage the reviewed registry, pins and .gitmodules before registering gitlinks')
    pending = []
    for source in sources:
        patch_series(root, source)
        _, path = repository_location(root, source['vendorPath'])
        check_registration(repository, path, source)
        actual = index_entries(repository, path)
        wanted = {'mode': '160000', 'commit': source['commit'], 'stage': '0', 'path': path}
        if actual == [wanted]:
            continue
        if actual:
            raise ValueError(f'{path}: existing index entry differs; refusing to overwrite {actual}')
        _, committed = committed_entry(root, source['vendorPath'])
        if committed is not None:
            raise ValueError(f'{path}: removal of a committed entry is not a missing-patch migration')
        check_existing_directory(repository, path, source)
        pending.append(wanted)
    if pending:
        records = ''.join(f"160000 {item['commit']}\t{item['path']}\0" for item in pending)
        subprocess.run(['git', '--no-replace-objects', 'update-index', '-z', '--index-info'],
                       cwd=repository, input=records, text=True, check=True, timeout=30)
        for item in pending:
            if index_entries(repository, item['path']) != [item]:
                raise ValueError('Index registration verification failed')
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'llama-cpp')
    parser.add_argument('--source', action='append')
    parser.add_argument('--stage-missing', action='store_true')
    args = parser.parse_args()
    try:
        names = args.source or list(load_sources(args.root)['sources'])
        if not names or len(set(names)) != len(names):
            raise ValueError('Select distinct configured sources')
        if args.stage_missing:
            pending = stage_missing(args.root, names)
            print(json.dumps({'staged': pending, 'next': 'Review git diff --cached and commit. '
                              'HEAD validation remains strict until that commit exists.'}, indent=2))
        else:
            errors = []
            for name in names:
                source = get_source(args.root, name)
                try:
                    check_gitlink(args.root, source)
                except ValueError as error:
                    errors.append(str(error))
            if errors:
                raise ValueError('\n'.join(errors))
            print(json.dumps({'checkedRevision': 'HEAD', 'sources': names, 'valid': True}))
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
