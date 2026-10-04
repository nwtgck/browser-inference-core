"""Read committed source identities; never repair a checkout during a build.

A directory, .gitmodules entry or staged gitlink is not a committed gitlink.
Use repository-relative, NUL-delimited Git plumbing so diagnostics identify the
actual source even from a subdirectory or a linked worktree.
"""
from __future__ import annotations

from pathlib import Path, PurePosixPath
import subprocess


def git(root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['git', '--no-replace-objects', *arguments], cwd=root, check=check,
        capture_output=True, text=True, timeout=30,
    )


def repository_location(root: Path, vendor: str) -> tuple[Path, str]:
    relative = PurePosixPath(vendor)
    if (relative.is_absolute() or '\\' in vendor or
            any(part in ('', '.', '..') for part in vendor.split('/'))):
        raise ValueError('Unsafe source vendor path')
    root = root.resolve()
    repository = Path(git(root, 'rev-parse', '--show-toplevel').stdout.strip()).resolve()
    path = (root.relative_to(repository) / vendor).as_posix()
    return repository, path


def parse_entry(record: str) -> dict | None:
    if not record:
        return None
    rows = record.rstrip('\0').split('\0')
    if len(rows) != 1:
        raise ValueError('Expected exactly one source tree entry')
    metadata, path = rows[0].split('\t', 1)
    mode, kind, commit = metadata.split()
    return {'mode': mode, 'type': kind, 'commit': commit, 'path': path}


def committed_entry(root: Path, vendor: str, revision: str = 'HEAD') -> tuple[str, dict | None]:
    repository, path = repository_location(root, vendor)
    tree = git(repository, 'rev-parse', '--verify', '--end-of-options',
               revision + '^{tree}').stdout.strip()
    entry = parse_entry(git(repository, 'ls-tree', '-z', '--full-tree', tree,
                            '--', path).stdout)
    if entry is not None and entry['path'] != path:
        raise ValueError('Unexpected source tree path')
    return path, entry


def check_gitlink(root: Path, source: dict, revision: str = 'HEAD') -> None:
    path, actual = committed_entry(root, source['vendorPath'], revision)
    expected = source['commit']
    if actual is not None and (actual['mode'], actual['type'], actual['commit']) == (
            '160000', 'commit', expected):
        return
    description = ('missing (not recorded in the committed tree)' if actual is None else
                   f"{actual['mode']} {actual['type']} {actual['commit']}")
    hint = (' A patch applied without --index does not register submodule commits. '
            'Check scripts/register_source_gitlinks.py; registration is an explicit '
            'local index change that must be committed, never a CI repair.' if actual is None else
            ' Do not bypass this check or replace the pin with the checked-out HEAD. '
            'Review the pin and committed gitlink together.')
    raise ValueError(
        f"Source gitlink/pin mismatch for {source.get('id', '<selected-source>')}: "
        f"{revision}:{path}; expected 160000 commit {expected} "
        f"from {source.get('pinFile', 'source pin')}; actual {description}." + hint
    )
