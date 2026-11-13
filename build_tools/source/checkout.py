"""Resolve local source identities from the index or a captured commit."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
from typing import Mapping

from build_tools.source.files import directory, relative_path


@dataclass(frozen=True)
class Checkout:
    path: Path
    commit: str


def git_environment() -> dict[str, str]:
    # Repository reads must not inherit another worktree, a credential helper,
    # global fsmonitor, or an implicit partial-clone network fetch.
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update({"GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": os.devnull,
                "GIT_TERMINAL_PROMPT": "0", "GIT_NO_LAZY_FETCH": "1",
                "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_REPLACE_OBJECTS": "1"})
    return env


def git(directory: Path, *args: str) -> str:
    result = subprocess.run(["git", "-c", "protocol.allow=never", "-c", "core.fsmonitor=false",
                             "-C", str(directory), *args], env=git_environment(), capture_output=True, text=True)
    if result.returncode:
        raise ValueError("Local source check failed: " + result.stderr.strip())
    return result.stdout


def _check_commit(repository: Path, revision: str) -> None:
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise ValueError('Expected a complete pinned source commit')
    if git(repository, 'cat-file', '-t', revision).strip() != 'commit':
        raise ValueError('Source identity is not a commit')


def commit_file(repository: Path, revision: str, relative: str) -> str:
    """Read a regular UTF-8 configuration blob from an immutable parent commit."""
    repository = directory(repository)
    relative_path(relative)
    _check_commit(repository, revision)
    fields = git(repository, 'ls-tree', '-z', revision, '--', relative).rstrip('\0').split('\t')
    if len(fields) != 2 or fields[1] != relative:
        raise ValueError('Missing or ambiguous committed input: ' + relative)
    mode, kind, oid = fields[0].split()
    if mode not in ('100644', '100755') or kind != 'blob':
        raise ValueError('Expected a regular committed input: ' + relative)
    return git(repository, 'cat-file', 'blob', oid)


def selected_gitlinks(repository: Path, paths: Mapping[str, str], *, revision: str | None = None) -> dict[str, str]:
    """Read committed or stage-zero identities without fetching a checkout."""
    repository = directory(repository)
    if Path(git(repository, "rev-parse", "--show-toplevel").strip()).resolve() != repository:
        raise ValueError("Expected the source repository root")
    if not paths or len(set(paths.values())) != len(paths):
        raise ValueError("Source roles must select distinct gitlinks")
    if revision is not None:
        _check_commit(repository, revision)
    result = {}
    for role, relative in paths.items():
        relative_path(relative)
        command = ('ls-files', '--stage', '-z') if revision is None else ('ls-tree', '-z', revision)
        fields = git(repository, *command, '--', relative).rstrip('\0').split('\t')
        if len(fields) != 2 or fields[1] != relative:
            raise ValueError(f"Missing or ambiguous source gitlink: {role}: {relative}")
        identity = fields[0].split()
        if len(identity) != 3:
            raise ValueError(f'Expected a stage-zero or committed source gitlink: {role}: {relative}')
        if revision is None:
            expected, commit = identity[0] == '160000' and identity[2] == '0', identity[1]
        else:
            expected, commit = identity[:2] == ['160000', 'commit'], identity[2]
        if not expected or not re.fullmatch(r'[0-9a-f]{40}', commit):
            raise ValueError(f'Expected a stage-zero or committed source gitlink: {role}: {relative}')
        result[role] = commit
    return result


def checked_sources(repository: Path, paths: Mapping[str, str], *, revision: str | None = None) -> dict[str, Checkout]:
    """Require exact, clean checkouts for the selected parent gitlinks.

    The index deliberately supports a locally staged upstream update. It is not
    proof that the parent source tree is clean/publishable; publication checks
    the parent source snapshot separately. No historical source-body hash is used.
    """
    repository = directory(repository)
    pins = selected_gitlinks(repository, paths, revision=revision)
    result = {}
    for role, relative in paths.items():
        path = repository / relative_path(relative)
        path = directory(path)
        metadata = path / ".git"
        if metadata.is_symlink() or not (metadata.is_file() or metadata.is_dir()):
            raise ValueError(f"Initialize the local upstream checkout: {role}: {relative}")
        if Path(git(path, "rev-parse", "--show-toplevel").strip()).resolve() != path:
            raise ValueError(f"Upstream path is not its own checkout: {role}")
        commit = git(path, "rev-parse", "HEAD").strip()
        if commit != pins[role]:
            raise ValueError(f"Upstream HEAD differs from the selected gitlink: {role}")
        status = git(path, "status", "--porcelain=v1", "--untracked-files=all", "--ignore-submodules=none")
        if status:
            raise ValueError(f"Dirty upstream checkout: {role}\n{status}")
        nested = git(path, "submodule", "status", "--recursive")
        if any(not line.startswith(" ") for line in nested.splitlines()):
            raise ValueError(f"Initialize the pinned nested source checkouts: {role}\n{nested}")
        result[role] = Checkout(path, commit)
    return result


def tracked_files(checkout: Path) -> tuple[str, ...]:
    """Enumerate local tracked inputs, including initialized nested submodules.

    Call after checked_sources. Ignored caches, local model files and temporary
    headers are not source inputs merely because they sit inside a clean checkout.
    """
    paths = tuple(git(directory(checkout), "ls-files", "--recurse-submodules", "-z").split("\0")[:-1])
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate tracked source paths")
    for path in paths:
        relative_path(path)
    return paths
