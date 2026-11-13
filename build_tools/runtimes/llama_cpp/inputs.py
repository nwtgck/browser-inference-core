"""Capture selected llama build inputs before CMake or any patch application.

The upstream tree always comes from its pinned Git objects. A clean parent uses
committed objects too; an explicitly dirty local build copies its current owned
inputs and remains unpublishable. Original worktree edits after capture cannot
change these inputs. This is an ownership boundary, not a security sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from build_tools.github import full_sha
from build_tools.output import output_directories, preserve_unchanged
from build_tools.source.checkout import Checkout, git
from build_tools.source.files import absolute, directory
from build_tools.source.snapshot import export_commit
from build_tools.source.inputs import copy_local, file_identities, tree_identity

RUNTIME = 'runtimes/llama-cpp'
UPSTREAM = RUNTIME + '/vendor/llama.cpp'
PATCH_DIRECTORY = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'
INCLUDE = ('build_tools', 'toolchain/config.json', RUNTIME)
EXCLUDE = (RUNTIME + '/vendor',)


@dataclass(frozen=True)
class Captured:
    root: Path
    receipt: dict
    build_info: dict | None = None

    @property
    def runtime(self) -> Path:
        return self.root / 'repository' / RUNTIME

    @property
    def upstream(self) -> Path:
        return self.root / 'repository' / UPSTREAM


def _copy_local(repository: Path, destination: Path) -> None:
    copy_local(repository, destination, include=INCLUDE, exclude=EXCLUDE)


def capture(repository: Path, revision: str, upstream: Checkout, output: Path, *, dirty: bool = False) -> Captured:
    repository, output = directory(repository), absolute(output)
    full_sha(revision)
    full_sha(upstream.commit)
    if git(repository, 'cat-file', '-t', revision).strip() != 'commit':
        raise ValueError('Captured parent identity must be a commit')
    owned_build = repository / RUNTIME / 'build'
    if output == repository or output in repository.parents or (output.is_relative_to(repository) and not output.is_relative_to(owned_build)):
        raise ValueError('Captured inputs must not replace source-owned files')
    if output == upstream.path or output.is_relative_to(upstream.path) or output in upstream.path.parents:
        raise ValueError('Captured inputs must remain outside the upstream checkout')
    # Derive the same version values as upstream CMake while Git is available.
    build_info = {'commit': git(upstream.path, 'rev-parse', '--short', upstream.commit).strip(),
                  'number': int(git(upstream.path, 'rev-list', '--count', upstream.commit).strip())}
    with output_directories(output, replace=True, reuse_if=preserve_unchanged) as (prepared,):
        exported = prepared / 'repository'
        if dirty:
            _copy_local(repository, exported)
        else:
            export_commit(repository, revision, exported, include=INCLUDE, exclude=EXCLUDE)
        source = exported / UPSTREAM
        source.parent.mkdir(parents=True, exist_ok=True)
        export_commit(upstream.path, upstream.commit, source)
        owned = file_identities(exported, exclude=(UPSTREAM,))
        receipt = {'formatVersion': 1, 'sourceCommit': revision,
                   'parentInputMode': 'dirty-local-worktree' if dirty else 'commit-objects',
                   'repositoryTree': tree_identity(owned),
                   'upstream': {'commit': upstream.commit, 'tree': tree_identity(file_identities(source))},
                   'patchFiles': {name: value for name, value in owned.items()
                                  if name.startswith(RUNTIME + '/' + PATCH_DIRECTORY + '/') and name.endswith('.patch')}}
        # Commit count and abbreviation reflect available history/config, so
        # they are observed build settings rather than source-tree identities.
        (prepared / 'build-info.json').write_text(json.dumps(build_info, sort_keys=True) + '\n')
        (prepared / 'inputs.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    return Captured(output, receipt, build_info)


def prepared_sources(build: Path, *, webgpu: bool, file_rows: bool) -> dict:
    """Observe actual generated source files used by this completed build."""
    result = {}
    active = ['model-graph-chat-overlay', 'qsa-selection-overlay', 'mtmd-source-overlay']
    if webgpu:
        active += ['webgpu-source-overlay']
    if file_rows:
        active.append('file-backed-rows-overlay')
    for relative in active:
        root = build / relative
        if root.exists():
            result[relative] = file_identities(directory(root))
    return result


def verify(captured: Captured) -> None:
    """Reject changed captured input bytes before writing successful provenance."""
    if captured.build_info is not None and json.loads((captured.root / 'build-info.json').read_text()) != captured.build_info:
        raise ValueError('Captured build settings changed during build')
    repository = captured.root / 'repository'
    if tree_identity(file_identities(repository, exclude=(UPSTREAM,))) != captured.receipt['repositoryTree']:
        raise ValueError('Captured repository inputs changed during build')
    if tree_identity(file_identities(captured.upstream)) != captured.receipt['upstream']['tree']:
        raise ValueError('Captured upstream inputs changed during build')
