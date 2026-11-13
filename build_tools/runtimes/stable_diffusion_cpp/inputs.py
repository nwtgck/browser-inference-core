"""Capture image-runtime CMake, binding and patch inputs before compilation.

The running driver is bootstrap code. Native sources, CMake files, patch bodies
and subprocess Python modules come from the captured parent selection. This is
an ownership boundary for ordinary builds, not a concurrent-writer sandbox.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from build_tools.output import output_directories, preserve_unchanged
from build_tools.source.checkout import git
from build_tools.source.files import absolute, directory, regular_file
from build_tools.source.inputs import copy_local, file_identities, tree_identity
from build_tools.source.snapshot import export_commit

RUNTIME = 'runtimes/stable-diffusion-cpp'
INCLUDE = ('build_tools', 'toolchain/config.json', RUNTIME)
EXCLUDE = (RUNTIME + '/vendor',)


@dataclass(frozen=True)
class Captured:
    root: Path
    receipt: dict

    @property
    def repository(self) -> Path:
        return self.root / 'repository'

    @property
    def runtime(self) -> Path:
        return self.repository / RUNTIME


def capture(repository: Path, revision: str, output: Path, *, dirty: bool = False) -> Captured:
    repository, output = directory(repository), absolute(output)
    if (not re.fullmatch(r'[0-9a-f]{40}', revision) or
            git(repository, 'cat-file', '-t', revision).strip() != 'commit'):
        raise ValueError('Captured parent identity must be a complete commit')
    owned_build = repository / RUNTIME / 'build'
    if (output == repository or output in repository.parents or
            (output.is_relative_to(repository) and not output.is_relative_to(owned_build))):
        raise ValueError('Captured inputs must not replace source-owned files')
    with output_directories(output, replace=True, reuse_if=preserve_unchanged) as (prepared,):
        exported = prepared / 'repository'
        if dirty:
            copy_local(repository, exported, include=INCLUDE, exclude=EXCLUDE)
        else:
            export_commit(repository, revision, exported, include=INCLUDE, exclude=EXCLUDE)
        files = file_identities(exported)
        receipt = {'formatVersion': 1, 'sourceCommit': revision,
                   'parentInputMode': 'dirty-local-worktree' if dirty else 'commit-objects',
                   'repositoryTree': tree_identity(files),
                   'patchFiles': {name: value for name, value in files.items()
                                  if name.startswith(RUNTIME + '/upstream-patches/') and name.endswith('.patch')}}
        (prepared / 'inputs.json').write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    return Captured(output, receipt)


def verify(captured: Captured) -> None:
    if tree_identity(file_identities(captured.repository)) != captured.receipt['repositoryTree']:
        raise ValueError('Captured image-runtime inputs changed during build')


def reset_legacy_cmake_cache(build: Path, source: Path) -> None:
    """Retire a cache that still points at the former live source directory."""
    cache = build / 'CMakeCache.txt'
    if not cache.exists():
        return
    home = re.search(r'^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$', regular_file(cache).read_text(), re.MULTILINE)
    if home is None or Path(home[1]).resolve() == source:
        return
    cache.unlink()
    metadata = build / 'CMakeFiles'
    if metadata.exists():
        import shutil
        shutil.rmtree(directory(metadata))
