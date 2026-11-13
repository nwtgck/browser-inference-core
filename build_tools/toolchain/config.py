"""Shared browser compiler pins; runtime upstream pins are deliberately separate."""
from __future__ import annotations
import json
from pathlib import Path

from build_tools.project import REPOSITORY_ROOT as ROOT
from build_tools.source.checkout import commit_file

def load_toolchain(repo: Path = ROOT, *, revision: str | None = None) -> dict:
    text = ((repo / 'toolchain/config.json').read_text() if revision is None
            else commit_file(repo, revision, 'toolchain/config.json'))
    return json.loads(text)

def runtime_toolchain(runtime: Path, *, revision: str | None = None) -> dict:
    """Keep the existing llama provenance schema without duplicating shared pins."""
    repository = runtime.parents[1]
    shared = load_toolchain(repository, revision=revision)
    relative = (runtime / 'config/toolchain.json').relative_to(repository).as_posix()
    local = json.loads((runtime / 'config/toolchain.json').read_text() if revision is None
                       else commit_file(repository, revision, relative))
    if shared.keys() & local.keys():
        raise ValueError('Runtime configuration must not override shared toolchain pins')
    return {**shared, **local}
