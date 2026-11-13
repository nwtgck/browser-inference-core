"""Transactional source composition shared by independent runtimes."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import stat
from typing import Mapping

from build_tools.output import output_directories, preserve_unchanged as _preserve_unchanged
from build_tools.source.files import absolute, directory, relative_path
from build_tools.source.patches import apply_patch, read_series
from build_tools.source.snapshot import export_commit


# Build products, repository internals and workflow files are not build inputs.
# This is the existing source-copy boundary, not a new size-based exclusion list.
_IGNORED_NAMES = frozenset({".git", "build", ".github"})


def _inspect_source(root: Path) -> None:
    for current, directories, files in os.walk(root, followlinks=False):
        directories[:] = [name for name in directories if name not in _IGNORED_NAMES]
        for name in directories + [name for name in files if name not in _IGNORED_NAMES]:
            path = Path(current) / name
            mode = path.lstat().st_mode
            if stat.S_ISLNK(mode):
                link = os.readlink(path)
                if Path(link).is_absolute() or not path.resolve().is_relative_to(root):
                    raise ValueError(f"Source link escapes its tree: {path}")
            elif not (stat.S_ISDIR(mode) or stat.S_ISREG(mode)):
                raise ValueError(f"Non-regular source input: {path}")


def _copy_selected(source: Path, destination: Path, files: tuple[str, ...]) -> None:
    destination.mkdir()
    if len(set(files)) != len(files):
        raise ValueError("Duplicate selected source paths")
    for relative in sorted(files):
        parts = relative_path(relative)
        if any(name in _IGNORED_NAMES for name in parts.parts):
            continue
        original = source / parts
        directory(original.parent)
        mode = original.lstat().st_mode
        target = destination / parts
        target.parent.mkdir(parents=True, exist_ok=True)
        if stat.S_ISLNK(mode):
            link = os.readlink(original)
            if Path(link).is_absolute() or not original.resolve().is_relative_to(source):
                raise ValueError(f"Source link escapes its tree: {original}")
            target.symlink_to(link)
        elif stat.S_ISREG(mode):
            shutil.copy2(original, target)
        else:
            raise ValueError(f"Non-regular selected source input: {original}")


def prepare_sources(sources: Mapping[str, Path], destination: Path, patches: Path,
                    *, source_files: Mapping[str, tuple[str, ...]] | None = None,
                    source_commits: Mapping[str, str] | None = None,
                    upstreams: Mapping[str, dict] | None = None,
                    replace: bool = False) -> dict:
    """Copy once per role; only publish the complete, successfully patched tree.

    Roles and upstream revisions are selected by the runtime adapter. This
    mechanism neither downloads dependencies nor invents semantic compatibility.
    The output is exclusively owned by this invocation, not a shared mutable cache.
    """
    sources = {name: directory(path) for name, path in sources.items()}
    destination = absolute(destination)
    if source_commits is not None:
        source_commits = dict(source_commits)
    if not sources:
        raise ValueError("A prepared workspace needs at least one source")
    for name, source in sources.items():
        # Validate roles independently of whether a patch targets them.
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in name) or not name[0].isalpha():
            raise ValueError("Invalid source role")
        if destination == source or destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("Prepared source must be outside and not contain an upstream checkout")
    if len(set(sources.values())) != len(sources):
        raise ValueError("Independent source roles must not alias the same directory")
    series = read_series(patches / "series.json", targets=sources)
    if source_files is not None and set(source_files) != set(sources):
        raise ValueError("Selected source paths must cover exactly the source roles")
    if source_commits is not None:
        if source_files is not None or set(source_commits) != set(sources):
            raise ValueError("Pinned commits must select exactly the source roles, not worktree files")
        if upstreams is None:
            upstreams = {role: {"commit": commit} for role, commit in source_commits.items()}
        if set(upstreams) != set(sources) or any(upstreams[role].get("commit") != commit
                                                for role, commit in source_commits.items()):
            raise ValueError("Source provenance differs from the selected commits")
    elif upstreams is not None:
        raise ValueError("Upstream provenance requires pinned commit inputs")
    # Snapshot serializable metadata before composition; caller mutation cannot
    # change the identity recorded after the input has already been selected.
    recorded_upstreams = json.loads(json.dumps(upstreams)) if upstreams is not None else None
    if source_files is None and source_commits is None:
        for source in sources.values():
            _inspect_source(source)
    records = []
    with output_directories(destination, replace=replace, reuse_if=_preserve_unchanged) as (work,):
        for name, source in sources.items():
            if source_commits is not None:
                export_commit(source, source_commits[name], work / name)
            elif source_files is None:
                shutil.copytree(source, work / name, ignore=shutil.ignore_patterns(*_IGNORED_NAMES), symlinks=True)
            else:
                _copy_selected(source, work / name, source_files[name])
        for patch in series.patches:
            records.append(apply_patch(work / patch.target, patches, patch))
        report = {"inventorySha256": series.sha256, "patches": records, "vendorCheckoutModified": False}
        if recorded_upstreams is not None:
            report["upstreams"] = recorded_upstreams
        # The receipt and sources have one completion boundary. A failed write
        # leaves neither a partial destination nor a destroyed previous build.
        (work / "patch-provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
