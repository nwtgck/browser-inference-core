"""Materialize pinned Git objects, without worktree filters or archive attributes.

A clean status is useful feedback, not proof of the bytes being compiled. Reading
objects also avoids assume-unchanged, skip-worktree, replacement refs and a file
changing between status and copy. Nested trees use their parent's exact gitlink.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile

from build_tools.source.checkout import git, git_environment
from build_tools.source.files import directory, relative_path


# Preserve the existing prepared-source boundary, not an export-ignore filter.
IGNORED_NAMES = frozenset({".git", "build", ".github"})


def _entries(checkout: Path, commit: str, include: tuple[str, ...] | None = None,
             exclude: tuple[str, ...] = ()) -> list[tuple[str, str, str, str]]:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected a complete pinned source commit")
    if git(checkout, "cat-file", "-t", commit).strip() != "commit":
        raise ValueError("Source identity is not a commit")
    for value in (*(include or ()), *exclude):
        relative_path(value)
    if include == ():
        raise ValueError("Expected at least one included source path")
    selected = set()
    result = []
    for row in git(checkout, "ls-tree", "-rz", "--full-tree", commit).split("\0"):
        if not row:
            continue
        metadata, name = row.split("\t", 1)
        mode, kind, oid = metadata.split()
        parts = relative_path(name)
        matching = {path for path in include or () if name == path or name.startswith(path + '/')}
        if include is not None and not matching:
            continue
        selected.update(matching)
        if any(name == path or name.startswith(path + '/') for path in exclude):
            continue
        if any(part in IGNORED_NAMES for part in parts.parts):
            continue
        if (mode, kind) not in {("100644", "blob"), ("100755", "blob"),
                              ("120000", "blob"), ("160000", "commit")}:
            raise ValueError(f"Unsupported source tree entry: {name}: {mode} {kind}")
        if not re.fullmatch(r"[0-9a-f]{40}", oid):
            raise ValueError(f"Invalid source object identity: {name}")
        result.append((mode, kind, oid, name))
    if include is not None and selected != set(include):
        raise ValueError("Missing committed source selection: " + ", ".join(sorted(set(include) - selected)))
    return result


def _export(checkout: Path, commit: str, destination: Path, *,
            include: tuple[str, ...] | None = None, exclude: tuple[str, ...] = ()) -> None:
    checkout = directory(checkout)
    if Path(git(checkout, "rev-parse", "--show-toplevel").strip()).resolve() != checkout:
        raise ValueError(f"Initialize the exact source checkout: {checkout}")
    entries = _entries(checkout, commit, include, exclude)
    destination.mkdir()
    # One reader per repository, not a subprocess per source file. Object reads
    # bypass checkout/smudge filters and .gitattributes export-ignore/export-subst.
    with tempfile.TemporaryFile() as errors, subprocess.Popen(
        ["git", "-c", "protocol.allow=never", "-C", str(checkout), "cat-file", "--batch"],
        env=git_environment(), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=errors,
    ) as reader:
        assert reader.stdin is not None and reader.stdout is not None
        try:
            for mode, kind, oid, name in entries:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if kind == "commit":
                    _export(checkout / name, oid, target)
                    continue
                reader.stdin.write((oid + "\n").encode("ascii"))
                reader.stdin.flush()
                fields = reader.stdout.readline().split()
                if len(fields) != 3 or fields[:2] != [oid.encode(), b"blob"] or not fields[2].isdigit():
                    raise ValueError(f"Missing or invalid local source blob: {name}")
                remaining = int(fields[2])
                if mode == "120000":
                    # A link is a small pathname; never allocate an arbitrary blob
                    # as a target string. No link is followed while writing files.
                    if remaining > 65536:
                        raise ValueError(f"Invalid source link: {name}")
                    raw = reader.stdout.read(remaining)
                    if len(raw) != remaining or not raw or b"\0" in raw:
                        raise ValueError(f"Invalid source link: {name}")
                    link = os.fsdecode(raw)
                    if Path(link).is_absolute():
                        raise ValueError(f"Source link escapes its tree: {name}")
                    target.symlink_to(link)
                else:
                    with target.open("xb") as stream:
                        while remaining:
                            block = reader.stdout.read(min(remaining, 1024 * 1024))
                            if not block:
                                raise ValueError(f"Truncated source blob: {name}")
                            stream.write(block)
                            remaining -= len(block)
                    target.chmod(0o755 if mode == "100755" else 0o644)
                if reader.stdout.read(1) != b"\n":
                    raise ValueError(f"Invalid source object boundary: {name}")
            reader.stdin.close()
            if reader.wait() != 0:
                errors.seek(0)
                raise ValueError("Source object reader failed: " + errors.read().decode(errors="replace"))
        except BaseException:
            # Do not wait for a child blocked on an unread blob after an error.
            reader.kill()
            raise

    # Validate after all entries exist so link chains and nested inputs are
    # checked as they will actually appear, without following them during copy.
    for current, directories, files in os.walk(destination, followlinks=False):
        for name in directories + files:
            path = Path(current) / name
            if path.is_symlink():
                try:
                    resolved = path.resolve()
                except (OSError, RuntimeError) as error:
                    raise ValueError(f"Invalid source link: {path}") from error
                if not resolved.is_relative_to(destination):
                    raise ValueError(f"Source link escapes its tree: {path}")


def export_commit(checkout: Path, commit: str, destination: Path, *,
                  include: tuple[str, ...] | None = None, exclude: tuple[str, ...] = ()) -> None:
    """Export into an absent private directory; the caller owns publication.

    Initialized nested repositories are required, but no worktree bytes, HEADs
    or index entries select their contents: the parent's gitlinks do that.
    Optional selections name exact files or whole subtrees in this repository;
    a selected gitlink exports its complete nested tree. Exclusions apply before
    entering gitlinks, so unrelated runtime dependencies need not be initialized.
    """
    _export(checkout, commit, destination, include=include, exclude=exclude)
