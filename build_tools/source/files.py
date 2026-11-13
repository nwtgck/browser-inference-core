"""Small filesystem contracts for private, reproducible source workspaces.

These checks protect against accidental links and malformed inputs. They do not
claim isolation from another process concurrently modifying the same workspace.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import stat


def relative_path(value: str) -> PurePosixPath:
    """Accept a canonical, relative POSIX source path, never Git metadata."""
    if not isinstance(value, str) or not value or "\\" in value or any(ord(c) < 32 or ord(c) == 127 for c in value):
        raise ValueError(f"Invalid relative source path: {value!r}")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or str(path) != value or any(p in (".", "..") or p.casefold() == ".git" for p in path.parts):
        raise ValueError(f"Invalid relative source path: {value!r}")
    return path


def absolute(path: Path) -> Path:
    # resolve() would hide a linked root before it can be rejected.
    return Path(os.path.abspath(path))


def directory(path: Path, *, create: bool = False) -> Path:
    path = absolute(path)
    for part in (*reversed(path.parents), path):
        try:
            mode = part.lstat().st_mode
        except FileNotFoundError:
            if not create:
                raise ValueError(f"Missing source directory: {part}") from None
            # Independent profile preparations may create a shared parent.
            part.mkdir(exist_ok=True)
            mode = part.lstat().st_mode
        if not stat.S_ISDIR(mode):
            raise ValueError(f"Linked or non-directory source path: {part}")
    return path


def regular_file(path: Path) -> Path:
    path = absolute(path)
    directory(path.parent)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        raise ValueError(f"Missing source file: {path}") from None
    if not stat.S_ISREG(mode):
        raise ValueError(f"Linked or non-regular source file: {path}")
    return path


def file_digest(path: Path) -> str:
    path = regular_file(path)
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def optional_file_digest(root: Path, relative: str) -> str | None:
    path = root / relative_path(relative)
    # Reject a linked parent even when the final file does not exist.
    for parent in reversed(path.parents):
        if parent.is_relative_to(root) and parent.exists():
            directory(parent)
        elif parent.is_relative_to(root) and parent.is_symlink():
            raise ValueError(f"Linked source parent: {parent}")
    if not path.exists() and not path.is_symlink():
        return None
    return file_digest(path)
