"""Checked package inputs; reject links before copies can dereference them."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat


def parse_json(raw: bytes):
    """Parse identity-bearing metadata without last-key-wins or non-JSON numbers."""
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError('Invalid JSON number: ' + value)

    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)


def regular_path(path: Path, *, directory: bool = False) -> Path:
    # absolute(), unlike resolve(), does not hide a linked input or its parents.
    path = Path(os.path.abspath(path))
    for parent in reversed(path.parents):
        if not stat.S_ISDIR(parent.lstat().st_mode):
            raise ValueError('Linked or non-directory input parent: ' + str(parent))
    mode = path.lstat().st_mode
    if not (stat.S_ISDIR(mode) if directory else stat.S_ISREG(mode)):
        raise ValueError('Linked or non-regular package input: ' + str(path))
    return path


def read_regular(path: Path, *, limit: int = 4 * 1024 * 1024) -> bytes:
    path = regular_path(path)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('Non-regular opened input')
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Oversized metadata input: ' + str(path))
    return raw


def file_identity(path: Path) -> dict:
    path = regular_path(path)
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
    with os.fdopen(os.open(path, flags), 'rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('Non-regular opened input')
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = os.fstat(stream.fileno())
    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError('Package input changed while reading: ' + str(path))
    return {'bytes': after.st_size, 'sha256': digest,
            'executable': bool(after.st_mode & 0o111)}


def tree_identity(root: Path) -> dict[str, dict]:
    root = regular_path(root, directory=True)
    entries = {}; folded = set()
    for path in sorted(root.rglob('*')):
        name = path.relative_to(root).as_posix()
        if ('\\' in name or any(ord(c) < 32 for c in name) or
                name.casefold() in folded):
            raise ValueError('Unsafe or case-colliding package path: ' + repr(name))
        folded.add(name.casefold())
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            entries[name] = {'directory': True}
        elif stat.S_ISREG(mode):
            entries[name] = file_identity(path)
        else:
            raise ValueError('Linked or non-regular package input: ' + str(path))
    return entries


def copy_regular(source: Path, destination: Path) -> None:
    expected = file_identity(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    regular_path(destination.parent, directory=True)
    if destination.exists() or destination.is_symlink():
        regular_path(destination)
    shutil.copy2(source, destination, follow_symlinks=False)
    if file_identity(destination) != expected:
        raise ValueError('Copied package input changed: ' + str(source))


def copy_regular_tree(source: Path, destination: Path) -> None:
    expected = tree_identity(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    regular_path(destination.parent, directory=True)
    # Preserve links, rather than following a link substituted after inspection;
    # the destination check will reject it without reading outside the snapshot.
    shutil.copytree(source, destination, symlinks=True)
    if tree_identity(destination) != expected:
        raise ValueError('Copied package tree changed: ' + str(source))


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True).encode()).hexdigest()
