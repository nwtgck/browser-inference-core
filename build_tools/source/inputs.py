"""Small identities and local-copy primitives for owned build input snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import stat

from build_tools.source.checkout import git
from build_tools.source.files import directory, regular_file, relative_path
from build_tools.source.snapshot import IGNORED_NAMES


def identity(path: Path) -> dict:
    raw = regular_file(path).read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def file_identities(root: Path, *, exclude: tuple[str, ...] = ()) -> dict:
    root = directory(root)
    for name in exclude:
        relative_path(name)
    result = {}
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if any(relative == prefix or relative.startswith(prefix + '/') for prefix in exclude):
            continue
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            # The object exporter checks containment; identify a link itself,
            # without hashing its target twice or traversing directory aliases.
            result[relative] = {'symlink': str(path.readlink())}
        elif stat.S_ISREG(mode):
            result[relative] = identity(path)
        elif not stat.S_ISDIR(mode):
            raise ValueError('Non-regular captured source entry: ' + str(path))
    return result


def tree_identity(files: dict) -> dict:
    encoded = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
    return {'files': len(files), 'sha256': hashlib.sha256(encoded).hexdigest(),
            'encoding': 'UTF-8 compact JSON of sorted relative paths to file identities or relative symlink targets'}


def copy_local(repository: Path, destination: Path, *, include: tuple[str, ...],
               exclude: tuple[str, ...] = ()) -> None:
    """Copy selected tracked/unignored dirty inputs, never ignored build caches.

    This deliberately local development mode is not proof of a committed tree.
    Callers record it as dirty and refuse to publish its build outputs.
    """
    if not include:
        raise ValueError('Expected at least one included source path')
    for name in (*include, *exclude):
        relative_path(name)
    destination.mkdir()
    names = git(repository, 'ls-files', '--cached', '--others', '--exclude-standard',
                '--deduplicate', '-z').split('\0')[:-1]
    for name in sorted(set(names)):
        parts = relative_path(name)
        if not any(name == prefix or name.startswith(prefix + '/') for prefix in include):
            continue
        if any(name == prefix or name.startswith(prefix + '/') for prefix in exclude):
            continue
        if any(part in IGNORED_NAMES for part in parts.parts):
            continue
        source = repository / name
        if not source.exists() and not source.is_symlink():
            continue  # Local deletions are part of the dirty snapshot.
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(regular_file(source), target)
