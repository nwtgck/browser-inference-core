"""Prepare complete local outputs before replacing their previous directories.

This provides rollback for ordinary filesystem failures, not a cross-directory
atomic view for concurrent readers or isolation from other workspace writers.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Callable, Iterator

from build_tools.source.files import absolute, directory


def _identity(path: Path) -> tuple[int, int] | None:
    if path.is_symlink():
        raise ValueError('Output destination must not already exist as a link: ' + str(path))
    if not path.exists():
        return None
    current = directory(path).stat()
    return current.st_dev, current.st_ino


@dataclass
class _Output:
    destination: Path
    temporary: Path
    original: tuple[int, int] | None
    moved: bool = False
    installed: tuple[int, int] | None = None
    reuse: bool = False
    retain: bool = False

    @property
    def prepared(self) -> Path:
        return self.temporary / 'prepared'

    @property
    def previous(self) -> Path:
        return self.temporary / 'previous'

    def check_original(self) -> None:
        current = _identity(self.destination)
        if current != self.original:
            change = 'disappeared' if current is None else 'changed or appeared'
            raise ValueError(f'Output destination {change} during preparation: {self.destination}')


@contextmanager
def output_directories(*destinations: Path, replace: bool = False,
                       reuse_if: Callable[[Path, Path], bool] | None = None) -> Iterator[tuple[Path, ...]]:
    """Yield empty private sibling directories; activate them only after success.

    Callers write and validate every payload inside the context. A later failed
    activation restores earlier outputs in the same group. If restoration also
    fails, the recovery directory is retained and named in the raised error.
    """
    destinations = tuple(absolute(path) for path in destinations)
    if not destinations or any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
                               for index, a in enumerate(destinations) for b in destinations[index + 1:]):
        raise ValueError('Output destinations must be nonempty, distinct and non-overlapping')
    outputs = []
    try:
        for destination in destinations:
            directory(destination.parent, create=True)
            original = _identity(destination)
            if original is not None and not replace:
                raise ValueError('Output destination must not already exist: ' + str(destination))
            temporary = Path(tempfile.mkdtemp(prefix='bicore-output-', dir=destination.parent))
            output = _Output(destination, temporary, original)
            outputs.append(output)
            output.prepared.mkdir(mode=0o700)
        yield tuple(output.prepared for output in outputs)
        # Check the whole group before moving anything. Recheck after callbacks
        # and immediately before each move; a callback cannot silently replace a
        # destination and then authorize reuse of that unrelated directory.
        for output in outputs:
            output.check_original()
        for output in outputs:
            if output.original is not None and reuse_if is not None:
                output.reuse = reuse_if(output.prepared, output.destination)
        for output in outputs:
            output.check_original()
        try:
            for output in outputs:
                output.check_original()
                if output.reuse:
                    continue
                prepared_identity = _identity(output.prepared)
                if prepared_identity is None:
                    raise ValueError('Prepared output disappeared: ' + str(output.prepared))
                if output.original is not None:
                    output.destination.rename(output.previous)
                    output.moved = True
                output.prepared.rename(output.destination)
                output.installed = prepared_identity
        except BaseException:
            failures = []
            for output in reversed(outputs):
                if output.installed is None and not output.moved:
                    continue
                try:
                    if output.installed is not None:
                        if _identity(output.destination) != output.installed:
                            raise ValueError('Activated output changed before recovery')
                        output.destination.rename(output.prepared)
                    if output.moved:
                        if _identity(output.destination) is not None:
                            raise ValueError('Output destination appeared before recovery')
                        output.previous.rename(output.destination)
                except BaseException as recovery_error:
                    output.retain = True
                    failures.append((output, recovery_error))
            if failures:
                retained = ', '.join(str(output.previous if output.previous.exists() else output.temporary)
                                     for output, _ in failures)
                raise RuntimeError('Output activation and recovery failed; previous output retained at ' + retained) from failures[0][1]
            raise
    finally:
        for output in outputs:
            if not output.retain:
                shutil.rmtree(output.temporary)


def _tree_entries(root: Path) -> dict[str, Path]:
    entries = {}
    for current, directories, files in os.walk(root, followlinks=False):
        for name in directories + files:
            path = Path(current) / name
            entries[path.relative_to(root).as_posix()] = path
    return entries


def _same_bytes(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open('rb') as a, right.open('rb') as b:
        while block := a.read(1024 * 1024):
            if block != b.read(len(block)):
                return False
        return b.read(1) == b''


def preserve_unchanged(work: Path, previous: Path) -> bool:
    """Reuse an identical tree and keep equal files' mtimes in a changed tree.

    The prepared tree never hardlinks the previous one. Compare actual outputs
    so exporting unchanged Git objects does not force native recompilation.
    """
    new, old = _tree_entries(work), _tree_entries(previous)
    identical = new.keys() == old.keys() and stat.S_IMODE(work.stat().st_mode) == stat.S_IMODE(previous.stat().st_mode)
    for name, path in new.items():
        earlier = old.get(name)
        if earlier is None:
            identical = False
            continue
        before, after = earlier.lstat(), path.lstat()
        equal = before.st_mode == after.st_mode
        if stat.S_ISREG(after.st_mode):
            equal = equal and _same_bytes(path, earlier)
        elif stat.S_ISLNK(after.st_mode):
            equal = equal and os.readlink(path) == os.readlink(earlier)
        elif not stat.S_ISDIR(after.st_mode):
            equal = False
        if equal and not stat.S_ISDIR(after.st_mode):
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns), follow_symlinks=False)
        elif not equal:
            identical = False
            if stat.S_ISREG(after.st_mode):
                os.utime(path, follow_symlinks=False)
    return identical
