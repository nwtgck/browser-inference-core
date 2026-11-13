"""Validate and unpack the already acquired Dawn archive; never download it."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import stat
import tempfile
import zipfile

from build_tools.source.files import directory, relative_path


def _members(archive: zipfile.ZipFile) -> list[tuple[zipfile.ZipInfo, str]]:
    selected = []
    names: set[str] = set()
    files: set[str] = set()
    for info in archive.infolist():
        name = info.filename[:-1] if info.is_dir() else info.filename
        try:
            relative_path(name)
        except (ValueError, TypeError) as error:
            raise RuntimeError('Unsafe Dawn archive member: ' + info.filename) from error
        mode = info.external_attr >> 16
        kind = stat.S_IFMT(mode)
        if (not (name == 'emdawnwebgpu_pkg' or name.startswith('emdawnwebgpu_pkg/'))
                or kind not in (0, stat.S_IFREG, stat.S_IFDIR)
                or name in names or (info.is_dir() and kind == stat.S_IFREG)
                or (not info.is_dir() and kind == stat.S_IFDIR)):
            raise RuntimeError('Unsafe Dawn archive member: ' + info.filename)
        names.add(name)
        if not info.is_dir():
            files.add(name)
        selected.append((info, name))
    if 'emdawnwebgpu_pkg/emdawnwebgpu.port.py' not in files:
        raise RuntimeError('Dawn archive does not contain the expected port')
    for name in names:
        if any(str(parent) in files for parent in Path(name).parents):
            raise RuntimeError('Unsafe Dawn archive file/directory overlap: ' + name)
    return selected


def prepare_dawn(config: dict, tools: Path) -> None:
    """Verify every restore, stage privately, and replace only after validation.

    Downloads are explicit workflow inputs. A missing/corrupt archive never
    triggers a fallback network request or destroys an existing unpacked port.
    """
    tools = directory(tools)
    downloads = tools / 'downloads'
    if downloads.is_symlink():
        raise RuntimeError('Refusing a linked download directory')
    archive = downloads / f'emdawnwebgpu_pkg-{config["dawnTag"]}.zip'
    if archive.is_symlink():
        raise RuntimeError('Refusing a linked Dawn archive')
    if not archive.is_file():
        raise RuntimeError(f'Missing pinned Dawn archive: {archive}; acquire it before offline preparation')
    package = tools / 'emdawnwebgpu_pkg'
    if package.is_symlink():
        raise RuntimeError('Refusing a linked Dawn package')
    if package.exists() and not package.is_dir():
        raise RuntimeError('Expected a Dawn package directory')
    work = Path(tempfile.mkdtemp(prefix='dawn-prepare-', dir=tools))
    keep_recovery = False
    try:
        # Hash and unpack the same private copy, not a mutable cache entry twice.
        private_archive = work / 'input.zip'
        shutil.copyfile(archive, private_archive)
        with private_archive.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != config['dawnSha256']:
                raise RuntimeError('Dawn archive checksum mismatch; refusing extraction')
        staged = work / 'unpacked'
        staged.mkdir()
        with zipfile.ZipFile(private_archive) as source:
            for info, name in _members(source):
                target = staged / name
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with source.open(info) as src, target.open('xb') as dst:
                        shutil.copyfileobj(src, dst)
                    # Preserve executability, not owner/group/set-id bits.
                    target.chmod(0o755 if info.external_attr >> 16 & 0o111 else 0o644)
        old = work / 'previous'
        if package.exists():
            os.replace(package, old)
        try:
            os.replace(staged / 'emdawnwebgpu_pkg', package)
        except BaseException:
            if old.exists():
                try:
                    os.replace(old, package)
                except BaseException as restore_error:
                    keep_recovery = True
                    raise RuntimeError(f'Dawn activation and recovery failed; previous package retained at {old}') from restore_error
            raise
    finally:
        if not keep_recovery:
            shutil.rmtree(work)
