#!/usr/bin/env python3
"""Apply an exact, ordered patch inventory to isolated build-tree copies."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent / 'scripts'))
from pipeline_metrics import measured, span

def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or linked input: {path}')
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def safe_file(root: Path, relative: str) -> Path:
    parts = Path(relative).parts
    if not relative or Path(relative).is_absolute() or '..' in parts or '\\' in relative:
        raise ValueError(f'Unsafe patch path: {relative}')
    result = root / relative
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError('Patch path escapes the source')
    return result

@measured('build.prepare_upstream', runtime='stable-diffusion-cpp')
def prepare(sources: dict[str, Path], destination: Path, patches: Path = ROOT / 'upstream-patches') -> dict:
    inventory = json.loads((patches / 'series.json').read_text())
    if inventory['formatVersion'] != 1 or set(sources) != {'stable-diffusion', 'ggml'}:
        raise ValueError('Unknown patch inventory/source set')
    if destination.exists() or destination.is_symlink():
        raise ValueError('Prepared source destination must not already exist')
    for source in sources.values():
        if destination.resolve().is_relative_to(source.resolve()):
            raise ValueError('Do not prepare inside an upstream checkout')
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix='sdb-prepare-', dir=destination.parent) as tmp:
        # Keep a container for TemporaryDirectory to clean; the completed child
        # can be renamed without copying the entire source tree a second time.
        work = Path(tmp) / 'prepared'
        work.mkdir(mode=0o700)  # Match the old TemporaryDirectory root mode.
        for name, source in sources.items():
            with span('build.copy_upstream', context=name):
                shutil.copytree(source, work / name, ignore=shutil.ignore_patterns('.git', 'build', '.github'), symlinks=True)
        for entry in inventory['patches']:
            if entry['target'] not in sources: raise ValueError('Unknown patch target')
            if not entry['files'] or len({f['path'] for f in entry['files']}) != len(entry['files']):
                raise ValueError('Empty or duplicate patch inventory')
            target = work / entry['target']
            patch = safe_file(patches, entry['file'])
            # Patches run outside every git worktree, including an enclosing repo.
            env = {**os.environ, 'GIT_CEILING_DIRECTORIES': str(work)}
            stat = subprocess.check_output(['git', 'apply', '--numstat', '-z', str(patch.resolve())], cwd=target, env=env)
            touched = [row.split(b'\t', 2)[-1].decode('utf8') for row in stat.split(b'\0') if row]
            if len(touched) != len(entry['files']) or set(touched) != {f['path'] for f in entry['files']}:
                raise ValueError('Patch changes files outside its inventory')
            for file in entry['files']:
                if digest(safe_file(target, file['path'])) != file['beforeSha256']:
                    raise ValueError(f'Unreviewed upstream input: {entry["file"]}: {file["path"]}')
            subprocess.run(['git', 'apply', '--check', '--whitespace=error-all', str(patch.resolve())], cwd=target, env=env, check=True)
            subprocess.run(['git', 'apply', '--whitespace=error-all', str(patch.resolve())], cwd=target, env=env, check=True)
            for file in entry['files']:
                if digest(safe_file(target, file['path'])) != file['afterSha256']:
                    raise ValueError(f'Unexpected patch output: {file["path"]}')
            records.append({**entry, 'sha256': digest(patch)})
        # Fail rather than overwriting an output created while patches ran.
        # Callers must own the output path; this is not isolation from a process
        # concurrently renaming arbitrary paths with the same OS credentials.
        if destination.exists() or destination.is_symlink():
            raise ValueError('Prepared source destination appeared during preparation')
        with span('build.install_prepared'):
            work.rename(destination)
    return {'inventorySha256': digest(patches / 'series.json'), 'patches': records,
            'vendorCheckoutModified': False}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sd-source', type=Path, default=ROOT / 'vendor/stable-diffusion.cpp')
    parser.add_argument('--ggml-source', type=Path, default=ROOT / 'vendor/ggml-webgpu-source')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not args.output.resolve().is_relative_to(ROOT / 'build'):
        parser.error('Output must be inside this runtime\'s build directory')
    report = prepare({'stable-diffusion': args.sd_source, 'ggml': args.ggml_source}, args.output)
    (args.output / 'patch-provenance.json').write_text(json.dumps(report, indent=2) + '\n')
