#!/usr/bin/env python3
"""Prepare a coherent model-definition overlay; never modify the vendor tree."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PATCH_NAME = 'qwen4exp-file-backed-rows.patch'
PATCH_DIRECTORY = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'
INPUTS_PATH = ROOT / 'tests/file-backed-rows/inputs.json'


def prepare(source: Path, output: Path) -> Path:
    source, output = source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('file-backed overlay must be outside the upstream source tree')
    inputs = json.loads(INPUTS_PATH.read_text())
    for name, expected in inputs.items():
        path = source / name
        if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'file-backed rows needs upstream review: {name}')
    # Every model TU includes models.h. Relocating just qwen4exp.cpp would leave
    # allocations in llama-model.cpp using a different sizeof(model), an ABI bug.
    for path in (source / 'src').rglob('*'):
        if path.is_symlink():
            raise ValueError('linked upstream sources are not accepted')
    output.mkdir(parents=True, exist_ok=True)
    if any(path.is_symlink() for path in output.rglob('*')):
        raise ValueError('linked overlay destinations are not accepted')
    with tempfile.TemporaryDirectory(prefix='file-rows-') as temporary:
        work = Path(temporary)
        shutil.copytree(source / 'src', work / 'src')
        patch = ROOT / PATCH_DIRECTORY / PATCH_NAME
        for flags in (['--check'], []):
            subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch)],
                           cwd=work, check=True, capture_output=True)
        shutil.copyfile(ROOT / 'bridge/file-backed-rows.h', work / 'src/models/file-backed-rows.h')
        desired = set()
        for path in (work / 'src').rglob('*'):
            if not path.is_file():
                continue
            relative = path.relative_to(work)
            desired.add(relative)
            target = output / relative
            content = path.read_bytes()
            if not target.exists() or target.read_bytes() != content:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Same-directory replacement also works across file systems.
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
                    stream.write(content)
                    pending = Path(stream.name)
                pending.replace(target)
        for path in (output / 'src').rglob('*'):
            if path.is_file() and path.relative_to(output) not in desired:
                path.unlink()
    return output / 'src'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == '__main__':
    main()
