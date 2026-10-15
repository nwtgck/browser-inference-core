#!/usr/bin/env python3
"""Compose all-pools selection with either reviewed model source, never upstream.

This graph optimization is independent of file-backed PLE. CMake preserves the
current model header's ownership with a per-source include directory.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIRECTORY = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'
PATCH_NAME = 'qwen4exp-all-pools-selection.patch'
SOURCE_PATH = 'src/models/qwen4exp.cpp'
INPUTS_PATH = ROOT / 'tests/qsa-selection/inputs.json'


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing or linked QSA input: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, model_source: Path, output: Path) -> Path:
    if output.is_symlink() or model_source.is_symlink():
        raise ValueError('Linked QSA paths are not accepted')
    source, model_source, output = source.resolve(), model_source.resolve(), output.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('QSA overlay must be outside the upstream tree')
    if output == model_source.parent or output in model_source.parents:
        raise ValueError('QSA overlay must not replace the input translation unit')
    reviewed = json.loads(INPUTS_PATH.read_text())
    for relative, expected in reviewed['upstream'].items():
        if digest(source / relative) != expected:
            raise ValueError('QSA selection needs upstream semantic review: ' + relative)
    patch = ROOT / PATCH_DIRECTORY / PATCH_NAME
    if digest(patch) != reviewed['patchSha256']:
        raise ValueError('QSA patch identity changed; review the resulting source')
    original = digest(model_source)
    variants = [v for v in reviewed['variants'].values() if v['inputSha256'] == original]
    if len(variants) != 1:
        raise ValueError('Unreviewed QSA model translation unit')
    result = output / 'qwen4exp.cpp'
    if result.is_symlink():
        raise ValueError('Linked QSA destination is not accepted')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='qsa-selection-', dir=output) as temporary:
        work = Path(temporary)
        target = work / SOURCE_PATH
        target.parent.mkdir(parents=True)
        target.write_bytes(model_source.read_bytes())
        env = os.environ.copy()
        for name in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE', 'GIT_PREFIX'):
            env.pop(name, None)
        env['GIT_CEILING_DIRECTORIES'] = str(work.parent)
        for flags in (['--check'], []):
            subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch)],
                           cwd=work, env=env, check=True, capture_output=True)
        if digest(target) != variants[0]['outputSha256']:
            raise ValueError('Unexpected QSA output identity')
        if not result.exists() or result.read_bytes() != target.read_bytes():
            target.replace(result)
    return result



def preflight(source: Path, output: Path) -> None:
    """Review both header ownership modes when updating the pinned upstream."""
    import prepare_file_backed_rows
    source, output = Path(source), Path(output)
    prepare(source, source / SOURCE_PATH, output / 'pristine')
    prepare_file_backed_rows.prepare(source, output / 'file-rows')
    prepare(source, output / 'file-rows' / SOURCE_PATH, output / 'composed')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.input, args.output)


if __name__ == '__main__':
    main()
