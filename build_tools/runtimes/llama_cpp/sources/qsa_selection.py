#!/usr/bin/env python3
"""Compose all-pools selection with either reviewed model source, never upstream.

This graph optimization is independent of file-backed PLE. CMake preserves the
current model header's ownership with a per-source include directory.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from build_tools.project import runtime_root
from build_tools.output import output_directories, preserve_unchanged
from build_tools.source.files import absolute, directory, file_digest, regular_file
from build_tools.source.patches import Patch, apply_patch
import tempfile

ROOT = runtime_root('llama-cpp')
PATCH_DIRECTORY = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'
PATCH_NAME = 'qwen4exp-all-pools-selection.patch'
SOURCE_PATH = 'src/models/qwen4exp.cpp'
RECEIPT = 'source-preparation.json'
DECLARATION = Patch(PATCH_NAME, 'llama-model',
    'Skip ranking only when every pool is selected, preserving the current model header owner.',
    (SOURCE_PATH,))


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing or linked QSA input: ' + str(path))
    return file_digest(path)


def prepare(source: Path, model_source: Path, output: Path) -> Path:
    if output.is_symlink() or model_source.is_symlink():
        raise ValueError('Linked QSA paths are not accepted')
    source, model_source, output = directory(source), regular_file(model_source), absolute(output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('QSA overlay must be outside the upstream tree')
    if output == model_source.parent or output in model_source.parents:
        raise ValueError('QSA overlay must not replace the input translation unit')
    patch_root = ROOT / PATCH_DIRECTORY
    if output == patch_root or patch_root in output.parents or output in patch_root.parents:
        raise ValueError('QSA overlay must not replace preparation inputs')
    result = output / 'qwen4exp.cpp'
    if result.is_symlink():
        raise ValueError('Linked QSA destination is not accepted')
    with output_directories(output, replace=True, reuse_if=preserve_unchanged) as (prepared,), \
            tempfile.TemporaryDirectory(prefix='qsa-selection-', dir=prepared.parent) as temporary:
        work = Path(temporary)
        target = work / SOURCE_PATH
        target.parent.mkdir(parents=True)
        target.write_bytes(model_source.read_bytes())
        record = apply_patch(work, patch_root, DECLARATION)
        target.replace(prepared / 'qwen4exp.cpp')
        receipt = {'formatVersion': 1, 'patches': [record],
                   'output': {'path': 'qwen4exp.cpp', 'sha256': digest(prepared / 'qwen4exp.cpp')}}
        (prepared / RECEIPT).write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    return result



def preflight(source: Path, output: Path) -> None:
    """Review both header ownership modes when updating the pinned upstream."""
    from build_tools.runtimes.llama_cpp.sources import file_backed_rows as prepare_file_backed_rows
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
