#!/usr/bin/env python3
"""Prepare a coherent model-definition overlay; never modify the vendor tree."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from build_tools.project import runtime_root
from build_tools.output import output_directories, preserve_unchanged
from build_tools.source.files import absolute, directory, file_digest, regular_file
from build_tools.source.patches import Patch, apply_patch
import shutil

ROOT = runtime_root('llama-cpp')
PATCH_NAME = 'qwen4exp-file-backed-rows.patch'
PATCH_DIRECTORY = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'
RECEIPT = 'source-preparation.json'
DECLARATION = Patch(PATCH_NAME, 'llama-models',
    'Keep all model-definition consumers coherent while adding bounded file-backed rows.',
    ('src/llama-model.cpp', 'src/llama.cpp', 'src/models/models.h', 'src/models/qwen4exp.cpp'))


def prepare(source: Path, output: Path) -> Path:
    # Check the requested root before resolve erases its link identity.
    if output.is_symlink():
        raise ValueError('linked overlay destinations are not accepted')
    source, output = directory(source), absolute(output)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('file-backed overlay must be outside the upstream source tree')
    for input_root in (ROOT / PATCH_DIRECTORY, ROOT / 'native'):
        if output == input_root or input_root in output.parents or output in input_root.parents:
            raise ValueError('file-backed overlay must not replace preparation inputs')
    # Every model TU includes models.h. Relocating just qwen4exp.cpp would leave
    # allocations in llama-model.cpp using a different sizeof(model), an ABI bug.
    for path in (source / 'src').rglob('*'):
        if path.is_symlink():
            raise ValueError('linked upstream sources are not accepted')
    if output.exists() and any(path.is_symlink() for path in output.rglob('*')):
        raise ValueError('linked overlay destinations are not accepted')
    with output_directories(output, replace=True, reuse_if=preserve_unchanged) as (work,):
        shutil.copytree(source / 'src', work / 'src')
        record = apply_patch(work, ROOT / PATCH_DIRECTORY, DECLARATION)
        companion = regular_file(ROOT / 'native/io/file-backed-rows.h')
        destination = work / 'src/models/file-backed-rows.h'
        if destination.exists() or destination.is_symlink():
            raise ValueError('File-backed companion header now exists upstream')
        shutil.copyfile(companion, destination)
        identity_header = regular_file(ROOT / 'native/io/file-backed-identity.h')
        identity_destination = work / 'src/models/file-backed-identity.h'
        if identity_destination.exists() or identity_destination.is_symlink():
            raise ValueError('File-backed identity header now exists upstream')
        shutil.copyfile(identity_header, identity_destination)
        receipt = {'formatVersion': 1, 'patches': [record],
                   'companion': {'path': 'native/io/file-backed-rows.h',
                                 'sha256': file_digest(destination)},
                   'identityCompanion': {'path': 'native/io/file-backed-identity.h',
                                         'sha256': file_digest(identity_destination)}}
        (work / RECEIPT).write_text(json.dumps(receipt, sort_keys=True, indent=2) + '\n')
    return output / 'src'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == '__main__':
    main()
