"""Prepare the approved vocabulary-independent chat source in a private tree.

Observed identities describe the exact inputs, patch and output. C++ compilation
and boundary tests, not a digest stored in a test fixture, check compatibility.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_tools.output import output_directories, preserve_unchanged
from build_tools.project import runtime_root
from build_tools.source.files import absolute, directory, file_digest, regular_file
from build_tools.source.patches import Patch, apply_patch

ROOT = runtime_root('llama-cpp')
PATCH_DIRECTORY = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'
PATCH_NAME = 'common-chat-independent-vocabulary.patch'
PATCH = ROOT / PATCH_DIRECTORY / PATCH_NAME
INPUTS = (
    'common/chat.cpp', 'common/chat.h', 'src/llama-vocab.h',
    'src/llama-model-loader.h', 'src/llama-model-loader.cpp',
    'src/llama-vocab.cpp', 'src/llama-arch.h', 'src/llama-sampler.cpp',
)
DECLARATION = Patch(PATCH_NAME, 'llama-common',
    'Initialize independent vocabulary chat metadata without a dummy model.',
    ('common/chat.cpp',))


def prepare(source: Path, output: Path, *, runtime: Path | None = None) -> dict:
    if source.is_symlink() or output.is_symlink():
        raise ValueError('Linked model graph paths are not accepted')
    source, output = directory(source), absolute(output)
    runtime = directory(ROOT if runtime is None else runtime)
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Model graph overlay must be outside the upstream checkout')
    patch_root = runtime / PATCH_DIRECTORY
    for protected in (patch_root, runtime / 'native'):
        if output == protected or output in protected.parents or protected in output.parents:
            raise ValueError('Model graph overlay must not replace preparation inputs')
    if output.exists() and any(p.is_symlink() for p in output.rglob('*')):
        raise ValueError('Linked model graph output is not accepted')
    # Read once: the bytes copied for patch application are the bytes recorded.
    captured = {name: regular_file(source / name).read_bytes() for name in INPUTS}
    with output_directories(output, replace=True, reuse_if=preserve_unchanged) as (work,):
        for name in DECLARATION.files:
            destination = work / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(captured[name])
        record = apply_patch(work, patch_root, DECLARATION)
        provenance = {
            'patch': PATCH_NAME, 'patchSha256': record['sha256'],
            'upstream': {name: hashlib.sha256(data).hexdigest() for name, data in captured.items()},
            'outputs': {name: file_digest(work / name) for name in DECLARATION.files},
        }
        (work / 'provenance.json').write_text(json.dumps(provenance, sort_keys=True, indent=2) + '\n')
    return provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == '__main__':
    main()
