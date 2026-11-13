#!/usr/bin/env python3
"""Prepare the explicitly approved independent-vocabulary chat entry point.

Pristine upstream is never modified. Input, patch, and output identities are
checked; an upstream change requires semantic review rather than fuzzy apply.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PATCH_RELATIVE = Path('upstream-patches-only-as-a-last-resort-with-explicit-user-approval/common-chat-independent-vocabulary.patch')
PATCH = ROOT / PATCH_RELATIVE


def sha(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing or linked model graph input: ' + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(source: Path, output: Path, *, root: Path = ROOT) -> dict:
    if source.is_symlink() or output.is_symlink() or root.is_symlink():
        raise ValueError('Linked model graph paths are not accepted')
    source, output, root = source.resolve(), output.resolve(), root.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Model graph overlay must be outside the upstream checkout')
    reviewed = json.loads((root / 'tests/model-graph/inputs.json').read_text())
    patch = root / PATCH_RELATIVE
    for relative, expected in reviewed['upstream'].items():
        if sha(source / relative) != expected:
            raise ValueError('Independent vocabulary requires upstream semantic review: ' + relative)
    if sha(patch) != reviewed['patchSha256']:
        raise ValueError('Independent vocabulary patch identity changed')
    output.mkdir(parents=True, exist_ok=True)
    # Never shadow upstream's pragma-once chat header with a second file.
    stale_header = output / 'common/chat.h'
    if stale_header.is_symlink():
        raise ValueError('Linked stale chat header')
    if stale_header.exists():
        stale_header.unlink()
    with tempfile.TemporaryDirectory(prefix='model-graph-', dir=output) as temporary:
        work = Path(temporary)
        for relative in reviewed['outputs']:
            target = work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((source / relative).read_bytes())
        env = os.environ.copy()
        for name in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE', 'GIT_PREFIX'):
            env.pop(name, None)
        env['GIT_CEILING_DIRECTORIES'] = str(work.parent)
        for flags in (['--check'], []):
            subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch)], cwd=work, env=env, check=True, capture_output=True)
        for relative, expected in reviewed['outputs'].items():
            if sha(work / relative) != expected:
                raise ValueError('Unexpected independent vocabulary output: ' + relative)
            target = output / relative
            if target.is_symlink():
                raise ValueError('Linked output: ' + relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (work / relative).read_bytes()
            if not target.exists() or target.read_bytes() != data:
                target.write_bytes(data)
    provenance = {'patch': patch.name, 'patchSha256': reviewed['patchSha256'], 'upstream': reviewed['upstream'], 'outputs': reviewed['outputs']}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    return provenance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
