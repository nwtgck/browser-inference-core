"""Prepare already downloaded compiler inputs. No network operations are allowed."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from build_tools.project import REPOSITORY_ROOT
from build_tools.source.files import directory, regular_file
from build_tools.toolchain.archive import prepare_dawn
from build_tools.toolchain.config import load_toolchain
from build_tools.toolchain.patch_emscripten import apply_asyncify_bigint_patch


def verify_sdk(config: dict, sdk: Path) -> None:
    sdk = directory(sdk)
    tags = regular_file(sdk / 'emscripten-releases-tags.json')
    releases = json.loads(tags.read_text())
    if releases.get('releases', {}).get(config['emsdkVersion']) != config['emscriptenRelease']:
        raise RuntimeError('emsdk release identity differs from the pinned configuration')


def prepare(repository: Path, *, dawn_only: bool = False) -> None:
    config = load_toolchain(repository)
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', config['dawnTag']):
        raise ValueError('Invalid Dawn tag')
    tools = directory(repository / '.tools')
    if not dawn_only:
        sdk = tools / 'emsdk'
        verify_sdk(config, sdk)
        apply_asyncify_bigint_patch(sdk / 'upstream/emscripten', config['emscriptenAsyncifyBigIntPatch'])
    prepare_dawn(config, tools)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dawn-only', action='store_true')
    args = parser.parse_args()
    prepare(REPOSITORY_ROOT, dawn_only=args.dawn_only)


if __name__ == '__main__':
    main()
