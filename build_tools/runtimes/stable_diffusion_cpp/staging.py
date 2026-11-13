"""Stage a complete image runtime and optional toolchain notices as one group."""
from __future__ import annotations

from pathlib import Path

from build_tools.output import output_directories
from build_tools.packaging.files import copy_regular, copy_regular_tree, regular_path
from build_tools.packaging.notices import collect_notices

API_FILES = ('schema.json', 'schema.mjs', 'functions.d.ts', 'exports.json')


def stage_profile(build: Path, destination: Path, *, notice_roots: tuple[Path, ...] = (),
                  notice_destination: Path | None = None) -> Path:
    if bool(notice_roots) != (notice_destination is not None):
        raise ValueError('Notice roots and destination must be provided together')
    names = [path.name for path in notice_roots]
    if len(set(names)) != len(names):
        raise ValueError('Toolchain license roots must have distinct names')
    targets = (destination,) if notice_destination is None else (destination, notice_destination)
    with output_directories(*targets, replace=True) as outputs:
        staged = outputs[0]
        copy_regular_tree(build / 'runtime', staged / 'runtime')
        for extension in ('mjs', 'wasm', 'd.ts'):
            regular_path(staged / 'runtime' / ('core.' + extension))
        copy_regular(build / 'provenance.json', staged / 'provenance.json')
        (staged / 'generated').mkdir()
        for name in API_FILES:
            copy_regular(build / 'generated' / name, staged / 'generated' / name)
        if notice_roots:
            for root in notice_roots:
                collect_notices(root, outputs[1] / root.name)
    return destination
