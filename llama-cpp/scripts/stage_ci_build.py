#!/usr/bin/env python3
"""Stage one profile/variant pair's package inputs between CI runners."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / 'scripts'))
from browser_toolchain import runtime_toolchain
import re
import shutil
import subprocess
from package_runtime import VARIANTS, copy_license_notices

ROOT=Path(__file__).resolve().parents[1]
API_FILES=('schema.json','schema.mjs','functions.d.ts','exports.json')


def stage_profile(build_root: Path, output: Path, profile: str, *, variant: str, source_commit: str,
                  toolchain: dict, configuration: dict, source_id: str | None = None):
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',profile):
        raise ValueError('Invalid profile name')
    if variant not in VARIANTS: raise ValueError('Invalid variant name')
    build=build_root/profile/variant
    target=output/profile/variant
    if target.exists() or target.is_symlink():
        raise ValueError(f'Profile variant already staged: {profile}/{variant}')
    # Never follow links to unrelated build inputs or silently omit a missing asset.
    if ((build_root/profile).is_symlink() or build.is_symlink()
            or (build/'runtime').is_symlink() or (build/'generated').is_symlink()):
        raise ValueError('Unexpected symlink in build output')
    for path in (build/'runtime').rglob('*'):
        if path.is_symlink(): raise ValueError('Unexpected symlink in runtime output')
    required=[build/'provenance.json',*(build/'generated'/name for name in API_FILES),
              *(build/'runtime'/('core.'+ext) for ext in ('mjs','wasm','d.ts'))]
    for path in required:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Missing or linked build input: {path}')
    provenance=json.loads((build/'provenance.json').read_text())
    if source_id is not None and provenance.get('sourceId')!=source_id:
        raise ValueError('Build source track differs from the requested source')
    if provenance['profile']!=profile or provenance['sourceCommit']!=source_commit:
        raise ValueError('Build provenance does not match this source/profile')
    if provenance['variant']!=variant or provenance['variantConfiguration']!=VARIANTS[variant]:
        raise ValueError('Build provenance does not match the requested variant')
    if provenance['llamaCommit']!=toolchain['llamaCommit'] or provenance['toolchain']!=toolchain:
        raise ValueError('Build provenance does not match the pinned toolchain')
    if provenance['configuration']!=configuration:
        raise ValueError('Build configuration differs from the requested profile')
    if provenance['sourceDirty'] is not False:
        raise ValueError('Refusing to stage a dirty source build')
    if provenance['validation'].get('compiled') is not True:
        raise ValueError('Refusing to stage an uncompiled profile')
    shutil.copytree(build/'runtime',target/'runtime')
    (target/'generated').mkdir()
    for name in API_FILES:
        shutil.copy2(build/'generated'/name,target/'generated'/name)
    shutil.copy2(build/'provenance.json',target/'provenance.json')
    return target


def stage_toolchain_notices(roots: list[Path], output: Path):
    names=[root.name for root in roots]
    if len(set(names))!=len(names): raise ValueError('Toolchain license roots must have distinct names')
    target=output/'toolchain-licenses'
    if target.exists() or target.is_symlink():
        raise ValueError('Toolchain notices already staged')
    for root in roots:
        if copy_license_notices(root,target/root.name)==0:
            raise ValueError(f'No standalone notices found: {root}')


def main():
    profiles=json.loads((ROOT/'config/profiles.json').read_text())
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default='upstream-stable')
    parser.add_argument('--profile',choices=list(profiles),required=True)
    parser.add_argument('--variant',choices=list(VARIANTS),required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'build/ci-upload')
    parser.add_argument('--include-toolchain-notices',action='store_true')
    args=parser.parse_args()
    output=args.output.resolve()
    if not output.is_relative_to(ROOT/'build') or output==ROOT/'build':
        parser.error('The staging directory must be inside build/')
    if any(output.is_relative_to(ROOT/'build'/name) for name in profiles):
        parser.error('Do not stage inside a profile build directory')
    from source_config import get_source
    entry=get_source(ROOT,args.source)
    if args.profile not in entry['profiles']:parser.error('Unavailable source profile')
    toolchain=runtime_toolchain(ROOT);toolchain['llamaCommit']=entry['commit']
    build_root=ROOT/'build' if args.source=='upstream-stable' else ROOT/'build/sources'/args.source
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    stage_profile(build_root,output,args.profile,source_commit=source,
                  variant=args.variant,toolchain=toolchain,configuration=profiles[args.profile],source_id=args.source)
    if args.include_toolchain_notices:
        stage_toolchain_notices([ROOT.parent/'.tools/emsdk/upstream/emscripten',ROOT.parent/'.tools/emdawnwebgpu_pkg'],output)
    print(output)

if __name__=='__main__': main()
