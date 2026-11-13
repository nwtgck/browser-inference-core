#!/usr/bin/env python3
"""Stage one profile/variant pair's package inputs between CI runners."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from build_tools.toolchain.config import runtime_toolchain
import re
import shutil
from build_tools.project import runtime_root
from build_tools.source.files import absolute
from build_tools.source.checkout import commit_file, git
from build_tools.runtimes.llama_cpp.package import copy_license_notices
from build_tools.output import output_directories
from build_tools.packaging.files import copy_regular, copy_regular_tree

ROOT = runtime_root('llama-cpp')
API_FILES=('schema.json','schema.mjs','functions.d.ts','exports.json')


def stage_profile(build_root: Path, output: Path, profile: str, *, variant: str, source_commit: str,
                  toolchain: dict, configuration: dict, notice_roots: tuple[Path, ...] = (),
                  variant_configuration: dict | None = None, source_repository: Path | None = None):
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',profile):
        raise ValueError('Invalid profile name')
    if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', variant): raise ValueError('Invalid variant name')
    if variant_configuration is None:
        variants = json.loads((ROOT / 'config/variants.json').read_text())
        if variant not in variants: raise ValueError('Invalid variant name')
        variant_configuration = variants[variant]
    output = absolute(output)
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
    if provenance['profile']!=profile or provenance['sourceCommit']!=source_commit:
        raise ValueError('Build provenance does not match this source/profile')
    if provenance['variant']!=variant or provenance['variantConfiguration']!=variant_configuration:
        raise ValueError('Build provenance does not match the requested variant')
    if provenance['llamaCommit']!=toolchain['llamaCommit'] or provenance['toolchain']!=toolchain:
        raise ValueError('Build provenance does not match the pinned toolchain')
    if provenance['configuration']!=configuration:
        raise ValueError('Build configuration differs from the requested profile')
    if provenance['sourceDirty'] is not False:
        raise ValueError('Refusing to stage a dirty source build')
    if provenance['validation'].get('compiled') is not True:
        raise ValueError('Refusing to stage an uncompiled profile')
    captured = provenance.get('sourceInputs')
    if captured is not None and (not isinstance(captured, dict) or captured.get('sourceCommit') != source_commit
                                 or captured.get('parentInputMode') != 'commit-objects'):
        raise ValueError('Captured build inputs do not match the selected committed source')
    names = [root.name for root in notice_roots]
    if len(set(names)) != len(names): raise ValueError('Toolchain license roots must have distinct names')
    targets = (target, output/'toolchain-licenses') if notice_roots else (target,)
    # Runtime, provenance, API and requested notices have one activation boundary.
    with output_directories(*targets) as outputs:
        staged = outputs[0]
        copy_regular_tree(build/'runtime', staged/'runtime')
        (staged/'generated').mkdir()
        for name in API_FILES:
            copy_regular(build/'generated'/name, staged/'generated'/name)
        copy_regular(build/'provenance.json', staged/'provenance.json')
        if notice_roots:
            for root in notice_roots:
                if copy_license_notices(root, outputs[1]/root.name) == 0:
                    raise ValueError(f'No standalone notices found: {root}')
        if source_repository is not None and git(source_repository, 'rev-parse', 'HEAD').strip() != source_commit:
            raise ValueError('Source commit changed while staging')
    return target


def stage_toolchain_notices(roots: list[Path], output: Path):
    names=[root.name for root in roots]
    if len(set(names))!=len(names): raise ValueError('Toolchain license roots must have distinct names')
    target=output/'toolchain-licenses'
    if target.exists() or target.is_symlink():
        raise ValueError('Toolchain notices already staged')
    with output_directories(target) as (staged,):
        for root in roots:
            if copy_license_notices(root, staged/root.name)==0:
                raise ValueError(f'No standalone notices found: {root}')


def main(argv: list[str] | None = None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',required=True)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'build/ci-upload')
    parser.add_argument('--include-toolchain-notices',action='store_true')
    args=parser.parse_args(argv)
    # All transfer checks use the same selected commit, even if worktree flags
    # hide an edit or HEAD moves while the configuration is being read.
    repository = ROOT.parents[1]
    source = git(repository, 'rev-parse', 'HEAD').strip()
    def configuration(name):
        return json.loads(commit_file(repository, source, 'runtimes/llama-cpp/config/' + name))
    profiles, variants = configuration('profiles.json'), configuration('variants.json')
    if args.profile not in profiles or args.variant not in variants:
        parser.error('Unknown profile or variant')
    output=absolute(args.output)
    if not output.is_relative_to(ROOT/'build') or output==ROOT/'build':
        parser.error('The staging directory must be inside build/')
    if any(output.is_relative_to(ROOT/'build'/name) for name in profiles):
        parser.error('Do not stage inside a profile build directory')
    toolchain=runtime_toolchain(ROOT, revision=source)
    notices = ()
    if args.include_toolchain_notices:
        notices = (ROOT.parents[1]/'.tools/emsdk/upstream/emscripten', ROOT.parents[1]/'.tools/emdawnwebgpu_pkg')
    stage_profile(ROOT/'build',output,args.profile,source_commit=source,
                  variant=args.variant,toolchain=toolchain,configuration=profiles[args.profile], notice_roots=notices,
                  variant_configuration=variants[args.variant], source_repository=repository)
    print(output)

if __name__=='__main__': main()
