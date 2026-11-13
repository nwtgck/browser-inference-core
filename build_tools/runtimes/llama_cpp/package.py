#!/usr/bin/env python3
"""Assemble a runtime-only npm package and verify every payload file."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from build_tools.runtimes.llama_cpp.bf16 import validate_metadata as validate_bf16_metadata

from build_tools.runtimes.llama_cpp.model_graph_contract import validate as validate_model_graph

from build_tools.project import runtime_root
ROOT = runtime_root('llama-cpp')
from build_tools.packaging.files import copy_regular, copy_regular_tree, read_regular, regular_path
from build_tools.metrics import measured, span
from build_tools.packaging.licenses import write_index, validate_index
from build_tools.output import output_directories
from build_tools.packaging.layout import RUNTIME_MANIFEST_VERSION, runtime_exports
from build_tools.packaging.types import write_host_profiles

RUNTIME_NAME='llama-cpp-browser-core'
VARIANTS=json.loads((ROOT/'config/variants.json').read_text())
# Preserve complete original files where notices are embedded in source. This
# avoids extracting only one of a header's licenses or dropping contributor text.
EMBEDDED_NOTICE_FILES = (
    'vendor/miniaudio/miniaudio.h',
    'vendor/stb/stb_image.h',
    'vendor/nlohmann/json.hpp',
    'vendor/nlohmann/json_fwd.hpp',
    'vendor/sheredom/subprocess.h',
    'vendor/hash/sha1/sha1.c',
)
HOST_FILES = ('index.mjs', 'index.d.ts', 'bindings.mjs', 'read-only-file.mjs', 'README.md')
GUIDE_FILES = ('chat-and-multimodal.md', 'browser-host.md')


@measured('package.copy_embedded_notices', packageKind='llama')
def copy_embedded_notices(source: Path, destination: Path):
    for relative in EMBEDDED_NOTICE_FILES:
        original=source/relative
        if not original.is_file() or original.is_symlink():
            raise ValueError(f'Missing embedded notice source: {original}')
        target=destination/(relative+'.txt')
        target.parent.mkdir(parents=True,exist_ok=True)
        copy_regular(original,target)


@measured('package.copy_notices', packageKind='llama')
def copy_license_notices(source: Path, destination: Path):
    """Copy the same standalone notices for local packaging and CI transfer."""
    if not source.exists(): raise ValueError(f'Missing license source: {source}')
    regular_path(source, directory=True)
    copied=0
    for path in sorted(source.rglob('*')):
        if not path.is_file() or path.is_symlink() or '.git' in path.parts: continue
        if path.name.upper().startswith(('LICENSE','COPYING','COPYRIGHT')):
            target=destination/path.relative_to(source)
            target.parent.mkdir(parents=True,exist_ok=True)
            copy_regular(path,target); copied+=1
    return copied


def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

@measured('package.validate', packageKind='llama')
def validate(directory: Path, require_clean=True, *, check_npm_pack: bool = True):
    # Assembly may defer compression to the publisher. Every payload and
    # provenance check still runs; standalone validation includes npm pack.
    directory=regular_path(directory, directory=True)
    entries=list(directory.rglob('*'))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in entries):
        raise ValueError('Linked or non-regular entry in runtime package')
    package=json.loads((directory/'package.json').read_text())
    for key in ('scripts','workspaces','devDependencies','dependencies','optionalDependencies',
                'peerDependencies','peerDependenciesMeta','bundleDependencies','bundledDependencies'):
        if key in package: raise ValueError(f'Unexpected install-time input: {key}')
    if package['name']!=RUNTIME_NAME: raise ValueError('Wrong package name')
    if package.get('exports') != runtime_exports('llama-cpp'):
        raise ValueError('Unexpected llama public exports')
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['formatVersion']!=RUNTIME_MANIFEST_VERSION: raise ValueError('Unsupported manifest format')
    expected={f['path']:f for f in manifest['files']}
    if len(expected) != len(manifest['files']): raise ValueError('Duplicate manifest entries')
    required_notices={'licenses/embedded/'+path+'.txt' for path in EMBEDDED_NOTICE_FILES}
    if not required_notices.issubset(expected):
        raise ValueError('Missing embedded third-party license notices')
    if not {'host/'+path for path in (*HOST_FILES, 'profiles.d.ts')}.issubset(expected):
        raise ValueError('Missing host runtime files')
    if not {'api/' + name for name in ('schema.json', 'schema.mjs', 'functions.d.ts', 'exports.json')}.issubset(expected):
        raise ValueError('Missing llama API files')
    actual={p.relative_to(directory).as_posix() for p in entries if p.is_file()}
    if actual != set(expected)|{'manifest.json'}: raise ValueError('Manifest does not exactly cover the package tree')
    if (directory/'.gitmodules').exists() or (directory/'binding.gyp').exists(): raise ValueError('Source/build input in runtime')
    if not manifest['profiles']: raise ValueError('No profiles')
    for file in manifest['files']:
        rel=Path(file['path'])
        if rel.is_absolute() or '..' in rel.parts or '\\' in file['path']: raise ValueError('Unsafe manifest path')
        path=directory/rel
        if path.is_symlink() or path.stat().st_size!=file['bytes'] or sha(path)!=file['sha256']:
            raise ValueError(f'Invalid payload: {rel}')
        if path.stat().st_size >= 100*1024**2: raise ValueError(f'GitHub single-file size guard exceeded: {rel}')
    model_graph_contracts = set()
    for name, profile in manifest['profiles'].items():
        if set(profile['variants'])!=set(VARIANTS): raise ValueError(f'Missing or unknown variant: {name}')
        for variant, info in profile['variants'].items():
            validate_bf16_metadata(info)
            model_graph_contracts.add(validate_model_graph(directory / f'profiles/{name}/{variant}', info))
            if info['profile']!=name or info['variant']!=variant or info['variantConfiguration']!=VARIANTS[variant]:
                raise ValueError(f'Variant provenance mismatch: {name}/{variant}')
            if info['sourceCommit']!=manifest['sourceCommit'] or info['llamaCommit']!=manifest['llamaCommit']:
                raise ValueError('Mixed source commits in manifest')
            if info.get('validation', {}).get('compiled') is not True:
                raise ValueError('Cannot package an uncompiled runtime')
            if require_clean and info['sourceDirty'] is not False:
                details={key:info.get(key,'not recorded') for key in
                         ('sourceStatusBeforeBuild','sourceStatusAfterBuild')}
                raise ValueError(f'Cannot publish a dirty source build: {name}/{variant}\n'+json.dumps(details,indent=2))
            for ext in ('mjs','wasm','d.ts'):
                if f'profiles/{name}/{variant}/core.{ext}' not in expected:
                    raise ValueError(f'Missing {name}/{variant} runtime')
            preparation = info.get('browserModulePreparation')
            if preparation is not None:
                module = expected[f'profiles/{name}/{variant}/core.mjs']
                if variant != 'browser' or preparation['output'] != {key: module[key] for key in ('bytes', 'sha256')}:
                    raise ValueError(f'Browser module preparation output mismatch: {name}/{variant}')
            with (directory/f'profiles/{name}/{variant}/core.wasm').open('rb') as wasm:
                if wasm.read(8) != b'\x00asm\x01\x00\x00\x00':
                    raise ValueError('Not a WebAssembly module')
    if len(model_graph_contracts) != 1:
        raise ValueError('Mixed model graph boundary contracts across profile variants')
    validate_index(directory, runtime='llama-cpp', license_roots=manifest['licenseRoots'])
    if check_npm_pack:
        with span('package.npm_pack', packageKind='llama'):
            packed=json.loads(subprocess.check_output(['npm','pack','--dry-run','--json'],cwd=directory,text=True))
        pack_paths={x['path'] for x in packed[0]['files']}
        if pack_paths != actual: raise ValueError(f'npm pack file mismatch: {sorted(actual ^ pack_paths)}')
    return {'files':len(actual),'bytes':sum(p.stat().st_size for p in directory.rglob('*') if p.is_file()),
            'profiles':list(manifest['profiles'])}

@measured('package.assemble', packageKind='llama')
def build_package(build_root: Path, destination: Path, profiles: list[str], *, license_roots: list[Path], check_npm_pack: bool = True):
    # Check names before forming either input or private output paths. The CLI
    # also accepts a selected subset, so these are not always configuration keys.
    if (not profiles or len(set(profiles)) != len(profiles) or
            any(not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', name) for name in profiles)):
        raise ValueError('Invalid package profile selection')
    source=None; upstream=None; schema=None; provenance={}
    with output_directories(destination, replace=True) as (out,):
        for name in profiles:
            provenance[name]={'variants':{}}
            for variant in VARIANTS:
                build=build_root/name/variant
                data=json.loads(read_regular(build/'provenance.json'))
                validate_bf16_metadata(data)
                if data['profile']!=name: raise ValueError('Profile mismatch')
                if data['variant']!=variant or data['variantConfiguration']!=VARIANTS[variant]:
                    raise ValueError(f'Variant provenance mismatch: {name}/{variant}')
                if source is not None and (source!=data['sourceCommit'] or upstream!=data['llamaCommit']):
                    raise ValueError('Mixed source commits in one package')
                source=data['sourceCommit']; upstream=data['llamaCommit']
                provenance[name]['variants'][variant]=data
                generated=build/'generated'
                current=read_regular(generated/'schema.json')
                if schema is not None and current!=schema: raise ValueError('Mixed binding schemas')
                schema=current
                runtime=out/'profiles'/name/variant
                with span('package.copy_payload', packageKind='llama', profile=name, variant=variant):
                    copy_regular_tree(build/'runtime',runtime)
                if not (out/'api').exists():
                    (out/'api').mkdir()
                    for file in ('schema.json','schema.mjs','functions.d.ts','exports.json'):
                        copy_regular(generated/file,out/'api'/file)
        host=out/'host'; host.mkdir(parents=True)
        for file in HOST_FILES:
            copy_regular(ROOT/'host'/file,host/file)
        write_host_profiles(host/'profiles.d.ts', 'llama-cpp', profiles, VARIANTS)
        copy_regular(ROOT/'packaging/README.md',out/'README.md')
        for file in GUIDE_FILES:
            copy_regular(ROOT/'docs'/file,out/file)
        copy_regular(ROOT/'LICENSE',out/'LICENSE')
        licenses=out/'licenses'; licenses.mkdir()
        copied=0
        for i,root in enumerate(license_roots):
            copied+=copy_license_notices(root,licenses/str(i))
        if not copied: raise ValueError('No third-party license notices collected')
        copy_embedded_notices(ROOT/'vendor/llama.cpp', licenses/'embedded')
        pkg={'name':RUNTIME_NAME,'version':'0.1.0','private':True,'type':'module','license':'MIT',
             'files':['host/','profiles/','api/','manifest.json','native-license-index.json','licenses/','README.md',*GUIDE_FILES,'LICENSE'],
             'exports':runtime_exports('llama-cpp')}
        (out/'package.json').write_text(json.dumps(pkg,indent=2)+'\n')
        write_index(out, runtime='llama-cpp', license_roots=[p.name for p in license_roots])
        manifest={'formatVersion':RUNTIME_MANIFEST_VERSION,'sourceCommit':source,'llamaCommit':upstream,'profiles':provenance,
                  'licenseRoots':[p.name for p in license_roots],
                  'files':[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
                           for p in sorted(out.rglob('*')) if p.is_file()]}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        result = validate(out,require_clean=False,check_npm_pack=check_npm_pack)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-root',type=Path,default=ROOT/'build')
    p.add_argument('--output',type=Path,default=ROOT/'dist/package')
    p.add_argument('--profiles',nargs='+')
    p.add_argument('--license-root',type=Path,action='append')
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--defer-npm-pack',action='store_true',
                   help='Defer compression only; the publication step must check npm packing before upload')
    a=p.parse_args()
    if a.verify_only and a.defer_npm_pack: p.error('--verify-only always includes npm packing')
    if a.verify_only:
        result = validate(a.output)
    else:
        profiles=a.profiles or list(json.loads((ROOT/'config/profiles.json').read_text()))
        roots=a.license_root or [ROOT/'vendor/llama.cpp',ROOT.parents[1]/'.tools/emsdk/upstream/emscripten',ROOT.parents[1]/'.tools/emdawnwebgpu_pkg']
        result = build_package(a.build_root,a.output,profiles,license_roots=roots,check_npm_pack=not a.defer_npm_pack)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
