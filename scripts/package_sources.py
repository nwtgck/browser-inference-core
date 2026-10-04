#!/usr/bin/env python3
"""Assemble source-namespaced runtime artifacts without rewriting build provenance.

The main workflow uses this v4 entry point. A source package may come from
another verified build, but its original sourceCommit is retained. Reuse records
bind that package to the immutable previous artifact selected by the planner.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT/'llama-cpp/scripts'))
from source_config import load_sources,get_source
from package_inputs import copy_regular_tree,read_regular,tree_identity,file_identity,regular_path

NAME='llama-cpp-browser-core'

def identity(path):
    info=file_identity(path)
    return {key:info[key] for key in ('bytes','sha256')}

def runtime_validator(runtime):
    if runtime not in ('llama-cpp','stable-diffusion-cpp'):raise ValueError('Unknown runtime')
    spec=importlib.util.spec_from_file_location('source_package_'+runtime.replace('-','_'),ROOT/runtime/'scripts/package_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.validate

def safe_relative(value):
    if not isinstance(value,str) or not re.fullmatch(r'[A-Za-z0-9_./-]+',value) or any(p in ('','.','..') for p in value.split('/')):raise ValueError('Unsafe package path')
    return value

def validate_reuse_inputs(record, runtimes):
    if not isinstance(record, dict) or set(record) != {'artifactCommit', 'manifestIdentity', 'packages'}:
        raise ValueError('Invalid reuse input receipt')
    if not isinstance(record['artifactCommit'], str) or not re.fullmatch(r'[a-f0-9]{40}', record['artifactCommit']):
        raise ValueError('Invalid reuse artifact commit')
    info = record['manifestIdentity']
    if (not isinstance(info, dict) or set(info) != {'bytes', 'sha256'} or
        type(info['bytes']) is not int or not 0 < info['bytes'] < 100*1024*1024 or
        not isinstance(info['sha256'], str) or not re.fullmatch(r'[a-f0-9]{64}', info['sha256'])):
        raise ValueError('Invalid reuse manifest identity')
    if not isinstance(record['packages'], list) or not record['packages']:
        raise ValueError('Empty reuse input receipt')
    seen = set()
    for package in record['packages']:
        if not isinstance(package, dict) or set(package) != {'runtime', 'source', 'manifestIdentity'}:
            raise ValueError('Invalid reused package receipt')
        runtime, source = package['runtime'], package['source']
        if not isinstance(runtime, str) or not isinstance(source, str):
            raise ValueError('Invalid reused package coordinate')
        entry = runtimes.get(runtime, {}).get('sources', {}).get(source)
        if entry is None or (runtime, source) in seen or package['manifestIdentity'] != entry['manifestIdentity']:
            raise ValueError('Unbound or duplicate reused package receipt')
        seen.add((runtime, source))

def validate(directory: Path,require_clean=True,*,check_npm_pack=True, verify_packed=True):
    directory=regular_path(directory,directory=True);tree=tree_identity(directory)
    manifest=json.loads(read_regular(directory/'manifest.json'));pkg=json.loads(read_regular(directory/'package.json'))
    if manifest.get('formatVersion')!=4 or not re.fullmatch(r'[a-f0-9]{40}',manifest.get('sourceCommit','')):raise ValueError('Invalid source catalog manifest')
    if pkg.get('name')!=NAME or any(k in pkg for k in ('scripts','dependencies','devDependencies','optionalDependencies','workspaces','peerDependencies','peerDependenciesMeta','bundleDependencies','bundledDependencies')):raise ValueError('Install-time code/dependencies in runtime package')
    files=manifest.get('files',[]);expected={};actual={name for name,meta in tree.items() if not meta.get('directory')}
    for f in files:
        rel=safe_relative(f['path'])
        if rel in expected or f['bytes']>=100*1024*1024:raise ValueError('Duplicate or excessive package payload')
        expected[rel]=f
        if identity(directory/rel)!={k:f[k] for k in ('bytes','sha256')}:raise ValueError('Package payload identity mismatch: '+rel)
    if actual!=set(expected)|{'manifest.json'}:raise ValueError('Package tree coverage mismatch')
    runtimes=manifest.get('runtimes')
    if not isinstance(runtimes,dict) or set(runtimes)!={'llama-cpp','stable-diffusion-cpp'}:raise ValueError('Incomplete runtime collection')
    for runtime,group in runtimes.items():
        sources=group.get('sources')
        if not isinstance(sources,dict) or not sources:raise ValueError('No runtime sources')
        for source,entry in sources.items():
            if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',source):raise ValueError('Invalid source identity')
            folder=f'runtimes/{runtime}/sources/{source}';relative=folder+'/manifest.json'
            if entry.get('manifest')!=relative or entry.get('manifestIdentity')!=identity(directory/relative):raise ValueError('Unbound source manifest')
            runtime_validator(runtime)(directory/folder,require_clean=require_clean,check_npm_pack=check_npm_pack)
            inner=json.loads(read_regular(directory/relative))
            if entry.get('buildSourceCommit')!=inner['sourceCommit'] or entry.get('profiles')!=sorted(inner['profiles']):raise ValueError('Source provenance or profile mismatch')
            if runtime=='llama-cpp':
                if entry.get('upstreamCommit')!=inner['llamaCommit']:raise ValueError('Upstream commit mismatch')
                if inner.get('sourceId',source)!=source:raise ValueError('Mixed source IDs')
    if 'reuseInputs' in manifest:
        validate_reuse_inputs(manifest['reuseInputs'], runtimes)
    packed=manifest.get('packedWasm')
    if packed is not None:
        if packed!={'catalog':'packed/llama-cpp/catalog.json','decoderApiVersion':1}:raise ValueError('Unknown packed entry')
        # This also verifies every alternative, not only selected/full targets.
        if verify_packed:
            subprocess.run(['node',str(ROOT/'wasm-pack/verify-pack.mjs'),str(directory/'packed/llama-cpp'),str(directory)],check=True,timeout=600)
    if check_npm_pack:
        result=json.loads(subprocess.check_output(['npm','pack','--dry-run','--json'],cwd=directory,text=True))
        if {f['path'] for f in result[0]['files']}!=actual:raise ValueError('npm pack differs from manifest')
    return manifest

def assemble(inputs: dict, image: Path, output: Path, source_commit: str, *, packing=None,check_npm_pack=True,reuse_inputs=None):
    config=load_sources(ROOT/'llama-cpp')
    if set(inputs)!=set(config['sources']):raise ValueError('Provide exactly the configured llama sources')
    if not re.fullmatch(r'[a-f0-9]{40}',source_commit):raise ValueError('Invalid assembler source identity')
    if output.exists() or output.is_symlink():raise ValueError('Output exists; select a fresh directory')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.source-package-',dir=output.parent) as td:
        out=Path(td)/'package';out.mkdir();runtimes={};pack_spec={'targets':{}}
        groups={'llama-cpp':inputs,'stable-diffusion-cpp':{'upstream-default':image}}
        for runtime,sources in groups.items():
            runtimes[runtime]={'sources':{}}
            for source,path in sorted(sources.items()):
                runtime_validator(runtime)(path,check_npm_pack=False)
                inner=json.loads(read_regular(path/'manifest.json'))
                if runtime=='llama-cpp':
                    entry=get_source(ROOT/'llama-cpp',source)
                    if inner['llamaCommit']!=entry['commit'] or set(inner['profiles'])!=set(entry['profiles']):raise ValueError('Package does not match configured source pin/profiles')
                    if inner.get('sourceId',source)!=source:raise ValueError('Wrong source package')
                folder=f'runtimes/{runtime}/sources/{source}';copy_regular_tree(path,out/folder)
                metadata={'manifest':folder+'/manifest.json','manifestIdentity':identity(out/folder/'manifest.json'),'buildSourceCommit':inner['sourceCommit'],'profiles':sorted(inner['profiles'])}
                if runtime=='llama-cpp':
                    metadata['upstreamCommit']=inner['llamaCommit']
                    files={f['path']:f for f in inner['files']}
                    for profile in sorted(inner['profiles']):
                        relative=f'profiles/{profile}/browser/core.wasm';f=files[relative];name=source+'--'+profile
                        pack_spec['targets'][name]={'identity':{'runtime':runtime,'source':source,'profile':profile,'variant':'browser'},'raw':{'path':folder+'/'+relative,'bytes':f['bytes'],'sha256':f['sha256']}}
                runtimes[runtime]['sources'][source]=metadata
        if packing is not None:
            sys.path.insert(0,str(ROOT/'wasm-pack/encoder'))
            from pack import build
            build(out,pack_spec,out/'packed/llama-cpp',**packing)
        shutil.copy2(ROOT/'LICENSE',out/'LICENSE')
        (out/'README.md').write_text('# Browser inference runtime artifacts\n\nEach source directory contains its original Wasm and paired glue, API and notices.\nUse manifest.json for immutable identities. Packed data is optional and includes\nits matching decoder. A packed target must resolve to its original Wasm hash.\n')
        pkg={'name':NAME,'version':'0.1.0','private':True,'type':'module','license':'MIT','files':['runtimes/','packed/','manifest.json','README.md','LICENSE'],
             'exports':{'./manifest.json':'./manifest.json','./llama-cpp/*':'./runtimes/llama-cpp/sources/*','./stable-diffusion-cpp/*':'./runtimes/stable-diffusion-cpp/sources/*'}}
        if packing is not None:pkg['exports']['./wasm-pack/*']='./packed/llama-cpp/*'
        (out/'package.json').write_text(json.dumps(pkg,indent=2)+'\n')
        manifest={'formatVersion':4,'sourceCommit':source_commit,'runtimes':runtimes,'files':[{'path':p.relative_to(out).as_posix(),**identity(p)} for p in sorted(out.rglob('*')) if p.is_file()]}
        if reuse_inputs is not None:
            validate_reuse_inputs(reuse_inputs, runtimes)
            manifest['reuseInputs'] = reuse_inputs
        if packing is not None:manifest['packedWasm']={'catalog':'packed/llama-cpp/catalog.json','decoderApiVersion':1}
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        validate(out,check_npm_pack=check_npm_pack)
        if output.exists():raise ValueError('Output appeared concurrently')
        out.rename(output)
    return manifest

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--llama-input',action='append',required=True,help='source=directory');p.add_argument('--image-input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--encoder',type=Path);p.add_argument('--zstd-decoder',type=Path);p.add_argument('--codecs',nargs='+',choices=['gzip','brotli','zstd'],default=['gzip','brotli']);p.add_argument('--defer-npm-pack',action='store_true')
    a=p.parse_args();inputs={}
    for item in a.llama_input:
        name,sep,path=item.partition('=')
        if not sep or name in inputs:p.error('Duplicate or invalid source=directory')
        inputs[name]=Path(path)
    packing=None if not a.encoder else {'encoder':a.encoder,'zstd_decoder':a.zstd_decoder,'codecs':a.codecs}
    assemble(inputs,a.image_input,a.output,a.source_commit,packing=packing,check_npm_pack=not a.defer_npm_pack)
if __name__=='__main__':main()
