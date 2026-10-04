#!/usr/bin/env python3
"""Join verified source packages using this run's plan, then pack and validate.

Reused inputs keep their original build commits and must match admitted manifest
identities. Fresh inputs must come from this run's source commit. Both must match
current build/validation inputs; no missing source or profile is silently filled.
"""
import argparse
import json
from pathlib import Path
from package_inputs import read_regular
from package_sources import ROOT, identity, assemble
from plan_source_build import can_reuse
from pipeline_plan import image_reuse, git
from source_config import load_sources, get_source


def assemble_plan(plan_path, inputs, image, output, *, packing):
    plan=json.loads(read_regular(plan_path))
    commit=git('rev-parse','HEAD',cwd=ROOT).stdout.decode().strip()
    if plan.get('sourceCommit')!=commit: raise ValueError('Plan/source checkout mismatch')
    source_names=set(load_sources(ROOT/'llama-cpp')['sources'])
    if set(inputs)!=source_names: raise ValueError('Missing or additional source inputs')
    for name,path in inputs.items():
        entry=get_source(ROOT/'llama-cpp',name)
        inner=json.loads(read_regular(path/'manifest.json'))
        reused=plan['reuse'].get(name)
        if reused:
            if identity(path/'manifest.json')!=reused['manifestIdentity']: raise ValueError('Reused input no longer matches admission')
        elif inner['sourceCommit']!=commit: raise ValueError('Fresh source package is from another commit')
        ok,reason=can_reuse(Path('/'),{'manifest':str((path/'manifest.json').absolute())},entry,ROOT/'llama-cpp')
        if not ok: raise ValueError(name+': '+reason)
    inner=json.loads(read_regular(image/'manifest.json'))
    reused=plan.get('imageReuse')
    if reused:
        if identity(image/'manifest.json')!=reused['manifestIdentity']: raise ValueError('Reused image identity mismatch')
    elif inner['sourceCommit']!=commit: raise ValueError('Fresh image source mismatch')
    metadata={'manifest':str((image/'manifest.json').absolute()),'manifestIdentity':identity(image/'manifest.json'),'buildSourceCommit':inner['sourceCommit']}
    ok,reason=image_reuse(Path('/'),{'runtimes':{'stable-diffusion-cpp':{'sources':{'upstream-default':metadata}}}},ROOT)
    if ok is None: raise ValueError('Image: '+reason)
    receipts = [{'runtime': 'llama-cpp', 'source': name, 'manifestIdentity': item['manifestIdentity']}
                for name, item in sorted(plan['reuse'].items())]
    if plan.get('imageReuse'):
        receipts.append({'runtime': 'stable-diffusion-cpp', 'source': 'upstream-default',
                         'manifestIdentity': plan['imageReuse']['manifestIdentity']})
    reuse_inputs = None
    if receipts:
        if not isinstance(plan.get('previous'), dict):
            raise ValueError('Reused packages require their previous artifact receipt')
        reuse_inputs = {**plan['previous'], 'packages': receipts}
    return assemble(inputs,image,output,commit,packing=packing,check_npm_pack=False,reuse_inputs=reuse_inputs)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--image',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tools',type=Path,required=True)
    a=p.parse_args();names=load_sources(ROOT/'llama-cpp')['sources']
    assemble_plan(a.plan,{name:a.inputs/name for name in names},a.image,a.output,packing={'codecs':['gzip','brotli','zstd'],
        'encoder':a.tools/'wasm-encoder','zstd_decoder':a.tools/'zstd-decoder.wasm'})
if __name__=='__main__':main()
