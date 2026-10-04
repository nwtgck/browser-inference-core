#!/usr/bin/env python3
"""Restore a run-local admitted raw package; never rewrite its build provenance."""
import argparse
import json
import os
from pathlib import Path
from package_inputs import read_regular, copy_regular_tree
from package_sources import ROOT, identity, runtime_validator
from plan_source_build import can_reuse
from source_config import get_source
from pipeline_plan import image_reuse, git


def restore(download: Path, output: Path, runtime: str, source: str | None):
    plan = json.loads(read_regular(download/'plan.json'))
    selected = git('rev-parse','HEAD',cwd=ROOT).stdout.decode().strip()
    if plan.get('sourceCommit') != selected or os.environ.get('LCB_SOURCE_COMMIT',selected) != selected:
        raise ValueError('Reuse plan belongs to another source revision')
    if runtime == 'llama-cpp':
        config = get_source(ROOT/'llama-cpp',source)
        entry = plan['reuse'].get(source)
        if entry is None: raise ValueError('Source was not admitted for reuse')
        package = download/'llama-cpp'/source
        metadata = {**entry,'manifest':str(package/'manifest.json')}
        ok,reason = can_reuse(Path('/'),metadata,config,ROOT/'llama-cpp')
        if not ok: raise ValueError('Source no longer reusable: '+reason)
    elif runtime == 'stable-diffusion-cpp':
        entry = plan.get('imageReuse')
        if entry is None: raise ValueError('Image was not admitted for reuse')
        package = download/'stable-diffusion-cpp'
        metadata = {**entry,'manifest':str(package/'manifest.json')}
        result,reason = image_reuse(Path('/'),{'runtimes':{'stable-diffusion-cpp':{'sources':{'upstream-default':metadata}}}},ROOT)
        if result is None: raise ValueError('Image no longer reusable: '+reason)
    else: raise ValueError('Unknown runtime')
    if identity(package/'manifest.json') != entry['manifestIdentity']:
        raise ValueError('Reused manifest differs from admitted snapshot')
    runtime_validator(runtime)(package,check_npm_pack=True)
    copy_regular_tree(package,output)
    if identity(output/'manifest.json') != entry['manifestIdentity']:
        raise ValueError('Copied reuse manifest changed')
    return entry


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--runtime',choices=['llama-cpp','stable-diffusion-cpp'],required=True);p.add_argument('--source')
    a=p.parse_args();print(json.dumps(restore(a.download,a.output,a.runtime,a.source),indent=2))
if __name__=='__main__':main()
