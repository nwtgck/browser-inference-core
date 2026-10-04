#!/usr/bin/env python3
"""Build one profile/variant pair without committing generated artifacts."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
sys.path.append(str(Path(__file__).resolve().parents[2] / 'scripts'))
from browser_toolchain import runtime_toolchain
from source_config import get_source, fingerprint, patch_series, DEFAULT_SOURCE
from patch_emscripten import verify_asyncify_bigint_patch

ROOT = Path(__file__).resolve().parents[1]

def output(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()

def source_status(directory: Path) -> list[str]:
    return subprocess.check_output(
        ['git', 'status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=none'],
        cwd=directory, text=True,
    ).splitlines()

def main():
    profiles=json.loads((ROOT/'config/profiles.json').read_text())
    variants=json.loads((ROOT/'config/variants.json').read_text())
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile', choices=list(profiles), required=True)
    p.add_argument('--variant', choices=list(variants), required=True)
    p.add_argument('--source', default=DEFAULT_SOURCE)
    p.add_argument('--fresh', action='store_true', help='Discard this profile/variant build tree before configuration')
    p.add_argument('--jobs',type=int,default=min(os.cpu_count() or 2, 8))
    a=p.parse_args(); cfg=profiles[a.profile]
    source=get_source(ROOT,a.source)
    if a.profile not in source['profiles']: p.error('Profile is not enabled for this source')
    toolchain={**runtime_toolchain(ROOT),'llamaCommit':source['commit']}
    patches=patch_series(ROOT,source)
    source_inputs=fingerprint(ROOT,a.source,a.profile,a.variant)
    src=ROOT/source['vendorPath']
    if not (src/'include/llama.h').exists(): p.error('Run git submodule update --init --recursive')
    sha=output('git','rev-parse','HEAD',cwd=src)
    if sha != toolchain['llamaCommit']: p.error('Submodule HEAD does not match toolchain.json')
    upstream_status=source_status(src)
    if upstream_status:
        p.error('Refusing a dirty upstream checkout:\n'+'\n'.join(upstream_status))
    version=output('emcc','--version').splitlines()[0]
    if not re.search(r'(?<!\d)'+re.escape(toolchain['emsdkVersion'])+r'(?!\d)',version):
        p.error(f'Expected Emscripten {toolchain["emsdkVersion"]}; found {version}')
    if cfg['asyncify']:
        compiler = shutil.which('emcc')
        if compiler is None: p.error('Cannot locate emcc to verify its Asyncify runtime')
        verify_asyncify_bigint_patch(Path(compiler).resolve().parent, toolchain['emscriptenAsyncifyBigIntPatch'])
    source_commit=output('git','rev-parse','HEAD')
    status_before=source_status(ROOT)
    # Assertions affect linked Wasm as well as JavaScript. Each variant owns a
    # separate build tree; never reuse another variant's generated runtime files.
    build_root=ROOT/'build' if a.source==DEFAULT_SOURCE else ROOT/'build/sources'/a.source
    build=build_root/a.profile/a.variant
    if a.fresh and build.exists(): shutil.rmtree(build)
    command=['emcmake','cmake','-S',str(ROOT),'-B',str(build),'-G','Ninja',
             '-DCMAKE_BUILD_TYPE=Release', '-DLCB_VARIANT='+a.variant,
             '-DLCB_LLAMA_SOURCE='+str(src), '-DLCB_SOURCE_ID='+a.source,
             '-DLCB_MEMORY64='+('ON' if cfg['memory64'] else 'OFF'),
             '-DLCB_WEBGPU='+('ON' if cfg['webgpu'] else 'OFF'),
             # Reset cached comparison overrides and record the workaround in
             # cmakeCommand provenance; CPU artifacts must keep the original loader.
             '-DLCB_WEBGPU_BF16_PROJECTOR='+('ON' if cfg['webgpu'] else 'OFF'),
             '-DLCB_JSPI='+('ON' if cfg['jspi'] else 'OFF'),
             '-DLCB_ASYNCIFY='+('ON' if cfg['asyncify'] else 'OFF'),
             '-DLCB_MAXIMUM_MEMORY='+str(cfg['maximumMemory'])]
    if cfg['webgpu']:
        dawn=ROOT.parent/'.tools/emdawnwebgpu_pkg'
        if not (dawn/'emdawnwebgpu.port.py').exists(): p.error('Run ../scripts/setup_toolchain.py to obtain Dawn')
        command.append('-DEMDAWNWEBGPU_DIR='+str(dawn))
    # CMake configure-time probes can write into the process working directory.
    # Keep those incidental outputs in the ignored build tree, not in the source tree.
    build.mkdir(parents=True,exist_ok=True)
    subprocess.run(command,cwd=build,check=True)
    subprocess.run(['cmake','--build',str(build),'--target','core','-j',str(a.jobs)],cwd=build,check=True)
    status_after=source_status(ROOT)
    if status_before or status_after:
        print('Source changes detected; this build cannot be published:\n'+json.dumps({
            'beforeBuild':status_before,'afterBuild':status_after,
        },indent=2),file=sys.stderr)
    provenance={'sourceId':a.source,'sourceRepository':source['repository'],
                'buildInputFingerprint':source_inputs['sha256'],'patchSeries':patches,
                'profile':a.profile,'variant':a.variant,'configuration':cfg,
                'variantConfiguration':variants[a.variant],'sourceCommit':source_commit,
                'sourceDirty': bool(status_before or status_after),
                'sourceStatusBeforeBuild':status_before,'sourceStatusAfterBuild':status_after,
                'llamaCommit':sha,'toolchain':toolchain,'emccVersion':version,
                'cmakeCommand':command,'builtAtUnix':int(time.time()),
                'buildCache': {'identity': os.environ.get('BIC_CACHE_ID'), 'compilerWrapper': os.environ.get('EM_COMPILER_WRAPPER')},
                'validation':{'compiled':True,'realModelInference':False,'browserSmoke':False}}
    (build/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(build/'runtime/core.mjs')
if __name__=='__main__': main()
