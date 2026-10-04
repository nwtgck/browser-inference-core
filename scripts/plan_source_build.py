#!/usr/bin/env python3
"""Plan source compilation against an explicitly pinned, validated v4 artifact.

A matching upstream pin alone is not sufficient. Reuse requires every configured
profile/variant fingerprint and recorded validation scope. This local planner does
not fetch remote artifacts, trust Actions caches, push or publish anything.
"""
import argparse
import json
from pathlib import Path
import sys
from package_sources import validate,identity,ROOT,read_regular
sys.path.append(str(ROOT/'llama-cpp/scripts'))
from source_config import load_sources,get_source,fingerprint,matrix

def plan(previous=None, expected_manifest_sha256=None, *, root=ROOT/'llama-cpp'):
    sources=load_sources(root)['sources']; reuse={}; reasons={};manifest=None
    if previous is not None:
        if not isinstance(expected_manifest_sha256,str) or identity(previous/'manifest.json')['sha256']!=expected_manifest_sha256:
            raise ValueError('Previous manifest is not the pinned snapshot')
        manifest=validate(previous,check_npm_pack=False,verify_packed=False)
    return plan_validated_sources(previous, manifest, expected_manifest_sha256, root=root)


def plan_validated_sources(previous, manifest, expected_manifest_sha256, *, root=ROOT/'llama-cpp'):
    """Internal planner after full snapshot identity/coverage admission."""
    sources=load_sources(root)['sources']; reuse={}; reasons={}
    for name in sources:
        entry=get_source(root,name);reason='No verified previous source'
        old=manifest['runtimes']['llama-cpp']['sources'].get(name) if manifest else None
        if old is not None:
            _,reason=can_reuse(previous,old,entry,root)
            if reason is None:
                reuse[name]={'path':str(Path(old['manifest']).parent),'manifestIdentity':old['manifestIdentity'],
                             'buildSourceCommit':old['buildSourceCommit'],'previousManifestSha256':expected_manifest_sha256}
                continue
        reasons[name]=reason
    build=[name for name in sources if name not in reuse]
    return {'formatVersion':1,'compile':{'include':matrix(root,build) if build else []},'reuse':reuse,'reasons':reasons,
            'scope':'Only llama source compilation is planned; root publication must retain and independently validate the image runtime.'}

def can_reuse(previous,metadata,entry,root):
    inner=json.loads(read_regular(previous/metadata['manifest']))
    if inner.get('sourceId')!=entry['id'] or inner.get('llamaCommit')!=entry['commit']:return False,'Source pin/identity changed'
    if set(inner['profiles'])!=set(entry['profiles']):return False,'Provided profile set changed'
    variants=json.loads((root/'config/variants.json').read_text())
    for profile in entry['profiles']:
        actual=inner['profiles'][profile]['variants']
        if set(actual)!=set(variants):return False,'Variant set changed'
        for variant,record in actual.items():
            if record.get('buildInputFingerprint')!=fingerprint(root,entry['id'],profile,variant)['sha256']:return False,'Build inputs changed or unrecorded'
            v=record.get('validation',{})
            if record.get('sourceDirty') is not False or v.get('compiled') is not True or v.get('browserSmoke') is not True:return False,'Missing clean compiled/browser validation'
            scope='syntheticUntrainedModel' if profile.startswith('cpu-') else 'mockedAdapterSuspension'
            result=v.get(scope,{})
            if result.get('passed') is not True or result.get('profile')!=profile or result.get('variant')!=variant:return False,'Missing exact profile/variant smoke evidence'
            if profile.startswith('cpu-'):
                if result.get('syntheticModel') is not True:return False,'Missing CPU scope'
            elif result.get('mockedAdapter') is not True or result.get('suspension') is not True:return False,'Missing WebGPU suspension scope'
    return True,None

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--previous',type=Path);p.add_argument('--previous-manifest-sha256');p.add_argument('--output',type=Path)
    a=p.parse_args();result=plan(a.previous,a.previous_manifest_sha256);text=json.dumps(result,indent=2)+'\n'
    if a.output:a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    else:print(text)
if __name__=='__main__':main()
