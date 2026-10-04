"""Independent upstream tracks; a source is not a browser/test variant.

Only selected-source inputs belong to that source's build fingerprint. This
module deliberately does not infer compatible patches from apply failures.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import subprocess

DEFAULT_SOURCE='upstream-stable'
NAME=re.compile(r'[a-z0-9]+(?:-[a-z0-9]+)*\Z')
SHA=re.compile(r'[a-f0-9]{40}\Z')

def relative_path(value):
    if not isinstance(value,str) or value.startswith('/') or '\\' in value or any(p in ('','.','..') for p in value.split('/')) or not re.fullmatch(r'[A-Za-z0-9_./-]+',value):
        raise ValueError('Unsafe source path')
    return value

def load_sources(root: Path):
    data=json.loads((root/'config/sources.json').read_text())
    if data.get('formatVersion')!=1 or not isinstance(data.get('sources'),dict) or not data['sources']: raise ValueError('Invalid sources configuration')
    profiles=json.loads((root/'config/profiles.json').read_text()); paths=set();pins=set()
    for name,entry in data['sources'].items():
        if not NAME.fullmatch(name): raise ValueError('Invalid source ID')
        if not isinstance(entry,dict) or set(entry)!={'repository','vendorPath','pinFile','profiles','patchSeries','updateTarget'}: raise ValueError('Invalid source entry')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',entry['repository']): raise ValueError('Invalid upstream repository')
        for field in ('vendorPath','pinFile','patchSeries'): relative_path(entry[field])
        if not entry['vendorPath'].startswith('vendor/') or entry['vendorPath'] in paths or entry['pinFile'] in pins: raise ValueError('Shared source pin or vendor')
        paths.add(entry['vendorPath']);pins.add(entry['pinFile'])
        selected=entry['profiles']
        if not isinstance(selected,list) or not selected or len(set(selected))!=len(selected) or any(p not in profiles for p in selected): raise ValueError('Unknown or duplicate source profile')
        if entry['updateTarget'] not in ('latest','latest-unstable','custom'): raise ValueError('Unknown update policy')
    if data.get('defaultSource') not in data['sources']: raise ValueError('Unknown default source')
    return data

def get_source(root: Path, name=DEFAULT_SOURCE):
    data=load_sources(root)
    if name not in data['sources']: raise ValueError('Unknown source: '+name)
    entry=dict(data['sources'][name]);pin=json.loads((root/entry['pinFile']).read_text())
    if not isinstance(pin.get('llamaCommit'),str) or not SHA.fullmatch(pin['llamaCommit']): raise ValueError('Invalid upstream pin')
    return {'id':name,**entry,'commit':pin['llamaCommit']}

def patch_series(root: Path, source):
    entry=get_source(root,source) if isinstance(source,str) else source
    data=json.loads((root/entry['patchSeries']).read_text())
    if data.get('formatVersion')!=1 or not isinstance(data.get('patches'),list): raise ValueError('Invalid patch series')
    components=set(); result=[]
    for p in data['patches']:
        if set(p)!={'component','file','sha256','translationUnit'} or p['component'] in components or p['component'] not in ('audio','vision'): raise ValueError('Invalid patch component')
        components.add(p['component']);relative_path(p['file'])
        if not p['file'].startswith('upstream-patches-only-as-a-last-resort-with-explicit-user-approval/'): raise ValueError('Patch outside exception register')
        if p['translationUnit']!={'audio':'mtmd-audio.cpp','vision':'clip.cpp'}[p['component']]: raise ValueError('Expanded patch scope')
        file=root/p['file']
        if file.is_symlink() or not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=p['sha256']: raise ValueError('Patch identity mismatch')
        result.append(dict(p))
    if components!={'audio','vision'}: raise ValueError('An accepted exception was silently removed')
    return result

def check_gitlink(root: Path, entry, revision='HEAD'):
    fields=subprocess.check_output(['git','ls-tree',revision,'--',entry['vendorPath']],cwd=root,text=True).split()
    if len(fields)!=4 or fields[:3]!=['160000','commit',entry['commit']]: raise ValueError('Source gitlink/pin mismatch')

def matrix(root: Path, selected=None):
    config=load_sources(root); variants=json.loads((root/'config/variants.json').read_text())
    names=list(config['sources']) if selected is None else selected
    if not names or len(set(names))!=len(names): raise ValueError('Invalid source selection')
    return [{'source':name,'profile':profile,'variant':variant}
            for name in names for profile in get_source(root,name)['profiles'] for variant in variants]

def fingerprint(root: Path, name, profile, variant):
    """Conservative build identity: no timestamp or parent repository commit.

    Other tracks' pins/profile lists are excluded. Shared bridge/build/toolchain
    inputs intentionally propagate. This is an identity, not permission to trust
    unverified cached output or reuse failed/missing validation evidence.
    """
    entry=get_source(root,name)
    profiles=json.loads((root/'config/profiles.json').read_text());variants=json.loads((root/'config/variants.json').read_text())
    if profile not in entry['profiles'] or variant not in variants: raise ValueError('Unavailable source/profile/variant')
    inputs={}
    for folder in ('bridge','cmake','scripts','tests'):
        if (root/folder).is_symlink(): raise ValueError('Linked source build directory')
        for p in sorted((root/folder).rglob('*')):
            if p.is_symlink(): raise ValueError('Linked source build input')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc',):
                inputs[p.relative_to(root).as_posix()]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [root/'CMakeLists.txt',*sorted((root.parent/'toolchain').glob('*')), *sorted((root.parent/'scripts').glob('*.py'))]:
        if p.is_symlink(): raise ValueError('Linked shared build input')
        if p.is_file():inputs[str(p.relative_to(root.parent))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for path in [root.parent/'.github/workflows/build.yml', root.parent/'.github/workflows/build-llama-source.yml',
                 *sorted((root.parent/'.github/actions').rglob('*.yml'))]:
        if path.is_symlink(): raise ValueError('Linked workflow build input')
        if path.is_file(): inputs[path.relative_to(root.parent).as_posix()]=hashlib.sha256(path.read_bytes()).hexdigest()
    value={'formatVersion':2,'source':{k:entry[k] for k in ('id','repository','vendorPath','commit')},'profileId':profile,'variantId':variant,'profile':profiles[profile],'variant':variants[variant],'patches':patch_series(root,entry),'inputs':inputs}
    return {'sha256':hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest(),'inputs':value}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--source',action='append');p.add_argument('--matrix',action='store_true')
    a=p.parse_args();print(json.dumps({'include':matrix(a.root,a.source)} if a.matrix else load_sources(a.root),sort_keys=True))
