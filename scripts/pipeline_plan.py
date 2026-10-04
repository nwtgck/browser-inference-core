#!/usr/bin/env python3
"""Pin one published Git snapshot, admit reusable raw packages, and plan CI work.

Only this repository's artifacts branch is a reuse source. Its mutable ref is
resolved once, and every subsequent operation uses the returned full commit.
No archive program, install script, decoder or Git checkout/filter from that
snapshot is executed. All bytes, inner manifests, validation and fingerprints
are checked. Unknown old formats cause a cold build; corrupt v4 packages fail.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

from package_inputs import read_regular, copy_regular_tree, file_identity, tree_identity
from package_sources import ROOT, identity, validate, safe_relative
from plan_source_build import plan_validated_sources as llama_plan
from build_identity import image_fingerprint
from validate_runtime_package import image_scope, parse_json
from source_config import get_source, load_sources, check_gitlink, patch_series

MAX_TOTAL = 2 * 1024**3
MAX_FILE = 100 * 1024**2
MAX_FILES = 20000

def git(*args, cwd=None, check=True):
    return subprocess.run(['git', '--no-replace-objects', *map(str, args)], cwd=cwd,
                          check=check, capture_output=True, timeout=180)

def extract_snapshot(repository: Path, commit: str, output: Path):
    """Extract only regular files/directories, without tarfile.extract*."""
    if not re.fullmatch(r'[a-f0-9]{40}', commit):
        raise ValueError('Expected a full artifact commit')
    if output.exists() or output.is_symlink():
        raise ValueError('Snapshot output must be fresh')
    output.mkdir(parents=True)
    seen = set(); total = 0; count = 0
    with subprocess.Popen(['git', '--no-replace-objects', 'archive', '--format=tar', commit],
                          cwd=repository, stdout=subprocess.PIPE) as process:
        try:
            with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
                for member in archive:
                    name = member.name.rstrip('/') if member.isdir() else member.name
                    safe_relative(name)
                    if any(part.startswith('.') for part in name.split('/')):
                        raise ValueError('Hidden repository/control file in runtime snapshot')
                    key = name.casefold(); count += 1
                    if key in seen or count > MAX_FILES:
                        raise ValueError('Duplicate or excessive archive entries')
                    seen.add(key)
                    if member.isdir():
                        (output / name).mkdir(parents=True, exist_ok=True)
                        continue
                    if not member.isfile() or not 0 <= member.size < MAX_FILE:
                        raise ValueError('Linked, non-regular or oversized archive input')
                    total += member.size
                    if total > MAX_TOTAL:
                        raise ValueError('Archive byte budget exceeded')
                    path = output / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.extractfile(member) as src, path.open('xb') as dst:
                        remaining = member.size
                        while remaining:
                            block = src.read(min(remaining, 1024**2))
                            if not block: raise ValueError('Truncated archive')
                            dst.write(block); remaining -= len(block)
                    path.chmod(0o755 if member.mode & 0o111 else 0o644)
            if process.wait(timeout=30):
                raise ValueError('Git snapshot extraction failed')
        except BaseException:
            process.kill()
            raise
    return output

def fetch_snapshot(remote: str, output: Path, *, ref='refs/heads/artifacts'):
    if not re.fullmatch(r'refs/heads/artifacts(?:/[A-Za-z0-9_-]+)*', ref):
        raise ValueError('Only artifact branches can seed reuse')
    result = git('ls-remote', '--exit-code', '--refs', remote, ref, check=False)
    if result.returncode == 2 and not result.stdout:
        return None
    if result.returncode:
        raise RuntimeError('Cannot resolve previous artifact; use cold mode explicitly')
    rows = result.stdout.decode().splitlines()
    if len(rows) != 1:
        raise ValueError('Ambiguous artifact ref')
    commit, name = rows[0].split('\t')
    if name != ref or not re.fullmatch(r'[a-f0-9]{40}', commit):
        raise ValueError('Invalid artifact ref response')
    with tempfile.TemporaryDirectory(prefix='bic-artifact-git-') as td:
        work = Path(td)
        git('init', '--bare', '-q', work)
        git('fetch', '--no-tags', '--depth=1', remote, commit, cwd=work)
        actual = git('rev-parse', 'FETCH_HEAD^{commit}', cwd=work).stdout.decode().strip()
        if actual != commit:
            raise ValueError('Fetched artifact identity changed')
        extract_snapshot(work, commit, output)
    return {'artifactCommit': commit, 'manifestIdentity': identity(output / 'manifest.json')}

def validate_raw_snapshot(directory: Path):
    """Bind all bytes without assuming the old codec/profile catalog matches today.

    Only unchanged raw packages may be reused. Their runtime-specific validators
    run again before restoration/publication, after fingerprint admission.
    """
    manifest = parse_json(read_regular(directory/'manifest.json'))
    tree = tree_identity(directory)
    expected = {}; entries = manifest.get('files')
    if manifest.get('formatVersion') != 4 or not isinstance(entries,list):
        raise ValueError('Invalid previous catalog')
    for entry in entries:
        name = safe_relative(entry['path'])
        if name == 'manifest.json' or name in expected:
            raise ValueError('Duplicate/self-referential previous entry')
        value = {'bytes':entry['bytes'],'sha256':entry['sha256']}
        if type(value['bytes']) is not int or not 0 <= value['bytes'] < MAX_FILE:
            raise ValueError('Invalid previous size')
        if not isinstance(value['sha256'],str) or not re.fullmatch(r'[a-f0-9]{64}',value['sha256']):
            raise ValueError('Invalid previous digest')
        if name not in tree or {k:tree[name].get(k) for k in value} != value:
            raise ValueError('Previous payload hash/size mismatch')
        expected[name] = value
    if {n for n,v in tree.items() if not v.get('directory')} != set(expected)|{'manifest.json'}:
        raise ValueError('Previous payload coverage mismatch')
    groups = manifest.get('runtimes',{})
    if set(groups) != {'llama-cpp','stable-diffusion-cpp'}:
        raise ValueError('Incomplete previous runtime catalog')
    for runtime,group in groups.items():
        if not isinstance(group.get('sources'),dict) or not group['sources']:
            raise ValueError('Empty previous source inventory')
        for source,entry in group['sources'].items():
            if not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*',source):
                raise ValueError('Unsafe previous source')
            name=f'runtimes/{runtime}/sources/{source}/manifest.json'
            if entry.get('manifest')!=name or name not in expected or entry.get('manifestIdentity')!=expected[name]:
                raise ValueError('Unbound previous source manifest')
            inner=parse_json(read_regular(directory/name))
            if entry.get('buildSourceCommit')!=inner.get('sourceCommit'):
                raise ValueError('Previous build provenance mismatch')
    return manifest

def image_reuse(previous: Path, manifest: dict, repo: Path):
    entry = manifest['runtimes']['stable-diffusion-cpp']['sources'].get('upstream-default')
    if entry is None: return None, 'No previous image source'
    inner = json.loads(read_regular(previous / entry['manifest']))
    root = repo / 'stable-diffusion-cpp'
    if inner['upstreams'] != json.loads(read_regular(root / 'config/upstreams.json')):
        return None, 'Image upstream pins changed'
    profiles = json.loads(read_regular(root / 'config/profiles.json'))
    variants = json.loads(read_regular(root / 'config/variants.json'))
    if set(inner['profiles']) != set(profiles): return None, 'Image profile set changed'
    for profile in profiles:
        records = inner['profiles'][profile]['variants']
        if set(records) != set(variants): return None, 'Image variant set changed'
        for variant, record in records.items():
            if record.get('buildInputFingerprint') != image_fingerprint(repo, profile, variant):
                return None, 'Image build/validation inputs changed or unrecorded'
            v = record.get('validation', {})
            if record.get('sourceDirty') is not False or v.get('compiled') is not True or v.get('browserSmoke') is not True:
                return None, 'Missing clean compiled image browser validation'
            if v.get('browserSmokeScope') not in (image_scope(False), image_scope(True)):
                return None, 'Missing image validation scope'
    return {'path': str(Path(entry['manifest']).parent), 'manifestIdentity': entry['manifestIdentity'],
            'buildSourceCommit': entry['buildSourceCommit']}, None

def create_plan(previous, receipt, reuse_output, *, repo=ROOT):
    if reuse_output.exists() or reuse_output.is_symlink(): raise ValueError('Reuse output exists')
    root = repo / 'llama-cpp'; previous_manifest = None
    reason = 'No published artifact'
    if previous is not None:
        if receipt is None or identity(previous/'manifest.json') != receipt['manifestIdentity']:
            raise ValueError('Previous snapshot receipt mismatch')
        old = parse_json(read_regular(previous/'manifest.json'))
        if old.get('formatVersion') == 4:
            # Raw reuse never executes a previous artifact's optional decoder.
            previous_manifest = validate_raw_snapshot(previous)
        elif old.get('formatVersion') == 3:
            reason = 'Legacy artifact has no source-aware reuse identity'
        else:
            raise ValueError('Unknown artifact manifest format')
    result = llama_plan(previous if previous_manifest else None, previous_manifest,
                        receipt['manifestIdentity']['sha256'] if previous_manifest else None, root=root)
    image, image_reason = image_reuse(previous, previous_manifest, repo) if previous_manifest else (None, reason)
    # Preserve each original manifest and its build commit; never mint fresh smoke evidence.
    reuse_output.mkdir(parents=True)
    for name, entry in result['reuse'].items():
        copy_regular_tree(previous/entry['path'], reuse_output/'llama-cpp'/name)
    if image:
        copy_regular_tree(previous/image['path'], reuse_output/'stable-diffusion-cpp')
    sources=[]
    for name in load_sources(root)['sources']:
        entry=get_source(root,name)
        sources.append({'source':name,'vendor':entry['vendorPath'],
                        'noticeProfile':entry['profiles'][0], 'reuse':name in result['reuse']})
    for item in result['compile']['include']:
        entry=get_source(root,item['source']);item.update(vendor=entry['vendorPath'],noticeProfile=entry['profiles'][0])
    result.update(sources={'include':sources},imageReuse=image, imageReason=image_reason,
                  previous=receipt, scope='Validated raw packages only; changed inputs rebuild. Assembly retains original provenance.')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repository', required=True)
    p.add_argument('--cold', action='store_true')
    p.add_argument('--output',type=Path,default=ROOT/'build/source-plan.json')
    p.add_argument('--reuse-output',type=Path,default=ROOT/'build/admitted-reuse')
    a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',a.repository):p.error('Invalid repository')
    for name in load_sources(ROOT/'llama-cpp')['sources']:
        entry = get_source(ROOT / 'llama-cpp', name)
        try:
            check_gitlink(ROOT / 'llama-cpp', entry)
            patch_series(ROOT / 'llama-cpp', entry)
        except ValueError as error:
            p.error(str(error))
    with tempfile.TemporaryDirectory(prefix='bic-previous-') as td:
        snapshot=Path(td)/'package'
        receipt=None if a.cold else fetch_snapshot('https://github.com/'+a.repository+'.git',snapshot)
        result=create_plan(snapshot if receipt else None,receipt,a.reuse_output)
    result['sourceCommit']=git('rev-parse','HEAD',cwd=ROOT).stdout.decode().strip()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    (a.reuse_output/'plan.json').write_bytes(a.output.read_bytes())
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as out:
            for name,value in [('compile-matrix',result['compile']),('source-matrix',result['sources']),
                               ('has-compile',bool(result['compile']['include'])),('reuse-image',bool(result['imageReuse']))]:
                out.write(name+'='+json.dumps(value,separators=(',',':'))+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
