#!/usr/bin/env python3
"""Append artifact commits, retrying conflicting pushes without rewriting history."""
from __future__ import annotations
import argparse
import base64
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from package_runtime import identity, validate
from pipeline_metrics import measured, span


def run(*args,cwd=None,check=True,env=None):
    phases = {'ls-remote': 'git.remote_head', 'fetch': 'git.fetch_parent',
              'add': 'git.stage', 'write-tree': 'git.write_tree',
              'commit-tree': 'git.commit_tree', 'push': 'git.push'}
    with span(phases.get(args[1] if len(args) > 1 else '', 'git.setup')) as timing:
        result = subprocess.run(args,cwd=cwd,check=check,text=True,capture_output=True,env=env)
        timing['exitCode'] = result.returncode
        return result


def source_commit_subject(source: str, repository: Path) -> str:
    if not re.fullmatch(r'[0-9a-f]{40}', source):
        raise ValueError('Invalid source commit')
    subject = run('git', 'show', '-s', '--format=%s', source, cwd=repository).stdout.rstrip('\n')
    subject = ''.join(char if char.isprintable() else ' ' for char in subject).strip()
    if not subject:
        raise ValueError('Source commit has no subject')
    return subject

@measured('publish.git_blob_verify')
def verify_git_tree(work: Path, tree: str, manifest: dict, manifest_identity: dict) -> None:
    """Check immutable Git blobs, not the working tree filtered by git add.

    Git ignore rules, attributes and changes during npm packing must not make
    the successful publication receipt describe bytes other than those pushed.
    One cat-file process streams raw blobs with bounded memory; no re-packing.
    """
    expected = {item['path']: {key: item[key] for key in ('bytes', 'sha256')}
                for item in manifest['files']}
    expected['manifest.json'] = manifest_identity
    listing = subprocess.check_output(
        ['git', '--no-replace-objects', 'ls-tree', '-r', '-z', '--full-tree', tree], cwd=work)
    objects = {}
    for record in listing.split(b'\0'):
        if not record: continue
        attributes, raw_path = record.split(b'\t', 1)
        mode, kind, oid = attributes.split()
        path = os.fsdecode(raw_path)
        if mode not in (b'100644', b'100755') or kind != b'blob' or path in objects:
            raise ValueError('Non-regular or duplicate entry in Git publication tree: ' + path)
        objects[path] = oid
    if set(objects) != set(expected):
        raise ValueError('Git publication tree differs from the validated package: ' +
                         repr(sorted(set(objects) ^ set(expected))))
    command = ['git', '--no-replace-objects', 'cat-file', '--batch']
    with subprocess.Popen(command, cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as reader:
        try:
            for path, oid in objects.items():
                wanted = expected[path]
                reader.stdin.write(oid + b'\n'); reader.stdin.flush()
                header = reader.stdout.readline(256)
                if header != oid + b' blob ' + str(wanted['bytes']).encode() + b'\n':
                    raise ValueError('Git publication payload size/type mismatch: ' + path)
                remaining = wanted['bytes']; digest = hashlib.sha256()
                while remaining:
                    chunk = reader.stdout.read(min(1024 * 1024, remaining))
                    if not chunk: raise ValueError('Truncated Git publication payload: ' + path)
                    digest.update(chunk); remaining -= len(chunk)
                if reader.stdout.read(1) != b'\n' or digest.hexdigest() != wanted['sha256']:
                    raise ValueError('Git publication payload hash mismatch: ' + path)
            reader.stdin.close()
            if reader.wait() != 0: raise subprocess.CalledProcessError(reader.returncode, command)
        except BaseException:
            # Do not wait for a child blocked writing a rejected, unread blob.
            reader.kill()
            raise


@measured('publish.total')
def publish(package: Path, remote: str, branch='artifacts', attempts=20, *, expected_manifest_sha256: str | None = None):
    if branch != 'artifacts' and not branch.startswith('artifacts/'):
        raise ValueError('Refusing to publish over a source branch')
    run('git','check-ref-format','--branch',branch)
    started=time.monotonic()
    manifest_identity=identity(package/'manifest.json')
    manifest_sha256=manifest_identity['sha256']
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError('Package manifest changed before publication')
    # Check inputs before copytree can dereference links. npm packing belongs to
    # the private snapshot below: that is the tree actually committed and pushed.
    with span('publish.input_check'):
        manifest=validate(package,check_npm_pack=False)
    source=manifest['sourceCommit']
    with tempfile.TemporaryDirectory(prefix='lcb-publish-') as tmp:
        work=Path(tmp)
        with span('publish.snapshot_copy'):
            for p in package.iterdir():
                if p.is_dir(): shutil.copytree(p,work/p.name)
                else: shutil.copy2(p,work/p.name)
        if identity(work/'manifest.json')['sha256'] != manifest_sha256:
            raise ValueError('Package manifest changed while staging publication')
        # Always run all three npm checks, including for standalone publication.
        # Do this before git init, since .git is not part of the runtime payload.
        with span('publish.snapshot_validate'):
            if validate(work) != manifest:
                raise ValueError('Package manifest changed while staging publication')
        print(f'[publication] stage-and-validate: {time.monotonic()-started:.3f}s',file=sys.stderr)
        git_started=time.monotonic()
        run('git','init','-q',cwd=work)
        run('git','config','user.name','github-actions[bot]',cwd=work)
        run('git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com',cwd=work)
        run('git','remote','add','origin',remote,cwd=work)
        run('git','add','--all',cwd=work)
        tree=run('git','write-tree',cwd=work).stdout.strip()
        verify_git_tree(work,tree,manifest,manifest_identity)
        print(f'[publication] git-stage-and-verify: {time.monotonic()-git_started:.3f}s',file=sys.stderr)
        git_started=time.monotonic()
        ref='refs/heads/'+branch
        for attempt in range(attempts):
            result=run('git','ls-remote','--exit-code','--heads','origin',ref,cwd=work,check=False)
            if result.returncode not in (0,2): raise RuntimeError(result.stderr)
            parent=None
            if result.stdout.strip():
                run('git','fetch','--no-tags','--depth=1','origin',ref,cwd=work)
                parent=run('git','rev-parse','FETCH_HEAD',cwd=work).stdout.strip()
            command=['git','commit-tree',tree]
            if parent: command+=['-p',parent]
            command+=['-m',f'build(artifacts): publish runtime from {source[:12]}',
                      '-m',f'Source-commit: {source}']
            commit_env=os.environ.copy()
            commit_env.update({
                'GIT_AUTHOR_NAME':'github-actions[bot]',
                'GIT_AUTHOR_EMAIL':'41898282+github-actions[bot]@users.noreply.github.com',
                'GIT_COMMITTER_NAME':'github-actions[bot]',
                'GIT_COMMITTER_EMAIL':'41898282+github-actions[bot]@users.noreply.github.com',
            })
            commit=run(*command,cwd=work,env=commit_env).stdout.strip()
            pushed=run('git','push','origin',f'{commit}:{ref}',cwd=work,check=False)
            if pushed.returncode==0:
                print(f'[publication] git-publish: {time.monotonic()-git_started:.3f}s',file=sys.stderr)
                return commit
            if 'non-fast-forward' not in pushed.stderr and 'fetch first' not in pushed.stderr and 'cannot lock ref' not in pushed.stderr:
                raise RuntimeError(pushed.stderr)
            with span('git.retry_wait', attempt=attempt+1):
                time.sleep(min(attempt+1,5))
        raise RuntimeError('Publication conflicted repeatedly; rerun this job. No history was overwritten.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--remote',required=True)
    p.add_argument('--branch',default='artifacts')
    p.add_argument('--source-repo',type=Path,
                   help='Checked-out source repository for the Actions summary title')
    a=p.parse_args()
    token=os.environ.get('GH_TOKEN')
    if token and a.remote.startswith('https://github.com/'):
        os.environ['GIT_CONFIG_COUNT']='1'
        os.environ['GIT_CONFIG_KEY_0']='http.https://github.com/.extraheader'
        encoded=base64.b64encode(('x-access-token:'+token).encode()).decode()
        os.environ['GIT_CONFIG_VALUE_0']='AUTHORIZATION: basic '+encoded
    package=a.package.resolve()
    manifest_sha256=identity(package/'manifest.json')['sha256']
    source=json.loads((package/'manifest.json').read_text())['sourceCommit']
    subject=None
    if a.source_repo is not None:
        try:
            subject=source_commit_subject(source,a.source_repo.resolve())
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            print(f'[publication] source commit title unavailable: {error}',file=sys.stderr)
    commit=publish(package,a.remote,a.branch,expected_manifest_sha256=manifest_sha256)
    print(commit)
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as f:
            # This receipt is paired with the commit from this successful push.
            # It is job output only, never a runtime payload or a cross-run cache.
            f.write('commit='+commit+'\nmanifest-sha256='+manifest_sha256+'\n')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        repo=os.environ.get('GITHUB_REPOSITORY','nwtgck/browser-inference-core')
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:
            f.write(f'## Runtime artifact commit\n\n`{commit}`\n\n'
                    f'**Source commit:** `{source}`\n\n')
            if subject is not None:
                f.write(f'**Source commit title:** <code>{escape(subject)}</code>\n\n')
            f.write(f'```sh\nnpm install github:{repo}#{commit}\n```\n')
if __name__=='__main__': main()
