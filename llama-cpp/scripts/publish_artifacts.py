#!/usr/bin/env python3
"""Append artifact commits, retrying conflicting pushes without rewriting history."""
from __future__ import annotations
import argparse
import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
from package_runtime import validate


def run(*args,cwd=None,check=True,env=None):
    return subprocess.run(args,cwd=cwd,check=check,text=True,capture_output=True,env=env)

def publish(package: Path, remote: str, branch='artifacts', attempts=20):
    if branch != 'artifacts' and not branch.startswith('artifacts/'):
        raise ValueError('Refusing to publish over a source branch')
    run('git','check-ref-format','--branch',branch)
    validate(package)
    manifest=json.loads((package/'manifest.json').read_text())
    source=manifest['sourceCommit']
    with tempfile.TemporaryDirectory(prefix='lcb-publish-') as tmp:
        work=Path(tmp)
        run('git','init','-q',cwd=work)
        run('git','config','user.name','github-actions[bot]',cwd=work)
        run('git','config','user.email','41898282+github-actions[bot]@users.noreply.github.com',cwd=work)
        run('git','remote','add','origin',remote,cwd=work)
        for p in package.iterdir():
            if p.is_dir(): shutil.copytree(p,work/p.name)
            else: shutil.copy2(p,work/p.name)
        run('git','add','--all',cwd=work)
        tree=run('git','write-tree',cwd=work).stdout.strip()
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
            if pushed.returncode==0: return commit
            if 'non-fast-forward' not in pushed.stderr and 'fetch first' not in pushed.stderr and 'cannot lock ref' not in pushed.stderr:
                raise RuntimeError(pushed.stderr)
            time.sleep(min(attempt+1,5))
        raise RuntimeError('Publication conflicted repeatedly; rerun this job. No history was overwritten.')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,required=True)
    p.add_argument('--remote',required=True)
    p.add_argument('--branch',default='artifacts')
    a=p.parse_args()
    token=os.environ.get('GH_TOKEN')
    if token and a.remote.startswith('https://github.com/'):
        os.environ['GIT_CONFIG_COUNT']='1'
        os.environ['GIT_CONFIG_KEY_0']='http.https://github.com/.extraheader'
        encoded=base64.b64encode(('x-access-token:'+token).encode()).decode()
        os.environ['GIT_CONFIG_VALUE_0']='AUTHORIZATION: basic '+encoded
    commit=publish(a.package.resolve(),a.remote,a.branch)
    print(commit)
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'],'a') as f: f.write('commit='+commit+'\n')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        repo=os.environ.get('GITHUB_REPOSITORY','nwtgck/browser-inference-core')
        with open(os.environ['GITHUB_STEP_SUMMARY'],'a') as f:
            f.write(f'## Runtime artifact commit\n\n`{commit}`\n\n```sh\nnpm install github:{repo}#{commit}\n```\n')
if __name__=='__main__': main()
