"""Freeze and verify the exact runtime tree that the workflow publishes."""
from __future__ import annotations
import argparse
import hashlib
from html import escape
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from build_tools.github import full_sha, repository_name
from build_tools.metrics import measured, span
from build_tools.packaging.assemble import identity, validate
from build_tools.source.checkout import git_environment


def local_environment() -> dict[str, str]:
    # A caller's GIT_DIR/worktree/index/config must not redirect a private
    # publication snapshot into the source repository or another checkout.
    env = git_environment()
    env['GIT_ALLOW_PROTOCOL'] = ''
    return env


def run(*args, cwd=None, check=True, env=None):
    if args[0] != 'git' or args[1] in ('fetch', 'push', 'ls-remote', 'clone', 'pull'):
        raise ValueError('Publication preparation accepts local Git operations only')
    return subprocess.run(args, cwd=cwd, check=check, text=True, capture_output=True,
                          env=local_environment() if env is None else env)


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
        ['git', '--no-replace-objects', 'ls-tree', '-r', '-z', '--full-tree', tree], cwd=work, env=local_environment())
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
    with subprocess.Popen(command, cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          env=local_environment()) as reader:
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


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + '\n')


def outputs(values: dict[str, str]) -> None:
    destination = os.environ.get('GITHUB_OUTPUT')
    if destination:
        with open(destination, 'a') as stream:
            for key, value in values.items():
                if any(char in value for char in '\r\n\0'):
                    raise ValueError('Invalid workflow output')
                stream.write(key + '=' + value + '\n')


@measured('publish.freeze')
def freeze(package: Path, work: Path, receipt: Path, *, branch: str = 'artifacts',
           expected_source: str | None = None, expected_manifest_sha256: str | None = None,
           source_repo: Path | None = None) -> dict:
    if branch != 'artifacts' and not branch.startswith('artifacts/'):
        raise ValueError('Refusing to publish over a source branch')
    run('git', 'check-ref-format', '--branch', branch)
    package, work, receipt = package.resolve(), work.absolute(), receipt.absolute()
    if work.exists() or work.is_symlink():
        raise ValueError('Publication snapshot must be a fresh directory')
    # Resolve parent links before creating either output. A receipt is state, not
    # a package member: writing it must never overwrite the package being frozen.
    resolved_work, resolved_receipt = work.resolve(), receipt.resolve()
    if (resolved_work.is_relative_to(package)
            or resolved_receipt.is_relative_to(package)
            or resolved_receipt.is_relative_to(resolved_work)
            or resolved_work.is_relative_to(resolved_receipt)):
        raise ValueError('Publication state must be outside the runtime package')
    if receipt.exists() or receipt.is_symlink():
        raise ValueError('Publication receipt must be a fresh file')
    manifest_identity = identity(package / 'manifest.json')
    if expected_manifest_sha256 is not None and manifest_identity['sha256'] != expected_manifest_sha256:
        raise ValueError('Package manifest changed before publication')
    # Validate before copying, so directory links and special files are rejected.
    manifest = validate(package, check_npm_pack=False)
    source = full_sha(manifest['sourceCommit'])
    if expected_source is not None and source != full_sha(expected_source):
        raise ValueError('Package source differs from the selected source commit')
    work.mkdir(parents=True)
    with span('publish.snapshot_copy'):
        for item in package.iterdir():
            if item.is_dir(): shutil.copytree(item, work / item.name)
            else: shutil.copy2(item, work / item.name)
    if identity(work / 'manifest.json') != manifest_identity:
        raise ValueError('Package manifest changed while staging publication')
    # All npm dry-run checks operate on this private immutable publication input.
    with span('publish.snapshot_validate'):
        if validate(work) != manifest:
            raise ValueError('Package manifest changed while staging publication')
    if source_repo is not None:
        # The production CLI requires this source check. Read the nested manifest
        # from the private snapshot and bind the exact parsed bytes to the root
        # manifest before reproducing captured source/preparation receipts.
        from build_tools.packaging.layout import ARTIFACT_DIRS
        from build_tools.runtimes.llama_cpp.provenance import collect
        nested = ARTIFACT_DIRS['llama-cpp'] + '/manifest.json'
        raw = (work / nested).read_bytes()
        expected = next(item for item in manifest['files'] if item['path'] == nested)
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('Source receipt manifest differs from the validated snapshot')
        with span('publish.source_receipts'):
            collect(source_repo.resolve() / 'runtimes/llama-cpp', json.loads(raw))
    run('git', 'init', '-q', cwd=work)
    run('git', 'config', 'user.name', 'github-actions[bot]', cwd=work)
    run('git', 'config', 'user.email', '41898282+github-actions[bot]@users.noreply.github.com', cwd=work)
    run('git', 'add', '--all', cwd=work)
    tree = full_sha(run('git', 'write-tree', cwd=work).stdout.strip())
    verify_git_tree(work, tree, manifest, manifest_identity)
    result = {'schemaVersion': 1, 'sourceCommit': source, 'tree': tree,
              'manifest': manifest_identity, 'branch': branch}
    write_json(receipt, result)
    outputs({'tree': tree, 'source-commit': source, 'manifest-sha256': manifest_identity['sha256']})
    return result


def confirm(work: Path, receipt: Path, commit: str, source_repo: Path | None = None) -> dict:
    """Bind the successful workflow push to the frozen tree before reporting."""
    commit = full_sha(commit)
    frozen = json.loads(receipt.read_text())
    tree = full_sha(run('git', 'rev-parse', '--verify', commit + '^{tree}', cwd=work).stdout.strip())
    if tree != frozen['tree']:
        raise ValueError('Published commit differs from the frozen Git tree')
    raw = subprocess.check_output(['git', '--no-replace-objects', 'show', commit + ':manifest.json'],
                                  cwd=work, env=local_environment())
    if {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} != frozen['manifest']:
        raise ValueError('Published manifest differs from the frozen receipt')
    if json.loads(raw)['sourceCommit'] != frozen['sourceCommit']:
        raise ValueError('Published source differs from the frozen receipt')
    values = {'commit': commit, 'manifest-sha256': frozen['manifest']['sha256']}
    outputs(values)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        repo = repository_name(os.environ.get('GITHUB_REPOSITORY', 'nwtgck/browser-inference-core'))
        source = full_sha(frozen['sourceCommit'])
        subject = None
        if source_repo is not None:
            try:
                subject = source_commit_subject(source, source_repo)
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                print('[publication] source title unavailable: ' + str(error), file=sys.stderr)
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as out:
            out.write(f'## Runtime artifact commit\n\n`{commit}`\n\n**Source commit:** `{source}`\n\n')
            if subject is not None:
                out.write(f'**Source commit title:** <code>{escape(subject)}</code>\n\n')
            out.write(f'```sh\nnpm install github:{repo}#{commit}\n```\n')
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('freeze')
    prepare.add_argument('--package', type=Path, required=True)
    prepare.add_argument('--work', type=Path, required=True)
    prepare.add_argument('--receipt', type=Path, required=True)
    prepare.add_argument('--branch', default='artifacts')
    prepare.add_argument('--source', required=True)
    prepare.add_argument('--source-repo', type=Path, required=True)
    finish = commands.add_parser('confirm')
    finish.add_argument('--work', type=Path, required=True)
    finish.add_argument('--receipt', type=Path, required=True)
    finish.add_argument('--commit', required=True)
    finish.add_argument('--source-repo', type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        freeze(args.package, args.work, args.receipt, branch=args.branch,
               expected_source=args.source, source_repo=args.source_repo)
    else:
        print(confirm(args.work, args.receipt, args.commit, args.source_repo)['commit'])


if __name__ == '__main__': main()
