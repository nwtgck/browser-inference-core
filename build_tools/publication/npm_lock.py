"""Prepare an isolated npm fixture and validate its locally produced lockfile.

Only workflow YAML runs npm install. This module never resolves a remote package.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
from urllib.parse import urlparse

from build_tools.github import full_sha, repository_name
from build_tools.packaging.layout import PACKAGE_NAME


def extract_lock(lock: dict, specifier: str, commit: str, package_version: str,
                 package: str = PACKAGE_NAME) -> dict:
    if lock.get('lockfileVersion') != 3:
        raise ValueError('Expected npm lockfileVersion 3')
    packages = lock.get('packages', {})
    key = 'node_modules/' + package
    if set(packages) != {'', key}:
        raise ValueError('Unexpected transitive dependency; this is not a standalone lock fragment')
    if packages[''].get('dependencies') != {package: specifier}:
        raise ValueError('npm changed the requested dependency specifier')
    entry = packages[key]
    if entry.get('version') != package_version or not entry.get('resolved', '').endswith('#' + full_sha(commit)):
        raise ValueError('npm resolved a different runtime version or commit')
    if not specifier.startswith('github:') or specifier.count('#') != 1:
        raise ValueError('Expected an immutable GitHub dependency specifier')
    repository, requested = specifier.removeprefix('github:').split('#')
    repository_name(repository)
    if requested != commit:
        raise ValueError('Requested and resolved artifact commits differ')
    resolved = urlparse(entry['resolved'])
    if (resolved.scheme not in ('git+ssh', 'git+https') or resolved.hostname != 'github.com'
            or resolved.path.removesuffix('.git').lower() != '/' + repository.lower()
            or resolved.query or resolved.password or resolved.port
            or resolved.username not in (None, 'git') or resolved.fragment != commit):
        raise ValueError('npm resolved a different repository or unsupported Git URL')
    if entry.get('hasInstallScript') or any(entry.get(key) for key in (
            'dependencies', 'optionalDependencies', 'peerDependencies', 'peerDependenciesMeta',
            'bundleDependencies', 'bundledDependencies')):
        raise ValueError('Unexpected install-time behavior in the lock entry')
    # Preserve npm's complete entry, including its own integrity, without reconstruction.
    return {'lockfileVersion': 3, 'rootDependencies': packages['']['dependencies'], 'packageEntry': entry}


def prepare(directory: Path, repository: str, commit: str, version: str,
            package: str = PACKAGE_NAME) -> dict:
    specifier = f'github:{repository_name(repository)}#{full_sha(commit)}'
    if directory.exists() or directory.is_symlink():
        raise ValueError('npm resolution requires a fresh directory')
    if not isinstance(version, str) or not version or package != PACKAGE_NAME:
        raise ValueError('Expected the aggregate runtime package and version')
    directory.mkdir(parents=True)
    request = {'repository': repository, 'commit': commit, 'version': version,
               'package': package, 'specifier': specifier}
    package_json = {'name': 'runtime-consumer-lock-fixture', 'version': '0.0.0', 'private': True,
                    'dependencies': {package: specifier}}
    (directory / 'package.json').write_text(json.dumps(package_json, indent=2) + '\n')
    (directory / 'request.json').write_text(json.dumps(request, indent=2) + '\n')
    (directory / 'empty.npmrc').touch()
    (directory / 'empty-global.npmrc').touch()
    return request


def read(directory: Path) -> dict:
    if directory.is_symlink():
        raise ValueError('npm resolution directory must not be linked')
    for name in ('request.json', 'package.json', 'package-lock.json'):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError('Missing or linked npm resolution input: ' + name)
    request = json.loads((directory / 'request.json').read_text())
    expected = {'name': 'runtime-consumer-lock-fixture', 'version': '0.0.0', 'private': True,
                'dependencies': {PACKAGE_NAME: request['specifier']}}
    if request['package'] != PACKAGE_NAME or json.loads((directory / 'package.json').read_text()) != expected:
        raise ValueError('npm resolution fixture changed')
    specifier = f'github:{repository_name(request["repository"])}#{full_sha(request["commit"])}'
    if request['specifier'] != specifier:
        raise ValueError('npm request does not identify the expected repository and commit')
    result = extract_lock(json.loads((directory / 'package-lock.json').read_text()), specifier,
                          request['commit'], request['version'], package=PACKAGE_NAME)
    result.update({'specifier': specifier,
        'nodeVersion': subprocess.check_output(['node', '--version'], text=True).strip(),
        'npmVersion': subprocess.check_output(['npm', '--version'], text=True).strip(),
        'resolutionScope': 'Isolated direct dependency; no transitive dependencies. Consumer root metadata and unrelated lock entries are not represented.'})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare_parser = commands.add_parser('prepare')
    prepare_parser.add_argument('--package', type=Path, required=True)
    prepare_parser.add_argument('--commit', required=True)
    prepare_parser.add_argument('--published-manifest-sha256', required=True)
    prepare_parser.add_argument('--directory', type=Path, required=True)
    validate_parser = commands.add_parser('validate')
    validate_parser.add_argument('--directory', type=Path, required=True)
    validate_parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        from build_tools.publication.consumer import validate_for_report, read_bound_snapshot
        manifest, _ = validate_for_report(args.package, args.published_manifest_sha256)
        files = {entry['path']: entry for entry in manifest['files']}
        raw, _ = read_bound_snapshot(args.package, 'package.json', files['package.json']['sha256'])
        package = json.loads(raw)
        if package['name'] != PACKAGE_NAME:
            raise ValueError('Unexpected runtime package name')
        prepare(args.directory, os.environ['GITHUB_REPOSITORY'], args.commit, package['version'])
    else:
        data = read(args.directory)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__': main()
