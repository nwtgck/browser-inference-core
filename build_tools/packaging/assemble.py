#!/usr/bin/env python3
"""Assemble and validate a runtime-only, namespaced monorepo artifact."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
from build_tools.metrics import measured, span
from build_tools.output import output_directories
from build_tools.packaging.files import copy_regular, copy_regular_tree, regular_path
from build_tools.packaging.layout import ARTIFACT_DIRS, MANIFEST_VERSION, PACKAGE_NAME, package_exports
from build_tools.project import REPOSITORY_ROOT as ROOT
RUNTIME_NAME = PACKAGE_NAME
RUNTIMES = ('llama-cpp', 'stable-diffusion-cpp')

def runtime_module(runtime: str):
    # Do not import the other runtime: its configuration is not an input of the
    # selected adapter. Aggregate assembly explicitly selects both when needed.
    if runtime == "llama-cpp":
        from build_tools.runtimes.llama_cpp import package
    elif runtime == "stable-diffusion-cpp":
        from build_tools.runtimes.stable_diffusion_cpp import package
    else:
        raise ValueError("Unknown runtime")
    return package


def identity(path: Path) -> dict:
    if path.is_symlink() or not path.is_file(): raise ValueError('Missing or linked payload')
    with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}

@measured('package.validate', packageKind='root')
def validate(directory: Path, require_clean: bool = True, *, check_npm_pack: bool = True) -> dict:
    """Check the complete tree; only npm compression can be explicitly deferred."""
    directory = regular_path(directory, directory=True)
    entries = list(directory.rglob('*'))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in entries):
        raise ValueError('Linked or non-regular entry in runtime package')
    package = json.loads((directory / 'package.json').read_text())
    if package['name'] != RUNTIME_NAME or any(k in package for k in ('scripts', 'dependencies', 'devDependencies', 'optionalDependencies', 'workspaces',
            'peerDependencies', 'peerDependenciesMeta', 'bundleDependencies', 'bundledDependencies')):
        raise ValueError('Runtime package must have no install/build hooks or dependencies')
    if package.get('exports') != package_exports():
        raise ValueError('Unexpected public package exports')
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['formatVersion'] != MANIFEST_VERSION or set(manifest['runtimes']) != set(RUNTIMES):
        raise ValueError('Incomplete or unknown multi-runtime manifest')
    if not isinstance(manifest['sourceCommit'], str) or not re.fullmatch(r'[0-9a-f]{40}', manifest['sourceCommit']):
        raise ValueError('Invalid source identity')
    files = manifest['files']; expected = {entry['path'] for entry in files}
    actual = {p.relative_to(directory).as_posix() for p in entries if p.is_file()}
    if len(files) != len(expected) or actual != expected | {'manifest.json'}:
        raise ValueError('Manifest does not exactly cover the complete package tree')
    for entry in files:
        relative = Path(entry['path'])
        path = directory / relative
        if relative.is_absolute() or '..' in relative.parts or '\\' in entry['path'] or not path.resolve().is_relative_to(directory):
            raise ValueError('Unsafe artifact path')
        if identity(path) != {k: entry[k] for k in ('bytes', 'sha256')}:
            raise ValueError('Artifact hash/size mismatch: ' + entry['path'])
        if entry['bytes'] >= 100 * 1024**2: raise ValueError('Single-file artifact size limit exceeded')
    for runtime in RUNTIMES:
        artifact_dir = ARTIFACT_DIRS[runtime]
        runtime_module(runtime).validate(directory / artifact_dir, require_clean=require_clean, check_npm_pack=check_npm_pack)
        inner = json.loads((directory / artifact_dir / 'manifest.json').read_text())
        if inner['sourceCommit'] != manifest['sourceCommit']: raise ValueError('Mixed source commits')
        if manifest['runtimes'][runtime] != {'manifest': artifact_dir + '/manifest.json', 'manifestFormatVersion': inner['formatVersion']}:
            raise ValueError('Wrong runtime manifest binding')
    if check_npm_pack:
        with span('package.npm_pack', packageKind='root'):
            packed = json.loads(subprocess.check_output(['npm', 'pack', '--dry-run', '--json'], cwd=directory, text=True))
        if {f['path'] for f in packed[0]['files']} != actual: raise ValueError('npm pack tree mismatch')
    return manifest

@measured('package.assemble', packageKind='root')
def assemble(inputs: Path, destination: Path, *, check_npm_pack: bool = True) -> dict:
    # CI defers npm packing until publication, not any integrity checks. Keep
    # full validation as the default for callers that only assemble a package.
    source = None; runtimes = {}
    with output_directories(destination, replace=True) as (out,):
        for runtime in RUNTIMES:
            artifact_dir = ARTIFACT_DIRS[runtime]
            runtime_module(runtime).validate(inputs / runtime, check_npm_pack=False)
            manifest = json.loads((inputs / runtime / 'manifest.json').read_text())
            if source is not None and source != manifest['sourceCommit']: raise ValueError('Cannot mix source revisions')
            source = manifest['sourceCommit']
            copy_regular_tree(inputs / runtime, out / artifact_dir)
            runtimes[runtime] = {'manifest': artifact_dir + '/manifest.json', 'manifestFormatVersion': manifest['formatVersion']}
        copy_regular(ROOT / 'LICENSE', out / 'LICENSE')
        copy_regular(ROOT / 'build_tools/packaging/runtime-readme.md', out / 'README.md')
        pkg = {'name': RUNTIME_NAME, 'version': '0.1.0', 'private': True, 'type': 'module', 'license': 'MIT',
               'files': ['runtimes/', 'manifest.json', 'README.md', 'LICENSE'],
               'exports': package_exports()}
        (out / 'package.json').write_text(json.dumps(pkg, indent=2) + '\n')
        manifest = {'formatVersion': MANIFEST_VERSION, 'sourceCommit': source, 'runtimes': runtimes,
                    'files': [{'path': p.relative_to(out).as_posix(), **identity(p)} for p in sorted(out.rglob('*')) if p.is_file()]}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        result = validate(out, check_npm_pack=check_npm_pack)
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'build/package-inputs')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/package')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--defer-npm-pack', action='store_true',
                        help='Validate payloads now; the publication step must check npm packing before upload')
    args = parser.parse_args()
    if args.verify_only and args.defer_npm_pack:
        parser.error('--verify-only always includes npm packing')
    result = (validate(args.output) if args.verify_only else
              assemble(args.inputs, args.output, check_npm_pack=not args.defer_npm_pack))
    print(json.dumps({'sourceCommit': result['sourceCommit']}))

if __name__ == '__main__': main()
