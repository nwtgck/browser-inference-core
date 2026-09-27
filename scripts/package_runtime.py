#!/usr/bin/env python3
"""Assemble and validate a runtime-only, namespaced monorepo artifact."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import sys
sys.path.append(str(Path(__file__).resolve().parent))
from pipeline_metrics import measured, span
ROOT = Path(__file__).resolve().parents[1]
RUNTIME_NAME = 'llama-cpp-browser-core'
RUNTIMES = ('llama-cpp', 'stable-diffusion-cpp')

def runtime_module(runtime: str):
    if runtime not in RUNTIMES: raise ValueError('Unknown runtime')
    spec = importlib.util.spec_from_file_location(runtime.replace('-', '_') + '_package', ROOT / runtime / 'scripts/package_runtime.py')
    if spec is None or spec.loader is None: raise RuntimeError('Missing runtime package validator')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def identity(path: Path) -> dict:
    if path.is_symlink() or not path.is_file(): raise ValueError('Missing or linked payload')
    with path.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}

@measured('package.validate', packageKind='root')
def validate(directory: Path, require_clean: bool = True, *, check_npm_pack: bool = True) -> dict:
    """Check the complete tree; only npm compression can be explicitly deferred."""
    directory = directory.resolve()
    entries = list(directory.rglob('*'))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in entries):
        raise ValueError('Linked or non-regular entry in runtime package')
    package = json.loads((directory / 'package.json').read_text())
    if package['name'] != RUNTIME_NAME or any(k in package for k in ('scripts', 'dependencies', 'devDependencies', 'optionalDependencies', 'workspaces',
            'peerDependencies', 'peerDependenciesMeta', 'bundleDependencies', 'bundledDependencies')):
        raise ValueError('Runtime package must have no install/build hooks or dependencies')
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['formatVersion'] != 3 or set(manifest['runtimes']) != set(RUNTIMES):
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
        runtime_module(runtime).validate(directory / runtime, require_clean=require_clean, check_npm_pack=check_npm_pack)
        inner = json.loads((directory / runtime / 'manifest.json').read_text())
        if inner['sourceCommit'] != manifest['sourceCommit']: raise ValueError('Mixed source commits')
        if manifest['runtimes'][runtime] != {'manifest': runtime + '/manifest.json', 'manifestFormatVersion': inner['formatVersion']}:
            raise ValueError('Wrong runtime manifest binding')
    if check_npm_pack:
        with span('package.npm_pack', packageKind='root'):
            packed = json.loads(subprocess.check_output(['npm', 'pack', '--dry-run', '--json'], cwd=directory, text=True))
        if {f['path'] for f in packed[0]['files']} != actual: raise ValueError('npm pack tree mismatch')
    return manifest

@measured('package.assemble', packageKind='root')
def assemble(inputs: Path, destination: Path, *, check_npm_pack: bool = True) -> None:
    # CI defers npm packing until publication, not any integrity checks. Keep
    # full validation as the default for callers that only assemble a package.
    source = None; runtimes = {}
    with tempfile.TemporaryDirectory(prefix='lcore-package-') as temporary:
        out = Path(temporary)
        for runtime in RUNTIMES:
            runtime_module(runtime).validate(inputs / runtime, check_npm_pack=check_npm_pack)
            manifest = json.loads((inputs / runtime / 'manifest.json').read_text())
            if source is not None and source != manifest['sourceCommit']: raise ValueError('Cannot mix source revisions')
            source = manifest['sourceCommit']
            shutil.copytree(inputs / runtime, out / runtime)
            runtimes[runtime] = {'manifest': runtime + '/manifest.json', 'manifestFormatVersion': manifest['formatVersion']}
        shutil.copy2(ROOT / 'LICENSE', out / 'LICENSE')
        shutil.copy2(ROOT / 'README.md', out / 'README.md')
        pkg = {'name': RUNTIME_NAME, 'version': '0.1.0', 'private': True, 'type': 'module', 'license': 'MIT',
               'files': [*RUNTIMES, 'manifest.json', 'README.md', 'LICENSE'],
               'exports': {'./stable-diffusion-cpp/examples/runtime': {'types': './stable-diffusion-cpp/examples/runtime/index.d.ts', 'import': './stable-diffusion-cpp/examples/runtime/index.mjs'}, './llama-cpp/*': './llama-cpp/*', './stable-diffusion-cpp/*': './stable-diffusion-cpp/*',
                           './manifest.json': './manifest.json',
                           # Legacy imports remain aliases, not duplicate multi-megabyte payloads.
                           './api/*': './llama-cpp/api/*', './profiles/*': './llama-cpp/profiles/*',
                           './profiles/*/core.mjs': {'types': './llama-cpp/profiles/*/core.d.ts', 'import': './llama-cpp/profiles/*/core.mjs'},
                           './examples/runtime': {'types': './llama-cpp/examples/runtime/index.d.ts', 'import': './llama-cpp/examples/runtime/index.mjs'}}}
        (out / 'package.json').write_text(json.dumps(pkg, indent=2) + '\n')
        manifest = {'formatVersion': 3, 'sourceCommit': source, 'runtimes': runtimes,
                    'files': [{'path': p.relative_to(out).as_posix(), **identity(p)} for p in sorted(out.rglob('*')) if p.is_file()]}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        validate(out, check_npm_pack=check_npm_pack)
        if destination.exists(): shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(out, destination)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, default=ROOT / 'build/package-inputs')
    parser.add_argument('--output', type=Path, default=ROOT / 'dist/package')
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--defer-npm-pack', action='store_true',
                        help='Validate payloads now; require publish_artifacts.py to check npm packing before publication')
    args = parser.parse_args()
    if not args.verify_only: assemble(args.inputs, args.output, check_npm_pack=not args.defer_npm_pack)
    print(json.dumps({'sourceCommit': validate(args.output, check_npm_pack=not args.defer_npm_pack)['sourceCommit']}))

if __name__ == '__main__': main()
