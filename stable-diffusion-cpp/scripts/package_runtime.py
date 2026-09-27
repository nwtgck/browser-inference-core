#!/usr/bin/env python3
"""Package the image bridge, Wasm, contract, provenance, and complete notices."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT.parent / 'scripts'))
from package_inputs import copy_regular, copy_regular_tree, read_regular, regular_path
from pipeline_metrics import measured, span

VARIANTS = json.loads((ROOT / 'config/variants.json').read_text())
PROFILES = json.loads((ROOT / 'config/profiles.json').read_text())
NAME = 'stable-diffusion-cpp-browser-core'
HELPERS = ('index.mjs', 'index.d.ts', 'bindings.mjs', 'read-only-file.mjs', 'README.md')
API_FILES = ('schema.json', 'schema.mjs', 'functions.d.ts')
CAPABILITIES = {'ggufFileOffsetBits': 64, 'callerOwnedRandomAccess': True, 'upstreamApi': True, 'safetensorsFileOffsetBits': 64, 'ggufShards': True}

def sha(path: Path) -> str:
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

@measured('package.validate', packageKind='image')
def validate(directory: Path, require_clean: bool = True, *, check_npm_pack: bool = True) -> dict:
    # This switch defers only compression, never the payload/provenance checks.
    directory = regular_path(directory, directory=True)
    entries = list(directory.rglob('*'))
    if any(p.is_symlink() or not (p.is_file() or p.is_dir()) for p in entries):
        raise ValueError('Linked or non-regular entry in runtime package')
    package = json.loads((directory / 'package.json').read_text())
    if package['name'] != NAME or any(k in package for k in ('scripts', 'dependencies', 'devDependencies', 'optionalDependencies', 'workspaces',
            'peerDependencies', 'peerDependenciesMeta', 'bundleDependencies', 'bundledDependencies')):
        raise ValueError('Not a runtime-only image package')
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['formatVersion'] != 2 or manifest['abiVersion'] != 2 or manifest['runtime'] != 'stable-diffusion-cpp':
        raise ValueError('Unknown image runtime format')
    files = manifest['files']
    expected = {entry['path'] for entry in files}
    actual = {p.relative_to(directory).as_posix() for p in entries if p.is_file()}
    if len(expected) != len(files) or actual != expected | {'manifest.json'}:
        raise ValueError('Manifest must exactly cover the payload')
    for entry in files:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or '\\' in entry['path']:
            raise ValueError('Unsafe manifest path')
        path = directory / relative
        if path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()) or path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            raise ValueError('Invalid image payload: ' + entry['path'])
        if path.stat().st_size >= 100 * 1024**2: raise ValueError('Artifact single-file size limit exceeded')
    for name in API_FILES:
        if 'api/' + name not in expected: raise ValueError('Missing image API schema')
    for name in HELPERS:
        if 'examples/runtime/' + name not in expected: raise ValueError('Missing thin host helper')
    schema = json.loads((directory / 'api/schema.json').read_text())
    digest = sha(directory / 'api/schema.json')
    expected_module = 'export default ' + json.dumps({**schema, 'schemaSha256': digest}, separators=(',', ':')) + ';\n'
    if (schema['abiVersion'] != 2 or manifest.get('schemaSha256') != digest or
            (directory / 'api/schema.mjs').read_text() != expected_module or
            manifest.get('capabilities') != CAPABILITIES):
        raise ValueError('Image schema/capability fingerprint mismatch')
    if set(manifest['profiles']) != set(PROFILES): raise ValueError('Incomplete image profiles')
    for profile, info in manifest['profiles'].items():
        if set(info['variants']) != set(VARIANTS): raise ValueError('Incomplete image variants')
        for variant, data in info['variants'].items():
            if data['sourceCommit'] != manifest['sourceCommit'] or data['upstreams'] != manifest['upstreams']:
                raise ValueError('Mixed image source identity')
            if (data['profile'] != profile or data['variant'] != variant or
                    data['configuration'] != PROFILES[profile] or data['variantConfiguration'] != VARIANTS[variant]):
                raise ValueError('Image profile provenance mismatch')
            if data['validation'].get('compiled') is not True or (require_clean and data['sourceDirty'] is not False):
                raise ValueError('Image runtime is uncompiled or dirty')
            if not data.get('patches', {}).get('patches'): raise ValueError('Missing upstream patch provenance')
            for ext in ('mjs', 'wasm', 'd.ts'):
                if f'profiles/{profile}/{variant}/core.{ext}' not in expected: raise ValueError('Missing image runtime')
            with (directory / f'profiles/{profile}/{variant}/core.wasm').open('rb') as wasm:
                if wasm.read(8) != b'\0asm\1\0\0\0':
                    raise ValueError('Not WebAssembly')
    for required in ('LICENSE', 'licenses/stable-diffusion/LICENSE', 'licenses/embedded/json.hpp.txt', 'licenses/embedded/stb_image.h.txt', 'licenses/embedded/stb_image_resize.h.txt', 'licenses/embedded/stb_image_write.h.txt'):
        if required not in expected: raise ValueError('Missing required image license: ' + required)
    if 'licenses/ggml/LICENSE' not in expected: raise ValueError('Missing ggml repository license')
    for name in ('emscripten', 'emdawnwebgpu_pkg'):
        if not any(path.startswith(f'licenses/toolchain/{name}/') for path in expected):
            raise ValueError('Missing image toolchain notices: ' + name)
    if check_npm_pack:
        with span('package.npm_pack', packageKind='image'):
            packed = json.loads(subprocess.check_output(['npm', 'pack', '--dry-run', '--json'], cwd=directory, text=True))
        if {entry['path'] for entry in packed[0]['files']} != actual: raise ValueError('npm package tree differs')
    return manifest

@measured('package.assemble', packageKind='image')
def package(build_root: Path, destination: Path, license_roots: list[Path], *, check_npm_pack: bool = True) -> None:
    sys.path.insert(0, str(ROOT.parent / 'scripts'))
    from package_notices import collect_notices, collect_subtree_notices
    profiles = {}; source = None; upstreams = None; schema_bytes = None
    with tempfile.TemporaryDirectory(prefix='sdb-package-') as temporary:
        out = Path(temporary)
        for profile in PROFILES:
            profiles[profile] = {'variants': {}}
            for variant in VARIANTS:
                build = build_root / profile / variant
                data = json.loads(read_regular(build / 'provenance.json'))
                if source is not None and (source != data['sourceCommit'] or upstreams != data['upstreams']):
                    raise ValueError('Mixed source commits')
                source = data['sourceCommit']; upstreams = data['upstreams']
                profiles[profile]['variants'][variant] = data
                with span('package.copy_payload', packageKind='image', profile=profile, variant=variant):
                    copy_regular_tree(build / 'runtime', out / 'profiles' / profile / variant)
                current = read_regular(build / 'generated/schema.json')
                if schema_bytes is not None and current != schema_bytes: raise ValueError('Mixed image binding schemas')
                schema_bytes = current
                if not (out / 'api').exists():
                    (out / 'api').mkdir()
                    for name in API_FILES: copy_regular(build / 'generated' / name, out / 'api' / name)
        (out / 'examples/runtime').mkdir(parents=True)
        for name in HELPERS: copy_regular(ROOT / 'examples/runtime' / name, out / 'examples/runtime' / name)

        copy_regular(ROOT / 'LICENSE', out / 'LICENSE')
        copy_regular(ROOT / 'README.md', out / 'README.md')
        sd = ROOT / 'vendor/stable-diffusion.cpp'
        ggml = ROOT / 'vendor/ggml-webgpu-source'
        collect_notices(sd, out / 'licenses/stable-diffusion')
        collect_subtree_notices(ggml, 'ggml', out / 'licenses/ggml')
        (out / 'licenses/embedded').mkdir(parents=True)
        # Preserve the entire embedded notices rather than extract partial licenses.
        for name in ('json.hpp', 'stb_image.h', 'stb_image_resize.h', 'stb_image_write.h'):
            copy_regular(sd / 'thirdparty' / name, out / 'licenses/embedded' / (name + '.txt'))
        roots = {root.name: root for root in license_roots}
        if len(roots) != len(license_roots) or set(roots) != {'emscripten', 'emdawnwebgpu_pkg'}:
            raise ValueError('Provide exactly the Emscripten and Dawn notice roots')
        for name, root in roots.items(): collect_notices(root, out / 'licenses/toolchain' / name)
        pkg = {'name': NAME, 'version': '0.1.0', 'private': True, 'type': 'module', 'license': 'MIT',
               'files': ['api/', 'examples/', 'profiles/', 'licenses/', 'manifest.json', 'README.md', 'LICENSE'],
               'exports': {'./examples/runtime': {'types': './examples/runtime/index.d.ts', 'import': './examples/runtime/index.mjs'}, './api/*': './api/*', './profiles/*': './profiles/*', './manifest.json': './manifest.json'}}
        (out / 'package.json').write_text(json.dumps(pkg, indent=2) + '\n')
        manifest = {'formatVersion': 2, 'runtime': 'stable-diffusion-cpp', 'abiVersion': 2,
                    'schemaSha256': hashlib.sha256(schema_bytes).hexdigest(), 'capabilities': CAPABILITIES,
                    'sourceCommit': source, 'upstreams': upstreams, 'profiles': profiles,
                    'experimental': True, 'files': [{'path': p.relative_to(out).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)}
                    for p in sorted(out.rglob('*')) if p.is_file()]}
        (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        validate(out, check_npm_pack=False)
        if destination.exists() or destination.is_symlink():
            regular_path(destination, directory=True)
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with span('package.commit_output', packageKind='image'):
            copy_regular_tree(out, destination)
        validate(destination, check_npm_pack=check_npm_pack)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build-root', type=Path, default=ROOT / 'build')
    p.add_argument('--output', type=Path, default=ROOT / 'dist/package')
    p.add_argument('--license-root', type=Path, action='append', default=None)
    p.add_argument('--verify-only', action='store_true')
    p.add_argument('--defer-npm-pack', action='store_true',
                   help='Defer compression only; run validate_runtime_package.py before upload')
    a = p.parse_args()
    if a.verify_only and a.defer_npm_pack: p.error('--verify-only always includes npm packing')
    if not a.verify_only:
        roots = a.license_root or [ROOT.parent / '.tools/emsdk/upstream/emscripten', ROOT.parent / '.tools/emdawnwebgpu_pkg']
        package(a.build_root, a.output, roots, check_npm_pack=False)
    print(json.dumps({'sourceCommit': validate(a.output, check_npm_pack=not a.defer_npm_pack)['sourceCommit']}))
