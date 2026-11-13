#!/usr/bin/env python3
"""Strict cache partitions for independent browser builds; no runtime artifacts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from build_tools.toolchain.config import load_toolchain
from build_tools.source.checkout import selected_gitlinks

from build_tools.project import REPOSITORY_ROOT as ROOT
RUNTIMES = ('llama-cpp', 'stable-diffusion-cpp')
POLICY = 'bic-v1'


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def cache_plan(root: Path, runtime: str, profile: str, variant: str, source: str,
               *, environment: dict, ccache_version: str) -> dict[str, str]:
    if runtime not in RUNTIMES or not re.fullmatch(r'[0-9a-f]{40}', source):
        raise ValueError('Invalid runtime or source commit')
    profiles = json.loads((root / 'runtimes' / runtime / 'config/profiles.json').read_text())
    variants = json.loads((root / 'runtimes' / runtime / 'config/variants.json').read_text())
    if profile not in profiles or variant not in variants:
        raise ValueError('Unknown profile/variant')
    cfg = load_toolchain(root)
    shared_files = ['build_tools/toolchain/prepare.py', 'build_tools/toolchain/archive.py',
                    'build_tools/toolchain/patch_emscripten.py', 'build_tools/toolchain/config.py',
                    'build_tools/toolchain/cache.py']
    implementation = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in shared_files}
    container = environment.get('BIC_OFFLINE_ENVIRONMENT', '')
    if container and not re.fullmatch(r'[0-9a-f]{64}', container):
        raise ValueError('Invalid offline environment fingerprint')
    compiler_identity = digest({'pins': cfg, 'implementation': implementation,
        'platform': {k: environment.get(k, '') for k in ('RUNNER_OS', 'RUNNER_ARCH', 'ImageOS', 'ImageVersion')},
        'offlineEnvironment': container,
        'python': list(sys.version_info[:2]), 'ccache': ccache_version,
        # No path rewriting/sloppiness. Absolute include paths must agree.
        'workspace': str(root.resolve())})
    configuration = {'profile': profiles[profile], 'variant': variants[variant]}
    partition = f'{runtime}-{profile}-{variant}-{digest(configuration)[:16]}'
    prefix = f'{POLICY}-cc-{compiler_identity}-{partition}-'
    inputs = {'CMakeLists.txt': hashlib.sha256((root / 'runtimes' / runtime / 'CMakeLists.txt').read_bytes()).hexdigest()}
    build_entry = ('build_tools/runtimes/llama_cpp/build.py' if runtime == 'llama-cpp' else
                   'build_tools/runtimes/stable_diffusion_cpp/build.py')
    inputs[build_entry] = hashlib.sha256((root / build_entry).read_bytes()).hexdigest()
    if runtime == 'llama-cpp' and variant == 'browser':
        names = ['build_tools/runtimes/llama_cpp/browser_module.py']
        if profiles[profile]['webgpu']:
            names.append('runtimes/llama-cpp/host/emscripten/host-navigator-pre.js')
        for name in names:
            inputs[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    # A changed upstream or patch can require additional system-library variants.
    # Give EM_CACHE a fresh immutable snapshot then; ccache still reuses unchanged
    # translation units via its own content/command/dependency checks.
    for path in sorted((root / 'runtimes' / runtime / 'config').glob('*.json')):
        inputs['config/' + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('upstream-patches', 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'):
        for path in sorted((root / 'runtimes' / runtime / name).glob('*')):
            if path.suffix in ('.patch', '.json') and path.is_file():
                inputs[name + '/' + path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Source pins live in gitlinks, not duplicated JSON constants. A staged
    # upstream update invalidates the EM snapshot even when its role/path stays
    # unchanged; ccache still owns translation-unit content compatibility.
    if runtime == 'llama-cpp':
        source_paths = {'llama': f'runtimes/{runtime}/vendor/llama.cpp'}
    else:
        declarations = json.loads((root / 'runtimes' / runtime / 'config/upstreams.json').read_text())
        source_paths = {role: f'runtimes/{runtime}/{entry["path"]}' for role, entry in declarations.items()}
    inputs['upstreamCommits'] = selected_gitlinks(root, source_paths)
    return {
        'identity': compiler_identity,
        'dawn-key': f'{POLICY}-dawn-{cfg["dawnSha256"]}',
        'dawn-path': f'.tools/downloads/emdawnwebgpu_pkg-{cfg["dawnTag"]}.zip',
        # Restore after emsdk setup. Keep the SDK's populated sysroot on a miss.
        'em-path': '.tools/emsdk/upstream/emscripten/cache',
        'em-key': f'{POLICY}-em-{compiler_identity}-{partition}-{digest(inputs)[:16]}',
        'cc-path': f'.cache/ccache/{runtime}/{profile}/{variant}',
        'cc-prefix': prefix,
        # Actions caches are immutable. A new source snapshot can retain new objects.
        'cc-key': prefix + source,
    }


def configure(root: Path, plan: dict[str, str]) -> dict[str, str]:
    config = root / '.tools/ccache.conf'
    config.parent.mkdir(parents=True, exist_ok=True)
    # This configuration is recreated from source, never restored from the cache.
    config.write_text('compiler_check = content\ndirect_mode = false\n'
                      'sloppiness =\nhard_link = false\nfile_clone = false\n'
                      'max_size = 256M\ncompression = true\n'
                      f'namespace = {plan["identity"]}\n')
    return {'EM_CACHE': str(root / plan['em-path']),
            # Wrap real clang invocations, not Emscripten's Python driver. The
            # driver still runs its port/library setup even on an object hit.
            'EM_COMPILER_WRAPPER': 'ccache',
            'CCACHE_DIR': str(root / plan['cc-path']),
            'CCACHE_CONFIGPATH': str(config),
            'BIC_CACHE_ID': plan['identity']}


def write_values(file: str, values: dict[str, str]) -> None:
    with open(file, 'a') as stream:
        for key, value in values.items():
            if any(c in value for c in ('\n', '\r', '\0')):
                raise ValueError('Unsafe workflow output')
            stream.write(f'{key}={value}\n')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', choices=RUNTIMES, required=True)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--variant', required=True)
    args = parser.parse_args()
    version = subprocess.check_output(['ccache', '--version'], text=True).splitlines()[0]
    plan = cache_plan(ROOT, args.runtime, args.profile, args.variant,
                      os.environ['LCB_SOURCE_COMMIT'], environment=os.environ, ccache_version=version)
    values = configure(ROOT, plan)
    write_values(os.environ['GITHUB_OUTPUT'], plan)
    write_values(os.environ['GITHUB_ENV'], values)
    print(json.dumps(plan, indent=2))

if __name__ == '__main__':
    main()
