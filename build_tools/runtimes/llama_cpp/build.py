"""Build one profile/variant from captured inputs without modifying upstream."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time

from build_tools.project import runtime_root
from build_tools.source.checkout import checked_sources, commit_file, git, git_environment
from build_tools.source.files import directory, regular_file
from build_tools.toolchain.config import runtime_toolchain
from build_tools.toolchain.patch_emscripten import verify_asyncify_bigint_patch
from build_tools.runtimes.llama_cpp import inputs
from build_tools.runtimes.llama_cpp.gpu_profile import verify as verify_gpu_profile_build
from build_tools.runtimes.llama_cpp.bf16 import verify as verify_bf16_build, runtime_metadata

ROOT = runtime_root('llama-cpp')


def source_status(root: Path) -> list[str]:
    return git(root, 'status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=none').splitlines()


def _configure_cache(build: Path, source: Path) -> None:
    cache = build / 'CMakeCache.txt'
    if not cache.exists():
        return
    home = re.search(r'^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$', regular_file(cache).read_text(), re.MULTILINE)
    if home is None or Path(home[1]).resolve() == source:
        return
    # A cache from the former live-source layout cannot configure another -S.
    # Remove only CMake's cache; selected runtime artifacts remain unpublishable
    # until this complete invocation writes new provenance.
    cache.unlink()
    metadata = build / 'CMakeFiles'
    if metadata.exists():
        shutil.rmtree(directory(metadata))


def main(argv: list[str] | None = None) -> None:
    repository = ROOT.parents[1]
    revision = git(repository, 'rev-parse', 'HEAD').strip()
    profiles = json.loads(commit_file(repository, revision, 'runtimes/llama-cpp/config/profiles.json'))
    variants = json.loads(commit_file(repository, revision, 'runtimes/llama-cpp/config/variants.json'))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=list(profiles), required=True)
    parser.add_argument('--variant', choices=list(variants), required=True)
    parser.add_argument('--fresh', action='store_true', help='Discard this profile/variant build tree before configuration')
    parser.add_argument('--jobs', type=int, default=min(os.cpu_count() or 2, 8))
    args = parser.parse_args(argv)
    if any(not re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', value) for value in (args.profile, args.variant)):
        parser.error('Profile and variant must be simple names')
    if args.jobs <= 0:
        parser.error('--jobs must be a positive integer')
    build = ROOT / 'build' / args.profile / args.variant
    if build.exists():
        directory(build)
        if args.fresh:
            shutil.rmtree(build)
        elif (build / 'provenance.json').exists() or (build / 'provenance.json').is_symlink():
            regular_file(build / 'provenance.json').unlink()
    status_before = source_status(repository)
    toolchain = runtime_toolchain(ROOT, revision=revision)
    upstream = checked_sources(repository, {'llama': inputs.UPSTREAM}, revision=revision)['llama']
    if upstream.commit != toolchain['llamaCommit']:
        parser.error('Selected llama gitlink does not match toolchain.json')
    version = subprocess.check_output(['emcc', '--version'], text=True).splitlines()[0]
    if not re.search(r'(?<!\d)' + re.escape(toolchain['emsdkVersion']) + r'(?!\d)', version):
        parser.error(f'Expected Emscripten {toolchain["emsdkVersion"]}; found {version}')
    directory(build, create=True)
    captured = inputs.capture(repository, revision, upstream, build / 'source-inputs', dirty=bool(status_before))
    cfg = json.loads((captured.runtime / 'config/profiles.json').read_text())[args.profile]
    variant = json.loads((captured.runtime / 'config/variants.json').read_text())[args.variant]
    if runtime_toolchain(captured.runtime) != toolchain:
        parser.error('Captured local toolchain configuration differs from the selected commit')
    if cfg['asyncify']:
        compiler = shutil.which('emcc')
        if compiler is None:
            parser.error('Cannot locate emcc to verify its Asyncify runtime')
        verify_asyncify_bigint_patch(Path(compiler).resolve().parent, toolchain['emscriptenAsyncifyBigIntPatch'])
    build_info = captured.build_info
    assert build_info is not None
    command = ['emcmake', 'cmake', '-S', str(captured.runtime), '-B', str(build), '-G', 'Ninja',
               '-DCMAKE_BUILD_TYPE=Release', '-DLCB_VARIANT=' + args.variant,
               '-DLCB_LLAMA_SOURCE=' + str(captured.upstream),
               # Upstream's supported identity variables preserve build-info
               # without a synthetic .git directory or parent-repository probes.
               '-DLLAMA_BUILD_COMMIT=' + build_info['commit'],
               '-DLLAMA_BUILD_NUMBER=' + str(build_info['number']),
               '-DGIT_EXE:FILEPATH=',
               '-DLCB_MEMORY64=' + ('ON' if cfg['memory64'] else 'OFF'),
               '-DLCB_FILE_BACKED_PLE=' + ('ON' if cfg['memory64'] else 'OFF'),
               '-DLCB_QSA_ALL_POOLS=ON',
               '-DLCB_WEBGPU=' + ('ON' if cfg['webgpu'] else 'OFF'),
               '-DGGML_WEBGPU_GPU_PROFILE=' + ('ON' if cfg.get('gpuProfile', False) else 'OFF'),
               '-DLCB_WEBGPU_BF16_PROJECTOR=' + ('ON' if cfg['webgpu'] else 'OFF'),
               '-DLCB_JSPI=' + ('ON' if cfg['jspi'] else 'OFF'),
               '-DLCB_ASYNCIFY=' + ('ON' if cfg['asyncify'] else 'OFF'),
               '-DLCB_MAXIMUM_MEMORY=' + str(cfg['maximumMemory'])]
    if cfg['webgpu']:
        dawn = repository / '.tools/emdawnwebgpu_pkg'
        if not (dawn / 'emdawnwebgpu.port.py').exists():
            parser.error('Acquire the pinned Dawn archive, then run python3 -m build_tools.toolchain.prepare from the repository root')
        command.append('-DEMDAWNWEBGPU_DIR=' + str(dawn))
    environment = git_environment()
    environment.update({'GIT_CEILING_DIRECTORIES': str(captured.root), 'PYTHONDONTWRITEBYTECODE': '1'})
    _configure_cache(build, captured.runtime)
    subprocess.run(command, cwd=build, env=environment, check=True)
    verify_gpu_profile_build(build / 'compile_commands.json', webgpu=cfg['webgpu'], enabled=cfg.get('gpuProfile', False))
    verify_bf16_build(build / 'compile_commands.json', webgpu=cfg['webgpu'])
    subprocess.run(['cmake', '--build', str(build), '--target', 'core', '-j', str(args.jobs)], cwd=build, env=environment, check=True)
    browser_module = None
    if args.variant == 'browser':
        result = subprocess.run([sys.executable, '-m', 'build_tools.runtimes.llama_cpp.browser_module',
                                 '--module', str(build / 'runtime/core.mjs')],
                                cwd=captured.root / 'repository', env=environment, check=True, capture_output=True, text=True)
        browser_module = json.loads(result.stdout)
    inputs.verify(captured)
    status_after = source_status(repository)
    revision_after = git(repository, 'rev-parse', 'HEAD').strip()
    dirty = bool(status_before or status_after or revision_after != revision)
    if dirty:
        print('Source changes detected; this build cannot be published:\n' + json.dumps({
            'beforeBuild': status_before, 'afterBuild': status_after,
            'commitBeforeBuild': revision, 'commitAfterBuild': revision_after,
        }, indent=2), file=sys.stderr)
    provenance = {'profile': args.profile, 'variant': args.variant, 'configuration': cfg,
                  'variantConfiguration': variant, 'sourceCommit': revision, 'sourceCommitAfterBuild': revision_after,
                  'visionBf16Storage': runtime_metadata(), 'sourceDirty': dirty,
                  'sourceStatusBeforeBuild': status_before, 'sourceStatusAfterBuild': status_after,
                  'llamaCommit': upstream.commit, 'toolchain': toolchain, 'emccVersion': version,
                  'sourceInputs': captured.receipt, 'upstreamBuildInfo': build_info,
                  'preparedSources': inputs.prepared_sources(build, webgpu=cfg['webgpu'], file_rows=cfg['memory64']),
                  'cmakeCommand': command, 'builtAtUnix': int(time.time()),
                  'buildCache': {'identity': os.environ.get('BIC_CACHE_ID'), 'compilerWrapper': os.environ.get('EM_COMPILER_WRAPPER')},
                  'validation': {'compiled': True, 'realModelInference': False, 'browserSmoke': False}}
    provenance['modelGraphBoundary'] = json.loads((build / 'model-graph-chat-overlay/provenance.json').read_text())
    provenance['modelGraphBoundary']['schemaSha256'] = json.loads((build / 'generated/model-graph-v1.schema.json').read_text())['schemaSha256']
    if browser_module is not None:
        provenance['browserModulePreparation'] = browser_module
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=build, delete=False) as stream:
        pending = Path(stream.name)
        json.dump(provenance, stream, indent=2)
        stream.write('\n')
    pending.replace(build / 'provenance.json')
    print(build / 'runtime/core.mjs')


if __name__ == '__main__':
    main()
