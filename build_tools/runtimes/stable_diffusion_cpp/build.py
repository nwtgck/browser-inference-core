#!/usr/bin/env python3
"""Build one isolated image-runtime profile and record exact source provenance."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from build_tools.project import REPOSITORY_ROOT as REPO, runtime_root
ROOT = runtime_root('stable-diffusion-cpp')
from build_tools.runtimes.stable_diffusion_cpp.source import prepare, resolve_sources
from build_tools.runtimes.stable_diffusion_cpp.staging import stage_profile
from build_tools.runtimes.stable_diffusion_cpp import inputs
from build_tools.source.checkout import commit_file, git_environment, git as source_git
from build_tools.source.inputs import file_identities, tree_identity
from build_tools.toolchain.config import load_toolchain
from build_tools.toolchain.patch_emscripten import verify_asyncify_bigint_patch

def git(*args: str, cwd: Path | None = None) -> str:
    return source_git(REPO if cwd is None else cwd, *args).strip()

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--include-toolchain-notices', action='store_true')
    parser.add_argument('--fresh', action='store_true', help='Discard this profile/variant build tree before configuration')
    parser.add_argument('--jobs', type=int, default=min(os.cpu_count() or 2, 4))
    args = parser.parse_args(argv)
    if args.jobs < 1: parser.error('jobs must be positive')
    # Capture the parent before selecting any configuration or upstream pin.
    # Clean builds read those inputs from this immutable commit; development
    # builds can still use staged/worktree inputs but cannot become publishable.
    source_commit = git('rev-parse', 'HEAD')
    before = git('status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=none').splitlines()
    revision = source_commit if not before else None
    def configuration(relative: str) -> dict:
        path = ROOT / relative
        return json.loads(path.read_text() if revision is None
                          else commit_file(REPO, revision, path.relative_to(REPO).as_posix()))
    profiles = configuration('config/profiles.json')
    variants = configuration('config/variants.json')
    if args.profile not in profiles or args.variant not in variants:
        parser.error('Unknown profile or variant')
    config = profiles[args.profile]
    try:
        sources, upstreams = resolve_sources(REPO, ROOT, revision=revision)
    except ValueError as error:
        parser.error(str(error))
    toolchain = load_toolchain(REPO, revision=revision)
    version = subprocess.check_output(['emcc', '--version'], text=True).splitlines()[0]
    if not re.search(r'(?<!\d)' + re.escape(toolchain['emsdkVersion']) + r'(?!\d)', version):
        parser.error('Wrong Emscripten version')
    if config['asyncify']:
        compiler = shutil.which('emcc')
        if compiler is None: parser.error('Missing emcc')
        verify_asyncify_bigint_patch(Path(compiler).resolve().parent, toolchain['emscriptenAsyncifyBigIntPatch'])
    tools = REPO / '.tools'
    dawn = tools / 'emdawnwebgpu_pkg'
    if not (dawn / 'emdawnwebgpu.port.py').is_file(): parser.error('Acquire the pinned local inputs, then run python3 -m build_tools.toolchain.prepare from the repository root')
    build = ROOT / 'build' / args.profile / args.variant
    if args.fresh and build.exists(): shutil.rmtree(build)
    build.mkdir(parents=True, exist_ok=True)
    # A failed new build must not leave the previous success receipt available
    # to a later packaging invocation alongside incomplete compiler outputs.
    (build / 'provenance.json').unlink(missing_ok=True)
    captured = inputs.capture(REPO, source_commit, build / 'source-inputs', dirty=bool(before))
    captured_profiles = json.loads((captured.runtime / 'config/profiles.json').read_text())
    captured_variants = json.loads((captured.runtime / 'config/variants.json').read_text())
    captured_toolchain = json.loads((captured.repository / 'toolchain/config.json').read_text())
    if (captured_profiles.get(args.profile) != config or captured_variants.get(args.variant) != variants[args.variant]
            or captured_toolchain != toolchain):
        raise ValueError('Selected image configuration changed before input capture')
    prepared = build / 'prepared'
    patch_report = prepare(sources, prepared, patches=captured.runtime / 'upstream-patches',
                           upstreams=upstreams, replace=True)
    prepared_identity = file_identities(prepared)
    receipt = prepared / 'patch-provenance.json'
    command = ['emcmake', 'cmake', '-S', str(captured.runtime), '-B', str(build), '-G', 'Ninja',
               '-DCMAKE_BUILD_TYPE=Release', '-DSDCB_VARIANT=' + args.variant,
               '-DSDCB_SOURCE=' + str(prepared / 'stable-diffusion'),
               '-DSDCB_GGML_SOURCE=' + str(prepared / 'ggml' / upstreams['ggml']['sourceSubdirectory']),
               '-DSDCB_PROVENANCE=' + str(receipt),
               '-DSDCB_JSPI=' + ('ON' if config['jspi'] else 'OFF'),
               '-DSDCB_MEMORY64=' + ('ON' if config['memory64'] else 'OFF'),
               '-DSDCB_MAXIMUM_MEMORY=' + str(config['maximumMemory']),
               '-DEMDAWNWEBGPU_DIR=' + str(dawn)]
    environment = git_environment()
    environment.update({'PYTHONPATH': str(captured.repository), 'PYTHONDONTWRITEBYTECODE': '1',
                        'GIT_CEILING_DIRECTORIES': str(captured.root)})
    inputs.reset_legacy_cmake_cache(build, captured.runtime)
    subprocess.run(command, cwd=build, env=environment, check=True)
    subprocess.run(['cmake', '--build', str(build), '--target', 'core', '--parallel', str(args.jobs)],
                   cwd=build, env=environment, check=True)
    inputs.verify(captured)
    if file_identities(prepared) != prepared_identity:
        raise ValueError('Prepared image sources changed during build')
    after = git('status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=none').splitlines()
    source_after = git('rev-parse', 'HEAD')
    source_changed = bool(before or after) or source_after != source_commit
    provenance = {'runtime': 'stable-diffusion-cpp', 'profile': args.profile, 'variant': args.variant,
                  'sourceCommit': source_commit, 'sourceDirty': source_changed,
                  'sourceCommitAfterBuild': source_after,
                  'sourceStatusBeforeBuild': before, 'sourceStatusAfterBuild': after,
                  'upstreams': upstreams, 'configuration': config, 'variantConfiguration': variants[args.variant],
                  'toolchain': toolchain, 'emccVersion': version, 'cmakeCommand': command,
                  'patches': patch_report, 'builtAtUnix': int(time.time()),
                  'sourceInputs': captured.receipt, 'preparedSources': tree_identity(prepared_identity),
                  'buildCache': {'identity': os.environ.get('BIC_CACHE_ID'), 'compilerWrapper': os.environ.get('EM_COMPILER_WRAPPER')},
                  'validation': {'compiled': True, 'browserSmoke': False, 'realModelInference': False}}
    (build / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    if source_changed: raise RuntimeError('Source changed; refusing to stage a publishable build')
    # Only the runtime and provenance cross CI runners, never prepared sources.
    stage = ROOT / 'build/ci-upload' / args.profile / args.variant
    notices = {}
    if args.include_toolchain_notices:
        notices = {'notice_roots': (tools / 'emsdk/upstream/emscripten', dawn),
                   'notice_destination': ROOT / 'build/ci-upload/toolchain-licenses'}
    stage_profile(build, stage, **notices)
    print(stage)

if __name__ == '__main__': main()
