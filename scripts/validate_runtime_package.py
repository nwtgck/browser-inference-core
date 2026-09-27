#!/usr/bin/env python3
"""Test one package snapshot and pack those same bytes; never rebuild after smoke."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import uuid

from package_inputs import digest, file_identity, read_regular, regular_path, tree_identity
from package_runtime import ROOT, RUNTIMES, runtime_module
from pipeline_metrics import exit_like_child, run_command, span

TEST_INPUTS = {
    'llama-cpp': ('tests/asyncify-rewind.mjs', 'tests/browser-smoke.mjs',
                  'tests/chat-surface.mjs', 'scripts/make_test_model.py',
                  'vendor/llama.cpp/models/templates/Qwen-Qwen3-0.6B.jinja'),
    'stable-diffusion-cpp': ('tests/browser-smoke.mjs', 'tests/gguf-fixture.mjs',
                             'tests/model-io-fixtures.mjs'),
}
IMAGE_SCOPE = ('real-Wasm Worker, public records/callbacks, sparse GGUF/safetensors/shard I/O; '
               'test variants also check synthetic Qwen BF16 timestep graph arithmetic on ')


def parse_json(raw: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Invalid JSON number')))


def image_scope(test_webgpu: bool) -> str:
    return (IMAGE_SCOPE + ('CPU and WebGPU' if test_webgpu else 'CPU (no GPU inference)') +
            ', plus deep graph construction/selection; no trained-model image generation')


def checked_results(runtime: str, profiles: dict, variants: dict, results: object,
                    *, test_webgpu: bool = False) -> list[dict]:
    expected = {(profile, variant) for profile in profiles for variant in variants}
    if not isinstance(results, list) or len(results) != len(expected):
        raise ValueError('Missing or extra browser test results')
    seen = set()
    for result in results:
        if not isinstance(result, dict):
            raise ValueError('Invalid browser test result')
        pair = (result.get('profile'), result.get('variant'))
        if (not all(isinstance(value, str) for value in pair) or pair not in expected or
                pair in seen or result.get('passed') is not True):
            raise ValueError('Duplicate, unknown or failed browser test result')
        seen.add(pair)
        if runtime == 'llama-cpp':
            if pair[0].startswith('cpu-'):
                if result.get('syntheticModel') is not True:
                    raise ValueError('Wrong CPU smoke scope')
            elif (result.get('mockedAdapter') is not True or result.get('suspension') is not True or
                  result.get('syntheticModel', False) is not False):
                raise ValueError('Wrong GPU smoke scope')
        else:
            if result.get('scope') != image_scope(test_webgpu):
                raise ValueError('Wrong image smoke scope')
            if ((pair[1] == 'test' and result.get('graphWalk') is not True) or
                    (pair[1] == 'browser' and 'graphWalk' in result)):
                raise ValueError('Wrong image graph walk evidence')
    return results


def add_validation(manifest: dict, runtime: str, results: list[dict]) -> dict:
    updated = copy.deepcopy(manifest)
    for result in results:
        validation = updated['profiles'][result['profile']]['variants'][result['variant']]['validation']
        validation['browserSmoke'] = True
        if runtime == 'llama-cpp':
            scope = 'syntheticUntrainedModel' if result['profile'].startswith('cpu-') else 'mockedAdapterSuspension'
            validation[scope] = result
        else:
            validation['browserSmokeScope'] = result['scope']
    return updated


def test_identity(runtime_root: Path, runtime: str, *, fixture: bool = True) -> dict:
    names = [*TEST_INPUTS[runtime], 'config/profiles.json', 'config/variants.json']
    if runtime == 'llama-cpp' and fixture:
        names.append('build/fixture.gguf')
    return {name: file_identity(runtime_root / name) for name in names}


def execution_context(runtime: str, source: str) -> dict:
    if not re.fullmatch(r'[0-9a-f]{40}', source):
        raise ValueError('Invalid source commit')
    selected = os.environ.get('LCB_SOURCE_COMMIT', source)
    if selected != source:
        raise ValueError('Package does not match selected source commit')
    run = os.environ.get('GITHUB_RUN_ID', 'local')
    attempt = os.environ.get('GITHUB_RUN_ATTEMPT', '1')
    job = os.environ.get('GITHUB_JOB', 'local')
    if ((run != 'local' and not re.fullmatch(r'[0-9]{1,32}', run)) or
            not re.fullmatch(r'[1-9][0-9]{0,8}', attempt) or
            not re.fullmatch(r'[A-Za-z0-9_-]{1,100}', job)):
        raise ValueError('Invalid run identity')
    return {'schemaVersion': 1, 'runtime': runtime, 'sourceCommit': source,
            'runId': run, 'runAttempt': attempt, 'job': job, 'sessionId': uuid.uuid4().hex}


def assert_snapshot(package: Path, expected: dict, runtime_root: Path,
                    runtime: str, tests: dict) -> None:
    with span('test.snapshot_check', runtime=runtime):
        if tree_identity(package) != expected:
            raise ValueError('Tested package snapshot changed')
        if test_identity(runtime_root, runtime) != tests:
            raise ValueError('Test implementation or fixture changed')


def _run(command: list[str], phase: str, cwd: Path, env: dict | None = None) -> None:
    code = run_command(command, phase, cwd=cwd, env=env)
    if code:
        # Do not put all arguments (or environment contents) into an error log.
        raise subprocess.CalledProcessError(code, [phase])


def validate_package(runtime: str, package: Path, receipt_path: Path) -> dict:
    if runtime not in RUNTIMES:
        raise ValueError('Unknown runtime')
    runtime_root = ROOT / runtime
    package = regular_path(package, directory=True)
    if receipt_path.resolve().is_relative_to(package):
        raise ValueError('Validation receipts must stay outside the package')
    # Clear stale success before any validation or test can fail.
    if receipt_path.exists() or receipt_path.is_symlink():
        receipt_path.unlink()
    validator = runtime_module(runtime)
    with span('test.snapshot_prepare', runtime=runtime):
        before = read_regular(package / 'manifest.json')
        manifest = parse_json(before)
        validator.validate(package, check_npm_pack=False)  # includes clean source check
        snapshot = tree_identity(package)
        if snapshot['manifest.json']['sha256'] != hashlib.sha256(before).hexdigest():
            raise ValueError('Manifest changed while capturing test snapshot')
        context = execution_context(runtime, manifest['sourceCommit'])
        profiles = parse_json(read_regular(runtime_root / 'config/profiles.json'))
        variants = parse_json(read_regular(runtime_root / 'config/variants.json'))
        if set(manifest['profiles']) != set(profiles):
            raise ValueError('Smoke requires every configured profile')
        for data in manifest['profiles'].values():
            if set(data['variants']) != set(variants):
                raise ValueError('Smoke requires every configured variant')
        implementation = test_identity(runtime_root, runtime, fixture=False)
    if runtime == 'llama-cpp':
        _run([sys.executable, 'scripts/make_test_model.py', 'build/fixture.gguf'],
             'test.generate_fixture', runtime_root)
    if test_identity(runtime_root, runtime, fixture=False) != implementation:
        raise ValueError('Test implementation changed while generating fixture')
    tests = test_identity(runtime_root, runtime)
    context.update(snapshotSha256=digest(snapshot), testInputsSha256=digest(tests),
                   testWebGpu=(runtime == 'stable-diffusion-cpp' and os.environ.get('SDCB_TEST_WEBGPU') == '1'))
    assert_snapshot(package, snapshot, runtime_root, runtime, tests)
    # A fresh, private result path plus a random, run-bound envelope prevents
    # stale success JSON from a previous attempt from being mistaken for this run.
    with tempfile.TemporaryDirectory(prefix='bic-smoke-') as temporary:
        # Canonicalize our own directory (e.g. the system /var alias on macOS),
        # never untrusted payload paths whose links must remain visible.
        result_path = Path(temporary).resolve() / 'results.json'
        env = dict(os.environ, BIC_BROWSER_RESULTS_FILE=str(result_path),
                   BIC_BROWSER_SESSION_JSON=json.dumps(context, separators=(',', ':')))
        if runtime == 'llama-cpp':
            _run(['node', 'tests/asyncify-rewind.mjs', str(package / 'profiles/webgpu-wasm32-asyncify/test')],
                 'test.node_asyncify', runtime_root, env)
            assert_snapshot(package, snapshot, runtime_root, runtime, tests)
        command = ['node', 'tests/browser-smoke.mjs', str(package)]
        if runtime == 'llama-cpp':
            command.append(str(runtime_root / 'build/fixture.gguf'))
        _run(command, 'test.chromium_smoke', runtime_root, env)
        assert_snapshot(package, snapshot, runtime_root, runtime, tests)
        envelope = parse_json(read_regular(result_path))
        if (not isinstance(envelope, dict) or set(envelope) != {'session', 'results'} or
                digest(envelope['session']) != digest(context)):
            raise ValueError('Browser result is not bound to this test session')
        results = checked_results(runtime, profiles, variants, envelope['results'],
                                  test_webgpu=context['testWebGpu'])
    with span('package.finalize_validation', runtime=runtime):
        updated = add_validation(manifest, runtime, results)
        final_bytes = (json.dumps(updated, indent=2) + '\n').encode()
        expected = copy.deepcopy(snapshot)
        expected['manifest.json'].update(bytes=len(final_bytes), sha256=hashlib.sha256(final_bytes).hexdigest())
        manifest_path = package / 'manifest.json'
        # Only the allowed validation fields change. No payload is fetched again
        # from the mutable build directory, including generated API or notices.
        manifest_path.write_bytes(final_bytes)
        try:
            assert_snapshot(package, expected, runtime_root, runtime, tests)
            with span('package.final_verify', runtime=runtime, context='final'):
                validator.validate(package)  # Exactly one final npm pack; no skip option.
            # npm is allowed to read, never to change the snapshot that was tested.
            assert_snapshot(package, expected, runtime_root, runtime, tests)
        except BaseException:
            # A failure never leaves a partially completed validation receipt.
            manifest_path.write_bytes(before)
            raise
    receipt = {**context, 'status': 'complete', 'finalSnapshotSha256': digest(expected),
               'finalManifestSha256': expected['manifest.json']['sha256'],
               'nodeAsyncifyPassed': True if runtime == 'llama-cpp' else None,
               'results': results}
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    # Receipts are auxiliary local state, not published metadata or a cache.
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', choices=RUNTIMES, required=True)
    parser.add_argument('--package', type=Path, default=Path('dist/package'))
    parser.add_argument('--receipt', type=Path, default=Path('build/package-validation.json'))
    args = parser.parse_args()
    try:
        result = validate_package(args.runtime, args.package, args.receipt)
    except subprocess.CalledProcessError as error:
        exit_like_child(error.returncode)
    print(json.dumps({'runtime': args.runtime, 'sourceCommit': result['sourceCommit'], 'status': 'complete'}))


if __name__ == '__main__':
    main()
