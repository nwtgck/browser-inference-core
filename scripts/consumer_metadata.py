#!/usr/bin/env python3
"""Report both runtimes with the existing immutable, attempt-bound PR envelope."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
import re
import time
from pathlib import Path
import sys
from package_runtime import validate, identity
from pipeline_metrics import measured, span
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'llama-cpp/scripts'))
from upstream_provenance import collect
spec = importlib.util.spec_from_file_location('llama_consumer_metadata', ROOT / 'llama-cpp/scripts/consumer_metadata.py')
if spec is None or spec.loader is None: raise RuntimeError('Missing llama report helper')
legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)

@measured('report.input_validate')
def validate_for_report(package: Path, published_manifest_sha256: str | None = None) -> tuple[dict, str]:
    """Reuse npm packing only for the exact tree checked by this job's publisher.

    The caller must obtain the digest from publish_artifacts.py's successful
    output, not calculate it from an unverified package or restore a cached flag.
    The root manifest binds every file, including the two runtime manifests.
    """
    if published_manifest_sha256 is not None and not re.fullmatch(r'[0-9a-f]{64}', published_manifest_sha256):
        raise ValueError('Invalid published manifest SHA-256')
    digest = identity(package / 'manifest.json')['sha256']
    if published_manifest_sha256 is not None and digest != published_manifest_sha256:
        raise ValueError('Package differs from the published manifest')
    manifest = validate(package, check_npm_pack=published_manifest_sha256 is None)
    if identity(package / 'manifest.json')['sha256'] != digest:
        raise ValueError('Package manifest changed during report validation')
    return manifest, digest

def read_bound_snapshot(package: Path, relative: str, expected_sha256: str) -> tuple[bytes, dict]:
    """Capture small report inputs and bind the bytes actually parsed to validation."""
    raw = (package / relative).read_bytes()
    info = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    if info['sha256'] != expected_sha256:
        raise ValueError('Report input snapshot differs from the validated manifest: ' + relative)
    return raw, info


@measured('report.metadata')
def metadata(package: Path, repo: str, commit: str, lock: dict, divergences: dict, *,
             published_manifest_sha256: str | None = None) -> dict:
    root_manifest, validated_digest = validate_for_report(package, published_manifest_sha256)
    files = {item['path']: item for item in root_manifest['files']}
    _, root_identity = read_bound_snapshot(package, 'manifest.json', validated_digest)
    llama_bytes, _ = read_bound_snapshot(package, 'llama-cpp/manifest.json', files['llama-cpp/manifest.json']['sha256'])
    sd_bytes, sd_identity = read_bound_snapshot(package, 'stable-diffusion-cpp/manifest.json', files['stable-diffusion-cpp/manifest.json']['sha256'])
    sd = json.loads(sd_bytes)
    data = legacy.metadata(package / 'llama-cpp', repo, commit, lock, divergences, manifest_bytes=llama_bytes)
    data['runtime']['manifestFormatVersion'] = root_manifest['formatVersion']
    data['runtime']['llamaManifestPath'] = 'llama-cpp/manifest.json'
    data['retrieval']['sourceRepositoryRawBase'] = data['retrieval']['sourceRawBase']
    data['retrieval']['sourceRawBase'] += 'llama-cpp/'
    data['retrieval']['sourceRuntimePath'] = 'llama-cpp/'
    data['retrieval']['manifest'] = {'path': 'manifest.json', **root_identity}
    for profile in data['browserProfiles'].values():
        for file in profile.values(): file['path'] = 'llama-cpp/' + file['path']
    for file in data['interfaceFiles']: file['path'] = 'llama-cpp/' + file['path']
    # Link full per-variant provenance by digest instead of repeating toolchain and
    # patch inventories for every profile/variant in the bounded PR comment.
    data['stableDiffusion'] = {'manifest': {'path': 'stable-diffusion-cpp/manifest.json', **sd_identity},
        'abiVersion': sd['abiVersion'], 'schemaSha256': sd['schemaSha256'],
        'capabilities': sd['capabilities'], 'upstreams': sd['upstreams'],
        'profiles': {name: {'variants': {variant: {'validation': item['validation'],
            'sourceCommit': item['sourceCommit']} for variant, item in profile['variants'].items()}}
            for name, profile in sd['profiles'].items()},
        'experimental': True,
        'installSeparatelyForNaidan': f'npm install --save-exact stable-diffusion-cpp-browser-core@github:{repo}#{commit}',
        'consumerBoundary': 'Keep the existing llama dependency pinned. The optional image dependency is a separate npm name referencing this same immutable multi-runtime artifact.'}
    # Keep validation at the end of construction too, before writing any report.
    # Snapshot data cannot be poisoned by a changed-and-restored metadata file.
    validate_for_report(package, validated_digest)
    return data

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'build/consumer-update')
    p.add_argument('--published-manifest-sha256',
                   help='Successful publisher output for --commit; recheck all payloads but do not repeat npm packing')
    a = p.parse_args(); package = a.package.resolve()
    started = time.monotonic()
    root_manifest, validated_digest = validate_for_report(package, a.published_manifest_sha256)
    print(f'[publication] report-input-validation: {time.monotonic()-started:.3f}s', file=sys.stderr)
    files = {item['path']: item for item in root_manifest['files']}
    llama_bytes, _ = read_bound_snapshot(package, 'llama-cpp/manifest.json', files['llama-cpp/manifest.json']['sha256'])
    package_bytes, _ = read_bound_snapshot(package, 'package.json', files['package.json']['sha256'])
    llama = json.loads(llama_bytes)
    repo = os.environ['GITHUB_REPOSITORY']
    started = time.monotonic()
    with span('report.lock_resolve'):
        lock = legacy.generate_lock(repo, a.commit, json.loads(package_bytes)['version'])
    print(f'[publication] npm-lock-resolution: {time.monotonic()-started:.3f}s', file=sys.stderr)
    # Check again after network/provenance work. A concurrent modification must
    # fail rather than silently report a different tree or reuse stale evidence.
    with span('report.collect_provenance'):
        provenance = collect(ROOT / 'llama-cpp', llama)
    data = metadata(package, repo, a.commit, lock, provenance,
                    published_manifest_sha256=validated_digest)
    with span('report.write'):
        markdown = legacy.write_report(a.output, data, os.environ['GITHUB_RUN_ID'], os.environ.get('GITHUB_RUN_ATTEMPT', '1'))
    print(markdown)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as out: out.write(markdown)
if __name__ == '__main__': main()
