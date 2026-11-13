"""Report both runtimes from local, commit-bound publication and npm receipts."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

from build_tools.packaging.assemble import validate, identity
from build_tools.packaging.layout import ARTIFACT_DIRS, PACKAGE_NAME
from build_tools.metrics import measured, span
from build_tools.github import full_sha, repository_name
from build_tools.project import REPOSITORY_ROOT as ROOT
from build_tools.runtimes.llama_cpp.provenance import collect
from build_tools.runtimes.llama_cpp import reporting as legacy
from build_tools.publication import npm_lock

LLAMA_ARTIFACT_DIR = ARTIFACT_DIRS['llama-cpp']
IMAGE_ARTIFACT_DIR = ARTIFACT_DIRS['stable-diffusion-cpp']

@measured('report.input_validate')
def validate_for_report(package: Path, published_manifest_sha256: str | None = None) -> tuple[dict, str]:
    """Reuse npm packing only for the exact tree checked by this job's publisher.

    The caller must obtain the digest from the workflow's confirmed successful publication
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
    llama_manifest = LLAMA_ARTIFACT_DIR + '/manifest.json'
    image_manifest = IMAGE_ARTIFACT_DIR + '/manifest.json'
    llama_bytes, _ = read_bound_snapshot(package, llama_manifest, files[llama_manifest]['sha256'])
    sd_bytes, sd_identity = read_bound_snapshot(package, image_manifest, files[image_manifest]['sha256'])
    sd = json.loads(sd_bytes)
    data = legacy.metadata(package / LLAMA_ARTIFACT_DIR, repo, commit, lock, divergences, manifest_bytes=llama_bytes)
    data['runtime']['package'] = PACKAGE_NAME
    data['consumerIntegration'] = legacy.consumer_knowledge(package=PACKAGE_NAME)
    data['runtime']['manifestFormatVersion'] = root_manifest['formatVersion']
    data['runtime']['llamaManifestPath'] = llama_manifest
    data['retrieval']['sourceRepositoryRawBase'] = data['retrieval']['sourceRawBase']
    data['retrieval']['sourceRawBase'] += 'runtimes/llama-cpp/'
    data['retrieval']['sourceRuntimePath'] = 'runtimes/llama-cpp/'
    data['retrieval']['manifest'] = {'path': 'manifest.json', **root_identity}
    for profile in data['browserProfiles'].values():
        for file in profile.values(): file['path'] = LLAMA_ARTIFACT_DIR + '/' + file['path']
    for file in data['interfaceFiles']: file['path'] = LLAMA_ARTIFACT_DIR + '/' + file['path']
    # Link full per-variant provenance by digest instead of repeating toolchain and
    # patch inventories for every profile/variant in the bounded PR comment.
    data['stableDiffusion'] = {'manifest': {'path': image_manifest, **sd_identity},
        'abiVersion': sd['abiVersion'], 'schemaSha256': sd['schemaSha256'],
        'capabilities': sd['capabilities'], 'upstreams': sd['upstreams'],
        'profiles': {name: {'variants': {variant: {'validation': item['validation'],
            'sourceCommit': item['sourceCommit']} for variant, item in profile['variants'].items()}}
            for name, profile in sd['profiles'].items()},
        'experimental': True,
        'packageExports': {
            'host': PACKAGE_NAME + '/stable-diffusion-cpp/host',
            'api': PACKAGE_NAME + '/stable-diffusion-cpp/api/',
            'profiles': PACKAGE_NAME + '/stable-diffusion-cpp/profiles/'},
        'consumerBoundary': 'Both runtimes are independent modules in this immutable package. Consumer profile and Wasm instance selection remain explicit.'}
    # Keep validation at the end of construction too, before writing any report.
    # Snapshot data cannot be poisoned by a changed-and-restored metadata file.
    validate_for_report(package, validated_digest)
    return data

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--lock-directory', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'build/consumer-update')
    p.add_argument('--published-manifest-sha256',
                   help='Successful publisher output for --commit; recheck all payloads but do not repeat npm packing')
    a = p.parse_args(); package = a.package.resolve()
    started = time.monotonic()
    root_manifest, validated_digest = validate_for_report(package, a.published_manifest_sha256)
    print(f'[publication] report-input-validation: {time.monotonic()-started:.3f}s', file=sys.stderr)
    files = {item['path']: item for item in root_manifest['files']}
    llama_manifest = LLAMA_ARTIFACT_DIR + '/manifest.json'
    llama_bytes, _ = read_bound_snapshot(package, llama_manifest, files[llama_manifest]['sha256'])
    package_bytes, _ = read_bound_snapshot(package, 'package.json', files['package.json']['sha256'])
    llama = json.loads(llama_bytes)
    repo = repository_name(os.environ['GITHUB_REPOSITORY'])
    full_sha(a.commit)
    started = time.monotonic()
    with span('report.lock_validate'):
        lock = npm_lock.read(a.lock_directory)
        if (lock['specifier'] != f'github:{repo}#{a.commit}'
                or lock['packageEntry']['version'] != json.loads(package_bytes)['version']):
            raise ValueError('npm receipt differs from the published package')
    print(f'[publication] npm-lock-validation: {time.monotonic()-started:.3f}s', file=sys.stderr)
    # Check again after lock/provenance work. A concurrent modification must
    # fail rather than silently report a different tree or reuse stale evidence.
    with span('report.collect_provenance'):
        provenance = collect(ROOT / 'runtimes/llama-cpp', llama)
    data = metadata(package, repo, a.commit, lock, provenance,
                    published_manifest_sha256=validated_digest)
    with span('report.write'):
        markdown = legacy.write_report(a.output, data, os.environ['GITHUB_RUN_ID'], os.environ.get('GITHUB_RUN_ATTEMPT', '1'))
    print(markdown)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as out: out.write(legacy.render_summary_markdown(data))
if __name__ == '__main__': main()
