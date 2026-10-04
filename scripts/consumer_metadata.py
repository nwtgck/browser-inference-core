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
from package_runtime import ARTIFACT_DIRS, validate, identity
from pipeline_metrics import measured, span
ROOT = Path(__file__).resolve().parents[1]
LLAMA_ARTIFACT_DIR = ARTIFACT_DIRS['llama-cpp']
IMAGE_ARTIFACT_DIR = ARTIFACT_DIRS['stable-diffusion-cpp']
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
    if root_manifest['formatVersion'] == 4:
        return source_catalog_metadata(package, repo, commit, lock, root_manifest, validated_digest)
    files = {item['path']: item for item in root_manifest['files']}
    _, root_identity = read_bound_snapshot(package, 'manifest.json', validated_digest)
    llama_manifest = LLAMA_ARTIFACT_DIR + '/manifest.json'
    image_manifest = IMAGE_ARTIFACT_DIR + '/manifest.json'
    llama_bytes, _ = read_bound_snapshot(package, llama_manifest, files[llama_manifest]['sha256'])
    sd_bytes, sd_identity = read_bound_snapshot(package, image_manifest, files[image_manifest]['sha256'])
    sd = json.loads(sd_bytes)
    data = legacy.metadata(package / LLAMA_ARTIFACT_DIR, repo, commit, lock, divergences, manifest_bytes=llama_bytes)
    data['runtime']['manifestFormatVersion'] = root_manifest['formatVersion']
    data['runtime']['llamaManifestPath'] = llama_manifest
    data['retrieval']['sourceRepositoryRawBase'] = data['retrieval']['sourceRawBase']
    data['retrieval']['sourceRawBase'] += 'llama-cpp/'
    data['retrieval']['sourceRuntimePath'] = 'llama-cpp/'
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
        'installSeparatelyForNaidan': f'npm install --save-exact stable-diffusion-cpp-browser-core@github:{repo}#{commit}',
        'consumerBoundary': 'Keep the existing llama dependency pinned. The optional image dependency is a separate npm name referencing this same immutable multi-runtime artifact.'}
    # Keep validation at the end of construction too, before writing any report.
    # Snapshot data cannot be poisoned by a changed-and-restored metadata file.
    validate_for_report(package, validated_digest)
    return data

def source_catalog_metadata(package, repo, commit, lock, manifest, digest):
    """Report only artifact-bound evidence; never inspect today's vendor for reused builds."""
    legacy.repository_name(repo);legacy.full_sha(commit)
    files={entry['path']:entry for entry in manifest['files']}
    _,root_identity=read_bound_snapshot(package,'manifest.json',digest)
    sources={}
    for name,entry in manifest['runtimes']['llama-cpp']['sources'].items():
        raw,bound=read_bound_snapshot(package,entry['manifest'],files[entry['manifest']]['sha256'])
        inner=json.loads(raw);base=str(Path(entry['manifest']).parent)+'/'
        inventory={item['path']:item for item in inner['files']}
        sources[name]={
            'manifest':{'path':entry['manifest'],**bound},
            'buildSourceCommit':inner['sourceCommit'],
            'upstreamCommit':inner['llamaCommit'],
            'browserProfiles':{profile:{kind:{**inventory[f'profiles/{profile}/browser/core.{ext}'],
                'path':base+f'profiles/{profile}/browser/core.{ext}'} for kind,ext in [('wasm','wasm'),('mjs','mjs'),('types','d.ts')]}
                for profile in inner['profiles']},
            'interfaceFiles':[{**item,'path':base+item['path']} for item in inner['files'] if item['path'].startswith('api/')],
            'validation':{profile:{variant:info['validation'] for variant,info in p['variants'].items()} for profile,p in inner['profiles'].items()},
            'buildProvenanceScope':'The bound source manifest contains original build identities, patch series and toolchain evidence. The assembly commit is not substituted for reused build commits.',
        }
    images={name:{**entry} for name,entry in manifest['runtimes']['stable-diffusion-cpp']['sources'].items()}
    data={'schemaVersion':2,
        'runtime':{'repository':repo,'package':legacy.RUNTIME_NAME,'artifactCommit':commit,'sourceCommit':manifest['sourceCommit'],'manifestFormatVersion':4},
        'retrieval':{'artifactArchive':f'https://codeload.github.com/{repo}/tar.gz/{commit}',
            'artifactRawBase':f'https://raw.githubusercontent.com/{repo}/{commit}/',
            'sourceRepositoryRawBase':f'https://raw.githubusercontent.com/{repo}/{manifest["sourceCommit"]}/',
            'manifest':{'path':'manifest.json',**root_identity}},
        'npm':lock,'llamaSources':sources,'imageSources':images,
        'consumerIntegration':legacy.consumer_knowledge()}
    if 'reuseInputs' in manifest:
        data['reusedArtifactInputs'] = manifest['reuseInputs']
    data['consumerIntegration']['sourceCatalog']={
        'knownLocations':['src/features/llama-cpp-browser/build-artifact-package.ts','src/features/llama-cpp-browser/build-core.ts','src/features/llama-cpp-browser/build-runtime-assets.ts'],
        'meaning':'Historical search hints, not a live consumer scan. Select source/profile targets explicitly. Missing profiles are not stable fallbacks. Bind reviewed glue and Wasm to each source.',
        'embeddedSubset':'Standalone historically selects WebGPU32/64 JSPI and Brotli. A CPU or Asyncify full must not be introduced solely as an unselected delta base.',
        'devParity':'Select the same packed representation and decoder for development and release of the same distribution. Do not silently bypass decode with raw Wasm.',
    }
    if manifest.get('packedWasm'):
        path=manifest['packedWasm']['catalog'];raw,bound=read_bound_snapshot(package,path,files[path]['sha256']);catalog=json.loads(raw);prefix=str(Path(path).parent)+'/'
        data['packedWasm']={'catalog':{'path':path,**bound},'decoderApiVersion':catalog['decoderApiVersion'],
            'selector':prefix+'runtime/catalog.mjs',
            'defaultLoader':prefix+catalog['runtime']['entry'],
            'loaderEntries':{codec:prefix+entry for codec,entry in catalog['runtime'].get('entries',{}).items()},
            'codecs':sorted({a['codec'] for a in catalog['assets'].values()}),
            'selectionContract':'selectWasmAssets(catalog, {targets, codec, fullTargets}) returns exact file closure. Import plan.runtime.entry after selection; do not hard-code a loader or use defaultLoader for every codec. Only selected full targets can be delta bases. Empty selection has no assets. Full alternatives remain available.',
            'verificationScope':'Every stored representation was reconstructed with the matching shipped decoder and compared with the raw Wasm. This is not evidence of real GPU inference or consumer bundling.',
            'rawPreserved':True}
    validate_for_report(package,digest)
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
    llama_bytes = None
    if root_manifest['formatVersion'] != 4:
        llama_manifest = LLAMA_ARTIFACT_DIR + '/manifest.json'
        llama_bytes, _ = read_bound_snapshot(package, llama_manifest, files[llama_manifest]['sha256'])
    package_bytes, _ = read_bound_snapshot(package, 'package.json', files['package.json']['sha256'])
    llama = json.loads(llama_bytes) if llama_bytes is not None else None
    repo = os.environ['GITHUB_REPOSITORY']
    started = time.monotonic()
    with span('report.lock_resolve'):
        lock = legacy.generate_lock(repo, a.commit, json.loads(package_bytes)['version'])
    print(f'[publication] npm-lock-resolution: {time.monotonic()-started:.3f}s', file=sys.stderr)
    # Check again after network/provenance work. A concurrent modification must
    # fail rather than silently report a different tree or reuse stale evidence.
    with span('report.collect_provenance'):
        provenance = collect(ROOT / 'llama-cpp', llama) if llama is not None else {}
    data = metadata(package, repo, a.commit, lock, provenance,
                    published_manifest_sha256=validated_digest)
    with span('report.write'):
        markdown = legacy.write_report(a.output, data, os.environ['GITHUB_RUN_ID'], os.environ.get('GITHUB_RUN_ATTEMPT', '1'))
    print(markdown)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as out: out.write(legacy.render_summary_markdown(data))
if __name__ == '__main__': main()
