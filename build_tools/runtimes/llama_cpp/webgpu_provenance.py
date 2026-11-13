"""Report the selected source recipe and its observed final outputs.

Feature descriptions explain existing patches. They are not an executable
registry: webgpu_plan owns ordering and scope, and the build receipt owns the
actual inputs and outputs. Historical prefix hashes are deliberately absent.
"""
from __future__ import annotations

import json
from pathlib import Path

from build_tools.runtimes.llama_cpp import source
from build_tools.runtimes.llama_cpp import webgpu_plan as plan
from build_tools.source.files import regular_file
from build_tools.source.patches import read_series

SUPPORTING_FILES = (
    'build_tools/runtimes/llama_cpp/source.py',
    'build_tools/runtimes/llama_cpp/webgpu_plan.py',
    'build_tools/runtimes/llama_cpp/webgpu_provenance.py',
    'build_tools/source/patches.py', 'build_tools/source/wgsl.py',
    'build_tools/source/files.py', 'build_tools/output.py', 'build_tools/project.py',
    'build_tools/toolchain/dawn.py',
    'cmake/WebgpuSourceOverlay.cmake',
    'config/webgpu-patches.json', 'config/webgpu-features.json',
)
PATCH_ROLES = {'patch', 'profilePatch', 'integrationPatch', 'sharedPatch', 'sourceEquivalentPatch'}


def patch_names(root: Path) -> set[str]:
    return {entry.file for entry in read_series(root / 'config/webgpu-patches.json',
                                               targets=('llama-webgpu', 'iq3s', 'word-template')).patches}


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('Duplicate WebGPU feature key: ' + name)
        result[name] = value
    return result


def features(root: Path) -> list[dict]:
    document = json.loads(regular_file(root / 'config/webgpu-features.json').read_text(),
                          object_pairs_hook=_unique_object)
    if (not isinstance(document, dict) or set(document) != {'formatVersion', 'features'}
            or type(document['formatVersion']) is not int or document['formatVersion'] != 1
            or not isinstance(document['features'], list)):
        raise ValueError('Invalid WebGPU feature documentation')
    names = patch_names(root)
    owners: set[str] = set()
    identifiers: set[str] = set()
    for item in document['features']:
        if (not isinstance(item, dict) or set(item) != {'id', 'patches', 'documentation', 'behavior'}
                or not isinstance(item['id'], str) or not item['id'] or item['id'] in identifiers
                or not isinstance(item['behavior'], str) or not item['behavior']
                or not isinstance(item['patches'], dict) or 'patch' not in item['patches']
                or not set(item['patches']) <= PATCH_ROLES
                or any(not isinstance(name, str) or name not in names for name in item['patches'].values())
                or not isinstance(item['documentation'], list)
                or any(not isinstance(path, str) or not path.startswith('docs/')
                       or '..' in Path(path).parts or not path.endswith('.md')
                       for path in item['documentation'])):
            raise ValueError('Invalid or duplicate WebGPU feature description')
        identifiers.add(item['id'])
        for name in item['patches'].values():
            if name in owners:
                raise ValueError('WebGPU patch has more than one documented owner: ' + name)
            owners.add(name)
    if owners != names:
        raise ValueError('WebGPU feature documentation must cover every declared patch')
    return document['features']


def supporting_identity(root: Path, relative: str) -> dict:
    repository = relative.startswith('build_tools/')
    result = plan.identity((root.parents[1] if repository else root) / relative)
    if repository:
        result['pathBase'] = 'sourceRepositoryRawBase'
    return result


def describe(root: Path, vendor: Path, temporary: Path, enabled: list[str],
             diagnostic: list[str]) -> tuple[dict, list[dict], dict[str, dict]]:
    """Reprepare only selected modes and return final identities for comparison.

    The caller compares *all* prepared files, including the recipe receipt,
    against the built manifest under the existing immutable input-capture check.
    """
    catalog = features(root)
    if not set(diagnostic) <= set(enabled) or len(set(enabled)) != len(enabled):
        raise ValueError('Invalid WebGPU profile selection')
    selection = {key: 'profile' if key in diagnostic else 'normal' for key in sorted(enabled)}
    modes, files = {}, {}
    for mode in sorted(set(selection.values())):
        prepared = source.prepare(vendor, temporary / ('webgpu-' + mode),
                                  patch_root=root / plan.PATCH_DIRECTORY,
                                  gpu_profile=mode == 'profile',
                                  inventory=root / 'config/webgpu-patches.json').parent
        receipt_path = prepared / plan.RECEIPT
        receipt = json.loads(regular_file(receipt_path).read_text())
        observed = {path.name: plan.identity(path) for path in sorted(prepared.iterdir())}
        outputs = {name: identity for name, identity in observed.items() if name != plan.RECEIPT}
        if (receipt.get('formatVersion') != 1 or receipt.get('gpuProfile') is not (mode == 'profile')
                or receipt.get('outputs') != outputs):
            raise ValueError('WebGPU preparation receipt differs from observed outputs')
        modes[mode] = {'receipt': {'path': plan.RECEIPT, **observed[plan.RECEIPT]}, **receipt}
        files[mode] = observed
    details = {
        'formatVersion': 1,
        'modes': modes,
        'profileVariants': selection,
        'inputsPathBase': 'upstreamRepositoryRawBase',
        'runtimeInputPrefix': 'runtime/',
        'runtimeInputsPathBase': 'sourceRawBase',
        'patchPathBase': plan.PATCH_DIRECTORY,
        'implementation': {path: supporting_identity(root, path) for path in SUPPORTING_FILES},
        'scope': 'Actual selected patch order, shader inputs and final composed source sets. '
                 'Hashes record provenance; they are not semantic compatibility guards.',
    }
    overlays = []
    for item in catalog:
        # Profile-only instrumentation has no normal-mode activation. All other
        # accepted features apply to both WebGPU modes, never to CPU-only builds.
        profile_only = item['patches']['patch'] in {
            'ggml-webgpu-profile-query-budget.patch', 'ggml-webgpu-q8-diagnostic.patch'}
        variants = {key: mode for key, mode in selection.items() if not profile_only or mode == 'profile'}
        copies = {key: modes[mode]['outputs'] for key, mode in variants.items()}
        values = list(copies.values())
        common = values[0] if values and all(value == values[0] for value in values) else None
        patches = {role: {'path': f'{plan.PATCH_DIRECTORY}/{name}',
                          **plan.identity(root / plan.PATCH_DIRECTORY / name)}
                   for role, name in item['patches'].items()}
        overlays.append({
            'id': item['id'], 'kind': 'build-tree-source-overlay', **patches,
            'behavior': item['behavior'],
            'application': {
                'preparationScript': 'build_tools/runtimes/llama_cpp/source.py',
                'preparationScriptPathBase': 'sourceRepositoryRawBase',
                'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                'option': 'GGML_WEBGPU_GPU_PROFILE' if profile_only else 'LCB_WEBGPU',
                'enabledProfileVariants': sorted(variants),
            },
            'preparationModesByProfileVariant': variants,
            'compiledCopies': common,
            'compiledCopiesByProfileVariant': copies,
            'compiledCopiesScope': 'Final composed source set shared by the enabled features; '
                                   'not an intermediate result or ownership claim for each file.',
            'supportingFiles': {path: supporting_identity(root, path) for path in item['documentation']},
        })
        if item['id'] == 'webgpu-batched-write-scalar-fallback':
            overlays[-1]['application']['compileGuard'] = 'defined(__EMSCRIPTEN__)'
    return details, overlays, {key: files[mode] for key, mode in selection.items()}
