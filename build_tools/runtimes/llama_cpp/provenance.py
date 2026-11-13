#!/usr/bin/env python3
"""Describe source overlays for CI reports without changing the runtime manifest."""
from __future__ import annotations
from build_tools.project import runtime_root

import hashlib
import json
import re
from pathlib import Path
import sys
from build_tools.toolchain.config import runtime_toolchain
import tempfile
import subprocess

from build_tools.runtimes.llama_cpp import inputs as captured_inputs
from build_tools.runtimes.llama_cpp import webgpu_provenance

from build_tools.github import full_sha
from build_tools.source.checkout import checked_sources, git, git_environment
from build_tools.runtimes.llama_cpp.sources.mtmd import PATCH_DIRECTORY, MANIFEST_NAME, prepare
from build_tools.runtimes.llama_cpp.bf16 import runtime_metadata, validate_metadata as validate_bf16_metadata
from build_tools.runtimes.llama_cpp.sources import file_backed_rows as file_rows
from build_tools.runtimes.llama_cpp.sources import qsa_selection as qsa
from build_tools.runtimes.llama_cpp.sources import model_graph_chat as model_graph

def file_identity(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or linked provenance input: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def repository_input(root: Path, relative: str) -> Path:
    base = root.parents[1] if relative.startswith('build_tools/') else root
    return base / relative


def supporting_identity(root: Path, relative: str) -> dict:
    identity = file_identity(repository_input(root, relative))
    if relative.startswith('build_tools/'):
        identity['pathBase'] = 'sourceRepositoryRawBase'
    return identity


def browser_module_preparations(root: Path, manifest: dict) -> list[dict]:
    records = []
    implementation = None
    files = {item['path']: item for item in manifest.get('files', [])}
    for profile, info in sorted(manifest['profiles'].items()):
        for variant, provenance in sorted(info['variants'].items()):
            preparation = provenance.get('browserModulePreparation')
            if preparation is None:
                continue
            expected = {'path': 'build_tools/runtimes/llama_cpp/browser_module.py',
                        **supporting_identity(root, 'build_tools/runtimes/llama_cpp/browser_module.py')}
            output = files.get(f'profiles/{profile}/{variant}/core.mjs')
            if (variant != 'browser' or preparation['implementation'] != expected or output is None
                    or preparation['output'] != {key: output[key] for key in ('bytes', 'sha256')}):
                raise ValueError('Browser module preparation differs from published source/output')
            implementation = expected
            records.append({'profile': profile, 'variant': variant,
                            'input': preparation['input'], 'output': preparation['output'],
                            'alreadyAnnotated': preparation['alreadyAnnotated']})
    if not records:
        return []
    return [{'id': 'browser-wasm-url-annotation', 'kind': 'generated-javascript-comment',
             'implementation': implementation, 'profileVariants': records,
             'supportingFiles': {path: supporting_identity(root, path) for path in
                                 ['build_tools/runtimes/llama_cpp/build.py', 'host/emscripten/host-navigator-pre.js', 'docs/browser-host.md']},
             'behavior': 'Add one @vite-ignore comment to the generated sibling core.wasm URL in browser variants. Vite skips automatic Wasm discovery at this expression; other bundlers may differ. Runtime acquisition, supplied bytes and fallback branches are unchanged. Hosts explicitly deploy the matching Wasm or supply bytes. Input/output identities describe the actual preparation, including no-op reruns.'}]


def model_graph_preparations(root: Path, manifest: dict) -> list[dict]:
    records = {f'{profile}/{variant}': info['modelGraphBoundary']
               for profile, data in manifest['profiles'].items()
               for variant, info in data['variants'].items() if 'modelGraphBoundary' in info}
    if not records:
        return []
    total = sum(len(data['variants']) for data in manifest['profiles'].values())
    if len(records) != total:
        raise ValueError('Mixed model graph boundary activation in compiled provenance')
    with tempfile.TemporaryDirectory(prefix='lcb-report-model-graph-') as temporary:
        prepared = model_graph.prepare(root / 'vendor/llama.cpp', Path(temporary) / 'overlay', runtime=root)
        copies = {path: {'logicalUpstreamPath': path, **file_identity(Path(temporary) / 'overlay' / path)}
                  for path in prepared['outputs']}
    import re
    hashes = set()
    for record in records.values():
        if not isinstance(record, dict):
            raise ValueError('Invalid model graph boundary provenance')
        fingerprint = record.get('schemaSha256')
        if not isinstance(fingerprint, str) or not re.fullmatch(r'[a-f0-9]{64}', fingerprint):
            raise ValueError('Missing model graph boundary schema fingerprint')
        if {key: value for key, value in record.items() if key != 'schemaSha256'} != prepared:
            raise ValueError('Model graph boundary differs from the reviewed build overlay')
        hashes.add(fingerprint)
    if len(hashes) != 1:
        raise ValueError('Mixed model graph boundary schema fingerprints')
    patch = f'{PATCH_DIRECTORY}/{prepared["patch"]}'
    return [{
        'id': 'independent-vocabulary-chat', 'kind': 'build-tree-source-overlay',
        'patch': {'path': patch, **file_identity(root / patch)},
        'compiledCopies': copies,
        'schemaSha256': next(iter(hashes)),
        'application': {'preparationScript': 'build_tools/runtimes/llama_cpp/sources/model_graph_chat.py',
                        'preparationScriptPathBase': 'sourceRepositoryRawBase',
                        'cmakeHook': 'cmake/ModelGraphBoundary.cmake',
                        'enabledProfileVariants': sorted(records)},
        'supportingFiles': {path: supporting_identity(root, path) for path in
                            ['native/model-graph/model-graph.h', 'native/model-graph/model-graph.cpp', 'native/model-graph/model-graph-internal.h',
                             'build_tools/runtimes/llama_cpp/generate_model_graph.py',
                             'build_tools/runtimes/llama_cpp/sources/model_graph_chat.py',
                             'build_tools/source/patches.py', 'build_tools/source/files.py', 'build_tools/output.py']},
        'behavior': 'Initialize independently owned vocabulary/chat metadata without a dummy model. '
                    'Preserve the native model entry point; graph builders, decode and kernels stay upstream. '
                    'The optional typed leaf boundary reuses CPU samplers for external logits; host graph '
                    'definitions, file policies, validation and execution control remain outside the core.',
    }]


def _describe(root: Path, manifest: dict, *, captured_receipt: dict | None = None) -> dict:
    source = full_sha(manifest['sourceCommit'])
    upstream = full_sha(manifest['llamaCommit'])
    vendor = root / 'vendor/llama.cpp'
    patch_path = f'{PATCH_DIRECTORY}/mtmd-webgpu-bf16.patch'
    upstream_path = 'tools/mtmd/mtmd-model.cpp'
    runtime_patch = f'{PATCH_DIRECTORY}/mtmd-vision-bf16-runtime.patch'
    runtime_source = 'tools/mtmd/mtmd.cpp'
    pool_patch = f'{PATCH_DIRECTORY}/mtmd-non-pthread-threadpool.patch'
    pool_source = 'tools/mtmd/mtmd-subcontext.cpp'
    patch_files = sorted(path.relative_to(root).as_posix() for path in (root / PATCH_DIRECTORY).rglob('*.patch'))
    enabled = []
    audio_enabled = []
    copy_enabled = []
    diagnostic_enabled = []
    batch_enabled = []
    rows_enabled = []
    qsa_enabled = []
    toolchain = runtime_toolchain(root)
    for profile, info in manifest['profiles'].items():
        for variant, provenance in info['variants'].items():
            validate_bf16_metadata(provenance)
            webgpu_options = [option for option in provenance['cmakeCommand']
                              if option.startswith('-DLCB_WEBGPU=') or option.startswith('-DLCB_WEBGPU:')]
            if webgpu_options not in (['-DLCB_WEBGPU=ON'], ['-DLCB_WEBGPU=OFF']):
                raise ValueError('Unknown WebGPU activation in compiled provenance')
            webgpu_enabled = webgpu_options == ['-DLCB_WEBGPU=ON']
            if webgpu_enabled:
                copy_enabled.append(profile + '/' + variant)
                batch_enabled.append(profile + '/' + variant)
            profiling_options = [option for option in provenance['cmakeCommand']
                                 if option.startswith(('-DGGML_WEBGPU_GPU_PROFILE=', '-DGGML_WEBGPU_GPU_PROFILE:'))]
            if profiling_options and profiling_options not in (['-DGGML_WEBGPU_GPU_PROFILE=ON'], ['-DGGML_WEBGPU_GPU_PROFILE=OFF']):
                raise ValueError('Unknown GPU-profile activation in compiled provenance')
            if profiling_options == ['-DGGML_WEBGPU_GPU_PROFILE=ON']:
                if not webgpu_enabled:
                    raise ValueError('Q8 diagnostics require WebGPU')
                diagnostic_enabled.append(profile + '/' + variant)
            rows_options = [option for option in provenance['cmakeCommand']
                            if option.startswith('-DLCB_FILE_BACKED_PLE=') or option.startswith('-DLCB_FILE_BACKED_PLE:')]
            if rows_options not in (['-DLCB_FILE_BACKED_PLE=ON'], ['-DLCB_FILE_BACKED_PLE=OFF']):
                raise ValueError('Unknown file-backed PLE activation in compiled provenance')
            if rows_options == ['-DLCB_FILE_BACKED_PLE=ON']:
                rows_enabled.append(profile + '/' + variant)
            qsa_options = [option for option in provenance['cmakeCommand']
                           if option.startswith('-DLCB_QSA_ALL_POOLS=') or option.startswith('-DLCB_QSA_ALL_POOLS:')]
            if qsa_options not in (['-DLCB_QSA_ALL_POOLS=ON'], ['-DLCB_QSA_ALL_POOLS=OFF']):
                raise ValueError('Unknown QSA all-pools activation in compiled provenance')
            if qsa_options == ['-DLCB_QSA_ALL_POOLS=ON']:
                qsa_enabled.append(profile + '/' + variant)
            audio_enabled.append(profile + '/' + variant)
            if any(option.startswith(('-DLCB_WEBGPU_MOE_DIRECT_SLOT=', '-DLCB_WEBGPU_MOE_DIRECT_SLOT:'))
                   for option in provenance['cmakeCommand']):
                raise ValueError('Retired MoE direct-slot overlay option in compiled provenance')
            if provenance['toolchain'] != toolchain:
                raise ValueError('Reporting toolchain differs from compiled provenance')
            options = [option for option in provenance['cmakeCommand']
                       if option.startswith('-DLCB_WEBGPU_BF16_PROJECTOR=')]
            if len(options) != 1 or options[0] not in ('-DLCB_WEBGPU_BF16_PROJECTOR=ON', '-DLCB_WEBGPU_BF16_PROJECTOR=OFF'):
                raise ValueError('Unknown BF16 overlay activation in compiled provenance')
            if webgpu_enabled and options[0].endswith('=ON'):
                enabled.append(profile + '/' + variant)
    rows_compiled = None
    rows_tree = None
    rows_by_variant = {}
    rows_trees_by_variant = {}
    qsa_by_variant = {}
    rows_source = None
    rows_preparation = None
    qsa_preparations = {}
    rows_identities = None
    with tempfile.TemporaryDirectory(prefix='lcb-report-overlay-') as temporary:
        if rows_enabled:
            rows_source = file_rows.prepare(vendor, Path(temporary) / 'file-rows')
            rows_preparation = json.loads((rows_source.parent / file_rows.RECEIPT).read_text())
            # These model TUs must all use the same modified models.h definition.
            paths = [rows_source / 'llama.cpp', rows_source / 'llama-model.cpp', *sorted((rows_source / 'models').glob('*.cpp')),
                     rows_source / 'models/models.h', rows_source / 'models/file-backed-rows.h',
                     rows_source / 'models/file-backed-identity.h']
            identities = {p.relative_to(rows_source).as_posix(): file_identity(p) for p in paths}
            # Unchanged model TUs are covered by an aggregate rather than making
            # the publication summary grow by hundreds of per-file entries.
            encoded = json.dumps(identities, sort_keys=True, separators=(',', ':')).encode()
            rows_tree = {'files': len(identities), 'sha256': hashlib.sha256(encoded).hexdigest(),
                         'encoding': 'UTF-8 compact JSON of sorted relative paths to bytes/SHA-256 objects'}
            selected = ['llama.cpp', 'llama-model.cpp', 'models/models.h', 'models/qwen4exp.cpp',
                        'models/file-backed-rows.h', 'models/file-backed-identity.h']
            rows_compiled = {name: identities[name] for name in selected}
            rows_identities = identities
        # Describe the translation unit actually compiled, not just the first
        # preparation stage. QSA may follow either original or file-backed rows.
        qsa_by_mode = {}
        qsa_files_by_mode = {}
        for mode in sorted({variant in rows_enabled for variant in qsa_enabled}):
            model_source = (rows_source / 'models/qwen4exp.cpp' if mode else vendor / qsa.SOURCE_PATH)
            prepared_qsa = qsa.prepare(vendor, model_source, Path(temporary) / ('qsa-rows' if mode else 'qsa'))
            qsa_preparations['file-backed-rows' if mode else 'upstream'] = json.loads((prepared_qsa.parent / qsa.RECEIPT).read_text())
            qsa_files_by_mode[mode] = captured_inputs.file_identities(prepared_qsa.parent)
            qsa_by_mode[mode] = {
                'logicalUpstreamPath': qsa.SOURCE_PATH,
                **file_identity(prepared_qsa),
                'inputTranslationUnit': file_identity(model_source),
                'modelHeader': file_identity(model_source.parent / 'models.h'),
                'fileBackedRows': mode,
            }
        qsa_by_variant = {variant: qsa_by_mode[variant in rows_enabled] for variant in qsa_enabled}
        for variant in rows_enabled:
            final_identities = dict(rows_identities)
            if variant in qsa_enabled:
                composed = qsa_by_variant[variant]
                final_identities['models/qwen4exp.cpp'] = {key: composed[key] for key in ('bytes', 'sha256')}
            rows_by_variant[variant] = {name: final_identities[name] for name in selected}
            encoded = json.dumps(final_identities, sort_keys=True, separators=(',', ':')).encode()
            rows_trees_by_variant[variant] = {**rows_tree, 'sha256': hashlib.sha256(encoded).hexdigest()}
        # Keep the compact single-copy fields only when all activated builds
        # really compile one identical set. Mixed flags require the per-variant map.
        if rows_by_variant:
            values = list(rows_by_variant.values())
            rows_compiled = values[0] if all(value == values[0] for value in values) else None
            trees = list(rows_trees_by_variant.values())
            rows_tree = trees[0] if all(value == trees[0] for value in trees) else None
        # Each profile compiles a complete, coherent mtmd header universe.
        # Hash the entire tree, including unchanged builders and quoted headers.
        mtmd_by_mode = {}
        mtmd_trees = {}
        mtmd_tree_files = {}
        for mode in ('cpu', 'webgpu'):
            tree = prepare(vendor, Path(temporary) / ('mtmd-' + mode),
                           mode=mode, patch_root=root / PATCH_DIRECTORY)
            source_manifest = json.loads((tree / MANIFEST_NAME).read_text())
            actual = {name: file_identity(tree / name) for name in source_manifest['preparedFiles']}
            if actual != source_manifest['preparedFiles']:
                raise ValueError('mtmd prepared inventory differs from its source manifest')
            encoded = json.dumps(actual, sort_keys=True, separators=(',', ':')).encode()
            raw_encoded = json.dumps(source_manifest['sourceInputs'], sort_keys=True, separators=(',', ':')).encode()
            contract_encoded = json.dumps(source_manifest['contractInputs'], sort_keys=True, separators=(',', ':')).encode()
            mtmd_trees[mode] = {
                'files': len(actual), 'sha256': hashlib.sha256(encoded).hexdigest(),
                'encoding': 'UTF-8 compact JSON of sorted relative paths to bytes/SHA-256 objects',
                'sourceInputs': {'files': len(source_manifest['sourceInputs']),
                                 'sha256': hashlib.sha256(raw_encoded).hexdigest()},
                'contractInputs': {'files': len(source_manifest['contractInputs']),
                                   'sha256': hashlib.sha256(contract_encoded).hexdigest()},
                'manifest': file_identity(tree / MANIFEST_NAME),
                'coreSources': source_manifest['coreSources'],
                'modifiedFiles': source_manifest['modifiedFiles'],
            }
            mtmd_by_mode[mode] = actual
            mtmd_tree_files[mode] = captured_inputs.file_identities(tree)
        mtmd_by_variant = {variant: mtmd_by_mode['webgpu' if variant in enabled else 'cpu']
                           for variant in audio_enabled}
        mtmd_trees_by_variant = {variant: mtmd_trees['webgpu' if variant in enabled else 'cpu']
                                 for variant in audio_enabled}
        selected_mtmd = ('mtmd-model.h', 'mtmd-model.cpp', 'mtmd-subcontext.cpp', 'mtmd.cpp')
        vision_copies = {variant: {name: mtmd_by_variant[variant][name] for name in selected_mtmd}
                         for variant in enabled}
        runtime_identity = mtmd_by_mode['cpu']['mtmd.cpp']
        pool_copies = {variant: {'logicalUpstreamPath': pool_source,
                                **mtmd_by_variant[variant]['mtmd-subcontext.cpp']}
                       for variant in audio_enabled}
        webgpu_details, webgpu_overlays, webgpu_files = webgpu_provenance.describe(
            root, vendor, Path(temporary), copy_enabled, diagnostic_enabled)

        if captured_receipt is not None:
            for profile, info in manifest['profiles'].items():
                for variant, provenance in info['variants'].items():
                    key = profile + '/' + variant
                    expected = {'mtmd-source-overlay': mtmd_tree_files['webgpu' if key in enabled else 'cpu']}
                    if key in copy_enabled:
                        expected['webgpu-source-overlay'] = webgpu_files[key]
                    if key in rows_enabled:
                        expected['file-backed-rows-overlay'] = captured_inputs.file_identities(rows_source.parent)
                    if key in qsa_enabled:
                        expected['qsa-selection-overlay'] = qsa_files_by_mode[key in rows_enabled]
                    if provenance.get('preparedSources') != expected:
                        raise ValueError('Compiled prepared-source receipt differs from captured inputs: ' + key)

    supporting = ['build_tools/runtimes/llama_cpp/sources/mtmd.py', 'cmake/MtmdOverlay.cmake',
                  'native/vision/mtmd-bf16.h', 'native/vision/mtmd-retained-upload.h',
                  'native/vision/mtmd-retained-graph.h', 'native/vision/mtmd-retained-roles.h',
                  'native/vision/mtmd-vision-bf16.h', 'native/vision/mtmd-vision-bf16-policy.h',
                  'build_tools/runtimes/llama_cpp/sources/retained_bf16.py', 'config/retained-bf16-source.json',
                  'docs/webgpu-bf16-projector.md']
    described_patches = {patch_path, pool_patch, runtime_patch,
                         f'{PATCH_DIRECTORY}/{file_rows.PATCH_NAME}',
                         f'{PATCH_DIRECTORY}/{qsa.PATCH_NAME}',
                         *(f'{PATCH_DIRECTORY}/{name}' for name in webgpu_provenance.patch_names(root))}
    report = {
        'webgpuPreparation': webgpu_details,
        'upstreamRepository': 'ggml-org/llama.cpp',
        'baseCommit': upstream,
        'capturedInputs': captured_receipt,
        'upstreamBuildSettingsByProfileVariant': {profile + '/' + variant: value.get('upstreamBuildInfo')
              for profile, info in manifest['profiles'].items() for variant, value in info['variants'].items()},
        'vendorCheckoutModified': False,
        'inventoryScope': f'Known build-tree overlays, recorded generated browser-module preparations, plus every *.patch file under {PATCH_DIRECTORY}/; not an exhaustive compiler transformation inventory.',
        'generatedModuleTransformations': browser_module_preparations(root, manifest),
        'retiredSourceOverlays': [{
            'id': 'experimental-webgpu-moe-direct-slot',
            'reason': 'llama.cpp v0.7.0 provides direct selected-slot dispatch upstream; no downstream MoE shader/header transformation is compiled.',
            'upstreamedBy': '620d2225cc436ab0b001e3200e590e3b286ef9ab',
            'documentation': 'docs/moe-direct-slot.md',
        }, {
            'id': 'single-thread-wasm-audio-preprocessing',
            'reason': 'Upstream replaced shared and Parakeet worker loops with scheduler-backed log-mel graphs. The old loop patch is retired; non-pthread threadpool initialization is tracked separately.',
            'upstreamedBy': 'a2654ec43148b8ae827d956276ce2c1eacfd3853',
            'documentation': 'docs/audio-single-thread.md',
        }, {
            'id': 'vision-bf16-cpu-loader-plumbing',
            'reason': 'The former clip-only CPU guard is consolidated into model-owned immutable policy and modality classification in the common mtmd runtime overlay. Unsupported retained GPU vision is still rejected before allocation.',
            'replacedBy': 'vision-bf16-runtime-selection',
        }],
        'sourceOverlays': [{
            'id': 'webgpu-vision-bf16-projector',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': upstream_path, **file_identity(vendor / upstream_path)},
            'patch': {'path': patch_path, **file_identity(root / patch_path)},
            'compiledCopies': next(iter(vision_copies.values()), None),
            'compiledCopiesByProfileVariant': vision_copies,
            'compiledSourceTreesByProfileVariant': {variant: mtmd_trees_by_variant[variant] for variant in enabled},
            'application': {
                'preparationScript': 'build_tools/runtimes/llama_cpp/sources/mtmd.py',
                'cmakeHook': 'cmake/MtmdOverlay.cmake',
                'option': 'LCB_WEBGPU_BF16_PROJECTOR',
                'runtimeStorageSelection': runtime_metadata(),
                'enabledProfileVariants': sorted(enabled),
            },
            'supportingFiles': {path: supporting_identity(root, path) for path in supporting},
            'searchHints': ['mtmd-webgpu-bf16', 'LCB_WEBGPU_BF16_PROJECTOR', 'lcb_clip', 'bf16-f32', 'cpu_bf16'],
            'behavior': 'Legacy callers expand WebGPU vision BF16 weights to resident F32. Immutable model-owned retain-checked policy preserves original BF16 storage and bounded uploads, explicit role/graph/placement rejection, and scheduler-before-backend cleanup in the shared compute owner, including partial construction failure. Borrowed subcontexts never free shared backends. A temporary upload backend uses the pinned WebGPU device-global queue before compute subcontexts exist; this is not a physical peak-memory guarantee. All mtmd TUs use one copied header universe; model builders and upstream graph-allocation failure handling are preserved.',
        }, {
            'id': 'non-pthread-wasm-mtmd-threadpool',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': pool_source, **file_identity(vendor / pool_source)},
            'patch': {'path': pool_patch, **file_identity(root / pool_patch)},
            'compiledCopy': None,
            'compiledCopiesByProfileVariant': pool_copies,
            'application': {
                'preparationScript': 'build_tools/runtimes/llama_cpp/sources/mtmd.py --mode cpu|webgpu',
                'cmakeHook': 'cmake/MtmdOverlay.cmake',
                'compileGuard': 'defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)',
                'enabledProfileVariants': sorted(audio_enabled),
            },
            'supportingFiles': {path: supporting_identity(root, path) for path in
                                ['build_tools/runtimes/llama_cpp/sources/mtmd.py', 'cmake/MtmdOverlay.cmake',
                                 'docs/audio-single-thread.md']},
            'searchHints': ['mtmd-non-pthread-threadpool', '__EMSCRIPTEN_PTHREADS__', 'mtmd_threadpool_init'],
            'behavior': 'Normalize the private shared mtmd threadpool request to one only in non-pthread Emscripten, using the existing no-pool branch. Native/pthread requested counts, upstream scheduler audio graphs, and public parameter layouts are unchanged.',
        }, {
            'id': 'qwen4exp-file-backed-rows',
            'kind': 'build-tree-coherent-model-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{file_rows.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / file_rows.PATCH_NAME)},
            'compiledCopies': rows_compiled,
            'coherentSourceSet': rows_tree,
            'compiledCopiesByProfileVariant': rows_by_variant,
            'coherentSourceSetsByProfileVariant': rows_trees_by_variant,
            'sourcePreparation': rows_preparation,
            'application': {
                'preparationScript': 'build_tools/runtimes/llama_cpp/sources/file_backed_rows.py',
                'cmakeHook': 'cmake/FileBackedRowsOverlay.cmake',
                'option': 'LCB_FILE_BACKED_PLE',
                'runtimeOptIn': 'LLAMA_LOAD_MODE_NONE plus LLAMA_LAZY_MODE_ON; non-Qwen4Exp remains eager',
                'enabledProfileVariants': sorted(rows_enabled),
            },
            'supportingFiles': {path: supporting_identity(root, path) for path in
                                ['build_tools/runtimes/llama_cpp/sources/file_backed_rows.py', 'cmake/FileBackedRowsOverlay.cmake',
                                 'native/io/file-backed-rows.h', 'native/io/file-backed-identity.h', 'docs/file-backed-rows.md']},
            'behavior': 'Qwen4Exp PLE only: retain a positioned-read descriptor, gather bounded rows into F32 input, leave n-gram hashing and downstream graph unchanged. No mmap, worker pool, global cache or full-model memory guarantee. All model definition consumers compile a coherent private header copy.',
        },
            {
            'id': 'qsa-all-pools-selection',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': qsa.SOURCE_PATH, **file_identity(vendor / qsa.SOURCE_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{qsa.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / qsa.PATCH_NAME)},
            'compiledCopiesByProfileVariant': qsa_by_variant,
            'sourcePreparationByInputMode': qsa_preparations,
            'application': {
                'preparationScript': 'build_tools/runtimes/llama_cpp/sources/qsa_selection.py',
                'cmakeHook': 'cmake/QsaSelectionOverlay.cmake',
                'option': 'LCB_QSA_ALL_POOLS',
                'enabledProfileVariants': sorted(qsa_enabled),
            },
            'supportingFiles': {path: supporting_identity(root, path) for path in
                               ['build_tools/runtimes/llama_cpp/sources/qsa_selection.py', 'cmake/QsaSelectionOverlay.cmake',
                                'docs/qsa-all-pools-selection.md']},
            'behavior': 'Skip QSA scores, ranking and indirect pool gathers only when all padded pools fit the budget. Preserve visibility, tails, per-layer key writes and the original ranked path. Share only graph-local masks with the same input identity. Independent of file-backed PLE and quantization names; not a measured real-device speedup.',
        }
        ] + webgpu_overlays,
        'otherPatchFiles': {path: {**file_identity(root / path), 'application': 'not classified by this report'} for path in patch_files
                            if path not in described_patches},
        'toolchainDivergences': {
            'emscriptenAsyncifyBigInt': {
                'scope': 'Emscripten runtime, not upstream llama.cpp',
                'implementation': {'path': 'build_tools/toolchain/patch_emscripten.py', 'pathBase': 'sourceRepositoryRawBase', **file_identity(root.parents[1] / 'build_tools/toolchain/patch_emscripten.py')},
                'emscriptenRelease': toolchain['emscriptenRelease'],
                'guardPins': toolchain['emscriptenAsyncifyBigIntPatch'],
            },
        },
    }

    report['sourceOverlays'].append({
        'id': 'vision-bf16-runtime-selection',
        'kind': 'build-tree-source-overlay',
        'upstreamSource': {'path': runtime_source, **file_identity(vendor / runtime_source)},
        'patch': {'path': runtime_patch, **file_identity(root / runtime_patch)},
        'compiledCopy': {'logicalUpstreamPath': runtime_source, **runtime_identity},
        'compiledSourceTreesByProfileVariant': mtmd_trees_by_variant,
        'application': {'preparationScript': 'build_tools/runtimes/llama_cpp/sources/mtmd.py --mode cpu|webgpu',
                        'cmakeHook': 'cmake/MtmdOverlay.cmake',
                        'enabledProfileVariants': sorted(audio_enabled),
                        'runtimeStorageSelection': runtime_metadata()},
        'supportingFiles': {path: supporting_identity(root, path) for path in
                            ['native/vision/mtmd-vision-bf16.h', 'native/vision/mtmd-vision-bf16-policy.h',
                             'cmake/MtmdOverlay.cmake', 'build_tools/runtimes/llama_cpp/bindings.py',
                             'build_tools/runtimes/llama_cpp/bf16.py']},
        'behavior': 'Add explicit load-time immutable storage selection without changing upstream parameter layouts. Models own the policy shared by const subcontexts. Old entry points use legacy defaults; audio and CPU storage retain upstream behavior. Unsupported GPU retain requests fail after existing modality classification and before weight allocation. Every mtmd TU and quoted private header comes from the same checked subtree.',
    })
    report['otherPatchFiles'].pop(runtime_patch, None)
    for item in model_graph_preparations(root, manifest):
        report['sourceOverlays'].append(item)
        report['otherPatchFiles'].pop(item['patch']['path'], None)

    # Non-WebGPU overlays keep their existing per-variant output description.
    # WebGPU entries already refer to the selected final preparation, never to
    # historical intermediate-stage hashes or a normal TU for a profile build.
    for item in report['sourceOverlays']:
        application = item.get('application', {})
        if application.get('preparationScript', '').startswith('build_tools/'):
            application['preparationScriptPathBase'] = 'sourceRepositoryRawBase'
        field = 'compiledCopies' if 'compiledCopies' in item else 'compiledCopy'
        if field not in item:
            continue
        variants = item.get('compiledCopiesByProfileVariant')
        if variants is None:
            variants = {variant: item[field] for variant in application.get('enabledProfileVariants', [])}
            item['compiledCopiesByProfileVariant'] = variants
        values = list(variants.values())
        if values:
            item[field] = values[0] if all(value == values[0] for value in values) else None
    return report


def _check_receipts(manifest: dict, receipt: dict) -> None:
    if (receipt.get('formatVersion') != 1 or receipt.get('parentInputMode') != 'commit-objects'
            or receipt.get('sourceCommit') != manifest['sourceCommit']
            or receipt.get('upstream', {}).get('commit') != manifest['llamaCommit']):
        raise ValueError('Invalid or dirty captured build-input receipt')
    for profile, info in manifest['profiles'].items():
        for variant, provenance in info['variants'].items():
            if provenance.get('sourceInputs') != receipt:
                raise ValueError('Built input receipt differs from selected committed inputs: ' + profile + '/' + variant)
            if not isinstance(provenance.get('preparedSources'), dict):
                raise ValueError('Missing compiled prepared-source receipt: ' + profile + '/' + variant)
            settings = provenance.get('upstreamBuildInfo')
            if (not isinstance(settings, dict) or set(settings) != {'commit', 'number'}
                    or not isinstance(settings['commit'], str) or not re.fullmatch(r'[0-9a-f]{7,40}', settings['commit'])
                    or not manifest['llamaCommit'].startswith(settings['commit'])
                    or type(settings['number']) is not int or settings['number'] < 1):
                raise ValueError('Invalid recorded upstream build settings: ' + profile + '/' + variant)
            command = provenance.get('cmakeCommand', [])
            for name, value in [('LLAMA_BUILD_COMMIT', settings['commit']), ('LLAMA_BUILD_NUMBER', str(settings['number']))]:
                actual = [arg for arg in command if arg.startswith(('-D' + name + '=', '-D' + name + ':'))]
                if actual != ['-D' + name + '=' + value]:
                    raise ValueError('Recorded upstream build settings differ from CMake arguments: ' + name)
            if [arg for arg in command if arg.startswith(('-DGIT_EXE=', '-DGIT_EXE:'))] != ['-DGIT_EXE:FILEPATH=']:
                raise ValueError('Captured upstream build must disable ambient ggml Git discovery')



def collect(root: Path, manifest: dict) -> dict:
    """Verify built receipts against committed inputs, without reading worktree bodies."""
    revision, upstream_revision = full_sha(manifest['sourceCommit']), full_sha(manifest['llamaCommit'])
    repository = root.parents[1]
    upstream = checked_sources(repository, {'llama': captured_inputs.UPSTREAM}, revision=revision)['llama']
    if upstream.commit != upstream_revision:
        raise ValueError('Reporting gitlink does not match the built upstream commit')
    with tempfile.TemporaryDirectory(prefix='llama-report-inputs-') as temporary:
        work = Path(temporary)
        captured = captured_inputs.capture(repository, revision, upstream, work / 'inputs')
        _check_receipts(manifest, captured.receipt)
        manifest_path, output = work / 'manifest.json', work / 'report.json'
        manifest_path.write_text(json.dumps(manifest))
        environment = git_environment()
        environment.update({'GIT_CEILING_DIRECTORIES': str(captured.root), 'PYTHONDONTWRITEBYTECODE': '1'})
        subprocess.run([sys.executable, '-m', 'build_tools.runtimes.llama_cpp.provenance',
                        '--manifest', str(manifest_path), '--output', str(output), '--captured'],
                       cwd=captured.root / 'repository', env=environment, check=True)
        captured_inputs.verify(captured)
        return json.loads(output.read_text())


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--captured', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = runtime_root('llama-cpp')
    manifest = json.loads(args.manifest.read_text())
    if args.captured:
        capture_root = root.parents[2]
        captured = captured_inputs.Captured(capture_root, json.loads((capture_root / 'inputs.json').read_text()))
        _check_receipts(manifest, captured.receipt)
        captured_inputs.verify(captured)
        report = _describe(root, manifest, captured_receipt=captured.receipt)
    else:
        report = collect(root, manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
