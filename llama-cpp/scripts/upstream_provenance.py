#!/usr/bin/env python3
"""Describe source overlays for CI reports without changing the runtime manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[2] / 'scripts'))
from browser_toolchain import runtime_toolchain
import tempfile

from github_api import full_sha, git
from prepare_mtmd import PATCH_DIRECTORY, prepare
import prepare_webgpu_tensor_copy as tensor_copy
import prepare_webgpu_source as webgpu_source
import prepare_webgpu_hc as hc
import prepare_webgpu_q8_short as q8_short
import prepare_webgpu_moe_sum as moe_sum
import prepare_file_backed_rows as file_rows
import prepare_qsa_selection as qsa


def file_identity(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or linked provenance input: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


def browser_module_preparations(root: Path, manifest: dict) -> list[dict]:
    records = []
    implementation = None
    files = {item['path']: item for item in manifest.get('files', [])}
    for profile, info in sorted(manifest['profiles'].items()):
        for variant, provenance in sorted(info['variants'].items()):
            preparation = provenance.get('browserModulePreparation')
            if preparation is None:
                continue
            expected = {'path': 'scripts/prepare_browser_module.py',
                        **file_identity(root / 'scripts/prepare_browser_module.py')}
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
             'supportingFiles': {path: file_identity(root / path) for path in
                                 ['scripts/build.py', 'bridge/host-navigator-pre.js', 'docs/browser-host.md']},
             'behavior': 'Add one @vite-ignore comment to the generated sibling core.wasm URL in browser variants. Vite skips automatic Wasm discovery at this expression; other bundlers may differ. Runtime acquisition, supplied bytes and fallback branches are unchanged. Hosts explicitly deploy the matching Wasm or supply bytes. Input/output identities describe the actual preparation, including no-op reruns.'}]


def collect(root: Path, manifest: dict) -> dict:
    source = full_sha(manifest['sourceCommit'])
    upstream = full_sha(manifest['llamaCommit'])
    if git('rev-parse', 'HEAD', cwd=root).stdout.strip() != source:
        raise ValueError('Reporting checkout does not match the built source commit')
    vendor = root / 'vendor/llama.cpp'
    if git('rev-parse', 'HEAD', cwd=vendor).stdout.strip() != upstream:
        raise ValueError('Reporting submodule does not match the built upstream commit')
    patch_path = f'{PATCH_DIRECTORY}/mtmd-webgpu-bf16.patch'
    upstream_path = 'tools/mtmd/clip.cpp'
    audio_patch = f'{PATCH_DIRECTORY}/mtmd-audio-single-thread.patch'
    audio_source = 'tools/mtmd/mtmd-audio.cpp'
    patch_files = sorted(path.relative_to(root).as_posix() for path in (root / PATCH_DIRECTORY).rglob('*.patch'))
    enabled = []
    audio_enabled = []
    copy_enabled = []
    batch_enabled = []
    rows_enabled = []
    qsa_enabled = []
    toolchain = runtime_toolchain(root)
    for profile, info in manifest['profiles'].items():
        for variant, provenance in info['variants'].items():
            webgpu_options = [option for option in provenance['cmakeCommand']
                              if option.startswith('-DLCB_WEBGPU=') or option.startswith('-DLCB_WEBGPU:')]
            if webgpu_options not in (['-DLCB_WEBGPU=ON'], ['-DLCB_WEBGPU=OFF']):
                raise ValueError('Unknown WebGPU activation in compiled provenance')
            webgpu_enabled = webgpu_options == ['-DLCB_WEBGPU=ON']
            if webgpu_enabled:
                copy_enabled.append(profile + '/' + variant)
                batch_enabled.append(profile + '/' + variant)
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
            if options[0].endswith('=ON'):
                enabled.append(profile + '/' + variant)
    rows_compiled = None
    rows_tree = None
    rows_by_variant = {}
    rows_trees_by_variant = {}
    qsa_by_variant = {}
    rows_source = None
    rows_identities = None
    with tempfile.TemporaryDirectory(prefix='lcb-report-overlay-') as temporary:
        if rows_enabled:
            rows_source = file_rows.prepare(vendor, Path(temporary) / 'file-rows')
            # These model TUs must all use the same modified models.h definition.
            paths = [rows_source / 'llama.cpp', rows_source / 'llama-model.cpp', *sorted((rows_source / 'models').glob('*.cpp')),
                     rows_source / 'models/models.h', rows_source / 'models/file-backed-rows.h']
            identities = {p.relative_to(rows_source).as_posix(): file_identity(p) for p in paths}
            # Unchanged model TUs are covered by an aggregate rather than making
            # the publication summary grow by hundreds of per-file entries.
            encoded = json.dumps(identities, sort_keys=True, separators=(',', ':')).encode()
            rows_tree = {'files': len(identities), 'sha256': hashlib.sha256(encoded).hexdigest(),
                         'encoding': 'UTF-8 compact JSON of sorted relative paths to bytes/SHA-256 objects'}
            selected = ['llama.cpp', 'llama-model.cpp', 'models/models.h', 'models/qwen4exp.cpp',
                        'models/file-backed-rows.h']
            rows_compiled = {name: identities[name] for name in selected}
            rows_identities = identities
        # Describe the translation unit actually compiled, not just the first
        # preparation stage. QSA may follow either original or file-backed rows.
        qsa_by_mode = {}
        for mode in sorted({variant in rows_enabled for variant in qsa_enabled}):
            model_source = (rows_source / 'models/qwen4exp.cpp' if mode else vendor / qsa.SOURCE_PATH)
            prepared_qsa = qsa.prepare(vendor, model_source, Path(temporary) / ('qsa-rows' if mode else 'qsa'))
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
        patched = prepare(vendor, Path(temporary) / 'overlay', root / patch_path)
        patched_identity = file_identity(patched)
        audio = prepare(vendor, Path(temporary) / 'audio', root / audio_patch, filename='mtmd-audio.cpp')
        audio_identity = file_identity(audio)
        webgpu_compiled = None
        loader_compiled = None
        ssm_compiled = None
        hc_compiled = None
        moe_sum_compiled = None
        if copy_enabled:
            prepared = webgpu_source.prepare(vendor, Path(temporary) / 'webgpu',
                                             patch_root=root / PATCH_DIRECTORY)
            webgpu_compiled = {
                'logicalUpstreamPath': webgpu_source.SOURCE_PATH, **file_identity(prepared),
                'tensorCopy': True, 'parameterUploadBatching': True,
                'compileDefinitions': ['GGML_WEBGPU_BATCH_PARAM_UPLOADS'],
            }
            hc_compiled = {'ggml-webgpu.cpp': file_identity(prepared),
                           **{name: file_identity(prepared.parent / name) for name in
                              ['ggml-webgpu-shader-lib.hpp', 'ggml-webgpu-hc.hpp', hc.HEADER_NAME]}}
            moe_sum_compiled = {name: file_identity(prepared.parent / name) for name in
                                ['ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp',
                                 'ggml-webgpu-moe-sum.hpp', moe_sum.HEADER_NAME]}
            loader_compiled = {
                'logicalUpstreamPath': webgpu_source.LOADER_PATH,
                **file_identity(prepared.parent / 'llama-model-loader.cpp'),
            }
            ssm_compiled = {
                'logicalUpstreamPath': webgpu_source.SSM_HEADER_PATH,
                **file_identity(prepared.parent / 'ggml-webgpu-shader-lib.hpp'),
            }
        webgpu_copies = {variant: webgpu_compiled for variant in copy_enabled}
    supporting = ['scripts/prepare_mtmd.py', 'cmake/MtmdOverlay.cmake',
                  'bridge/mtmd-bf16.h', 'docs/webgpu-bf16-projector.md']
    return {
        'upstreamRepository': 'ggml-org/llama.cpp',
        'baseCommit': upstream,
        'vendorCheckoutModified': False,
        'inventoryScope': f'Known build-tree overlays, recorded generated browser-module preparations, plus every *.patch file under {PATCH_DIRECTORY}/; not an exhaustive compiler transformation inventory.',
        'generatedModuleTransformations': browser_module_preparations(root, manifest),
        'retiredSourceOverlays': [{
            'id': 'experimental-webgpu-moe-direct-slot',
            'reason': 'llama.cpp v0.7.0 provides direct selected-slot dispatch upstream; no downstream MoE shader/header transformation is compiled.',
            'upstreamedBy': '620d2225cc436ab0b001e3200e590e3b286ef9ab',
            'documentation': 'docs/moe-direct-slot.md',
        }],
        'sourceOverlays': [{
            'id': 'webgpu-vision-bf16-projector',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': upstream_path, **file_identity(vendor / upstream_path)},
            'patch': {'path': patch_path, **file_identity(root / patch_path)},
            'compiledCopy': {'logicalUpstreamPath': upstream_path, **patched_identity},
            'application': {
                'preparationScript': 'scripts/prepare_mtmd.py',
                'cmakeHook': 'cmake/MtmdOverlay.cmake',
                'option': 'LCB_WEBGPU_BF16_PROJECTOR',
                'enabledProfileVariants': sorted(enabled),
            },
            'supportingFiles': {path: file_identity(root / path) for path in supporting},
            'searchHints': ['mtmd-webgpu-bf16', 'LCB_WEBGPU_BF16_PROJECTOR', 'lcb_clip', 'bf16-f32', 'cpu_bf16'],
            'behavior': 'WebGPU vision BF16 weights become resident F32 weights (2x storage for converted weights); bounded upload, placement diagnostics and loader guards. Upstream graph-allocation failure handling is preserved. Model files remain unchanged.',
        }, {
            'id': 'single-thread-wasm-audio-preprocessing',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': audio_source, **file_identity(vendor / audio_source)},
            'patch': {'path': audio_patch, **file_identity(root / audio_patch)},
            'compiledCopy': {'logicalUpstreamPath': audio_source, **audio_identity},
            'application': {
                'preparationScript': 'scripts/prepare_mtmd.py --component audio',
                'cmakeHook': 'cmake/MtmdAudioOverlay.cmake',
                'compileGuard': 'defined(__EMSCRIPTEN__) && !defined(__EMSCRIPTEN_PTHREADS__)',
                'enabledProfileVariants': sorted(audio_enabled),
            },
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_mtmd.py', 'cmake/MtmdAudioOverlay.cmake',
                                 'docs/audio-single-thread.md']},
            'searchHints': ['mtmd-audio-single-thread', '__EMSCRIPTEN_PTHREADS__', 'log_mel_spectrogram'],
            'behavior': 'Execute shared mel and Parakeet preprocessing serially in non-pthread Emscripten builds. Native and pthread worker loops are unchanged.',
        }, {
            'id': 'webgpu-same-device-tensor-copy',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': tensor_copy.SOURCE_PATH, **file_identity(vendor / tensor_copy.SOURCE_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{tensor_copy.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / tensor_copy.PATCH_NAME)},
            'compiledCopy': webgpu_compiled,
            'compiledCopiesByProfileVariant': webgpu_copies,
            'reviewedCommit': tensor_copy.verify_reviewed_source(vendor) if copy_enabled else None,
            'reviewedInputs': next(iter(tensor_copy.REVIEWED_REVISIONS.values())),
            'reviewedDawn': {'release': tensor_copy.DAWN_RELEASE, 'headers': tensor_copy.DAWN_HEADERS},
            'application': {
                'preparationScript': 'scripts/prepare_webgpu_source.py',
                'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                'option': 'LCB_WEBGPU',
                'enabledProfileVariants': sorted(copy_enabled),
            },
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_tensor_copy.py', 'scripts/prepare_webgpu_source.py', 'cmake/WebgpuSourceOverlay.cmake',
                                 'docs/webgpu-tensor-copy.md']},
            'behavior': 'Eligible same-device aligned copies use the existing optional buffer hook and common GPU queue. Unsupported cases retain generic fallback. No checkpoint policy or capability API change.',
        }, {
            'id': 'webgpu-parameter-upload-batching',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': webgpu_source.SOURCE_PATH, **file_identity(vendor / webgpu_source.SOURCE_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{webgpu_source.PARAM_PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / webgpu_source.PARAM_PATCH_NAME)},
            'compiledCopy': webgpu_compiled,
            'compiledCopiesByProfileVariant': webgpu_copies,
            'reviewedInputs': next(iter(tensor_copy.REVIEWED_REVISIONS.values())),
            'reviewedDawn': {'release': tensor_copy.DAWN_RELEASE, 'headers': tensor_copy.DAWN_HEADERS},
            'application': {
                'preparationScript': 'scripts/prepare_webgpu_source.py',
                'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                'option': 'LCB_WEBGPU',
                'enabledProfileVariants': sorted(batch_enabled),
            },
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_source.py', 'cmake/WebgpuSourceOverlay.cmake',
                                 'docs/webgpu-param-upload-batching.md']},
            'behavior': 'Coalesce parameter writes only at existing graph submission boundaries. Slot padding increases transferred bytes; no measured performance claim. Both patches compose into one translation unit.',
        }, {
            'id': 'webgpu-chunked-model-upload',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': webgpu_source.LOADER_PATH, **file_identity(vendor / webgpu_source.LOADER_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{webgpu_source.LOADER_PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / webgpu_source.LOADER_PATCH_NAME)},
            'compiledCopy': loader_compiled,
            'reviewedInputs': webgpu_source.LOADER_INPUTS,
            'application': {
                'preparationScript': 'scripts/prepare_webgpu_source.py',
                'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                'option': 'LCB_WEBGPU',
                'enabledProfileVariants': sorted(copy_enabled),
            },
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_source.py', 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Bound the synchronous default-WebGPU-buffer tensor read vector to 8 MiB and charged outstanding model uploads to a provisional 32 MiB per load_all_data call, including 24 bytes per upload for auxiliary parameters. Synchronize the actual destination device before exceeding the budget, on device handoff and on scope exit. Other backends retain whole-tensor uploads. This is not a browser RAM limit or graceful device-loss recovery; target-device model-load throughput effects are unmeasured.',
        }, {
            'id': 'webgpu-ssm-conv-single-token',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': webgpu_source.SSM_HEADER_PATH, **file_identity(vendor / webgpu_source.SSM_HEADER_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{webgpu_source.SSM_PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / webgpu_source.SSM_PATCH_NAME)},
            'compiledCopy': ssm_compiled,
            'compiledCopiesByProfileVariant': {variant: ssm_compiled for variant in copy_enabled},
            'reviewedInputs': {**next(iter(tensor_copy.REVIEWED_REVISIONS.values())),
                               webgpu_source.SSM_SHADER_PATH: webgpu_source.SSM_SHADER_SHA256},
            'application': {
                'preparationScript': 'scripts/prepare_webgpu_source.py',
                'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                'option': 'LCB_WEBGPU',
                'enabledProfileVariants': sorted(copy_enabled),
            },
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_source.py', 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Use one token lane only for single-token SSM_CONV, retaining eight lanes otherwise. Cache key, shader define and dispatch decisions share the selected width; arithmetic and MoE embedded shaders remain unchanged. Software-adapter correctness is not a Mac performance claim.',
        }, {
            'id': 'qwen4exp-file-backed-rows',
            'kind': 'build-tree-coherent-model-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{file_rows.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / file_rows.PATCH_NAME)},
            'compiledCopies': rows_compiled,
            'coherentSourceSet': rows_tree,
            'compiledCopiesByProfileVariant': rows_by_variant,
            'coherentSourceSetsByProfileVariant': rows_trees_by_variant,
            'reviewedInputs': json.loads((root / 'tests/file-backed-rows/inputs.json').read_text()),
            'application': {
                'preparationScript': 'scripts/prepare_file_backed_rows.py',
                'cmakeHook': 'cmake/FileBackedRowsOverlay.cmake',
                'option': 'LCB_FILE_BACKED_PLE',
                'runtimeOptIn': 'LLAMA_LOAD_MODE_NONE plus LLAMA_LAZY_MODE_ON; non-Qwen4Exp remains eager',
                'enabledProfileVariants': sorted(rows_enabled),
            },
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_file_backed_rows.py', 'cmake/FileBackedRowsOverlay.cmake',
                                 'bridge/file-backed-rows.h', 'docs/file-backed-rows.md']},
            'behavior': 'Qwen4Exp PLE only: retain a positioned-read descriptor, gather bounded rows into F32 input, leave n-gram hashing and downstream graph unchanged. No mmap, worker pool, global cache or full-model memory guarantee. All model definition consumers compile a coherent private header copy.',
        }, {
            'id': 'qsa-all-pools-selection',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': qsa.SOURCE_PATH, **file_identity(vendor / qsa.SOURCE_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{qsa.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / qsa.PATCH_NAME)},
            'compiledCopiesByProfileVariant': qsa_by_variant,
            'reviewedInputs': json.loads((root / 'tests/qsa-selection/inputs.json').read_text()),
            'application': {
                'preparationScript': 'scripts/prepare_qsa_selection.py',
                'cmakeHook': 'cmake/QsaSelectionOverlay.cmake',
                'option': 'LCB_QSA_ALL_POOLS',
                'enabledProfileVariants': sorted(qsa_enabled),
            },
            'supportingFiles': {path: file_identity(root / path) for path in
                               ['scripts/prepare_qsa_selection.py', 'cmake/QsaSelectionOverlay.cmake',
                                'docs/qsa-all-pools-selection.md']},
            'behavior': 'Skip QSA scores, ranking and indirect pool gathers only when all padded pools fit the budget. Preserve visibility, tails, per-layer key writes and the original ranked path. Share only graph-local masks with the same input identity. Independent of file-backed PLE and quantization names; not a measured real-device speedup.',
        }, {
            'id': 'webgpu-hc-pre-post',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{hc.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / hc.PATCH_NAME)},
            'compiledCopies': hc_compiled,
            'application': {'preparationScript': 'scripts/prepare_webgpu_hc.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {name: file_identity(root / name) for name in
                                ['scripts/prepare_webgpu_hc.py', 'docs/webgpu-hc-pre-post.md']},
            'behavior': 'Support upstream gated HC PRE and identity-combination POST for contiguous F32 HC4 tensors. Use the upstream support probe and fallback; do not change graph construction, model weights, variants or sampling.',
        }, {
            'id': 'webgpu-q8-short-workgroup',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{q8_short.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / q8_short.PATCH_NAME)},
            'compiledCopy': ssm_compiled,
            'application': {'preparationScript': 'scripts/prepare_webgpu_q8_short.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU',
                            'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                               ['scripts/prepare_webgpu_q8_short.py', 'docs/webgpu-q8-short-workgroup.md']},
            'behavior': 'Use 32/64/128-lane workgroups for subgroup Q8_0 x F32 single-column matvecs with K <= 1024 and at least 1024 output rows. Upstream WGSL, quantization, tiling and lane-to-block mapping are unchanged. Include the workgroup size in the pipeline cache key; other paths retain 256 lanes. No target-device speed guarantee.',
        }, {
            'id': 'webgpu-moe-expert-sum',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{moe_sum.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / moe_sum.PATCH_NAME)},
            'compiledCopies': moe_sum_compiled,
            'application': {'preparationScript': 'scripts/prepare_webgpu_moe_sum.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                               ['scripts/prepare_webgpu_moe_sum.py', 'docs/webgpu-moe-expert-sum.md']},
            'behavior': 'Fuse complete slot-ordered contiguous F32 expert-output ADD chains, preserving accumulation order. Reject external intermediate users, unsupported device limits and overlapping aligned bindings; retain staged fallback. No new variants or model graph changes.',
        }],
        'otherPatchFiles': {path: {'application': 'not classified by this report', **file_identity(root / path)}
                            for path in patch_files if path not in (f'{PATCH_DIRECTORY}/{moe_sum.PATCH_NAME}', f'{PATCH_DIRECTORY}/{q8_short.PATCH_NAME}', f'{PATCH_DIRECTORY}/{hc.PATCH_NAME}', f'{PATCH_DIRECTORY}/{qsa.PATCH_NAME}', f'{PATCH_DIRECTORY}/{file_rows.PATCH_NAME}', patch_path, audio_patch, f'{PATCH_DIRECTORY}/{tensor_copy.PATCH_NAME}', f'{PATCH_DIRECTORY}/{webgpu_source.PARAM_PATCH_NAME}', f'{PATCH_DIRECTORY}/{webgpu_source.LOADER_PATCH_NAME}', f'{PATCH_DIRECTORY}/{webgpu_source.SSM_PATCH_NAME}')},
        'toolchainDivergences': {
            'emscriptenAsyncifyBigInt': {
                'scope': 'Emscripten runtime, not upstream llama.cpp',
                'implementation': {'path': 'scripts/patch_emscripten.py', 'pathBase': 'sourceRepositoryRawBase', **file_identity(root.parent / 'scripts/patch_emscripten.py')},
                'emscriptenRelease': toolchain['emscriptenRelease'],
                'guardPins': toolchain['emscriptenAsyncifyBigIntPatch'],
            },
        },
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    report = collect(root, json.loads(args.manifest.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
