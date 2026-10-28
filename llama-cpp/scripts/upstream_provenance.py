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
import prepare_webgpu_expert_short as expert_short
import prepare_webgpu_ordered_readback as ordered_readback
import prepare_webgpu_bind_group_cache as bind_group_cache
import prepare_webgpu_q4_short as q4_short
import prepare_sampler_single_sync as sampler_single_sync
import prepare_webgpu_scale_unary as scale_unary
import prepare_webgpu_q8_subgroup as q8_subgroup
import prepare_webgpu_rms_scale as rms_scale
import prepare_webgpu_async_tensor_copy as async_tensor_copy
import prepare_webgpu_iq3s_subgroup as iq3s_subgroup
import prepare_webgpu_iq3s_field_loads as iq3s_field_loads
import prepare_webgpu_matmul_vec4_layout as matmul_vec4_layout
import prepare_webgpu_moe_tight_tile_bound as moe_tight_tile_bound
import prepare_webgpu_iq3s_row_split as iq3s_row_split
import prepare_webgpu_f32_matvec_epilogue as f32_matvec_epilogue
import prepare_webgpu_host_input_snapshot as host_input_snapshot
import prepare_webgpu_ssm_conv_silu as ssm_conv_silu
import prepare_webgpu_moe_group_skip as moe_group_skip
import prepare_webgpu_router_tail as router_tail
import prepare_webgpu_q8_diagnostic as q8_diagnostic
import prepare_webgpu_host_encode_assembly as host_encode_assembly
import prepare_webgpu_f32_small_m as f32_small_m
import prepare_webgpu_moe_gather_logical32 as moe_gather_logical32
import prepare_webgpu_profile_query_budget as profile_query_budget
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
    diagnostic_enabled = []
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
        sampler_compiled = None
        ssm_compiled = None
        hc_compiled = None
        moe_sum_compiled = None
        scale_unary_compiled = None
        q8_subgroup_compiled = None
        rms_scale_compiled = None
        iq3s_subgroup_compiled = None
        iq3s_field_loads_compiled = None
        ssm_conv_silu_compiled = None
        moe_group_skip_compiled = None
        router_tail_compiled = None
        f32_small_m_compiled = None
        moe_gather_logical32_compiled = None
        host_snapshot_compiled = None
        if copy_enabled:
            prepared = webgpu_source.prepare(vendor, Path(temporary) / 'webgpu',
                                             patch_root=root / PATCH_DIRECTORY)
            webgpu_compiled = {
                'logicalUpstreamPath': webgpu_source.SOURCE_PATH, **file_identity(prepared),
                'tensorCopy': True, 'parameterUploadBatching': True, 'orderedReadback': True,
                'bindGroupCache': True,
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
            sampler_compiled = {
                'logicalUpstreamPath': sampler_single_sync.SOURCE_PATH,
                **file_identity(prepared.parent / 'llama-sampler.cpp'),
            }
            ssm_compiled = {
                'logicalUpstreamPath': webgpu_source.SSM_HEADER_PATH,
                **file_identity(prepared.parent / 'ggml-webgpu-shader-lib.hpp'),
            }
            scale_unary_compiled = {name: file_identity(prepared.parent / name) for name in
                                    ['ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp',
                                     'ggml-webgpu-scale-unary.hpp', scale_unary.HEADER_NAME]}
            q8_subgroup_compiled = {name: file_identity(prepared.parent / name) for name in
                                    ['ggml-webgpu-shader-lib.hpp', q8_subgroup.HEADER_NAME]}
            rms_scale_compiled = {name: file_identity(prepared.parent / name) for name in
                                  ['ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp',
                                   'ggml-webgpu-rms-scale.hpp', rms_scale.HEADER_NAME]}
            iq3s_subgroup_compiled = {name: file_identity(prepared.parent / name) for name in
                                      ['ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp', iq3s_subgroup.HEADER_NAME]}
            iq3s_field_loads_compiled = {iq3s_field_loads.HEADER_NAME:
                                        file_identity(prepared.parent / iq3s_field_loads.HEADER_NAME)}
            ssm_conv_silu_compiled = {name: file_identity(prepared.parent / name) for name in
                                      ['ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp',
                                       'ggml-webgpu-ssm-conv-silu.hpp', ssm_conv_silu.HEADER_NAME]}
            moe_group_skip_compiled = {name: file_identity(prepared.parent / name) for name in
                                       ['ggml-webgpu-shader-lib.hpp', moe_group_skip.HEADER_NAME]}
            router_tail_compiled = {name: file_identity(prepared.parent / name) for name in
                                     ['ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp',
                                      'ggml-webgpu-router-tail.hpp', router_tail.HEADER_NAME]}
            f32_small_m_compiled = {'ggml-webgpu-shader-lib.hpp': file_identity(prepared.parent / 'ggml-webgpu-shader-lib.hpp')}
            moe_gather_logical32_compiled = {name: file_identity(prepared.parent / name) for name in
                                            ['ggml-webgpu-shader-lib.hpp', moe_gather_logical32.HEADER_NAME]}
            host_snapshot_compiled = {name: file_identity(prepared.parent / name) for name in
                                      ['ggml-webgpu.cpp', 'ggml-backend.cpp', 'ggml-backend-host-snapshot.h']}
        webgpu_copies = {variant: webgpu_compiled for variant in copy_enabled}
        diagnostic_files = {}
        diagnostic_copies = {}
        if diagnostic_enabled:
            profiled = webgpu_source.prepare(vendor, Path(temporary) / 'webgpu-profile',
                                            patch_root=root / PATCH_DIRECTORY, gpu_profile=True).parent
            diagnostic_files = {path.name: file_identity(path) for path in profiled.iterdir() if path.is_file()}
            names = ['ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp', 'ggml-webgpu-q8-diagnostic.h',
                     'ggml-webgpu-q8-diagnostic-sha256.hpp']
            diagnostic_copies = {variant: {name: diagnostic_files[name] for name in names} for variant in diagnostic_enabled}
        query_budget_copies = {variant: {'ggml-webgpu.cpp': diagnostic_files['ggml-webgpu.cpp']}
                               for variant in diagnostic_enabled}
        vec4_copies = {variant: {name: hc_compiled[name] for name in matmul_vec4_layout.OUTPUTS[False]}
                       for variant in copy_enabled}
        tight_tile_copies = {variant: {'ggml-webgpu.cpp': hc_compiled['ggml-webgpu.cpp']}
                             for variant in copy_enabled}
        epilogue_names = ['ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp',
                          f32_matvec_epilogue.HELPER_NAME, f32_matvec_epilogue.HEADER_NAME]
        epilogue_copies = {variant: {name: file_identity(prepared.parent / name) for name in epilogue_names}
                           for variant in copy_enabled}
        epilogue_copies.update({variant: {name: diagnostic_files[name] for name in epilogue_names}
                               for variant in diagnostic_enabled})

    supporting = ['scripts/prepare_mtmd.py', 'cmake/MtmdOverlay.cmake',
                  'bridge/mtmd-bf16.h', 'docs/webgpu-bf16-projector.md']
    report = {
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
            'behavior': 'Coalesce parameter writes only at existing graph submission boundaries. Slot padding increases transferred bytes; no measured performance claim. The WebGPU overlays compose into one translation unit.',
        }, {
            'id': 'webgpu-ordered-readback',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': webgpu_source.SOURCE_PATH, **file_identity(vendor / webgpu_source.SOURCE_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{ordered_readback.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / ordered_readback.PATCH_NAME)},
            'compiledCopy': webgpu_compiled,
            'compiledCopiesByProfileVariant': webgpu_copies,
            'reviewedInputs': ordered_readback.CONTRACT_INPUTS,
            'reviewedDawn': {'release': tensor_copy.DAWN_RELEASE, 'headers': tensor_copy.DAWN_HEADERS},
            'application': {'preparationScript': 'scripts/prepare_webgpu_source.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_ordered_readback.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'For nonempty reads from the same WebGPU queue, skip the redundant preliminary queue wait and use the existing blocking copy and map; foreign and zero-byte reads retain the generic synchronization behavior. No measured speedup is claimed.',
        }, {
            'id': 'webgpu-host-encode-assembly',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': webgpu_source.SOURCE_PATH, **file_identity(vendor / webgpu_source.SOURCE_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{host_encode_assembly.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / host_encode_assembly.PATCH_NAME)},
            'compiledCopy': webgpu_compiled,
            'compiledCopiesByProfileVariant': webgpu_copies,
            'reviewedInputs': host_encode_assembly.CONTRACT_INPUTS,
            'composedInputSha256': host_encode_assembly.INPUT_SHA256,
            'application': {'preparationScript': 'scripts/prepare_webgpu_host_encode_assembly.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_host_encode_assembly.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Borrow single-dispatch inputs only during synchronous encoding and reserve bounded temporary vectors. Preserve parameter staging, exact binding identity, profile-owned names, API order, queue ownership and batching. No GPU or target-device speedup is claimed.',
        }, {
            'id': 'webgpu-bind-group-cache',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': webgpu_source.SOURCE_PATH, **file_identity(vendor / webgpu_source.SOURCE_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{bind_group_cache.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / bind_group_cache.PATCH_NAME)},
            'compiledCopy': webgpu_compiled,
            'compiledCopiesByProfileVariant': webgpu_copies,
            'reviewedInputs': bind_group_cache.CONTRACT_INPUTS,
            'reviewedDawn': {'release': tensor_copy.DAWN_RELEASE, 'headers': tensor_copy.DAWN_HEADERS},
            'application': {'preparationScript': 'scripts/prepare_webgpu_source.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_bind_group_cache.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Reuse exact buffer-only bind groups at matching dispatch positions across graph encodes; retain at most 8192 positions and 16 buffer entries per group, bypass unsupported entries and invalidate on buffer destruction. This bounds object counts, not retained resource bytes or VRAM; no target-device speedup is claimed.',
        }, {
            'id': 'webgpu-async-tensor-copy',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': webgpu_source.SOURCE_PATH, **file_identity(vendor / webgpu_source.SOURCE_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{async_tensor_copy.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / async_tensor_copy.PATCH_NAME)},
            'compiledCopy': webgpu_compiled,
            'compiledCopiesByProfileVariant': webgpu_copies,
            'reviewedInputs': async_tensor_copy.CONTRACT_INPUTS,
            'application': {'preparationScript': 'scripts/prepare_webgpu_source.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_async_tensor_copy.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Use the existing optional async copy hook for checked same-queue WebGPU copies and CPU-to-WebGPU uploads after synchronizing the CPU producer; unsupported cases retain the generic fallback. No target-device speedup is claimed.',
        }, {
            'id': 'webgpu-host-input-snapshot',
            'kind': 'build-tree-source-overlay',
            'upstreamSources': {path: file_identity(vendor / path) for path in
                                [webgpu_source.SOURCE_PATH, host_input_snapshot.BACKEND_SOURCE_PATH]},
            'patch': {'path': f'{PATCH_DIRECTORY}/{host_input_snapshot.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / host_input_snapshot.PATCH_NAME)},
            'compiledCopies': host_snapshot_compiled,
            'compiledCopiesByProfileVariant': {variant: host_snapshot_compiled for variant in copy_enabled},
            'reviewedInputs': host_input_snapshot.CONTRACT_INPUTS,
            'reviewedDawn': {'release': tensor_copy.DAWN_RELEASE, 'headers': tensor_copy.DAWN_HEADERS},
            'application': {'preparationScript': 'scripts/prepare_webgpu_source.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_host_input_snapshot.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'The scheduler optionally discovers a versioned host-input snapshot callback for eligible single-copy, event-free, non-weight inputs after the first input in each split. The first input retains the original wait path to bound queued work across repeated prefill calls without output readback. The WebGPU provider synchronizes the CPU producer and snapshots host bytes before return, ordering upload with destination work on the same queue. Other cases retain their existing paths; no target-device speedup is claimed.',
        }, {
            'id': 'webgpu-sampler-single-sync',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': sampler_single_sync.SOURCE_PATH, **file_identity(vendor / sampler_single_sync.SOURCE_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{sampler_single_sync.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / sampler_single_sync.PATCH_NAME)},
            'compiledCopy': sampler_compiled,
            'compiledCopiesByProfileVariant': {variant: sampler_compiled for variant in copy_enabled},
            'reviewedInputs': sampler_single_sync.CONTRACT_INPUTS,
            'application': {'preparationScript': 'scripts/prepare_webgpu_source.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_sampler_single_sync.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Synchronize the context once at the start of sampler_sample before reading internal host outputs; retain the ordinary logits fallback and public getter behavior. Enabled for WebGPU profiles, without a target-device speedup claim.',
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
            'id': 'webgpu-scale-unary-fusion',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{scale_unary.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / scale_unary.PATCH_NAME)},
            'compiledCopies': scale_unary_compiled,
            'compiledCopiesByProfileVariant': {variant: scale_unary_compiled for variant in copy_enabled},
            'reviewedInputs': scale_unary.CONTRACT_INPUTS,
            'application': {'preparationScript': 'scripts/prepare_webgpu_scale_unary.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_scale_unary.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Fuse eligible contiguous F32 scale, SILU or sigmoid, and optional final scale operations into one dispatch, retaining guarded fallback and upstream graph semantics. No target-device speedup is claimed.',
        }, {
            'id': 'webgpu-f32-small-m-prefill',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': str(f32_small_m.HEADER_PATH),
                               **file_identity(vendor / f32_small_m.HEADER_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{f32_small_m.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / f32_small_m.PATCH_NAME)},
            'compiledCopies': f32_small_m_compiled,
            'compiledCopiesByProfileVariant': {variant: f32_small_m_compiled for variant in copy_enabled},
            'reviewedInputs': f32_small_m.CONTRACT_INPUTS,
            'composedInputSha256': f32_small_m.INPUT_SHA256,
            'normalOutputSha256': f32_small_m.OUTPUT_SHA256,
            'application': {'preparationScript': 'scripts/prepare_webgpu_f32_small_m.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_f32_small_m.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Use narrower register tiles for F32/F32 small-M matrices with K>=128 and N>=5, excluding subgroup matrices and source overlap. Preserve canonical TK8 arithmetic, parameter bytes, binding layout and dispatch grid; cache keys and profile labels distinguish effective geometries. No target-device speedup is claimed.',
        }, {
            'id': 'webgpu-moe-gather-logical32',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': str(moe_gather_logical32.BASE / 'wgsl-shaders/mul_mat_id_gather.wgsl'),
                               **file_identity(vendor / moe_gather_logical32.BASE / 'wgsl-shaders/mul_mat_id_gather.wgsl')},
            'patch': {'path': f'{PATCH_DIRECTORY}/{moe_gather_logical32.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / moe_gather_logical32.PATCH_NAME)},
            'compiledCopies': moe_gather_logical32_compiled,
            'compiledCopiesByProfileVariant': {variant: moe_gather_logical32_compiled for variant in copy_enabled},
            'reviewedInputs': moe_gather_logical32.CONTRACT_INPUTS,
            'composedInputSha256': moe_gather_logical32.INPUT_SHA256,
            'normalOutputSha256': moe_gather_logical32.OUTPUT_SHA256,
            'specializedShaderSha256': moe_gather_logical32.SHADER_SHA256,
            'embeddedHeaderSha256': moe_gather_logical32.HEADER_SHA256,
            'application': {'preparationScript': 'scripts/prepare_webgpu_moe_gather_logical32.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_moe_gather_logical32.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Distribute only skipped-slot zeroing across logical groups of at most 32 lanes in the existing MoE gather. Preserve shared-memory capacity, atomics, barriers, keys, parameters, bindings, grid and matrix arithmetic. No hardware subgroup feature or target-device speedup is claimed.',
        }, {
            'id': 'webgpu-matmul-vec4-layout',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{matmul_vec4_layout.PATCHES[False][0]}',
                      **file_identity(root / PATCH_DIRECTORY / matmul_vec4_layout.PATCHES[False][0])},
            'profilePatch': {'path': f'{PATCH_DIRECTORY}/{matmul_vec4_layout.PATCHES[True][0]}',
                             **file_identity(root / PATCH_DIRECTORY / matmul_vec4_layout.PATCHES[True][0])},
            'compiledCopies': (next(iter(vec4_copies.values())) if vec4_copies else None),
            'compiledCopiesByProfileVariant': vec4_copies,
            'reviewedInputs': matmul_vec4_layout.CONTRACT_INPUTS,
            'prefixInputs': {('profile' if profile else 'normal'): values
                             for profile, values in matmul_vec4_layout.INPUTS.items()},
            'finalOutputs': {('profile' if profile else 'normal'): values
                             for profile, values in matmul_vec4_layout.OUTPUTS.items()},
            'finalOutputsScope': 'Outputs of this intermediate stage; compiledCopies identify the final composition.',
            'application': {'preparationScript': 'scripts/prepare_webgpu_matmul_vec4_layout.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_matmul_vec4_layout.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Require actual binding-relative input offsets and active strides to align to floating-point vec4 loads; check matrix output offsets too. Otherwise use the existing scalar shader. Preserve aligned keys and WGSL, uniform/binding ABI, dispatch geometry, quantized paths and gather semantics. Historical prefix pins remain distinct from final compiled identities.',
        }, {
            'id': 'webgpu-f32-matvec-epilogue',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{f32_matvec_epilogue.PATCHES[False][0]}',
                      **file_identity(root / PATCH_DIRECTORY / f32_matvec_epilogue.PATCHES[False][0])},
            'profilePatch': {'path': f'{PATCH_DIRECTORY}/{f32_matvec_epilogue.PATCHES[True][0]}',
                             **file_identity(root / PATCH_DIRECTORY / f32_matvec_epilogue.PATCHES[True][0])},
            'compiledCopies': (next(iter(epilogue_copies.values())) if epilogue_copies else None),
            'compiledCopiesByProfileVariant': epilogue_copies,
            'reviewedInputs': f32_matvec_epilogue.CONTRACT_INPUTS,
            'prefixInputs': {('profile' if profile else 'normal'): values
                             for profile, values in f32_matvec_epilogue.INPUTS.items()},
            'finalOutputs': {('profile' if profile else 'normal'): values
                             for profile, values in f32_matvec_epilogue.OUTPUTS.items()},
            'embeddedHeaderSha256': f32_matvec_epilogue.HEADER_SHA256,
            'application': {'preparationScript': 'scripts/prepare_webgpu_f32_matvec_epilogue.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_f32_matvec_epilogue.py', 'scripts/prepare_webgpu_scale_unary.py',
                                 'scripts/prepare_webgpu_source.py', 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Fuse only guarded dense F32 four-row single-column matvec followed by scale, sigmoid and scale. Retain original accumulation, reduction and geometry, use a distinct default-none pipeline key, copy four coefficients into 80-byte parameters and reject all live input/output payload and aligned binding overlaps. Existing graph, callback and scale-unary fallbacks remain. No model or target-device speedup is claimed.',
        }, {
            'id': 'webgpu-moe-tight-tile-bound',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{moe_tight_tile_bound.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / moe_tight_tile_bound.PATCH_NAME)},
            'compiledCopies': (next(iter(tight_tile_copies.values())) if tight_tile_copies else None),
            'compiledCopiesByProfileVariant': tight_tile_copies,
            'reviewedInputs': moe_tight_tile_bound.CONTRACT_INPUTS,
            'prefixInputs': {('profile' if profile else 'normal'): values
                             for profile, values in moe_tight_tile_bound.INPUTS.items()},
            'finalOutputs': {('profile' if profile else 'normal'): values
                             for profile, values in moe_tight_tile_bound.OUTPUTS.items()},
            'application': {'preparationScript': 'scripts/prepare_webgpu_moe_tight_tile_bound.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_moe_tight_tile_bound.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Tighten only the shared host tile-count upper bound to A+floor((S-A)/L), A=min(E,S), with the existing L=32. Allocation, descriptor binding and direct dispatch use the same helper. Preserve all shaders, parameters, cache keys and mathematical outputs. No indirect work or runtime probe is added. Savings count provably empty workgroups and descriptor capacity; target-device speed is not established.',
        }, {
            'id': 'webgpu-profile-query-budget',
            'kind': 'profile-only-build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{profile_query_budget.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / profile_query_budget.PATCH_NAME)},
            'compiledCopies': (next(iter(query_budget_copies.values())) if query_budget_copies else None),
            'compiledCopiesByProfileVariant': query_budget_copies,
            'reviewedInputs': profile_query_budget.CONTRACT_INPUTS,
            'composedInputSha256': profile_query_budget.INPUT_SHA256,
            'profilePrefixSha256': profile_query_budget.OUTPUT_SHA256,
            'application': {'preparationScript': 'scripts/prepare_webgpu_profile_query_budget.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'GGML_WEBGPU_GPU_PROFILE', 'enabledProfileVariants': sorted(diagnostic_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_profile_query_budget.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Reserve timestamp queries for the actual arena slot capacity, including multi-dispatch overshoot. Check that a fresh query set can hold that capacity. Preserve normal source, shaders, ABI, graph submission and parameter staging; only profile collection boundaries may move.',
        }, {
            'id': 'webgpu-q8-kernel-diagnostic-v1',
            'kind': 'profile-only-build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{q8_diagnostic.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / q8_diagnostic.PATCH_NAME)},
            'compiledCopies': (next(iter(diagnostic_copies.values())) if diagnostic_copies else None),
            'compiledCopiesByProfileVariant': diagnostic_copies,
            'reviewedInputs': q8_diagnostic.CONTRACT_INPUTS,
            'composedInputSha256': q8_diagnostic.INPUTS,
            'application': {'preparationScript': 'scripts/prepare_webgpu_q8_diagnostic.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'GGML_WEBGPU_GPU_PROFILE', 'enabledProfileVariants': sorted(diagnostic_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_q8_diagnostic.py', 'scripts/prepare_webgpu_router_tail.py', 'scripts/generate_webgpu_q8_diagnostic.py',
                                 'bridge/webgpu-q8-diagnostic.h', 'bridge/webgpu-q8-diagnostic.cpp',
                                 'scripts/prepare_webgpu_source.py', 'cmake/WebgpuSourceOverlay.cmake', 'CMakeLists.txt']},
            'behavior': 'Expose a typed optional Q8 kernel diagnostic only in the existing GPU-profile build. Reuse actual matvec factories, params/bindings/grid and graph submission. PREPARE caches actual processed-WGSL SHA-256; EXECUTE only looks up prepared keys. Normal source and base binding outputs remain unchanged. No target-device speedup is claimed.',
        }, {
            'id': 'webgpu-ssm-conv-silu-fusion',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{ssm_conv_silu.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / ssm_conv_silu.PATCH_NAME)},
            'compiledCopies': ssm_conv_silu_compiled,
            'compiledCopiesByProfileVariant': {variant: ssm_conv_silu_compiled for variant in copy_enabled},
            'reviewedInputs': ssm_conv_silu.CONTRACT_INPUTS,
            'application': {'preparationScript': 'scripts/prepare_webgpu_ssm_conv_silu.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_ssm_conv_silu.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Fuse eligible F32 SSM convolution followed by SiLU, preserving scalar/vec4 accumulation and single-token workgroup selection. Guard graph visibility, layout, physical buffers and aligned binding overlap; retain staged fallback. No target-device speedup is claimed.',
        }, {
            'id': 'webgpu-moe-matrix-group-skip',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': str(moe_group_skip.BASE / 'wgsl-shaders/mul_mat_id.wgsl'),
                               **file_identity(vendor / moe_group_skip.BASE / 'wgsl-shaders/mul_mat_id.wgsl')},
            'patch': {'path': f'{PATCH_DIRECTORY}/{moe_group_skip.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / moe_group_skip.PATCH_NAME)},
            'compiledCopies': moe_group_skip_compiled,
            'compiledCopiesByProfileVariant': {variant: moe_group_skip_compiled for variant in copy_enabled},
            'preparedShaderSha256': moe_group_skip.OUTPUTS['wgsl-shaders/' + moe_group_skip.SHADER_NAME],
            'reviewedInputs': moe_group_skip.CONTRACT_INPUTS,
            'application': {'preparationScript': 'scripts/prepare_webgpu_moe_group_skip.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_moe_group_skip.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Skip inactive four-column groups only in the MoE matrix register-tile calculation. Preserve constant column loops, shared initialization, barriers, gather, parameters, grid, valid F16 arithmetic and final stores. Canonical matvec remains unchanged. No target-device or model speedup is claimed.',
        }, {
            'id': 'webgpu-router-tail-fusion',
            'kind': 'build-tree-source-overlay',
            'upstreamSource': {'path': str(router_tail.BASE / 'wgsl-shaders/sum_rows.wgsl'),
                               **file_identity(vendor / router_tail.BASE / 'wgsl-shaders/sum_rows.wgsl')},
            'patch': {'path': f'{PATCH_DIRECTORY}/{router_tail.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / router_tail.PATCH_NAME)},
            'compiledCopies': router_tail_compiled,
            'compiledCopiesByProfileVariant': {variant: router_tail_compiled for variant in copy_enabled},
            'preparedShadersSha256': {name: value for name, value in router_tail.OUTPUTS.items() if name.startswith('wgsl-shaders/')},
            'reviewedInputs': router_tail.CONTRACT_INPUTS,
            'application': {'preparationScript': 'scripts/prepare_webgpu_router_tail.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_router_tail.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Fuse eligible contiguous F32 gather/reshape/sum/clamp/div/reshape subgraphs. Preserve the sum tree, clamp and direct division. Permit overlapping probability/output bindings only for one token and one workgroup through a merged storage binding and full-lane storage barrier; keep IDs disjoint and retain fallback elsewhere. No target-device speedup is claimed.',
        }, {
            'id': 'webgpu-rms-scale-fusion',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{rms_scale.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / rms_scale.PATCH_NAME)},
            'compiledCopies': rms_scale_compiled,
            'compiledCopiesByProfileVariant': {variant: rms_scale_compiled for variant in copy_enabled},
            'reviewedInputs': rms_scale.CONTRACT_INPUTS,
            'application': {'preparationScript': 'scripts/prepare_webgpu_rms_scale.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_rms_scale.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Fuse guarded F32 RMS normalization followed by scale into one dispatch, retaining existing RMS-multiply priority and eligible three-operation scale-unary-scale fusion. Unsupported graph and binding cases fall back. No target-device speedup is claimed.',
        }, {
            'id': 'webgpu-iq3s-indexed-subgroup-matvec',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{iq3s_subgroup.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / iq3s_subgroup.PATCH_NAME)},
            'compiledCopies': iq3s_subgroup_compiled,
            'compiledCopiesByProfileVariant': {variant: iq3s_subgroup_compiled for variant in copy_enabled},
            'reviewedInputs': iq3s_subgroup.CONTRACT_INPUTS,
            'intermediateEmbeddedHeaderSha256': iq3s_subgroup.HEADER_SHA256,
            'application': {'preparationScript': 'scripts/prepare_webgpu_iq3s_subgroup.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_iq3s_subgroup.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Use independent fixed-width subgroups for guarded single-token IQ3_S indexed matvecs, preserving the upstream decoder and accumulator; cover missing logical lanes when subgroups are partial and fall back for unsupported shapes, device limits or ranges. Geometry probes are test-only and add no production CPU readback. Reduction order differs; target model quality and performance are not established.',
        }, {
            'id': 'webgpu-iq3s-indexed-field-loads',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{iq3s_field_loads.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / iq3s_field_loads.PATCH_NAME)},
            'compiledCopies': iq3s_field_loads_compiled,
            'compiledCopiesByProfileVariant': {variant: iq3s_field_loads_compiled for variant in copy_enabled},
            'reviewedInputs': iq3s_field_loads.CONTRACT_INPUTS,
            'intermediateEmbeddedHeaderSha256': iq3s_field_loads.HEADER_SHA256,
            'upstreamSource': {'path': iq3s_field_loads.ACCUMULATOR_PATH,
                               **file_identity(vendor / iq3s_field_loads.ACCUMULATOR_PATH)},
            'priorStage': {'id': 'webgpu-iq3s-indexed-subgroup-matvec',
                           'intermediateEmbeddedHeaderSha256': iq3s_field_loads.PRIOR_HEADER_SHA256},
            'application': {'preparationScript': 'scripts/prepare_webgpu_iq3s_field_loads.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_iq3s_field_loads.py', 'scripts/prepare_webgpu_iq3s_subgroup.py',
                                 'scripts/prepare_webgpu_source.py', 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Replace only the existing independent indexed IQ3_S embedding with lane-selected qs and aligned-containing-word metadata loads. Preserve the decoder arithmetic, main/reduction, selector, cache key, parameter/binding layout and all fallback paths. The original embedding hash is an intermediate stage contract; compiled copies describe the final header. No model or target-device speedup is established.',
        }, {
            'id': 'webgpu-iq3s-indexed-row-split',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{iq3s_row_split.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / iq3s_row_split.PATCH_NAME)},
            'compiledCopies': iq3s_field_loads_compiled,
            'compiledCopiesByProfileVariant': {variant: iq3s_field_loads_compiled for variant in copy_enabled},
            'reviewedInputs': iq3s_row_split.CONTRACT_INPUTS,
            'priorStage': {'id': 'webgpu-iq3s-indexed-field-loads',
                           'intermediateEmbeddedHeaderSha256': iq3s_row_split.PRIOR_HEADER_SHA256,
                           'accumulatorSha256': iq3s_row_split.ACCUMULATOR_INPUT_SHA256},
            'accumulatorOutputSha256': iq3s_row_split.ACCUMULATOR_OUTPUT_SHA256,
            'rowsPerSubgroup': iq3s_row_split.ROWS_PER_SUBGROUP,
            'prefixInputs': {('profile' if profile else 'normal'): values
                             for profile, values in iq3s_row_split.INPUTS.items()},
            'finalEmbeddedHeaderSha256': iq3s_row_split.HEADER_SHA256,
            'application': {'preparationScript': 'scripts/prepare_webgpu_iq3s_row_split.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_iq3s_row_split.py', 'scripts/prepare_webgpu_iq3s_field_loads.py',
                                 'scripts/prepare_webgpu_source.py', 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Split the dedicated four-row independent IQ3_S accumulator loop into rows 0..1 and 2..3, retaining the four accumulators, per-row arithmetic and K order, main, partial-subgroup completion, reduction, selection and dispatch contracts. Canonical accumulators and fallback shaders remain upstream. Private software hot-only measurements are not model, streaming or target-device performance evidence.',
        }, {
            'id': 'webgpu-q8-fixed-subgroup-matvec',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{q8_subgroup.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / q8_subgroup.PATCH_NAME)},
            'compiledCopies': q8_subgroup_compiled,
            'compiledCopiesByProfileVariant': {variant: q8_subgroup_compiled for variant in copy_enabled},
            'reviewedInputs': q8_subgroup.CONTRACT_INPUTS,
            'application': {'preparationScript': 'scripts/prepare_webgpu_q8_subgroup.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_q8_subgroup.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'For guarded long-K Q8_0 single-column matvecs with fixed reported subgroup width, process rows independently per subgroup, reusing the pinned upstream decoder and accumulator; cover missing logical lanes when subgroups are partial and retain upstream fallback elsewhere. Geometry probes are test-only and add no production CPU readback. Numerical order changes and target-device speed or model quality are not established.',
        }, {
            'id': 'webgpu-q4-short-workgroup',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{q4_short.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / q4_short.PATCH_NAME)},
            'compiledCopy': ssm_compiled,
            'compiledCopiesByProfileVariant': {variant: ssm_compiled for variant in copy_enabled},
            'reviewedInputs': q4_short.CONTRACT_INPUTS,
            'application': {'preparationScript': 'scripts/prepare_webgpu_q4_short.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_webgpu_q4_short.py', 'scripts/prepare_webgpu_source.py',
                                 'cmake/WebgpuSourceOverlay.cmake']},
            'behavior': 'Select existing 32, 64, 128 or 256 lane subgroup Q4_K single-column ordinary and indexed matvec workgroups when the shape, device and alias guards permit. Preserve upstream shader arithmetic, output tiling and dispatch grid; otherwise use the upstream workgroup. No target-device speedup is claimed.',
        }, {
            'id': 'webgpu-expert-short-workgroup',
            'kind': 'build-tree-source-overlay',
            'patch': {'path': f'{PATCH_DIRECTORY}/{expert_short.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / expert_short.PATCH_NAME)},
            'compiledCopies': ({'ggml-webgpu.cpp': webgpu_compiled,
                                'ggml-webgpu-shader-lib.hpp': ssm_compiled} if copy_enabled else None),
            'reviewedInputs': expert_short.CONTRACT_INPUTS,
            'application': {'preparationScript': 'scripts/prepare_webgpu_expert_short.py',
                            'cmakeHook': 'cmake/WebgpuSourceOverlay.cmake',
                            'option': 'LCB_WEBGPU', 'enabledProfileVariants': sorted(copy_enabled)},
            'supportingFiles': {path: file_identity(root / path) for path in
                               ['scripts/prepare_webgpu_expert_short.py', 'docs/webgpu-expert-short-workgroup.md']},
            'behavior': 'Use 128 lanes for subgroup Q5_1/IQ4_NL x F32 single-token expert matvecs with K <= 1024 and at least 256 rows. Preserve upstream WGSL, selected-slot dispatch, quantization and per-lane block ownership. Forward actual subgroup limits and key pipelines by workgroup size; unsupported paths retain 256 lanes. No target-device speed guarantee.',
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
                            for path in patch_files if path not in (f'{PATCH_DIRECTORY}/{f32_matvec_epilogue.PATCHES[False][0]}', f'{PATCH_DIRECTORY}/{f32_matvec_epilogue.PATCHES[True][0]}', f'{PATCH_DIRECTORY}/{iq3s_row_split.PATCH_NAME}', f'{PATCH_DIRECTORY}/{moe_tight_tile_bound.PATCH_NAME}', f'{PATCH_DIRECTORY}/{matmul_vec4_layout.PATCHES[False][0]}', f'{PATCH_DIRECTORY}/{matmul_vec4_layout.PATCHES[True][0]}', f'{PATCH_DIRECTORY}/{iq3s_field_loads.PATCH_NAME}', f'{PATCH_DIRECTORY}/{moe_gather_logical32.PATCH_NAME}', f'{PATCH_DIRECTORY}/{f32_small_m.PATCH_NAME}', f'{PATCH_DIRECTORY}/{profile_query_budget.PATCH_NAME}', f'{PATCH_DIRECTORY}/{host_encode_assembly.PATCH_NAME}', f'{PATCH_DIRECTORY}/{q8_diagnostic.PATCH_NAME}', f'{PATCH_DIRECTORY}/{router_tail.PATCH_NAME}', f'{PATCH_DIRECTORY}/{moe_group_skip.PATCH_NAME}', f'{PATCH_DIRECTORY}/{ssm_conv_silu.PATCH_NAME}', f'{PATCH_DIRECTORY}/{host_input_snapshot.PATCH_NAME}', f'{PATCH_DIRECTORY}/{iq3s_subgroup.PATCH_NAME}', f'{PATCH_DIRECTORY}/{rms_scale.PATCH_NAME}', f'{PATCH_DIRECTORY}/{async_tensor_copy.PATCH_NAME}', f'{PATCH_DIRECTORY}/{scale_unary.PATCH_NAME}', f'{PATCH_DIRECTORY}/{q8_subgroup.PATCH_NAME}', f'{PATCH_DIRECTORY}/{sampler_single_sync.PATCH_NAME}', f'{PATCH_DIRECTORY}/{bind_group_cache.PATCH_NAME}', f'{PATCH_DIRECTORY}/{q4_short.PATCH_NAME}', f'{PATCH_DIRECTORY}/{ordered_readback.PATCH_NAME}', f'{PATCH_DIRECTORY}/{expert_short.PATCH_NAME}', f'{PATCH_DIRECTORY}/{moe_sum.PATCH_NAME}', f'{PATCH_DIRECTORY}/{q8_short.PATCH_NAME}', f'{PATCH_DIRECTORY}/{hc.PATCH_NAME}', f'{PATCH_DIRECTORY}/{qsa.PATCH_NAME}', f'{PATCH_DIRECTORY}/{file_rows.PATCH_NAME}', patch_path, audio_patch, f'{PATCH_DIRECTORY}/{tensor_copy.PATCH_NAME}', f'{PATCH_DIRECTORY}/{webgpu_source.PARAM_PATCH_NAME}', f'{PATCH_DIRECTORY}/{webgpu_source.LOADER_PATCH_NAME}', f'{PATCH_DIRECTORY}/{webgpu_source.SSM_PATCH_NAME}')},
        'toolchainDivergences': {
            'emscriptenAsyncifyBigInt': {
                'scope': 'Emscripten runtime, not upstream llama.cpp',
                'implementation': {'path': 'scripts/patch_emscripten.py', 'pathBase': 'sourceRepositoryRawBase', **file_identity(root.parent / 'scripts/patch_emscripten.py')},
                'emscriptenRelease': toolchain['emscriptenRelease'],
                'guardPins': toolchain['emscriptenAsyncifyBigIntPatch'],
            },
        },
    }

    # Profile and normal TUs now differ. Never attribute the normal final bytes
    # to a profile artifact, including earlier overlays that share the final TU.
    def actual_copy(value):
        if not isinstance(value, dict):
            return value
        result = {}
        for name, child in value.items():
            result[name] = actual_copy(child)
            if name in diagnostic_files and isinstance(child, dict) and 'sha256' in child and 'bytes' in child:
                result[name].update(diagnostic_files[name])
        logical = value.get('logicalUpstreamPath')
        if logical and Path(logical).name in diagnostic_files:
            result.update(diagnostic_files[Path(logical).name])
            if logical == webgpu_source.SOURCE_PATH and 'compileDefinitions' in result:
                result['compileDefinitions'] = result['compileDefinitions'] + ['GGML_WEBGPU_GPU_PROFILE=1']
        return result

    for item in report['sourceOverlays']:
        field = 'compiledCopies' if 'compiledCopies' in item else 'compiledCopy'
        if field not in item:
            continue
        variants = item.get('compiledCopiesByProfileVariant')
        if variants is None:
            variants = {variant: item[field] for variant in item.get('application', {}).get('enabledProfileVariants', [])}
        item['compiledCopiesByProfileVariant'] = {
            variant: actual_copy(value) if variant in diagnostic_enabled else value for variant, value in variants.items()}
        values = list(item['compiledCopiesByProfileVariant'].values())
        if values:
            item[field] = values[0] if all(value == values[0] for value in values) else None
    return report


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
