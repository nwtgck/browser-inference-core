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
import prepare_moe_direct_slot as moe
import prepare_webgpu_tensor_copy as tensor_copy
import prepare_webgpu_source as webgpu_source


def file_identity(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f'Missing or linked provenance input: {path}')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'bytes': path.stat().st_size, 'sha256': digest}


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
    moe_enabled = []
    copy_enabled = []
    batch_enabled = []
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
            audio_enabled.append(profile + '/' + variant)
            moe_options = [option for option in provenance['cmakeCommand']
                           if option.startswith('-DLCB_WEBGPU_MOE_DIRECT_SLOT=')]
            if len(moe_options) != 1 or moe_options[0] not in (
                    '-DLCB_WEBGPU_MOE_DIRECT_SLOT=ON', '-DLCB_WEBGPU_MOE_DIRECT_SLOT=OFF'):
                raise ValueError('Unknown MoE direct-slot activation in compiled provenance')
            if moe_options[0].endswith('=ON'):
                webgpu = [option for option in provenance['cmakeCommand']
                          if option.startswith('-DLCB_WEBGPU=')]
                if webgpu != ['-DLCB_WEBGPU=ON']:
                    raise ValueError('MoE direct-slot activation requires WebGPU')
                moe_enabled.append(profile + '/' + variant)
            if provenance['toolchain'] != toolchain:
                raise ValueError('Reporting toolchain differs from compiled provenance')
            options = [option for option in provenance['cmakeCommand']
                       if option.startswith('-DLCB_WEBGPU_BF16_PROJECTOR=')]
            if len(options) != 1 or options[0] not in ('-DLCB_WEBGPU_BF16_PROJECTOR=ON', '-DLCB_WEBGPU_BF16_PROJECTOR=OFF'):
                raise ValueError('Unknown BF16 overlay activation in compiled provenance')
            if options[0].endswith('=ON'):
                enabled.append(profile + '/' + variant)
    with tempfile.TemporaryDirectory(prefix='lcb-report-overlay-') as temporary:
        patched = prepare(vendor, Path(temporary) / 'overlay', root / patch_path)
        patched_identity = file_identity(patched)
        audio = prepare(vendor, Path(temporary) / 'audio', root / audio_patch, filename='mtmd-audio.cpp')
        audio_identity = file_identity(audio)
        webgpu_compiled = None
        loader_compiled = None
        if copy_enabled:
            prepared = webgpu_source.prepare(vendor, Path(temporary) / 'webgpu',
                                             patch_root=root / PATCH_DIRECTORY)
            webgpu_compiled = {
                'logicalUpstreamPath': webgpu_source.SOURCE_PATH, **file_identity(prepared),
                'tensorCopy': True, 'parameterUploadBatching': True,
                'compileDefinitions': ['GGML_WEBGPU_BATCH_PARAM_UPLOADS'],
            }
            loader_compiled = {
                'logicalUpstreamPath': webgpu_source.LOADER_PATH,
                **file_identity(prepared.parent / 'llama-model-loader.cpp'),
            }
        webgpu_copies = {variant: webgpu_compiled for variant in copy_enabled}
        moe_compiled = None
        if moe_enabled:
            outputs = moe.prepare(vendor, Path(temporary) / 'moe', root / PATCH_DIRECTORY / moe.PATCH_NAME)
            moe_compiled = {key: file_identity(path) for key, path in outputs.items()}
    supporting = ['scripts/prepare_mtmd.py', 'cmake/MtmdOverlay.cmake',
                  'bridge/mtmd-bf16.h', 'docs/webgpu-bf16-projector.md']
    return {
        'upstreamRepository': 'ggml-org/llama.cpp',
        'baseCommit': upstream,
        'vendorCheckoutModified': False,
        'inventoryScope': f'Known build-tree overlays plus every *.patch file under {PATCH_DIRECTORY}/; not an exhaustive compiler transformation inventory.',
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
            'id': 'experimental-webgpu-moe-direct-slot',
            'experimental': True,
            'kind': 'build-tree-shader-header-overlay',
            'upstreamSource': {'path': moe.SHADER_PATH, **file_identity(vendor / moe.SHADER_PATH)},
            'patch': {'path': f'{PATCH_DIRECTORY}/{moe.PATCH_NAME}',
                      **file_identity(root / PATCH_DIRECTORY / moe.PATCH_NAME)},
            'compiledCopy': moe_compiled,
            'reviewedCommit': upstream if upstream in moe.REVIEWED_REVISIONS else None,
            'reviewedInputs': moe.REVIEWED_REVISIONS.get(upstream),
            'application': {
                'preparationScript': 'scripts/prepare_moe_direct_slot.py',
                'cmakeHook': 'cmake/MoeDirectSlotOverlay.cmake',
                'option': 'LCB_WEBGPU_MOE_DIRECT_SLOT',
                'enabledProfileVariants': sorted(moe_enabled),
            },
            'supportingFiles': {path: file_identity(root / path) for path in
                                ['scripts/prepare_moe_direct_slot.py', 'cmake/MoeDirectSlotOverlay.cmake',
                                 'docs/moe-direct-slot.md']},
            'behavior': 'Single-token MoE vector dispatch indexes selected slots directly in all WebGPU profiles; CPU, routing, multi-token kernels and public interfaces are unchanged.',
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
        }],
        'otherPatchFiles': {path: {'application': 'not classified by this report', **file_identity(root / path)}
                            for path in patch_files if path not in (patch_path, audio_patch, f'{PATCH_DIRECTORY}/{moe.PATCH_NAME}', f'{PATCH_DIRECTORY}/{tensor_copy.PATCH_NAME}', f'{PATCH_DIRECTORY}/{webgpu_source.PARAM_PATCH_NAME}', f'{PATCH_DIRECTORY}/{webgpu_source.LOADER_PATCH_NAME}')},
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
