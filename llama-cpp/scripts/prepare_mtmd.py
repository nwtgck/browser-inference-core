#!/usr/bin/env python3
"""Prepare one coherent mtmd source/header tree without editing the checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
PATCH_DIRECTORY = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval'
REVIEWED_COMMIT = 'b1b770d90d94cad9e9ea5b993bdbc383a522fb28'
MANIFEST_NAME = 'mtmd-source-manifest.json'
CONTRACT_INPUTS = {'tools/mtmd/mtmd.cpp': '638227372e11b21f09fc933b6cad444ad55ab663939acb2752df0a1cea3698c2',
 'tools/mtmd/mtmd-model.cpp': '74cd1f3fed4de8282a305b4472c090ceb52c6a494f19afa3100b444c2850e88f',
 'tools/mtmd/mtmd-model.h': 'bdfb7b9e6a1d30edcec2b392c99e2b3d722ab549e82f7b2387b5d352025cf040',
 'tools/mtmd/mtmd-subcontext.cpp': '0f6d70980987378007299043f733625b4127a409c7e78c9e5335738774ab8fa6',
 'tools/mtmd/mtmd-subcontext.h': '645e39a6bc835e0439c8b523f079eae2767fcfa44ecd86366be321f44a3cafbd',
 'tools/mtmd/mtmd-graph.cpp': '18913199007af4f6e05f6044159078b9dcffc72a3c5cbe08bed9fb82c8fa4f45',
 'tools/mtmd/mtmd-graph.h': '5c32d78eb6f2c251377851869b178c47a809f23322fd9991d8e7a3a4183f156b',
 'tools/mtmd/mtmd-impl.h': '720d306aa4d664663454709e3832e105dad545156fefb993603c20b9fe260e2e',
 'tools/mtmd/CMakeLists.txt': '90b232eb29cc15bcc9d75eb3a3de7e03ece945192aa944a82a20ce1024eda93e',
 'ggml/include/ggml.h': 'bdb0e75be8eeb1fc14a742015025505f5810d8200cbb40bda7e981101413440b',
 'ggml/src/ggml-cpu/ggml-cpu.c': 'e6f33c7ecbe5eeb6cce487b6aad7ce4932800894d3e33269f0157feaaf74e769',
 'tools/mtmd/mtmd-audio.cpp': '93817f37593c2659f5e7b53c6bf4c37554d8a190778026393d27b6213921eabc',
 'tools/mtmd/mtmd-audio.h': '19a765db3826d49d617d097fe1494a95e3e1a8c17c08621ebee5b1835699e048',
 'tools/mtmd/mtmd-helper.cpp': '429cac1f55896c4e3cce8e7ba5f218b0e709c15fb94c610efbc9527071e7cfa4',
 'tools/mtmd/mtmd-helper.h': '6a4a684577311451ea9644cf51991cfcc6bd463f7ecd8d5f067c27f568eef888',
 'tools/mtmd/models/models.h': '77e9e3b751e6bdbe44e6000bfe8cca47ab472b9feafa9295a7e878de93a1f2f5',
 'ggml/include/ggml-cpp.h': 'bc15ba002c68685df23faae42df0d2b3d58032011f88b649021ab92485862c72'}
STAGES = {'mtmd-vision-bf16-runtime.patch': {'sha256': '2ddbfd02e1eee24d296b086adfe280c10b693ecd248028207e62292e393f26d3',
                                    'outputs': {'mtmd-model.h': '71d05f4c3bae5586181f0db8b83d1e834c68533b0ae0f5549776a284e7f8a8a7',
                                                'mtmd-model.cpp': '7efa86f383310c2da852ab60776cb1ed8a25ea9dc5a4554f07c1d59eee01aa9e',
                                                'mtmd-subcontext.cpp': '634dc7a55a5da1982dd1d16dce43d0b32217ca2020d631179070d57838c1953e',
                                                'mtmd.cpp': '02a7207d2b19c8491bd1859753cb873d006a481a662a57edabbb646843b2024f'}},
 'mtmd-webgpu-bf16.patch': {'sha256': '7e1c4c5f2039e1c125e46eb1124eb3e183f99d1c25178da73ed7dd5d52927a49',
                            'outputs': {'mtmd-model.cpp': 'dea6a51599e2595b28aa628484bc0da1d32f40db9a6e1b5c34191a7207645c5a',
                                        'mtmd-subcontext.cpp': 'edc234d0f8528fc619ed6d52f8750c57511d157f6e77f51ca633c1cc15daeaff'}},
 'mtmd-non-pthread-threadpool.patch': {'sha256': 'ea12ad40d7b095aa95a967a57a28c7b2e323478eb5f374a886815867ae23e6c1',
                                       'outputs': {'mtmd-subcontext.cpp': '483bf7222c78c2c2a150bd5b4c2ef2109c83bfd09fa813ad259273cebe2971bc'}}}
OUTPUTS = {'cpu': {'mtmd-model.h': '71d05f4c3bae5586181f0db8b83d1e834c68533b0ae0f5549776a284e7f8a8a7',
         'mtmd-model.cpp': '7efa86f383310c2da852ab60776cb1ed8a25ea9dc5a4554f07c1d59eee01aa9e',
         'mtmd-subcontext.cpp': 'a31a9f3b01713879b6ea1501304cebbdefcc337caf8d0f6fd767859d91cfd264',
         'mtmd.cpp': '02a7207d2b19c8491bd1859753cb873d006a481a662a57edabbb646843b2024f'},
 'webgpu': {'mtmd-model.h': '71d05f4c3bae5586181f0db8b83d1e834c68533b0ae0f5549776a284e7f8a8a7',
            'mtmd-model.cpp': 'dea6a51599e2595b28aa628484bc0da1d32f40db9a6e1b5c34191a7207645c5a',
            'mtmd-subcontext.cpp': '483bf7222c78c2c2a150bd5b4c2ef2109c83bfd09fa813ad259273cebe2971bc',
            'mtmd.cpp': '02a7207d2b19c8491bd1859753cb873d006a481a662a57edabbb646843b2024f'}}


def identity(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError('Missing or linked mtmd input: ' + str(path))
    raw = path.read_bytes()
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def inventory(directory: Path) -> dict:
    result = {}
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():
            raise ValueError('Linked mtmd source is not supported: ' + str(path))
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = identity(path)
    return result


def prepare(source: Path, output: Path, *, mode: str = 'webgpu',
            patch_root: Path = ROOT / PATCH_DIRECTORY) -> Path:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('The mtmd overlay must be outside the upstream source tree')
    if mode not in OUTPUTS:
        raise ValueError('Unknown mtmd composition mode')
    for relative, expected in CONTRACT_INPUTS.items():
        if identity(source / relative)['sha256'] != expected:
            raise ValueError('mtmd source needs semantic review: ' + relative)
    original = source / 'tools/mtmd'
    inputs = inventory(original)
    if 'clip.h' in inputs or 'clip.cpp' in inputs:
        raise ValueError('Obsolete clip source found in mtmd input')
    chosen = ['mtmd-vision-bf16-runtime.patch']
    if mode == 'webgpu':
        chosen.append('mtmd-webgpu-bf16.patch')
    chosen.append('mtmd-non-pthread-threadpool.patch')
    patches = {name: identity(patch_root / name) for name in chosen}
    for name, value in patches.items():
        if value['sha256'] != STAGES[name]['sha256']:
            raise ValueError('mtmd patch identity changed: ' + name)
    if output.exists():
        allowed = set(inputs) | {MANIFEST_NAME}
        for relative in inventory(output):
            if relative not in allowed:
                raise ValueError('Unexpected stale mtmd overlay file: ' + relative)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='mtmd-source-', dir=output.parent) as temporary:
        work = Path(temporary) / 'tree'
        shutil.copytree(original, work)
        env = os.environ.copy()
        for key in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE'):
            env.pop(key, None)
        env['GIT_CEILING_DIRECTORIES'] = str(work.parent)
        stage_receipts = []
        for name in chosen:
            for flags in (['--check'], []):
                subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags,
                                str((patch_root / name).resolve())], cwd=work, env=env,
                               check=True, capture_output=True, text=True)
            expected_outputs = STAGES[name]['outputs']
            if name == 'mtmd-non-pthread-threadpool.patch':
                expected_outputs = {'mtmd-subcontext.cpp': OUTPUTS[mode]['mtmd-subcontext.cpp']}
            actual = {relative: identity(work / relative) for relative in expected_outputs}
            if any(actual[relative]['sha256'] != digest for relative, digest in expected_outputs.items()):
                raise ValueError('Unexpected mtmd stage output: ' + name)
            stage_receipts.append({'patch': name, **patches[name], 'outputs': actual})
        for relative, digest in OUTPUTS[mode].items():
            if identity(work / relative)['sha256'] != digest:
                raise ValueError('Unexpected final mtmd source: ' + relative)
        prepared = inventory(work)
        if set(prepared) != set(inputs):
            raise ValueError('mtmd overlay source inventory changed')
        for relative in inputs:
            if relative.startswith('models/') and prepared[relative] != inputs[relative]:
                raise ValueError('Model builder bytes changed in mtmd overlay: ' + relative)
        cmake = (work / 'CMakeLists.txt').read_text()
        core_match = re.search(r'set\(MTMD_CORE_SOURCES\s+(.*?)\)', cmake, re.DOTALL)
        if not core_match:
            raise ValueError('mtmd core source inventory changed')
        core_sources = re.sub(r'#[^\n]*', '', core_match.group(1)).split()
        if any(relative not in prepared for relative in core_sources):
            raise ValueError('mtmd core source is outside the prepared tree')
        manifest = {'formatVersion': 1, 'reviewedCommit': REVIEWED_COMMIT, 'mode': mode,
                    'inputRoot': 'tools/mtmd', 'contractInputs': CONTRACT_INPUTS,
                    'sourceInputs': inputs, 'preparedFiles': prepared,
                    'coreSources': core_sources, 'stages': stage_receipts,
                    'modifiedFiles': [name for name in inputs if inputs[name] != prepared[name]]}
        (work / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + '\n')
        # Validate every stage before publishing. Existing successful files are
        # retained on failure and unchanged files keep their timestamps.
        output.mkdir(parents=True, exist_ok=True)
        for relative in [*prepared, MANIFEST_NAME]:
            destination, generated = output / relative, work / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists() or destination.read_bytes() != generated.read_bytes():
                generated.replace(destination)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('cpu', 'webgpu'), default='webgpu')
    args = parser.parse_args()
    prepare(args.source, args.output, mode=args.mode)


if __name__ == '__main__':
    main()
