#!/usr/bin/env python3
"""Compose required WebGPU changes into source-bound translation units."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import prepare_webgpu_tensor_copy as tensor_copy
import prepare_webgpu_hc as hc
import prepare_webgpu_q8_short as q8_short
import prepare_webgpu_moe_sum as moe_sum
import prepare_webgpu_expert_short as expert_short
import prepare_webgpu_ordered_readback as ordered_readback
import prepare_webgpu_bind_group_cache as bind_group_cache
import prepare_webgpu_q4_short as q4_short

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = tensor_copy.SOURCE_PATH
PATCH_DIRECTORY = tensor_copy.PATCH_DIRECTORY
PARAM_PATCH_NAME = 'ggml-webgpu-batch-param-uploads.patch'
PARAM_PATCH_SHA256 = 'f1fee64844addb636621412ef7227d6d5fd3ac4210cf7e001689550a321a8a26'
OUTPUT_SHA256 = 'b5ac93bbf834718ef60002d71b2a60d52dc605a3b64d5c90305599cc07f7d19e'
SSM_HEADER_PATH = 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'
SSM_SHADER_PATH = 'ggml/src/ggml-webgpu/wgsl-shaders/ssm_conv.wgsl'
SSM_SHADER_SHA256 = '868a7284cb858811b5ac70f466550340dba0df356b3b9ac1e7568053b7b9c809'
SSM_PATCH_NAME = 'ggml-webgpu-ssm-conv-single-token.patch'
SSM_PATCH_SHA256 = '632d0fb83a903b5005ceb061dffbb06a59a4d5d5d18e1eb6d947511981937ceb'
SSM_HEADER_OUTPUT_SHA256 = 'd0045d4cc1f4fc88a8ed6a24edbe64c6148f5d3eb2e70b7fcabb9346e17900a8'

# Why: bound read staging and queued model-load work across tensors.
# Why not wait after every write: amortize synchronization over a charged batch;
# browser transport thresholds are version-specific, not this loader's contract.
LOADER_PATH = 'src/llama-model-loader.cpp'
LOADER_PATCH_NAME = 'llama-model-loader-webgpu-chunked-upload.patch'
LOADER_PATCH_SHA256 = 'ddb01246fb2a7901359c0bc576023ce0a0f2c5898ee46623f71c86c9eb6f7f3f'
LOADER_OUTPUT_SHA256 = '6f3919c473df574163644661cbcc9741435397f1e0ea998518d93a49da0b0907'
LOADER_INPUTS = {
    "src/llama-model-loader.cpp": "7975d83406627e405d36876472495c6111696e70db66c65a629d6213ec54fc04",
    "src/llama-mmap.cpp": "130601ff8c2b6e16668b70a1dca07cc09ae84328d91bbd5bd259aea5658103e7",
    "src/llama-mmap.h": "0fff03ce62bc51b1649833f6017fbd84dc0aac7fce3b38484439153fc13ba8f5",
    "src/CMakeLists.txt": "802f9f3eced47f2396f2e8b7534050361e72882cffe2619fed6653d11b1f474d",
    "src/llama-model-loader.h": "4eb5318f7b70333015386d612738f507153770e4e5635895b2c1d8355ff22103",
    "ggml/src/ggml-quants.c": "5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab"
}


def patch_environment(work: Path) -> dict[str, str]:
    # --no-index still discovers enclosing repositories and can silently skip
    # Git-format patch paths when CI's build directory is inside the checkout.
    env = os.environ.copy()
    for key in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_COMMON_DIR', 'GIT_INDEX_FILE'):
        env.pop(key, None)
    env['GIT_CEILING_DIRECTORIES'] = str(work.parent)
    return env


def prepare(source: Path, output: Path, *, patch_root: Path = ROOT / PATCH_DIRECTORY) -> Path:
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('WebGPU overlay must be outside the upstream source tree')
    tensor_copy.verify_reviewed_source(source)
    # The existing reviewed inputs pin the host dispatch and shader-library header.
    if tensor_copy.digest(source / SSM_SHADER_PATH) != SSM_SHADER_SHA256:
        raise ValueError('SSM convolution needs semantic review: upstream shader changed')
    for relative, expected in LOADER_INPUTS.items():
        if tensor_copy.digest(source / relative) != expected:
            raise ValueError('Chunked loader needs semantic review: upstream input changed: ' + relative)
    selected = [(patch_root / tensor_copy.PATCH_NAME, tensor_copy.PATCH_SHA256),
                (patch_root / PARAM_PATCH_NAME, PARAM_PATCH_SHA256),
                (patch_root / LOADER_PATCH_NAME, LOADER_PATCH_SHA256),
                (patch_root / SSM_PATCH_NAME, SSM_PATCH_SHA256)]
    # HC shares this profile; it does not create another variant.
    if tensor_copy.digest(patch_root / hc.PATCH_NAME) != hc.PATCH_SHA256:
        raise ValueError('WebGPU HC patch identity changed')
    if tensor_copy.digest(patch_root / q8_short.PATCH_NAME) != q8_short.PATCH_SHA256:
        raise ValueError('WebGPU Q8 short matvec patch identity changed')
    for patch, expected in selected:
        if tensor_copy.digest(patch) != expected:
            raise ValueError('WebGPU patch identity changed: ' + patch.name)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='webgpu-source-', dir=output) as temporary:
        work = Path(temporary)
        target = work / SOURCE_PATH
        target.parent.mkdir(parents=True)
        shutil.copyfile(source / SOURCE_PATH, target)
        loader = work / LOADER_PATH
        loader.parent.mkdir(parents=True)
        shutil.copyfile(source / LOADER_PATH, loader)
        header = work / SSM_HEADER_PATH
        shutil.copyfile(source / SSM_HEADER_PATH, header)
        env = patch_environment(work)
        for patch, _ in selected:
            for flags in (['--check'], []):
                subprocess.run(['git', 'apply', '--no-index', '--whitespace=error', *flags, str(patch.resolve())],
                               cwd=work, env=env, check=True, capture_output=True)
        if (output / 'ggml-wgsl-shaders.hpp').exists():
            raise ValueError('Unexpected canonical WGSL header would shadow upstream shaders')
        hc_headers = hc.prepare(work, source, patch_root, env)
        q8_short.prepare(work, source, patch_root, env)
        hc_headers.update(moe_sum.prepare(work, source, patch_root, env))
        expert_short.prepare(work, source, patch_root, env)
        ordered_readback.prepare(work, source, patch_root, env)
        bind_group_cache.prepare(work, source, patch_root, env)
        q4_short.prepare(work, source, patch_root, env)
        if tensor_copy.digest(target) != OUTPUT_SHA256:
            raise ValueError('Unexpected combined WebGPU source identity')
        if tensor_copy.digest(loader) != LOADER_OUTPUT_SHA256:
            raise ValueError('Unexpected chunked loader source identity')
        if tensor_copy.digest(header) != SSM_HEADER_OUTPUT_SHA256:
            raise ValueError('Unexpected SSM convolution header identity')
        # Do not emit an embedded WGSL header here: the sibling shader library
        # must continue to use upstream's generated shaders, including MoE.
        for name, generated in hc_headers.items():
            destination = output / name
            if not destination.exists() or destination.read_bytes() != generated.read_bytes():
                generated.replace(destination)
        header_result = output / header.name
        if not header_result.exists() or header_result.read_bytes() != header.read_bytes():
            header.replace(header_result)
        loader_result = output / loader.name
        if not loader_result.exists() or loader_result.read_bytes() != loader.read_bytes():
            loader.replace(loader_result)
        result = output / target.name
        if not result.exists() or result.read_bytes() != target.read_bytes():
            target.replace(result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dawn-package', type=Path, required=True)
    args = parser.parse_args()
    tensor_copy.verify_dawn(args.dawn_package)
    prepare(args.source, args.output)


if __name__ == '__main__':
    main()
