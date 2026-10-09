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

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = tensor_copy.SOURCE_PATH
PATCH_DIRECTORY = tensor_copy.PATCH_DIRECTORY
PARAM_PATCH_NAME = 'ggml-webgpu-batch-param-uploads.patch'
PARAM_PATCH_SHA256 = 'f1fee64844addb636621412ef7227d6d5fd3ac4210cf7e001689550a321a8a26'
OUTPUT_SHA256 = 'd1fd96c0113ac2ffd60c6175fbe9ab8115a0a58d5c17bfd209a5a64f288a1f56'
SSM_HEADER_PATH = 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'
SSM_SHADER_PATH = 'ggml/src/ggml-webgpu/wgsl-shaders/ssm_conv.wgsl'
SSM_SHADER_SHA256 = '868a7284cb858811b5ac70f466550340dba0df356b3b9ac1e7568053b7b9c809'
SSM_PATCH_NAME = 'ggml-webgpu-ssm-conv-single-token.patch'
SSM_PATCH_SHA256 = '632d0fb83a903b5005ceb061dffbb06a59a4d5d5d18e1eb6d947511981937ceb'
SSM_HEADER_OUTPUT_SHA256 = '82ae3f5fd15ae83378a494312f6272b6b4128dbab205381b91ad09ce95ff4671'

# Why: bound read staging and queued model-load work across tensors.
# Why not wait after every write: amortize synchronization over a charged batch;
# browser transport thresholds are version-specific, not this loader's contract.
LOADER_PATH = 'src/llama-model-loader.cpp'
LOADER_PATCH_NAME = 'llama-model-loader-webgpu-chunked-upload.patch'
LOADER_PATCH_SHA256 = 'c89c88a047ffca6875cc924df04fc92e73caa2338cb8644dbdf2ce1582c181c8'
LOADER_OUTPUT_SHA256 = '775f94b93ef591f4313600ecfd23b2320718d6a9f8a626e5699dc800401eda65'
LOADER_INPUTS = {
    'src/llama-model-loader.cpp': '301f8250eef5b34d14617c7c9bf5c14bb60519d199751f4d829502a5107f0593',
    'src/llama-mmap.cpp': 'ed859abec87282d5dd13a08b03b93b4ff3efb3f852f289d5347904b0c1cc017d',
    'src/llama-mmap.h': '0fff03ce62bc51b1649833f6017fbd84dc0aac7fce3b38484439153fc13ba8f5',
    'src/CMakeLists.txt': '6602fed62f13557db0888ccf2fc69b5c49a01f1eea56a9a0c866e8e60084d5d2',
    'src/llama-model-loader.h': '687dac85ea01c1fb0db19a4e53286144e29f351233c411e88932b99553538422',
    'ggml/src/ggml-quants.c': '5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab',
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
        if tensor_copy.digest(target) != OUTPUT_SHA256:
            raise ValueError('Unexpected combined WebGPU source identity')
        if tensor_copy.digest(loader) != LOADER_OUTPUT_SHA256:
            raise ValueError('Unexpected chunked loader source identity')
        if tensor_copy.digest(header) != SSM_HEADER_OUTPUT_SHA256:
            raise ValueError('Unexpected SSM convolution header identity')
        # Do not emit an embedded WGSL header here: preserve the MoE BEFORE
        # include (or upstream generated header) selected by this sibling.
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
