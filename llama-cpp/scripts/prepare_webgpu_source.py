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
import prepare_sampler_single_sync as sampler_single_sync
import prepare_webgpu_scale_unary as scale_unary
import prepare_webgpu_q8_subgroup as q8_subgroup
import prepare_webgpu_rms_scale as rms_scale
import prepare_webgpu_async_tensor_copy as async_tensor_copy
import prepare_webgpu_iq3s_subgroup as iq3s_subgroup
import prepare_webgpu_host_input_snapshot as host_input_snapshot
import prepare_webgpu_ssm_conv_silu as ssm_conv_silu
import prepare_webgpu_moe_group_skip as moe_group_skip
import prepare_webgpu_router_tail as router_tail
import prepare_webgpu_q8_diagnostic as q8_diagnostic
import prepare_webgpu_host_encode_assembly as host_encode_assembly
import prepare_webgpu_profile_query_budget as profile_query_budget
import prepare_webgpu_f32_small_m as f32_small_m
import prepare_webgpu_moe_gather_logical32 as moe_gather_logical32
import prepare_webgpu_iq3s_field_loads as iq3s_field_loads
import prepare_webgpu_matmul_vec4_layout as matmul_vec4_layout
import prepare_webgpu_moe_tight_tile_bound as moe_tight_tile_bound

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = tensor_copy.SOURCE_PATH
PATCH_DIRECTORY = tensor_copy.PATCH_DIRECTORY
PARAM_PATCH_NAME = 'ggml-webgpu-batch-param-uploads.patch'
PARAM_PATCH_SHA256 = 'f1fee64844addb636621412ef7227d6d5fd3ac4210cf7e001689550a321a8a26'
OUTPUT_SHA256 = 'd5228a8a36613fa8ae7341bb96eee04f318aed5ccd73aa85e57164d118a70a2b'
SSM_HEADER_PATH = 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'
SSM_SHADER_PATH = 'ggml/src/ggml-webgpu/wgsl-shaders/ssm_conv.wgsl'
SSM_SHADER_SHA256 = '868a7284cb858811b5ac70f466550340dba0df356b3b9ac1e7568053b7b9c809'
SSM_PATCH_NAME = 'ggml-webgpu-ssm-conv-single-token.patch'
SSM_PATCH_SHA256 = '632d0fb83a903b5005ceb061dffbb06a59a4d5d5d18e1eb6d947511981937ceb'
# Historical name retained for consumers of the final shared shader library.
SSM_HEADER_OUTPUT_SHA256 = 'a302e654b7afd4ed1ba39a2031596c1d8cb1ca3ba19533be4f5b40f28e1281e2'
# OUTPUT_SHA256/SSM_HEADER_OUTPUT_SHA256 above pin the historical normal prefix.
# Later stages keep their own contracts; these identities describe compiled files.
FINAL_OUTPUTS = {profile: {**matmul_vec4_layout.OUTPUTS[profile], **moe_tight_tile_bound.OUTPUTS[profile]}
                 for profile in (False, True)}

# Why: bound read staging and queued model-load work across tensors.
# Why not wait after every write: amortize synchronization over a charged batch;
# browser transport thresholds are version-specific, not this loader's contract.
LOADER_PATH = 'src/llama-model-loader.cpp'
LOADER_PATCH_NAME = 'llama-model-loader-webgpu-chunked-upload.patch'
LOADER_PATCH_SHA256 = 'ddb01246fb2a7901359c0bc576023ce0a0f2c5898ee46623f71c86c9eb6f7f3f'
LOADER_OUTPUT_SHA256 = '96aefafbf214482f3116af5ee4de0c8a4e53d6da999077cf8492526f0c52837e'
LOADER_INPUTS = {
    "src/llama-model-loader.cpp": "fc7ab08ffda8f94ff2a1ae6ddff034b9a90d2eb1d532309bd14a388e1bfa83e3",
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


def replace_iq3s_field_embedding(work: Path, source: Path, patch_root: Path, env: dict[str, str], headers: dict[str, Path]) -> None:
    # This is mandatory in production; prefix-only tests exclude the whole stage.
    # Missing original-stage output must fail closed rather than skip replacement.
    headers[iq3s_field_loads.HEADER_NAME] = iq3s_field_loads.prepare(
        work, source, patch_root, env, previous_header=headers[iq3s_field_loads.HEADER_NAME])


def prepare(source: Path, output: Path, *, patch_root: Path = ROOT / PATCH_DIRECTORY, gpu_profile: bool = False) -> Path:
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
        hc_headers.update(scale_unary.prepare(work, source, patch_root, env))
        hc_headers.update(q8_subgroup.prepare(work, source, patch_root, env))
        hc_headers.update(rms_scale.prepare(work, source, patch_root, env))
        async_tensor_copy.prepare(work, source, patch_root, env)
        hc_headers.update(iq3s_subgroup.prepare(work, source, patch_root, env))
        hc_headers.update(host_input_snapshot.prepare(work, source, patch_root, env))
        hc_headers.update(ssm_conv_silu.prepare(work, source, patch_root, env))
        hc_headers.update(moe_group_skip.prepare(work, source, patch_root, env))
        hc_headers.update(router_tail.prepare(work, source, patch_root, env))
        host_encode_assembly.prepare(work, source, patch_root, env)
        f32_small_m.prepare(work, source, patch_root, env)
        hc_headers.update(moe_gather_logical32.prepare(work, source, patch_root, env))
        # Preserve the original IQ3 stage contract, then replace only its final
        # dedicated embedding. Every selector, cache key and host source stays.
        replace_iq3s_field_embedding(work, source, patch_root, env, hc_headers)
        sampler = sampler_single_sync.prepare(work, source, patch_root, env)
        if tensor_copy.digest(target) != OUTPUT_SHA256:
            raise ValueError('Unexpected combined WebGPU source identity')
        if tensor_copy.digest(loader) != LOADER_OUTPUT_SHA256:
            raise ValueError('Unexpected chunked loader source identity')
        if tensor_copy.digest(header) != SSM_HEADER_OUTPUT_SHA256:
            raise ValueError('Unexpected combined shader-library header identity')
        if gpu_profile:
            profile_query_budget.prepare(work, source, patch_root, env)
            hc_headers.update(q8_diagnostic.prepare(work, source, patch_root, env))
        matmul_vec4_layout.prepare(work, source, patch_root, env, gpu_profile=gpu_profile)
        moe_tight_tile_bound.prepare(work, source, patch_root, env, gpu_profile=gpu_profile)
        if not gpu_profile:
            for name in ('ggml-webgpu-q8-diagnostic.h', 'ggml-webgpu-q8-diagnostic-sha256.hpp'):
                (output / name).unlink(missing_ok=True)
        # Do not emit an embedded WGSL header here: the sibling shader library
        # must continue to use upstream's generated shaders. Narrow matrix-only
        # specializations use dedicated embedded headers, never shadowing it.
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
        sampler_result = output / sampler.name
        if not sampler_result.exists() or sampler_result.read_bytes() != sampler.read_bytes():
            sampler.replace(sampler_result)
        result = output / target.name
        if not result.exists() or result.read_bytes() != target.read_bytes():
            target.replace(result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dawn-package', type=Path, required=True)
    parser.add_argument('--gpu-profile', action='store_true')
    args = parser.parse_args()
    tensor_copy.verify_dawn(args.dawn_package)
    prepare(args.source, args.output, gpu_profile=args.gpu_profile)


if __name__ == '__main__':
    main()
