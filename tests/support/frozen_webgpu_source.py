"""Frozen pre-plan source composer for historical prefix/reference fixtures.

Not a production entrypoint. Final native/semantic tests use the production
composer; this builder exists only where an older prefix is itself the fixture.
Its byte guards protect those reference fixtures, not upstream compatibility.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from build_tools.project import runtime_root
import shutil
import subprocess
import tempfile
from build_tools.output import output_directories, preserve_unchanged
from build_tools.source.files import absolute
from build_tools.source.patches import patch_environment
from tests.support.llama_cpp.frozen_webgpu_stages import tensor_copy
from tests.support.llama_cpp.frozen_webgpu_stages import hc
from tests.support.llama_cpp.frozen_webgpu_stages import q8_short
from tests.support.llama_cpp.frozen_webgpu_stages import moe_sum
from tests.support.llama_cpp.frozen_webgpu_stages import expert_short
from tests.support.llama_cpp.frozen_webgpu_stages import ordered_readback
from tests.support.llama_cpp.frozen_webgpu_stages import bind_group_cache
from tests.support.llama_cpp.frozen_webgpu_stages import q4_short
from tests.support.llama_cpp.frozen_webgpu_stages import sampler_single_sync
from tests.support.llama_cpp.frozen_webgpu_stages import scale_unary
from tests.support.llama_cpp.frozen_webgpu_stages import q8_subgroup
from tests.support.llama_cpp.frozen_webgpu_stages import rms_scale
from tests.support.llama_cpp.frozen_webgpu_stages import async_tensor_copy
from tests.support.llama_cpp.frozen_webgpu_stages import iq3s_subgroup
from tests.support.llama_cpp.frozen_webgpu_stages import host_input_snapshot
from tests.support.llama_cpp.frozen_webgpu_stages import ssm_conv_silu
from tests.support.llama_cpp.frozen_webgpu_stages import moe_group_skip
from tests.support.llama_cpp.frozen_webgpu_stages import router_tail
from tests.support.llama_cpp.frozen_webgpu_stages import q8_diagnostic
from tests.support.llama_cpp.frozen_webgpu_stages import host_encode_assembly
from tests.support.llama_cpp.frozen_webgpu_stages import profile_query_budget
from tests.support.llama_cpp.frozen_webgpu_stages import f32_small_m
from tests.support.llama_cpp.frozen_webgpu_stages import moe_gather_logical32
from tests.support.llama_cpp.frozen_webgpu_stages import iq3s_field_loads
from tests.support.llama_cpp.frozen_webgpu_stages import matmul_vec4_layout
from tests.support.llama_cpp.frozen_webgpu_stages import moe_tight_tile_bound
from tests.support.llama_cpp.frozen_webgpu_stages import iq3s_row_split
from tests.support.llama_cpp.frozen_webgpu_stages import f32_matvec_epilogue
from tests.support.llama_cpp.frozen_webgpu_stages import q6_short
from tests.support.llama_cpp.frozen_webgpu_stages import quant_half_pairs
from tests.support.llama_cpp.frozen_webgpu_stages import direct_mapped_read
from tests.support.llama_cpp.frozen_webgpu_stages import quant_wide_m
from tests.support.llama_cpp.frozen_webgpu_stages import checkpoint_readback
from tests.support.llama_cpp.frozen_webgpu_stages import subgroup_size_control
from tests.support.llama_cpp.frozen_webgpu_stages import q4_subgroup
from tests.support.llama_cpp.frozen_webgpu_stages import q6_subgroup
from tests.support.llama_cpp.frozen_webgpu_stages import aligned_word_load
ROOT = runtime_root('llama-cpp')
SOURCE_PATH = tensor_copy.SOURCE_PATH
PATCH_DIRECTORY = tensor_copy.PATCH_DIRECTORY
PARAM_PATCH_NAME = 'ggml-webgpu-batch-param-uploads.patch'
PARAM_PATCH_SHA256 = 'f1fee64844addb636621412ef7227d6d5fd3ac4210cf7e001689550a321a8a26'
OUTPUT_SHA256 = '67940a20bba8d06205ad2f6cf1e909450c4fbb6805b5eba59f574c3576d69779'
SSM_HEADER_PATH = 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'
SSM_SHADER_PATH = 'ggml/src/ggml-webgpu/wgsl-shaders/ssm_conv.wgsl'
SSM_SHADER_SHA256 = '868a7284cb858811b5ac70f466550340dba0df356b3b9ac1e7568053b7b9c809'
SSM_PATCH_NAME = 'ggml-webgpu-ssm-conv-single-token.patch'
SSM_PATCH_SHA256 = '632d0fb83a903b5005ceb061dffbb06a59a4d5d5d18e1eb6d947511981937ceb'
# Historical name retained for consumers of the final shared shader library.
SSM_HEADER_OUTPUT_SHA256 = '7756a6ef0fb1521dd896fbb1a34b7431dac71ef52db93003c0db70527dff3536'
# OUTPUT_SHA256/SSM_HEADER_OUTPUT_SHA256 above pin the historical normal prefix.
# Later stages keep their own contracts; these identities describe compiled files.
FINAL_OUTPUTS = {profile: {name: q6_subgroup.OUTPUTS[profile][name]
                           for name in ('ggml-webgpu.cpp', 'ggml-webgpu-shader-lib.hpp')}
                 for profile in (False, True)}

# Why: bound read staging and queued model-load work across tensors.
# Why not wait after every write: amortize synchronization over a charged batch;
# browser transport thresholds are version-specific, not this loader's contract.
LOADER_PATH = 'src/llama-model-loader.cpp'
LOADER_PATCH_NAME = 'llama-model-loader-webgpu-chunked-upload.patch'
LOADER_PATCH_SHA256 = '6b1edd9a6a14e1b70d8cc79e548484b75db47d20c68d3a87a73c5a2be757eb2b'
LOADER_OUTPUT_SHA256 = '6bbfef6f691c432101e4eb1df6cea489c489c89940767c39b191ab955c8053af'
LOADER_INPUTS = {
    "src/llama-model-loader.cpp": "b856b513340e9e86ab81e2d8d110fedd07b60f508e0e4f8513690150f8efd7e1",
    "src/llama-mmap.cpp": "130601ff8c2b6e16668b70a1dca07cc09ae84328d91bbd5bd259aea5658103e7",
    "src/llama-mmap.h": "0fff03ce62bc51b1649833f6017fbd84dc0aac7fce3b38484439153fc13ba8f5",
    "src/CMakeLists.txt": "802f9f3eced47f2396f2e8b7534050361e72882cffe2619fed6653d11b1f474d",
    "src/llama-model-loader.h": "4eb5318f7b70333015386d612738f507153770e4e5635895b2c1d8355ff22103",
    "ggml/src/ggml-quants.c": "bcda33625707fdfcb1367cdd2bb1a2f017430162ec9623abc84ad17459698f73"
}


def replace_iq3s_field_embedding(work: Path, source: Path, patch_root: Path, env: dict[str, str], headers: dict[str, Path]) -> None:
    # This is mandatory in production; prefix-only tests exclude the whole stage.
    # Missing original-stage output must fail closed rather than skip replacement.
    headers[iq3s_field_loads.HEADER_NAME] = iq3s_field_loads.prepare(
        work, source, patch_root, env, previous_header=headers[iq3s_field_loads.HEADER_NAME])


def replace_iq3s_row_split_embedding(work: Path, source: Path, patch_root: Path, env: dict[str, str], headers: dict[str, Path], *, gpu_profile: bool) -> None:
    headers[iq3s_row_split.HEADER_NAME] = iq3s_row_split.prepare(
        work, source, patch_root, env, previous_header=headers[iq3s_row_split.HEADER_NAME], gpu_profile=gpu_profile)


def prepare(source: Path, output: Path, *, patch_root: Path | None = None, gpu_profile: bool = False) -> Path:
    source, output = source.resolve(), absolute(output)
    # Frozen references must never silently consume the live runtime's patches.
    patch_root = source / 'patches' if patch_root is None else patch_root
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('WebGPU overlay must be outside the upstream source tree')
    tensor_copy.verify_contract_inputs(source)
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
    with output_directories(output, replace=True, reuse_if=preserve_unchanged) as (prepared,), \
            tempfile.TemporaryDirectory(prefix='webgpu-source-', dir=prepared.parent) as temporary:
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
                               cwd=work, env=env, check=True)
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
        replace_iq3s_row_split_embedding(work, source, patch_root, env, hc_headers, gpu_profile=gpu_profile)
        hc_headers.update(f32_matvec_epilogue.prepare(work, source, patch_root, env, gpu_profile=gpu_profile))
        q6_short.prepare(work, source, patch_root, env, gpu_profile=gpu_profile)
        hc_headers.update(quant_half_pairs.prepare(work, source, patch_root, env, gpu_profile=gpu_profile))
        direct_mapped_read.prepare(work, source, patch_root, env, gpu_profile=gpu_profile)
        quant_wide_m.prepare(work, source, patch_root, env, gpu_profile=gpu_profile)
        q4_subgroup.prepare(work, source, patch_root, env, gpu_profile=gpu_profile)
        hc_headers.update(aligned_word_load.prepare(work, source, patch_root, env,
                                                   gpu_profile=gpu_profile, headers=hc_headers))
        hc_headers.update(checkpoint_readback.prepare(work, source, patch_root, env, gpu_profile=gpu_profile))
        hc_headers.update(subgroup_size_control.prepare(work, source, patch_root, env,
                                                       gpu_profile=gpu_profile, headers=hc_headers))
        q6_subgroup.prepare(work, source, patch_root, env, gpu_profile=gpu_profile)
        # Publish the complete selected profile only after every stage succeeds.
        for name, generated in hc_headers.items():
            generated.replace(prepared / name)
        for generated in (header, loader, sampler, target):
            generated.replace(prepared / generated.name)
    return output / target.name


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
