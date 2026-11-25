import type { Patch } from '#tools/source/patches';

export const patchDirectory = 'upstream-patches-only-as-a-last-resort-with-explicit-user-approval';

export type SourceMode = 'normal' | 'profile';
export type RecipePatch = Patch & { modes: readonly SourceMode[] };

// Order, accepted scope and purpose are owned together. No expected source digest
// is an acceptance condition; receipts observe the inputs actually consumed.
const webgpuRecipe: readonly RecipePatch[] = [
  {
    file: 'ggml-webgpu-same-device-tensor-copy.patch',
    target: 'llama-webgpu',
    reason: 'same device tensor copy',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-batch-param-uploads.patch',
    target: 'llama-webgpu',
    reason: 'batch param uploads',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'llama-model-loader-webgpu-chunked-upload.patch',
    target: 'llama-webgpu',
    reason: 'llama model loader webgpu chunked upload',
    files: [
      'src/llama-model-loader.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-ssm-conv-single-token.patch',
    target: 'llama-webgpu',
    reason: 'ssm conv single token',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-hc-pre-post.patch',
    target: 'llama-webgpu',
    reason: 'hc pre post',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-hc.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/hc_post.wgsl',
      'ggml/src/ggml-webgpu/wgsl-shaders/hc_pre.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-q8-short-workgroup.patch',
    target: 'llama-webgpu',
    reason: 'q8 short workgroup',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-moe-expert-sum.patch',
    target: 'llama-webgpu',
    reason: 'moe expert sum',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-moe-sum.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/moe_expert_sum.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-expert-short-workgroup.patch',
    target: 'llama-webgpu',
    reason: 'expert short workgroup',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-ordered-readback.patch',
    target: 'llama-webgpu',
    reason: 'ordered readback',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-bind-group-cache.patch',
    target: 'llama-webgpu',
    reason: 'bind group cache',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-q4-short-workgroup.patch',
    target: 'llama-webgpu',
    reason: 'q4 short workgroup',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-scale-unary.patch',
    target: 'llama-webgpu',
    reason: 'scale unary',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-scale-unary.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/scale_unary.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-q8-subgroup-matvec.patch',
    target: 'llama-webgpu',
    reason: 'q8 subgroup matvec',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/q8_subgroup_matvec.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-rms-scale.patch',
    target: 'llama-webgpu',
    reason: 'rms scale',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-rms-scale.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/row_norm.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-async-tensor-copy.patch',
    target: 'llama-webgpu',
    reason: 'async tensor copy',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-iq3s-id-subgroup.patch',
    target: 'llama-webgpu',
    reason: 'iq3s id subgroup',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/iq3s_id_subgroup.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-host-input-snapshot.patch',
    target: 'llama-webgpu',
    reason: 'host input snapshot',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-backend.cpp',
      'ggml/src/ggml-backend-host-snapshot.h',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-ssm-conv-silu.patch',
    target: 'llama-webgpu',
    reason: 'ssm conv silu',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-ssm-conv-silu.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/ssm_conv_silu.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-moe-group-skip.patch',
    target: 'llama-webgpu',
    reason: 'moe group skip',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_group.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-router-tail.patch',
    target: 'llama-webgpu',
    reason: 'router tail',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-router-tail.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/router_tail.wgsl',
      'ggml/src/ggml-webgpu/wgsl-shaders/router_tail_merged.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-router-tail-epsilon.patch',
    target: 'llama-webgpu',
    reason: 'router tail epsilon normalization',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-router-tail.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/router_tail.wgsl',
      'ggml/src/ggml-webgpu/wgsl-shaders/router_tail_merged.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-host-encode-assembly.patch',
    target: 'llama-webgpu',
    reason: 'host encode assembly',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-f32-small-m.patch',
    target: 'llama-webgpu',
    reason: 'f32 small m',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-moe-gather-logical32.patch',
    target: 'llama-webgpu',
    reason: 'moe gather logical32',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_gather_logical32.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'llama-sampler-single-sync.patch',
    target: 'llama-webgpu',
    reason: 'llama sampler single sync',
    files: [
      'src/llama-sampler.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-matmul-vec4-layout.patch',
    target: 'llama-webgpu',
    reason: 'matmul vec4 layout',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal'],
  },
  {
    file: 'ggml-webgpu-profile-query-budget.patch',
    target: 'llama-webgpu',
    reason: 'profile query budget',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['profile'],
  },
  {
    file: 'ggml-webgpu-q8-diagnostic.patch',
    target: 'llama-webgpu',
    reason: 'q8 diagnostic',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['profile'],
  },
  {
    file: 'ggml-webgpu-matmul-vec4-layout-profile.patch',
    target: 'llama-webgpu',
    reason: 'matmul vec4 layout profile',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['profile'],
  },
  {
    file: 'ggml-webgpu-moe-tight-tile-bound.patch',
    target: 'llama-webgpu',
    reason: 'moe tight tile bound',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-f32-matvec-epilogue.patch',
    target: 'llama-webgpu',
    reason: 'f32 matvec epilogue',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-f32-matvec-epilogue.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl',
    ],
    modes: ['normal'],
  },
  {
    file: 'ggml-webgpu-f32-matvec-epilogue-profile.patch',
    target: 'llama-webgpu',
    reason: 'f32 matvec epilogue profile',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-f32-matvec-epilogue.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_f32_epilogue.wgsl',
    ],
    modes: ['profile'],
  },
  {
    file: 'ggml-webgpu-q6-short-workgroup.patch',
    target: 'llama-webgpu',
    reason: 'q6 short workgroup',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-quant-half-pairs.patch',
    target: 'llama-webgpu',
    reason: 'quant half pairs',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_reg_tile_half_pairs.wgsl',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_half_pairs_decls.tmpl',
      'ggml/src/ggml-webgpu/wgsl-shaders/quant_half_pairs_inner_loops.tmpl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-nvfp4-expert-accumulation.patch',
    target: 'llama-webgpu',
    reason: 'honor NVFP4 expert matrix accumulator precision',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_group.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-moe-sparse-columns.patch',
    target: 'llama-webgpu',
    reason: 'moe sparse column assignments',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_group.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-indexed-wide-k.patch',
    target: 'llama-webgpu',
    reason: 'use a wider K tile for selected indexed prefill shapes',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-indexed-q4-k-paired.patch',
    target: 'llama-webgpu',
    reason: 'decode paired Q4_K nibbles in the indexed K64 weight tile',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_group.wgsl',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_group_decls.tmpl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-direct-mapped-read.patch',
    target: 'llama-webgpu',
    reason: 'direct mapped read',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-quant-wide-m.patch',
    target: 'llama-webgpu',
    reason: 'quant wide m',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-indexed-wide-m.patch',
    target: 'llama-webgpu',
    reason: 'use 64 output rows in selected indexed prefill workgroups',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-q8-prefill-wide-m.patch',
    target: 'llama-webgpu',
    reason: 'wide paired-half Q8 prefill',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/quant_half_pairs_inner_loops.tmpl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-q4-subgroup-matvec.patch',
    target: 'llama-webgpu',
    reason: 'q4 subgroup matvec',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-aligned-word-load-integration.patch',
    target: 'llama-webgpu',
    reason: 'aligned word load integration',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-checkpoint-batch-readback.patch',
    target: 'llama-webgpu',
    reason: 'checkpoint batch readback',
    files: [
      'src/llama-context.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-backend-tensor-get-batch.h',
      'ggml/src/ggml-webgpu/ggml-webgpu-batch-readback.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-batch-readback-geometry.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-subgroup-size-control-shared.patch',
    target: 'llama-webgpu',
    reason: 'subgroup size control shared',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-optional-pipeline.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-batch-readback.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-subgroup-size-control.patch',
    target: 'llama-webgpu',
    reason: 'subgroup size control',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal'],
  },
  {
    file: 'ggml-webgpu-subgroup-size-control-profile.patch',
    target: 'llama-webgpu',
    reason: 'subgroup size control profile',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['profile'],
  },
  {
    file: 'ggml-webgpu-subgroup-size-control-wgsl.patch',
    target: 'llama-webgpu',
    reason: 'subgroup size control wgsl',
    files: [
      'ggml/src/ggml-webgpu/wgsl-shaders/q8_subgroup_matvec.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-q6-subgroup-matvec.patch',
    target: 'llama-webgpu',
    reason: 'q6 subgroup matvec',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-long-k-matvec.patch',
    target: 'llama-webgpu',
    reason: 'select independent subgroup reduction for long-K matvecs',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-buffer-size-limit.patch',
    target: 'llama-webgpu',
    reason: 'Respect native and browser buffer limits.',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-emscripten-scalar-tensor-writes.patch',
    target: 'llama-webgpu',
    reason: 'Use safe scalar writes for browser batched tensor uploads.',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-iq3s-field-loads.patch',
    target: 'iq3s',
    reason: 'iq3s field loads',
    files: [
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-iq3s-row-split.patch',
    target: 'iq3s',
    reason: 'iq3s row split',
    files: [
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-aligned-word-load.patch',
    target: 'word-template',
    reason: 'aligned word load',
    files: [
      'ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-l2-clip.patch',
    target: 'llama-webgpu',
    reason: 'fuse F32 L2-clipping graph sequences',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-l2-clip.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/l2_clip.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-flash-attn-maskskip.patch',
    target: 'llama-webgpu',
    reason: 'skip fully masked KV tiles in long F16 attention',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_tile.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-flash-attn-wide-scalar-kv.patch',
    target: 'llama-webgpu',
    reason: 'use wider KV tiles within the scalar attention shared-memory limit',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-quantized-expert-subgroup-decode.patch',
    target: 'llama-webgpu',
    reason: 'independent subgroups for selected IQ4_NL and NVFP4 expert decode shapes',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-short-conv.patch',
    target: 'llama-webgpu',
    reason: 'single-token short convolution and state store',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-short-conv-residual.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-short-conv-state.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/short_conv_residual.wgsl',
      'ggml/src/ggml-webgpu/wgsl-shaders/short_conv_concat_state.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-depth-mix.patch',
    target: 'llama-webgpu',
    reason: 'single-token depth residual mixing',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-depth-mix.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/depth_normalize.wgsl',
      'ggml/src/ggml-webgpu/wgsl-shaders/depth_scores.wgsl',
      'ggml/src/ggml-webgpu/wgsl-shaders/depth_blend.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-indexed-q2-split-shaders.patch',
    target: 'llama-webgpu',
    reason: 'route indexed Q2_0 loads through two binding ranges',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_vec.wgsl',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_group.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-indexed-q2-split-word-load.patch',
    target: 'word-template',
    reason: 'select the Q2_0 binding for each aligned word load',
    files: ['ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl'],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-indexed-q2-split-host.patch',
    target: 'llama-webgpu',
    reason: 'admit resident indexed Q2_0 weights with checked binding ranges',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu.cpp'],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-indexed-q2-short-workgroup.patch',
    target: 'llama-webgpu',
    reason: 'reduce indexed Q2_0 decode workgroup width at K640 and M2560',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-depth-mix-view-sources.patch',
    target: 'llama-webgpu',
    reason: 'preserve depth mix intermediates read through views without source edges',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-fusion-view-sources.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-depth-mix.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-fusion-view-sources-normal-encode.patch',
    target: 'llama-webgpu',
    reason: 'share view source checks across the ordinary graph encoder',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu.cpp'],
    modes: ['normal'],
  },
  {
    file: 'ggml-webgpu-fusion-view-sources-profile-encode.patch',
    target: 'llama-webgpu',
    reason: 'share view source checks across the profiled graph encoder',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu.cpp'],
    modes: ['profile'],
  },
  {
    file: 'ggml-webgpu-situ-packed.patch',
    target: 'llama-webgpu',
    reason: 'fuse the packed expert SITU activation for single-token evaluation',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-situ.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/situ_packed.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-situ-packed-prefill.patch',
    target: 'llama-webgpu',
    reason: 'reuse the packed SITU activation across contiguous token batches',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu-situ.hpp', 'ggml/src/ggml-webgpu/ggml-webgpu.cpp'],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-bf16-small-m.patch',
    target: 'llama-webgpu',
    reason: 'reuse small-M register tiles for long-K BF16 prefill',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-indexed-q4-0-paired.patch',
    target: 'llama-webgpu',
    reason: 'reuse aligned Q4_0 block pairs in wider indexed prefill tiles',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_group_decls.tmpl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-loop-controller-context-normal.patch',
    target: 'llama-webgpu',
    reason: 'share the ordinary matvec factory context with controller fusion admission',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu.cpp'],
    modes: ['normal'],
  },
  {
    file: 'ggml-webgpu-loop-controller.patch',
    target: 'llama-webgpu',
    reason: 'fuse the paired F32 projections and elementwise update on single-token graphs',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-loop-controller.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/loop_controller.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-dense-q4-swiglu.patch',
    target: 'llama-webgpu',
    reason: 'share Q4_K input partial sums and fuse dense SwiGLU projections',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-fusion-view-sources.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-dense-q4-swiglu.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/dense_q4_swiglu.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-conv-router-view-sources.patch',
    target: 'llama-webgpu',
    reason: 'preserve convolution and router intermediates read through raw views',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-router-tail.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-short-conv-residual.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-ssm-conv-silu.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-indexed-wide-k-apple.patch',
    target: 'llama-webgpu',
    reason: 'keep K32 for indexed IQ4_NL and NVFP4 prefill on Apple adapters',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-indexed-live-grid.patch',
    target: 'llama-webgpu',
    reason: 'dispatch only live workgroups for selected indexed prefill matrices',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-optional-pipeline.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_gather_logical32.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-fusion-intermediate-views.patch',
    target: 'llama-webgpu',
    reason: 'preserve RMS, scale, matvec and expert-sum intermediates read through raw views',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-f32-matvec-epilogue.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-fusion-view-sources.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-moe-sum.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-rms-scale.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-scale-unary.hpp',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-argsort-compact-workgroup.patch',
    target: 'llama-webgpu',
    reason: 'skip padded comparisons in single-tile argsort and defer unused merge pipelines',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/argsort.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-flash-mask-block-workgroup.patch',
    target: 'llama-webgpu',
    reason: 'use a compact workgroup for vector Flash Attention mask classification',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-q8-sigmoid-mul.patch',
    target: 'llama-webgpu',
    reason: 'write the sigmoid-gated output directly from the fixed-subgroup Q8 matvec',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-checked-tensor-binding.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-dense-q4-swiglu.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-q8-sigmoid-mul.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/q8_subgroup_matvec.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-indexed-q8-q5-subgroup.patch',
    target: 'llama-webgpu',
    reason: 'use independent subgroups for matching Q8_0 and Q5_K expert projections',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp', 'ggml/src/ggml-webgpu/ggml-webgpu.cpp'],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-fwht-1024.patch',
    target: 'llama-webgpu',
    reason: 'replace explicitly hinted 1024-point Hadamard matrix products with a butterfly transform',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_fwht_1024.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-q2-prefill-wide-m.patch',
    target: 'llama-webgpu',
    reason: 'use 64-row paired-half tiles in matching dense Q2_0 prefill',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_half_pairs_decls.tmpl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-add-cast-f16.patch',
    target: 'llama-webgpu',
    reason: 'combine broadcast mask addition with its packed F16 conversion',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-add-cast-f16.hpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/add_cast_f16.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-unary-parameter-cache.patch',
    target: 'llama-webgpu',
    reason: 'reuse unary pipelines across changes to uniform parameters',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-matmul-host-dispatch.patch',
    target: 'llama-webgpu',
    reason: 'reduce temporary allocations in matrix dispatch encoding',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu.cpp', 'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-nvfp4-indexed-half-staging.patch',
    target: 'llama-webgpu',
    reason: 'stage F32-accumulated NVFP4 expert weights with half vectors',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_id_group_decls.tmpl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-indexed-half-flash.patch',
    target: 'llama-webgpu',
    reason: 'Compose rounded indexed attention masks using bounded GPU storage.',
    files: [
      'ggml/src/ggml-webgpu/ggml-webgpu.cpp',
      'ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp',
      'ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_tile.wgsl',
      'ggml/src/ggml-webgpu/wgsl-shaders/flash_attn_decls.tmpl',
      'ggml/src/ggml-webgpu/wgsl-shaders/indexed_vec_mask.wgsl',
    ],
    modes: ['normal', 'profile'],
  },
  {
    file: 'ggml-webgpu-q2-dense-subgroup-matvec.patch',
    target: 'llama-webgpu',
    reason: 'reduce selected dense Q2_0 matvecs within independent subgroups',
    files: ['ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp'],
    modes: ['normal', 'profile'],
  },
];

export function webgpuPatches({ mode }: { mode: SourceMode }): Patch[] {
  return webgpuRecipe.filter(patch => patch.modes.includes(mode)).map(({ modes: _modes, ...patch }) => patch);
}

export const modelPatches = {
  fileRows: {
    file: 'qwen4exp-file-backed-rows.patch',
    target: 'llama',
    reason: 'Keep all model-definition consumers coherent while adding bounded file-backed rows.',
    files: ['src/llama-model.cpp', 'src/llama.cpp', 'src/models/models.h', 'src/models/qwen4exp.cpp'],
  },
  qsa: {
    file: 'qwen4exp-all-pools-selection.patch',
    target: 'llama',
    reason: 'Skip ranking only when every pool is selected.',
    files: ['src/models/qwen4exp.cpp'],
  },
  inklingRowViews: {
    file: 'inkling-packed-row-views.patch',
    target: 'llama',
    reason: 'Read packed loop messages and attention inputs through row views.',
    files: ['src/models/inkling2.cpp'],
  },
  indexedHalfCpu: {
    file: 'ggml-cpu-indexed-half-flash.patch',
    target: 'llama',
    reason: 'Preserve half-mask rounding on CPU fallback.',
    files: ['ggml/src/ggml-cpu/flash-attn-indexed-mask.h', 'ggml/src/ggml-cpu/ggml-cpu.cpp', 'ggml/src/ggml-cpu/ops.cpp'],
  },
  inklingIndexedHalfFlash: {
    file: 'inkling-indexed-half-flash.patch',
    target: 'llama',
    reason: 'Keep the same compact relative-bias graph for prefill and generation.',
    files: ['src/llama-graph.h', 'src/llama-graph.cpp', 'src/models/inkling2.cpp'],
  },
  chat: {
    file: 'common-chat-independent-vocabulary.patch',
    target: 'llama',
    reason: 'Initialize independent vocabulary chat metadata without a dummy model.',
    files: ['common/chat.cpp'],
  },
} satisfies Record<string, Patch>;

export function mtmdPatches({ mode }: { mode: 'cpu' | 'webgpu' }): Patch[] {
  const patches: Patch[] = [{
    file: 'mtmd-vision-bf16-runtime.patch',
    target: 'mtmd',
    reason: 'Expose both vision BF16 policies through the existing runtime boundary.',
    files: ['mtmd-model.h', 'mtmd-model.cpp', 'mtmd-subcontext.cpp', 'mtmd.cpp'],
  }];
  switch (mode) {
  case 'webgpu':
    patches.push({
      file: 'mtmd-webgpu-bf16.patch',
      target: 'mtmd',
      reason: 'Retain checked BF16 projector weights on the WebGPU path.',
      files: ['mtmd-model.cpp', 'mtmd-subcontext.cpp'],
    });
    break;
  case 'cpu': break;
  default: {
    const _ex: never = mode;
    throw new Error(`Unknown mtmd mode: ${String(_ex)}`);
  }
  }
  patches.push({
    file: 'mtmd-non-pthread-threadpool.patch',
    target: 'mtmd',
    reason: 'Keep the single-threaded browser threadpool ownership valid.',
    files: ['mtmd-subcontext.cpp'],
  });
  return patches;
}
