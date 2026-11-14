import type { Patch } from '#tools/source/patches';

export type SourceSelection = { include?: readonly string[]; exclude?: readonly string[] };

/** Optional upstream projects are disabled in the owned CMake build graph. */
export const sourceSelections = {
  'stable-diffusion': {
    exclude: ['assets', 'examples', 'ggml', 'thirdparty/cutlass', 'thirdparty/libwebm', 'thirdparty/libwebp'],
  },
  ggml: { include: ['ggml', 'LICENSE'] },
} as const satisfies Record<'stable-diffusion' | 'ggml', SourceSelection>;

/** The ordered, independently reviewable image-runtime changes. */
export const patches = [
  {
    "file": "0001-single-thread-model-reader.patch",
    "target": "stable-diffusion",
    "reason": "Execute the model-reader closure inline for single-threaded Emscripten, preserving native/pthread behavior.",
    "files": [
      "src/model_loader.cpp"
    ],
  },
  {
    "file": "0002-webgpu-group-normalization.patch",
    "target": "stable-diffusion",
    "reason": "Lower supported group-normalization layouts to existing NORM where equivalent, with CPU reference tests.",
    "files": [
      "src/core/ggml_extend.cpp"
    ],
  },
  {
    "file": "0003-webgpu-unknown-memory-capacity.patch",
    "target": "ggml",
    "reason": "Do not misreport maxBufferSize as total/free GPU memory. Image callers must supply an explicit budget when live capacity is unknown.",
    "files": [
      "ggml/src/ggml-webgpu/ggml-webgpu.cpp"
    ],
  },
  {
    "file": "0004-browser-gguf-64bit-file-offsets.patch",
    "target": "stable-diffusion",
    "reason": "Read unsplit GGUF metadata without whole-file size_t accumulation; preserve 64-bit file positions, validate metadata and leave tensor payloads lazy.",
    "files": [
      "src/model_io/gguf_io.cpp",
      "src/model_io/gguf_reader_ext.h",
      "src/model_io/tensor_storage.h"
    ],
  },
  {
    "file": "0005-browser-gguf-logical-cursor.patch",
    "target": "stable-diffusion",
    "reason": "Keep a 64-bit logical metadata cursor; avoid tellg buffer invalidation and repeated FileReaderSync reads while retaining bounded skips and file checks.",
    "files": [
      "src/model_io/gguf_reader_ext.h"
    ],
  },
  {
    "file": "0006-browser-safetensors-and-gguf-shards.patch",
    "target": "stable-diffusion",
    "reason": "Keep safetensors file positions 64-bit, bound/validate metadata and local shard paths, and load complete standard GGUF shard groups without file conversion.",
    "files": [
      "src/model_loader.cpp",
      "src/model_io/gguf_reader_ext.h",
      "src/model_io/safetensors_io.cpp"
    ],
  },
  {
    "file": "0007-webgpu-qwen-timestep-out-of-place.patch",
    "target": "stable-diffusion",
    "reason": "Avoid an in-place Qwen timestep SiLU alias across a BF16 CPU matmul / WebGPU activation boundary. Preserve weights and arithmetic; no profile fallback or dtype conversion.",
    "files": [
      "src/model/diffusion/qwen_image.hpp"
    ],
  },
  {
    "file": "0008-browser-graph-placement-diagnostics.patch",
    "target": "stable-diffusion",
    "reason": "Report bounded allocation-time BF16 weight matmul placement through existing logs, without changing graph execution or reading tensor contents.",
    "files": [
      "src/core/compute_workspace.cpp"
    ],
  },
  {
    "file": "0009-webgpu-bf16-parameter-type.patch",
    "target": "stable-diffusion",
    "reason": "Convert only original BF16 parameters before WebGPU allocation to caller-selected F32 (default) or F16, preserving source files and other weight types.",
    "files": [
      "include/stable-diffusion.h",
      "src/stable-diffusion.cpp",
      "src/model_manager.cpp",
      "src/model_manager.h",
      "src/core/backend_fit.cpp",
      "src/core/backend_fit.h",
      "src/pipeline/diffusion_engine.cpp"
    ],
  },
  {
    "file": "0010-iterative-graph-traversal.patch",
    "target": "ggml",
    "reason": "Replace recursive graph-parent traversal with a heap-backed DFS stack, preserving order, compute flags and operand use counts for deep image graphs.",
    "files": [
      "ggml/src/ggml.c"
    ],
  },
  {
    "file": "0011-webgpu-conv3d-bias-out-of-place.patch",
    "target": "stable-diffusion",
    "reason": "Give browser 3D convolution bias additions their own output when CPU convolution and WebGPU addition use different backends, preserving arithmetic and parameter types.",
    "files": [
      "src/core/ggml_extend.cpp"
    ],
  },
  {
    "file": "0012-webgpu-single-depth-convolution.patch",
    "target": "stable-diffusion",
    "reason": "Use an equivalent supported 2D direct convolution for a single full-depth 3D window instead of CPU fallback, without changing parameter types or allocating a full im2col volume.",
    "files": [
      "src/core/ggml_extend.cpp"
    ],
  },
  {
    "file": "0013-passive-context-snapshots.patch",
    "target": "stable-diffusion",
    "reason": "Expose idle on-demand runtime, borrowed configuration and separately labelled memory bookkeeping snapshots without adding generation-path instrumentation or GPU synchronization; own the omitted audio-encoder configuration path.",
    "files": [
      "include/stable-diffusion.h",
      "src/stable-diffusion.cpp",
      "src/model_manager.cpp",
      "src/model_manager.h",
      "src/pipeline/diffusion_engine.h"
    ],
  },
  {
    "file": "0014-opt-in-graph-stage-diagnostics.patch",
    "target": "stable-diffusion",
    "reason": "Add explicit default-off, bounded graph-stage logs to distinguish preparation, scheduler reserve including sync, weights, allocation, input copy and execution without extra native timing, graph walks or GPU synchronization.",
    "files": [
      "include/stable-diffusion.h",
      "src/core/util.cpp",
      "src/core/util.h",
      "src/core/compute_workspace.cpp",
      "src/core/compute_workspace.h",
      "src/core/execution_engine.cpp",
      "src/core/execution_engine.h"
    ],
  },
  {
    "file": "0015-preserve-measurement-tensor-addresses.patch",
    "target": "stable-diffusion",
    "reason": "Preserve external tensor address metadata while measuring scheduler graphs; avoid invalid sentinel alignment in WebGPU GET_ROWS support checks without changing weights or backend policy.",
    "files": [
      "src/core/compute_workspace.cpp"
    ],
  },
  {
    "file": "0016-webgpu-temporal-pointwise-convolution.patch",
    "target": "stable-diffusion",
    "reason": "Lower unsupported multi-frame 1x1x1 convolution to supported matrix and contiguous operations without im2col expansion; retain direct, precision, single-frame and rejected-backend paths.",
    "files": [
      "src/core/ggml_extend.cpp"
    ],
  },
  {
    "file": "0017-browser-h3-checked-video-publication.patch",
    "target": "stable-diffusion",
    "reason": "H3 video output allocation is checked and cancellation-safe; completed frames are published together without an extra float crop, and unused frame outputs are released.",
    "files": [
      "src/core/util.cpp",
      "src/core/util.h",
      "src/pipeline/video.cpp"
    ],
  },
  {
    "file": "0018-browser-video-queue-wait-budget.patch",
    "target": "ggml",
    "reason": "Allow finite 180-second queue completion for video graphs that exceed 30 seconds; retain independent 30-second map/event deadlines and all wait/callback failure checks.",
    "files": [
      "ggml/src/ggml-webgpu/ggml-webgpu.cpp"
    ],
  },
  {
    "file": "0019-webgpu-completion-based-waits.patch",
    "target": "ggml",
    "reason": "Wait for submitted work, maps and events without an application wall-clock deadline; keep wait/callback failures and per-recording event ownership.",
    "files": [
      "ggml/src/ggml-webgpu/ggml-webgpu.cpp"
    ],
  },
  {
    "file": "0020-webgpu-single-image-causal-convolution.patch",
    "target": "stable-diffusion",
    "reason": "Lower zero-history single-image causal convolution to a packed temporal slice only when the selected backend supports the candidate; preserve circular/history/video fallbacks.",
    "files": [
      "src/model/vae/wan_vae.hpp"
    ],
  },
  {
    "file": "0021-browser-anima-conditioning.patch",
    "target": "stable-diffusion",
    "reason": "Keep structural Anima zero-padding mass with explicit sinks and reuse immutable sampling conditions through RunnerCache; remove when upstream provides equivalent scoped ownership, invalidation and fallback contracts.",
    "files": [
      "src/pipeline/diffusion_engine.cpp",
      "src/model/diffusion/anima.hpp",
      "src/model/diffusion/model.hpp",
      "src/model/integrations/anima.cpp",
      "src/core/ggml_extend.cpp",
      "src/core/ggml_extend.h"
    ],
  },
  {
    "file": "0022-webgpu-device-tensor-copy.patch",
    "target": "ggml",
    "reason": "Copy compatible same-device tensor ranges directly on the WebGPU queue; preserve synchronous and asynchronous contracts and retain fallback for unsupported ranges.",
    "files": [
      "ggml/src/ggml-webgpu/ggml-webgpu.cpp"
    ],
  },
  {
    "file": "0023-webgpu-cache-capture-batches.patch",
    "target": "stable-diffusion",
    "reason": "Allocate cache destinations before queuing same-device copies, then complete once before publishing ownership; preserve host/strided fallbacks and failure lifetimes.",
    "files": [
      "src/core/runner_cache.cpp"
    ],
  },
  {
    "file": "0024-webgpu-bounded-materialization.patch",
    "target": "stable-diffusion",
    "reason": "Bound rejected convolution expansions and manual attention scores using existing WebGPU operations; require every introduced compute node to be supported, retaining precision, masks, sinks and original fallbacks.",
    "files": [
      "src/core/ggml_extend.cpp"
    ],
  },
  {
    "file": "0025-webgpu-opt-in-operation-placement.patch",
    "target": "stable-diffusion",
    "reason": "Report an opt-in compute-operation placement histogram and current rejection counts without tensor names, readbacks, extra synchronization or execution-time claims.",
    "files": [
      "src/core/compute_workspace.cpp"
    ],
  },
  {
    "file": "0026-webgpu-batch-parameter-uploads.patch",
    "target": "ggml",
    "reason": "Stage aligned per-kernel uniform parameters in a bounded host mirror and flush once before each existing WebGPU command submission; retain queue order, throttling, graph boundaries, bindings and slot lifetimes. Remove when the pinned upstream provides equivalent batching.",
    "files": [
      "ggml/src/ggml-webgpu/ggml-webgpu.cpp"
    ],
  },
  {
    "file": "0027-skip-empty-cache-completion.patch",
    "target": "stable-diffusion",
    "reason": "Avoid a queue completion registration when a runner or graph-cut cache has no new outputs; retain waits for every non-empty capture and exceptional copy exit. Segment computation keeps its own completion boundary.",
    "files": [
      "src/core/runner_cache.cpp"
    ],
  },
  {
    "file": "0028-webgpu-softmax-private-cache.patch",
    "target": "ggml",
    "reason": "Index the softmax private cache by lane iteration, use guarded storage/cache loads instead of eager select, and retain the same per-lane sum/reduction/precision. Remove when pinned upstream supplies equivalent bounds-safe cache use.",
    "files": [
      "ggml/src/ggml-webgpu/wgsl-shaders/soft_max.wgsl"
    ],
  },
  {
    "file": "0029-webgpu-attention-softmax-scale.patch",
    "target": "stable-diffusion",
    "reason": "Fuse unmasked manual attention scaling into an admitted WebGPU softmax AFTER F32 score accumulation, without changing Q/K, masks, sinks, CPU behavior or Flash settings. Remove when upstream avoids the score-scale pass with equivalent admission and numerical contracts.",
    "files": [
      "src/core/ggml_extend.cpp"
    ],
  },
  {
    "file": "0030-webgpu-qwen-shared-rope-matrix.patch",
    "target": "stable-diffusion",
    "reason": "Share one graph-local position-matrix preparation across Qwen 2.1 WebGPU Q/K consumers. No persistent pointer-keyed cache or new numerical operation. Remove when upstream shares or fuses the equivalent RoPE position work with graph-cut-safe ownership.",
    "files": [
      "src/model/common/rope.hpp",
      "src/model/diffusion/qwen_image_2_1.hpp"
    ],
  },
  {
    "file": "0031-webgpu-qwen-shared-modulation-and-swiglu.patch",
    "target": "stable-diffusion",
    "reason": "Share graph-local Qwen Image 2.1 modulation, borrow read-only contiguous spans and select already-supported F32 SwiGLU; preserve the existing CPU and unsupported-operation paths, graph-cut lifetimes and an immutable context A/B opt-out. Remove when upstream supplies equivalent guarded work reduction.",
    "files": [
      "src/model/diffusion/qwen_image_2_1.hpp"
    ],
  },
  {
    "file": "0032-webgpu-compute-operation-details.patch",
    "target": "stable-diffusion",
    "reason": "Extend existing opt-in placement logs with bounded WebGPU operation/subtype counts and largest logical output; no readback, timestamps, tensor names, new synchronization or GPU-time/VRAM claims. Remove when upstream supplies equivalent allocation metadata.",
    "files": [
      "src/core/compute_workspace.cpp"
    ],
  },
  {
    "file": "0033-webgpu-linear-dispatch-grid.patch",
    "target": "ggml",
    "reason": "Dispatch row normalization, fused RMS multiply, softmax and GLU in a checked 2D grid; discard padding uniformly before barriers and bound linear arithmetic without JS shader rewriting. Remove when pinned WebGPU supports equivalent bounded dispatches.",
    "files": [
      "ggml/src/ggml-webgpu/ggml-webgpu.cpp",
      "ggml/src/ggml-webgpu/wgsl-shaders/row_norm.wgsl",
      "ggml/src/ggml-webgpu/wgsl-shaders/rms_norm_mul.wgsl",
      "ggml/src/ggml-webgpu/wgsl-shaders/soft_max.wgsl",
      "ggml/src/ggml-webgpu/wgsl-shaders/glu.wgsl"
    ],
  },
  {
    "file": "0034-webgpu-stable-registry-device.patch",
    "target": "ggml",
    "reason": "Preserve once-only registry/device identity across lookup and own adapter/device callback results through wait failures. Never reset live counts, re-request admitted adapters, replace queues beneath buffers, retry automatically or publish late failed requests. Remove when upstream provides equivalent lifetime contracts.",
    "files": [
      "ggml/src/ggml-webgpu/ggml-webgpu.cpp"
    ],
  },
  {
    "file": "0035-browser-gguf-original-shapes.patch",
    "target": "stable-diffusion",
    "reason": "Restore bounded Comfy GGUF original-shape metadata without changing payload, dtype, offsets or tensor elements; reject invalid, duplicate and conflicting shapes rather than bypassing validation through the generic parser. Remove when upstream preserves the same shape and validation contracts.",
    "files": [
      "src/model_io/gguf_reader_ext.h",
      "src/model_io/gguf_io.cpp"
    ],
  },
  {
    "file": "0036-webgpu-bf16-weight-kernels.patch",
    "target": "ggml",
    "reason": "Backport upstream 4b1622afb BF16-left WebGPU matrix and row-lookup kernels while preserving existing binding, alignment and dispatch contracts. Remove when the image ggml pin supplies the equivalent kernels.",
    "files": [
      "ggml/src/ggml-webgpu/ggml-webgpu.cpp",
      "ggml/src/ggml-webgpu/ggml-webgpu-shader-lib.hpp",
      "ggml/src/ggml-webgpu/wgsl-shaders/common_decls.tmpl",
      "ggml/src/ggml-webgpu/wgsl-shaders/get_rows.wgsl",
      "ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_decls.tmpl",
      "ggml/src/ggml-webgpu/wgsl-shaders/mul_mat_vec_acc.tmpl"
    ],
  },
  {
    "file": "0037-webgpu-bounded-tensor-uploads.patch",
    "target": "ggml",
    "reason": "Bound WebGPU tensor writes and queue-wide pending uploads without changing tensor bytes, command order or completion ownership. Remove when upstream provides equivalent bounded upload and failure contracts.",
    "files": [
      "ggml/src/ggml-webgpu/ggml-webgpu-bounded-upload.h",
      "ggml/src/ggml-webgpu/ggml-webgpu.cpp"
    ],
  },
  {
    "file": "0038-browser-bounded-model-reader.patch",
    "target": "stable-diffusion",
    "reason": "Read eligible original BF16 device weights through bounded loader scratch without changing file bytes or offsets, retaining conversion and generic-reader fallbacks. Remove when upstream supplies equivalent checked, bounded reads and cleanup.",
    "files": [
      "src/model_io/browser_weight_reader.h",
      "src/model_loader.cpp"
    ],
  },
  {
    "file": "0039-browser-bf16-weight-retention.patch",
    "target": "stable-diffusion",
    "reason": "Retain explicitly registered original BF16 matrix/lookup weights on WebGPU with role, identity and allocation checks; preserve F32/F16, CPU and other roles. Remove when upstream supplies the same checked storage and fallback contract.",
    "files": [
      "include/stable-diffusion.h",
      "src/core/backend_fit.cpp",
      "src/core/browser_bf16_retention.h",
      "src/core/compute_workspace.cpp",
      "src/core/compute_workspace.h",
      "src/core/execution_engine.cpp",
      "src/core/ggml_runner.h",
      "src/device_residency_manager.h",
      "src/model/common/ggml_block.hpp",
      "src/model/diffusion/minimax_h3.hpp",
      "src/model/te/llm.hpp",
      "src/model/te/ming_image_te.hpp",
      "src/model_manager.cpp",
      "src/model_manager.h",
      "src/pipeline/diffusion_engine.cpp",
      "src/stable-diffusion.cpp"
    ],
  }
] as const satisfies readonly Patch[];
