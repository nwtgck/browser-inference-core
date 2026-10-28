# WebGPU vision BF16 storage

The pinned llama.cpp revision `8642541b255bfb5dc39cd351a178be09013a55d6`
has BF16-left/F32-right matrix multiplication and BF16 row lookup in WebGPU.
Storage support alone does not establish complete-model compatibility or speed.
BF16 means Brain Floating Point 16; F16/F32 mean 16/32-bit floating point.

## Load-time selection in the same runtime

Every existing profile and variant exposes the same versioned storage API.
There is no extra Wasm asset, profile, variant or build directory for retention.
`LCB_WEBGPU_BF16_PROJECTOR` continues to control the reviewed WebGPU loader
integration. Ordinary builds include both storage policies in that integration.

`mtmd_init_from_file` remains unchanged and selects `legacy-expand`. Its parameter
record and `clip_context_params` keep their upstream layouts. New callers use the
C declarations in `bridge/mtmd-vision-bf16.h`:

```c
uint32_t mtmd_vision_bf16_storage_version(void);
struct mtmd_context * mtmd_init_from_file_with_vision_bf16_storage(
    const char * path, const struct llama_model * text_model,
    const struct mtmd_context_params params, int32_t storage);
enum mtmd_vision_bf16_storage mtmd_context_vision_bf16_storage(
    const struct mtmd_context * context);
```

Version 1 defines `MTMD_VISION_BF16_STORAGE_LEGACY_EXPAND=0` and
`MTMD_VISION_BF16_STORAGE_RETAIN_CHECKED=1`. The init argument is a raw i32 so
unknown values can be rejected before enum conversion, file I/O or allocation.
The generated schema, types, normalized pointer ABI and JSPI exports include
these functions and constants just like the other mtmd functions.

The selected mode is fixed before loading and immutable for that context.
Changing mode requires freeing and reloading the projector; include mode in any
host context-reuse identity. Separate contexts can use different modes in the
same module, including progress-callback reentry. No process-global or
thread-local selector is used. The query returns the requested policy, not proof
of actual backend placement; null returns `MTMD_VISION_BF16_STORAGE_INVALID=-1`.

| Request | Behavior |
| --- | --- |
| Legacy / old init | Established WebGPU BF16-to-F32 expansion |
| Retain with actual WebGPU | Original BF16 storage with role/graph/placement checks |
| Retain requesting GPU vision without WebGPU | Explicit failure, without CPU retry |
| Explicit CPU load | Original upstream storage/execution, not WebGPU acceptance |
| Unknown storage value | Failure before file I/O |

Audio-only loads keep their original behavior in every profile. The existing
loader modality classification rejects unsupported retained GPU vision before
weight allocation, without reparsing the file. Audio policy and non-BF16 weight
types remain unchanged. Model files are never
rewritten. Retention is opt-in; this core does not change consumer defaults.
New runtime capability metadata records both modes and the legacy default.
Normal distribution accepts the shared runtime; obsolete build-time storage
selection is rejected rather than published as a runtime-selected comparison.

## Checked retention contract

A WebGPU vision load first checks the actual weight roles against the pinned
private `clip_model` fields. Reviewed projector enums are MLP/MLP-norm,
Qwen2/2.5/3VL, Gemma 3, Gemma4V and LFM2/3VL. This is a conservative role
allowlist, not a complete real-model compatibility list. If BF16 weights exist,
unreviewed projectors, BF16 norms/patch kernels/interpolated position tables and
Gemma 3's custom BF16 transposed projection are rejected. Standard F16 Gemma 3
projection stays F16. A load without BF16 weights does not need that role gate.

Every reservation and encoding checks retained weights and their views:

- Accept only contiguous BF16-left/F32-input `MUL_MAT` and BF16 `GET_ROWS`.
  Metadata-only views must retain original storage.
- Reject weight-side CONT/CPY/DUP/casts, unsupported operands, shapes, strides,
  byte spans, view bounds and dispatch envelopes before large graph allocation.
  Row lookup is limited to 65,535 output rows, avoiding an unreviewed 2D path.
- Require target backend support for the actual operation and weight buffer type.
  Commit scheduler assignments only after the entire plan passes.
- After reserve/allocation, require unchanged weight-input identities. After
  allocation, also check actual backend assignment and reject a weight buffer
  shared with input or output. The mixed BF16/F32 overlap path is not assumed safe.

Failures stop load/encode; they do not silently retry in F32 or on CPU. Load
returns null and encode returns false through the existing failure boundaries.
With warmup disabled, graph rejection can first occur at encoding. Original
upstream graph-allocation failure handling is preserved. These checks are not a
generic malicious-GGUF validator or a complete detector of activation copies.

## Storage, loading and lifetime

For original BF16 payload B and other weight payload C, retention aims for C+B
instead of C+2B resident tensor bytes. Alignment, graph buffers, driver staging,
physical GPU memory and overlapping old/new contexts are separate quantities.
There is no all-F32 arithmetic or universal bit-identity claim: existing upstream
BF16 kernels and intermediate precision remain unchanged. Retention can be slower;
matched model/device/variant/image measurements are needed before adoption.

Retained WebGPU loads use a reusable 512 KiB raw reader for all loaded tensor
types and a loader-local 32 MiB logical queued-byte budget, including bounded
head/tail write overhead. Ordinary `ggml_backend_tensor_set` owns partial-word
BF16 writes; there is no replacement WebGPU upload API. Checked source-byte totals
and stream offsets differ from expanded destination sizes.

The queue drains before exceeding its budget, at completion and on stack
unwinding, before the owning weight buffer can be destroyed. Cancellation is
checked initially, around chunks, after waits, after the last write and after
final drain. Truncation and arithmetic overflow fail explicitly. The budget is
not a physical-memory guarantee. Existing backend fatal validation/device-loss
behavior is unchanged; no recoverable device-loss API is added.

Legacy conversion keeps its bounded 1.5 MiB scratch (0.5 MiB BF16 input + 1 MiB
F32 output) and twofold converted resident payload. Finite BF16 values are exactly
representable in F32. Neither policy promises equal whole-model output/speed.

## Diagnostics

Legacy successful allocated loads keep their existing record:

```text
lcb_clip: bf16-f32 tensors=<count> source_bytes=<original> destination_bytes=<expanded>
```

Retained WebGPU loads use separate metadata records:

```text
lcb_clip: bf16-storage mode=retain-checked tensors=<BF16 count> source_bytes=<BF16 payload> resident_bytes=<BF16 payload> scratch_bytes=<raw reader capacity> queue_budget_bytes=<logical window>
lcb_clip: retained graph consumers=<checked consumers> webgpu_compute_bytes=<scheduler buffers> cpu_compute_bytes=<scheduler buffers>
```

Both preserve `lcb_clip: matmul placement cpu=... webgpu=... other=... cpu_bf16=...`
after allocation. These are metadata, not timings or physical-memory readings.
Absence of a record is not zero or success; `no_alloc` emits no load success.
Requested mode, actual backend and retained-weight count are distinct facts.
New records omit model paths, tensor names and tensor values.

## Legacy execution cost

Storage selection is read at context initialization and graph boundaries, not
inside every WebGPU operation. Legacy mode skips role inspection and returns
from retained graph preparation before collecting weights or walking nodes.
The retained reader allocates scratch only when used; the legacy queue is empty
and submits no extra wait. Existing placement diagnostics remain unchanged in
scope. Empty helper state and a few branches still have a cost; these source
properties are not a measured guarantee of equal latency.

## Maintenance and validation

The vendor checkout is unchanged. A shared build-tree mtmd.cpp copy adds the
versioned entry and immutable policy; an independent clip.cpp copy contains the
existing loader integration and both policies. Audio and WebGPU kernel overlays
remain independent. Source hashes pin the reviewed upstream contracts, including
private model roles, scheduler, allocator and BF16 shaders. CPU-only builds use
a separate minimal clip plumbing copy; they do not compile the retained graph
checks. Configuration fails
on drift, and all contract files trigger ordinary CMake regeneration.

Native CPU contract/numerical tests and synthetic upstream model-graph fixtures
are distinct from actual model loading, WebGPU execution and image quality. The
source-extracted entry fixture uses a no-I/O constructor double and does not
certify loader execution. Generated API tests cover the new scalar signature and
old layout. Actual runtime loading and browser/device measurements remain
separate gates. Supplied host evidence reported a CPU function-type sanitizer
issue in pristine upstream; do not hide an upstream issue by weakening guards.

Retire legacy expansion only after matched target-model evidence supports the
change. Remove independent loader safety/diagnostics only when upstream provides
equivalent behavior. New shaders or widened model-role acceptance require their
own semantic review and are not implied by this storage API.
