# llama.cpp v0.7.0 integration review

## Identity and scope

This review compares v0.6.0 `d81235049384534c167caea52b85a694f6103d14` with
v0.7.0 `620d2225cc436ab0b001e3200e590e3b286ef9ab`. The supplied PR #7 source base
is `8a8d04b9d25f423fb816303d79a3c23153183eed`, which already pins v0.7.0.
Upstream working trees are kept pristine. No new upstream feature, model-specific
branch, toolchain workaround, runtime profile or publishing identity is invented.

A successful patch application is not the review criterion. Reviewed dependency
hashes and composed output hashes remain fail-closed, and existing negative
controls are retained. The reviewed input set is for the new pin; it is not a
promise that the removed v0.5/v0.6 integration can still be built by this source.
Tensor-copy preparation still permits unrelated commits with identical reviewed
contract files, as before.

## Upstream differences and decisions

| Area | Reviewed change | Integration decision |
| --- | --- | --- |
| MoE vector shader and dispatch | Upstream now directly indexes selected slots, guards rectangular workgroup tails, and preserves broadcast/output addressing. Multi-token gathering has separate upstream changes. | Retire the old vector-kernel overlay and all activation/include plumbing. Keep the upstream batched path unchanged. Record retirement separately from compiled overlays. |
| Qwen4Exp graph | Per-layer embedding input moves earlier in the split; multi-token prediction changes model/graph structure. The pool visibility, positive score and set-valued mask contracts required by the accepted optimization remain. | Mechanically rebase all-pools selection; preserve cache writes and ranked fallback. No widened architecture, quantization or context scope. |
| Hybrid indexed memory | Shared sequence pools, unique representative rows and state persistence changed. | Review `llama-memory-hybrid-idx.*` rather than just accepting `qwen4exp.cpp` context. Run independent state/shared-sequence/threshold traces. |
| File-backed rows | Shared model declarations and graph organization changed; accepted row-reader insertion points remain. | Keep the row patch bytes unchanged, refresh reviewed inputs, compile coherent model copies, and rerun native model/reader tests. |
| WebGPU copy and parameter batching | Backend scheduler and two-dimensional dispatch changed outside the accepted copy and batching paths. | Keep both patch payloads unchanged; review queue/fallback/header contracts and refresh full input/output identities. |
| Bounded WebGPU model upload | `load_all_data` changes the buffer collection from a map to a vector, with related mapped-range changes. | Rebase context around the existing loader helper. Retain the non-mmap/non-host/default-WebGPU restriction, block alignment, charged budget, drain and cancellation semantics. |
| Single-token SSM convolution | Host/header context changed; the selected token-width behavior remains applicable. | Keep the patch unchanged, update composed identities, and use the upstream-generated embedded shader header after MoE retirement. |
| Audio and BF16 vision | New architecture/support code appears outside the existing worker/weight-conversion insertion points. | Both existing patches still apply and remain byte-identical. Keep them independent. |
| Chat | Prompt parts, byte-indexed token-aware input, native parser arenas and session-owned continuation replace the old string/serialized-parser boundary. | Bind upstream values and public session interfaces directly. Do not recreate removed JSON serialization or flatten prompt segments at the boundary. |

The only retained patch payloads regenerated for changed context are
`qwen4exp-all-pools-selection.patch` and
`llama-model-loader-webgpu-chunked-upload.patch`. Three-way comparison used the
old pristine file, its previously accepted patched form and the new pristine
file; both merges were conflict-free. The resulting changes retain the accepted
behavior, not simply new checksums over an arbitrary successful application.

See [the exception register](../upstream-patches-only-as-a-last-resort-with-explicit-user-approval/README.md),
[MoE retirement](moe-direct-slot.md), and [chat migration](chat-and-multimodal.md).

## QSA regression: unchanged output and observable optimization

Two independent native libraries are configured and linked with all-pools OFF/ON,
with file-backed PLE enabled in both. Each executable consumes the same saved
synthetic fixture. F16, Q3_K and Q4_K all pass byte-identical, finite, nonconstant
logit comparison: **56,832 float values / 227,328 bytes per fixture**.

The suite exercises prefill, reuse, threshold crossing, ranked prefill, state
save/restore, shared prefixes, sequence divergence/removal, mixed token/embedding
input, unified cache and parallel sequences. The short-prefill observations are:

| Observed node | Reference | Optimized |
| --- | ---: | ---: |
| Indexer scoring | 2 | 0 |
| Top-k selection | 2 | 0 |
| Query projection | 2 | 0 |
| Raw key projection | 2 | 2 |
| Final selection mask | 2 | 1 |

The first comparison exposed an obsolete **observer label**, not unequal logits:
v0.7.0 uses the selected tensor directly in attention and renames it from
`indexer_sel` to `kq_mask_qsa`. The callback now counts that actual ADD node. The
comparison oracle is unchanged: it still requires mask sharing, per-layer key
writes, both threshold branches and byte-identical output. Negative controls
still reject a reference executable compared with itself, missing observations,
nonfinite/constant traces and changed output. No tolerance was added.

Final inspection of saved fixture metadata found another inherited test defect:
the F16 branch skipped conversion although the upstream synthetic generator wrote
F32 weights. It now converts eligible matrices to F16 just as the Q3_K/Q4_K cases
do, and every fixture reopens its saved metadata to verify the requested type
count. The three independent-build comparisons were rerun after this correction.
This repairs the test fixture, not the runtime arithmetic or comparison oracle.

These results are native synthetic parity, not throughput, trained-model quality,
WebAssembly execution or hardware GPU validation.

## Chat validation and ownership review

The full production bridge compiles against the new real upstream headers with
both Clang and GCC using a **type-only Embind shim**. This tests constructors,
fields and adapter bodies as well as explicit overload selectors. Negative
controls reintroduce a removed parser field and arena serializer and must fail.
The existing selector suite also covers future overload ambiguity and removal.

`tests/chat-native` links the actual upstream common/chat libraries and runs 24
checks with the same pinned Qwen3 template used by the browser smoke test. These
cover segmented prompts, UTF-8 byte/token alignment, input slicing/append/prepend,
parser reuse, partial tool-call parsing, session completion, deltas and values
surviving template/session destruction. Session adapters return value copies,
not borrowed references to a mutable session.

The updated browser smoke source tests these bound values and copy lifetimes.
Its `size_t` argument checks distinguish Wasm32 numbers from Wasm64 bigints.
Only JavaScript syntax was checked locally; the real Embind module was not built.
No claim is made that a type shim validates Embind registration or marshalling.

## Reproduction (Linux, existing offline dependencies)

Run from `llama-cpp/`. The native linker fixtures currently target Linux, matching
the existing QSA and file-row tests. They do not fetch repositories or models.

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
node --test tests/*.test.mjs
python3 tests/qsa-selection/run.py --source "$PWD/vendor/llama.cpp" \
  --work "$PWD/build/qsa-regression" --jobs 2
cmake -S tests/chat-native -B build/chat-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLCB_LLAMA_SOURCE="$PWD/vendor/llama.cpp" \
  -DLCB_NATIVE_BUILD="$PWD/build/qsa-regression/native-reference"
cmake --build build/chat-native --parallel 2
ctest --test-dir build/chat-native --output-on-failure
cmake -S tests/file-backed-rows -B build/file-rows-native -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DLCB_LLAMA_SOURCE="$PWD/vendor/llama.cpp" \
  -DLCB_NATIVE_BUILD="$PWD/build/qsa-regression/native-fast"
cmake --build build/file-rows-native --parallel 2
ctest --test-dir build/file-rows-native --output-on-failure
```

Set `LCB_TEST_DAWN_PACKAGE` to the collected `.tools/emdawnwebgpu_pkg` directory to
also compile the generated WebGPU hook against the reviewed Dawn headers. This
was enabled in local Python-suite validation. `LCB_TEST_CXX` can select a compiler
for the full chat-bridge probe. Root publication/metadata tests run separately
from the repository root with `python3 -m unittest discover -s tests -p 'test_*.py'`.

## Remaining validation boundary

Emscripten 6.0.9 and browser/GPU execution were not available in the supplied
source-only collection. The existing CI must still build all five profiles in
both browser/test variants, generate actual Embind declarations, run generated
module smoke tests, and publish source-bound artifacts. Do not reuse old runtime
hashes or treat the PR source commit as an artifact commit. Consumer chat usage
must adopt the new upstream types; separate Naidan runtime pin/hash changes are
not part of this patch. Stable-diffusion source and its upstream are unchanged.
