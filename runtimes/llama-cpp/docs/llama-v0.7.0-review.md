# llama.cpp v0.7.0 compatibility notes

These notes describe the transition from v0.6.0
`d81235049384534c167caea52b85a694f6103d14` to v0.7.0
`620d2225cc436ab0b001e3200e590e3b286ef9ab`. They retain the upstream API and
ownership decisions needed for consumers and later source reviews. The selected
source gitlink and current documentation govern builds; this is not a recipe for
reconstructing the retired integration.

## Upstream differences and decisions

| Area | Reviewed change | Integration decision |
| --- | --- | --- |
| MoE vector shader and dispatch | Upstream now directly indexes selected slots, guards rectangular workgroup tails, and preserves broadcast/output addressing. Multi-token gathering has separate upstream changes. | Retire the old vector-kernel overlay and all activation/include plumbing. Keep the upstream batched path unchanged. Record retirement separately from compiled overlays. |
| Qwen4Exp graph | Per-layer embedding input moves earlier in the split; multi-token prediction changes model/graph structure. The pool visibility, positive score and set-valued mask contracts required by the accepted optimization remain. | Mechanically rebase all-pools selection; preserve cache writes and ranked fallback. No widened architecture, quantization or context scope. |
| Hybrid indexed memory | Shared sequence pools, unique representative rows and state persistence changed. | Review `llama-memory-hybrid-idx.*` rather than just accepting `qwen4exp.cpp` context. Run independent state/shared-sequence/threshold traces. |
| File-backed rows | Shared model declarations and graph organization changed; accepted row-reader insertion points remain. | Keep the row patch bytes unchanged, review changed inputs, compile coherent model copies, and rerun native model/reader tests. |
| WebGPU copy and parameter batching | Backend scheduler and two-dimensional dispatch changed outside the accepted copy and batching paths. | Keep both patch payloads unchanged; review queue/fallback/header contracts and record the composed input/output identities. |
| Bounded WebGPU model upload | `load_all_data` changes the buffer collection from a map to a vector, with related mapped-range changes. | Rebase context around the existing loader helper. Retain the non-mmap/non-host/default-WebGPU restriction, block alignment, charged budget, drain and cancellation semantics. |
| Single-token SSM convolution | Host/header context changed; the selected token-width behavior remains applicable. | Keep the patch unchanged, update composed identities, and use the upstream-generated embedded shader header after MoE retirement. |
| Audio and BF16 vision | New architecture/support code appears outside the existing worker/weight-conversion insertion points. | Both existing patches still apply and remain byte-identical. Keep them independent. |
| Chat | Prompt parts, byte-indexed token-aware input, native parser arenas and session-owned continuation replace the old string/serialized-parser boundary. | Bind upstream values and public session interfaces directly. Do not recreate removed JSON serialization or flatten prompt segments at the boundary. |

See [the exception register](../upstream-patches-only-as-a-last-resort-with-explicit-user-approval/README.md),
[upstream MoE selection](moe-direct-slot.md) and [chat API](chat-and-multimodal.md).

## QSA observation and fixture requirements

The upstream selection tensor changed from `indexer_sel` to `kq_mask_qsa`.
Observe the actual final selection rather than a stale label or per-stream
reshape alias. The native comparison requires graph-local mask sharing,
per-layer key writes, both threshold branches and finite, nonconstant,
byte-identical reference/optimized outputs. Comparing the reference with itself
must fail the graph-coverage check.

A requested fixture quantization must be checked in the saved GGUF itself.
An F16 label alone does not establish that a generator converted F32 matrices.
The current QSA runner tests independent F16/Q3_K/Q4_K builds and fused/unfused
masks; see [the reproducible native gate](qsa-all-pools-selection.md).
These small synthetic models do not establish trained-model quality, throughput
or GPU correctness.

## Chat ownership and compatibility

The public chat API uses segmented prompts, byte-indexed token-aware inputs,
native parser arenas and session-owned continuation. Retain parser values as
Embind objects and release them with `delete()`. Do not reconstruct removed JSON
parser serialization or flatten prompt segments unless a consumer explicitly
wants text-only output. Session adapters return owned value copies rather than
references into mutable session state.

Explicitly selected C++ overloads establish which upstream signature the bridge
uses. A native type shim can check those signatures and adapter bodies; actual
Embind marshalling still requires the generated Wasm module. Browser smoke uses
the selected artifact's bindings and the matching template. Native chat, API
compilation and browser smoke remain separate evidence.

## Current verification

Use [verification](../../../docs/verification.md) for emitted Node tests, native
fixtures and saved-package browser validation. Linux native model fixtures require
the selected upstream and prepared native libraries but download no model weights.
A source commit is not an artifact commit, and local source/native results cannot
replace the actual published module/schema pair. The image runtime retains its
own independent source and validation.
