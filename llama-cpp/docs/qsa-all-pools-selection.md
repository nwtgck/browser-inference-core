# All-pools QSA selection

QSA is Qwen Sparse Attention. This optimization is in the architecture's graph,
not a model repository name, filename, quantization, browser, or device check.

## Exact scope

When the padded candidate pool count does not exceed the selection budget, all
pools are selected. The fast path also requires `n_kv + n_slots <= 2^24`,
so reordering remains exact through the original F32 dump-row redirection.
Larger shapes retain the original scoring/ranking path. The pinned graph has nonnegative visible scores (positive
head weights multiplied by rectified dot products), and its visibility input is
zero or negative infinity. After selection, only live/dead membership remains.

The optimized graph substitutes visibility for scoring, repeats the pool member
table without sorting or indirect gathering, and shares the final selection mask
within a graph only for the exact same base attention mask input tensor.

It does **not** replace sparse masking with an unconditional dense mask. The
original live/dead redirection, unique dump rows, sequence visibility, cell holes,
incomplete tail and final attention mask addition remain. Every layer still
updates raw and pooled key caches before entering the fast path. Once the padded
pool count exceeds the budget, the original scoring, fusion metadata and ranking
path runs. No cache format or public runtime ABI (Application Binary Interface)
changes. No context-length, tool, grammar, quantization or sampling compromise.

## b11804 fused-mask integration

The b11804 graph shares `new_pool_flat` across QSA layers and selects a fused
`TOP_K_MASK` only after probing backend support. The all-pools shortcut retains
the shared input and every cache write; it skips the fused ranking path only
when every pool is selected. Ranked selection still uses the upstream fused
mask when supported and the original gather/scatter fallback otherwise.

The native regression separately counts fused masks. A reference run that
silently loses fused-mask coverage is rejected even when its logits match.
A test-only `trace-unfused` mode disables the mask before first decode in each
native context, covering the ranked fallback without changing the distributed
runtime or turning off graph reuse. Both modes run on each fixture; the explicit
`--mask-mode unfused` oracle rejects any fused execution, and fused/unfused
outputs must agree within each independently compiled implementation.
The source checks also cover the fusion probe and mask construction dependencies;
CMake watches the same files so a warm build cannot bypass renewed review.
See [the b11804 review](llama-b11804-review.md) for the exact input identities and
remaining WebAssembly/browser validation boundaries.

## Build composition

`LCB_QSA_ALL_POOLS` is independent of `LCB_FILE_BACKED_PLE`. The checked overlay
accepts either the pinned original translation unit or the reviewed file-backed
translation unit. It preserves that source's model header include directory and
excludes the source from unity builds. The vendor checkout is not edited.
`tests/qsa-selection/inputs.json` records both input and output identities plus
semantic dependencies. Upstream changes deliberately require renewed review.

Disable the option explicitly for the reference build. Build and link reference
and optimized executables separately; changing a runtime label on one binary
would not constitute a comparison. Distribution build commands state the option
explicitly so provenance does not infer it from a default.

## Regression scope

The native fixture has four layers (two QSA attention layers), representative
Q3_K or Q4_K matrix quantization, an IQ4_NL file-backed embedding table, and an F16
control. It is a small synthetic model, **not** the real UD-Q3_K_XL or UD-Q4_K_XL
weights. Its 64-pool budget deliberately crosses the upstream padded-pool boundary.

The trace covers prefill, one-token reuse, a live boundary crossing, state restore
and re-crossing, copied-prefix sequence divergence/removal, EOS boundaries, and
mixed token/embedding input, plus simultaneous distinct sequences on a unified
cache after a supported prefix copy. PLE forbids assigning multiple sequence IDs
to one input token; the test does not bypass that upstream restriction. `compare.py` requires byte-identical finite logits
and checks that the executed native graph actually skipped scoring below the
threshold and resumed it above. Comparing a baseline binary with itself fails
that graph-coverage requirement.

The recorded node census is not an individual-operation timing. Do not use its
run time as a clean performance benchmark. `benchmark` omits the census; even that
is a synthetic native workload, not an Apple WebGPU or full-model speed result.

## Explicit upstream exception

The user requested continuation of the all-pools optimization previously supplied
as snapshot 008 and permitted bicore changes for Q3/Q4 and other models. This
small, hash-checked build-copy patch is within that requested scope. The source
layout and semantic hashes must remain fail-closed. The longer-context WebGPU
LIGHTNING_INDEXER prototype is separate; its standalone shader tests do not prove
integration into the distributed C++ backend.

## Reproducible required native gate

On Linux with the pinned submodule initialized, from `llama-cpp` run:

```sh
python3 tests/qsa-selection/run.py --source vendor/llama.cpp --work build/qsa-regression --jobs 2
```

The runner downloads nothing, configures independent reference/optimized builds,
links two different libraries, and generates local fixtures for F16, Q3_K and
Q4_K, each with fused and explicitly unfused masks. Commands, exit statuses, fixture/library identities, finite output traces
and graph censuses are saved in `build/qsa-regression/qsa-results`. `status.json`
is authoritative for the most recent run; a failed rerun does not make earlier
comparison files valid. Compiler intermediates are reusable, but configuration,
fixtures and traces are not skipped. Constant or nonfinite model outputs, a
baseline compared to itself, missing phases and changed observation schedules
are rejected. Capturing metadata does not request per-node completion. The census also
requires fewer final selection masks, proving that graph-local sharing executes
rather than merely being present in source code. An independent F32 arithmetic
test exhibits the rounding counterexample beyond the exact-index range.

The `qsa-native` workflow job runs independently of WebAssembly compilation and
is required before packaging. It uploads comparison results even on failure,
not the native build trees. This gate is native CPU correctness coverage, not
WebGPU numerical validation or a throughput claim.

## Composed-source provenance

Every distribution variant explicitly states `LCB_QSA_ALL_POOLS=ON/OFF`.
Reporting rejects missing, repeated, typed or otherwise ambiguous activations.
The report prepares the same reviewed source composition as CMake and records
the **final compiled** Qwen translation unit and owning model header for each
profile/variant. For file-backed rows followed by QSA, the coherent source-set
hash uses the QSA result rather than its intermediate input. Compact single-set
fields are emitted only when all affected variants compile the same set;
mixed activations use the per-variant maps. The upstream updater preflights
both original-header and file-backed-header compositions before proceeding.

Disabling QSA is an explicit reference/diagnostic build option, not a model
name heuristic. No generated runtime commit or consumer integrity hash is
invented by this source patch; consumers should update their pins only after
building and validating real distribution artifacts.
