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
within a graph only through the same pool-input object and the same mask type.
The pinned upstream derives selection from pool visibility and tail membership;
it no longer takes a separate base attention mask. Flash attention uses an F16
selection, while the other path uses F32. The selected rows become one contiguous
two-dimensional mask before upstream reshapes it for the K/V cache streams.

It does **not** replace sparse masking with an unconditional dense mask. The
original live/dead redirection, unique dump rows, sequence visibility, cell holes,
incomplete tail and upstream mask layout remain. Every layer still
updates raw and pooled key caches before entering the fast path. Once the padded
pool count exceeds the budget, the original scoring, fusion metadata and ranking
path runs, including upstream's device-probed TOP_K_MASK and its scatter
fallback. No cache format or public runtime ABI (Application Binary Interface)
changes. No context-length, tool, grammar, quantization or sampling compromise.

## Build composition

`LCB_QSA_ALL_POOLS` is independent of `LCB_FILE_BACKED_PLE`. The checked overlay
accepts either the original translation unit or the file-backed
translation unit. It preserves that source's model header include directory and
excludes the source from unity builds. The vendor checkout is not edited.
`source-preparation.json` records the actual patch, input and output identities.
A changed patch context fails preparation. C++ compilation and the native
reference/optimized regressions check type and behavioral compatibility; a
comment change alone does not require refreshing hand-maintained hashes.

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
to one input token; the test does not bypass that upstream restriction. `compareQsaTraces` in the typed regression runner requires byte-identical finite logits
and checks that the executed native graph actually skipped scoring below the
threshold and resumed it above. Comparing a baseline binary with itself fails
that graph-coverage requirement.

The runner checks both the native fused mask and an explicitly forced unfused
mode in each independently built implementation. The census observes the actual
CONT or TOP_K_MASK selection rather than its per-stream reshape aliases. It
requires fused execution in the fused reference and ranked phases, rejects it in
the unfused mode, and also requires byte-identical fused/unfused output traces.
The forced mode uses the pinned context only inside the test executable before
its first decode; it adds no production setting.
An additional F16-weight control explicitly enables flash attention, requires
observed selection masks to be F16, and repeats both comparisons. Native AUTO
policy alone is not evidence that the F16 mask path ran.

The recorded node census is not an individual-operation timing. Do not use its
run time as a clean performance benchmark. `benchmark` omits the census; even that
is a synthetic native workload, not an Apple WebGPU or full-model speed result.

## Upstream coupling

The proof depends on nonnegative scoring, visibility semantics and set-valued
selection in the pinned Qwen graph. Review those properties when updating the
upstream; successful patch application or a recorded digest cannot prove them.
The longer-context WebGPU LIGHTNING_INDEXER prototype is separate. Its isolated
shader tests do not establish integration into the distributed C++ backend.

## Reproducible required native gate

On Linux with the pinned submodule, Node dependencies and native compiler
already available, run from the repository root:

```sh
npm run tooling:build
node build/tooling/tests/runtimes/llama-cpp/tools/qsa.test-support.js \
  --checkout-root "$PWD" --tooling-root "$PWD/build/tooling" \
  --output-root /tmp/bicore-qsa-regression --toolchain-root "$PWD/.tools" \
  --local --jobs 2
```

The runner downloads nothing, configures independent reference/optimized builds,
links two different libraries, and generates local fixtures for F16, Q3_K and
Q4_K. Commands, exit statuses, fixture/library identities, finite output traces
and graph censuses are saved in `<output-root>/qsa-results`. `status.json`
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

The runtime build selects QSA explicitly in its typed preparation options. The
prepared-source receipt identifies the applied patch, final coherent source tree
and selected options. Freeze reconstructs the expected preparation for each
profile and compares it with the variant's recorded preparation and actual
compiler inputs. File-backed PLE and QSA compose in the same private source tree;
the final Qwen translation unit includes both selected changes.

Disabling QSA is an explicit reference/diagnostic build option, not a model-name
heuristic. Source and artifact digests identify bytes; consumers update their
pins only after building and validating real distribution artifacts.
