# Current contracts and historical references

Production source preparation must not import `tests.support` or require a
historical whole-file digest to accept a source change. Input/patch/output
identities in preparation receipts describe provenance, not semantic validity.

## Current-source checks

`tests/runtimes/llama_cpp/source/test_plan.py` checks the actual production plan,
its declared patch order, input capture, atomic output, and change detection.
`source/test_full_headers.py` compiles the complete current shader and optional
pipeline headers and backend translation units against the real pinned Dawn declarations.
The provider check poisons the obsolete canonical header and verifies that only
the current generated embedding is included. Its mutations prove
that unrelated comments pass while invalid unreferenced declarations fail.
Neither native syntax checking nor a mocked GPU API is browser inference.

`test_webgpu_source_overlay.py` checks the current CMake integration against a
fresh direct preparation of the same current input. That comparison checks graph
selection and source identity; it is not an independent numerical oracle. The
native semantic tests, source-plan mutations, and browser matrix are separate.
These current-source checks use the selected `LCB_TEST_LLAMA_SOURCE` or checkout.

The host CI job acquires only the small pinned Dawn archive online, verifies and
unpacks it inside its explicit offline container, and runs the header contracts.
It does not install Emscripten. Local absence of optional tooling can produce a
reported skip; the complete-header test refuses missing inputs when `CI=true`.
Emscripten compilation/linking and actual browser execution remain separate CI
obligations and are not claimed by native results.

### Current Q8 behavioral recording

The short-Q8 and Q8 subgroup emitters include the complete prepared shader
library and the pinned Dawn C++ declarations. They mock only C WebGPU entrypoints
using the same owned header as the optional-pipeline recorder. Factories, cache
keys and selectors are not reconstructed, reversed to a historical prefix, or
silently redirected to frozen input. Existing geometry/cache/shader assertions
remain in the native executables. A harmless current-source comment succeeds;
an invalid unused declaration fails compilation. Native dependencies are system
headers, but the generated library and recorder retain `-Werror`.

For the small host-dispatch calculation, Clang type-checks the complete current
provider, then supplies physical AST source ranges. The contract identifies the
workgroup branch by declarations/references, not comments or brace searches.
Normal and diagnostic-template definitions are selected explicitly. Ambiguous
branches/declarations, macro-expanded boundaries, foreign-file definitions and
invalid unrelated code fail rather than producing a plausible fragment. Byte
ranges retain correct positions with non-ASCII source comments. The compiler's
active `CEIL_DIV` definition accompanies the range, instead of substituting the
recorder's handwritten formula; ambiguous macro lifetimes are rejected. This is still
execution of a dispatch fragment, not a full backend or GPU execution; complete
provider syntax checks and browser obligations remain separate.

The Q8 standalone commands record observed patch identities from the actual
preparation receipt. They do not report a historical constant as a current input
identity. Clang and Dawn are required for these native contracts, not Emscripten.

### Current Q4/Q6 behavioral recording

`test_quant_matvec_recording.py` exercises the complete current header in normal
and diagnostic modes. It retains Q4 shape/capability/batch admission checks,
current pipeline-cache reuse, effective workgroup decisions, shader construction,
and AST-selected grid coverage. Q6 checks cover short widths, invalid dimensions,
MMVQ/overlap/indexed exclusions, dtype and broadcast boundaries. The public
shader-library entry points run unchanged; only C WebGPU calls are recorded.
There are no reverse patches or factory/body slicing in this recorder.
Subgroup-capable and scalar-only devices have separate caches; capability changes
are not simulated by reusing one device cache with contradictory contexts.

Harmless source comments pass without historical hash updates; an invalid unused
declaration fails compilation. Normal and diagnostic shaders are compared with
each other, not a historical digest. This establishes configuration equivalence,
not GPU numerical correctness. Historical byte/fallback comparisons remain
separate and retain their original assertions.

`CurrentQ6Subgroup` runs the complete optional-pipeline recorder independently of
historical preparation. A broken historical prefix cannot prevent current tests
from setting up.

### Current short-Q4 and IQ3 standalone recording

The short-Q4 and IQ3 recorders now compile the complete current shader library
and real Dawn declarations. The shared `current_matvec.py` helper type-checks the
whole provider before selecting dispatcher definitions by Clang source ranges.
It preserves the provider's active `CEIL_DIV` macro. The runtime-specific cases
retain all existing selector, cache, fallback, parameter, binding and grid checks.
Ordinary and indexed Q4 dispatch are both exercised. In the diagnostic build,
the ordinary entry point calls the original primary template (with its actual
template declaration and context builder), not a copied normal-build body.
This does not run GPU profiling or the diagnostic device command itself.
This is still execution of
selected dispatcher functions with a resource-boundary model, not a whole GPU
backend or device execution.

The fallback comparator includes the complete unmodified upstream shader library
in its own namespace, using the same current common shader input as the candidate.
An include adapter changes only lookup, not either factory body. Different C++
decision types are translated by named fields rather than aliased across types.
Observed preparation and compiler receipts accompany the real returned executable.
Standalone runners must not label frozen fixture hashes as current provenance.
Report-wiring tests consume real production preparation outputs, so inventing
a matching but nonexistent embedded-header filename cannot hide a broken path.

The IQ3 subgroup/field-load/row-split historical classes now own frozen source and
patch inputs through `historical_source()`. Existing inverse patches, golden bytes,
rejection cases and numerical comparisons remain explicit history checks. The
current recorder independently covers normal/diagnostic preparation, unrelated
comments and invalid unused declarations. Its setup does not run a frozen prefix.

### Current CPU numerical references

The Q4 short checker and IQ3 fixture generator compile the complete unmodified
`ggml-quants.c` translation unit and include its actual public declarations.
Function/data sections and linker garbage collection discard unused compiled
routines; they do not slice source before type-checking it. Existing known-block,
shape, dtype and float64 accumulation checks remain in their original harnesses.
A comment change passes, while an invalid unused declaration outside the selected
dequantizer fails compilation. The native checks do not execute WebGPU shaders.

The compilation receipt records observed input/harness/binary identities, commands
and compiler dependency data, not a frozen semantic acceptance hash. A source-only
capture reports no Git commit; it must not borrow its containing repository's HEAD.
The checker reads the receipt next to the executable actually returned by the
compiler helper. Its mocked CLI regression checks report wiring, not inference.

### Current copy, host snapshots and scheduler

`test_current_copy_recording.py` type-checks the complete current prepared
WebGPU provider, prepared generic backend, and unmodified CPU backend before
selecting the copy/snapshot functions. It executes the original functions in
an owned deferred-queue/resource model in normal, batched and diagnostic modes.
Ordinary copies use the prepared backend as well, not an old pristine scheduler.
The original ordering, borrowing/lifetime, fallback, registry-cache, first-input
barrier, bounds and failure cases remain. This is selected-function native
behavior, not execution of the complete backend or a GPU.

`copy_ast.py` matches the actual interface record fields to the semantic aggregate
initializer, not comments or fixed positions. Scheduler discovery is selected
by references to the actual parameter/field declaration IDs. Ambiguous, foreign,
macro-expanded or indirect registrations fail rather than selecting a plausible
body. All selected sources and directly captured headers must remain unchanged.
The same tools reject unused type errors outside selected functions and accept
unrelated source comments. They do not claim an atomic snapshot of every include.

The diagnostic registry also exposes a Q8 diagnostic entry. Its real typedef and
active macro accompany a typed, fail-on-call boundary. The registry branch is
kept; diagnostic kernel execution is deliberately outside this queue model.
Missing native dependencies fail when `CI=true`; local absence is a reported skip.
`source/test_copy_ast.py` covers actual Clang declarations, shadowing, field order,
UTF-8 ranges, macros, ambiguities and source mutation with small standalone inputs.

The old copy and snapshot byte guards, inverse/prefix comparisons and lifecycle
regressions remain explicitly historical. Their source and patches come from the
licensed fixed archive, including the CPU registry checked against the original
contract. The old full-syntax checks moved to current complete translation units;
they were not redirected to old source and relabeled as current coverage.

## Historical checks

`source/test_historical_guards.py` checks the byte guards and native arena
regressions of the old staged composer. Its source, accepted patches, and typed
bridge header come from the compact, licensed fixture under
`tests/fixtures/llama-webgpu-history-cd959bae/`. They do not follow an environment
variable or the current vendor checkout. Output digests are meaningful for that
fixed reference only. Private extraction prevents one test mutating another.

The historical aligned-word regression also owns this frozen input, including
its compiled shader-string and bounded byte-oracle comparisons. Its former live
provider check moved to `source/test_full_headers.py`; current standalone
preparation remains in `source/test_recording_inputs.py`, in a separate class
with live inputs. Do not describe the historical oracle as current GPU execution.

The positive reference check covers both normal and diagnostic composition; the
archive's inventory/bytes are checked before use. It contains no Git repository
and does not fabricate the provenance of the original source-only input.

Short-Q8 and Q8-subgroup byte guards also own that private fixture and its
patches. The optional-pipeline recorder builds its before-state from the owned
historical input, while its candidate and dispatch remain current. Their passing
reference comparisons are not relabeled as current-upstream acceptance.

Q4-subgroup, Q6-short and Q6-subgroup prefix/inverse/byte guards now use the owned
historical source and patch set explicitly. Their native before/after recorders
are historical comparisons, not claims about arbitrary current upstream input.
The fixture additionally owns the public WebGPU header needed to compile these
historical providers; all pre-existing fixture files remain byte-identical.

## Remaining migration

This separation is not complete for every historical prefix helper. Several
stage-specific test modules still couple historical identities or fragments to
live inputs. Snapshot/scheduler selections now have the current/historical
boundary described above; several other specialized stages remain to classify.
They must be migrated with their semantic coverage retained, not mass-skipped,
blindly rehashed, or redirected to old inputs and called current compatibility.
No production function body or accepted upstream patch was changed for the new
validation tiers. Keep this limitation visible until those migrations finish.
