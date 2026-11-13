# Source preparation and its evidence

## Production composition

`build_tools.runtimes.llama_cpp.webgpu_plan` owns one concrete recipe for the
accepted normal and GPU-profile compositions. It is not a plugin framework or a
language for rewriting arbitrary source. Runtime behavior and the individual
approved upstream patches remain independently reviewable.

1. The build-input capture selects the upstream Git tree and repository inputs.
2. The recipe copies the required original files, selected patches and owned
   diagnostic headers into private temporary trees. It never edits the checkout.
3. `config/webgpu-patches.json` declares the exact files each patch may change.
   The shared patch runner rejects unlisted paths, duplicate declarations,
   symlinks, binary patches and rename/copy representations.
4. Accepted patches run in the explicit order in `webgpu_plan.py`. Dedicated
   shader variants start from private source copies, not generated C++ strings.
   The IQ3 accumulator and aligned-word template have distinct patch scopes.
5. The unchanged upstream `embed_wgsl.py` generates each of the 13 final shader
   headers once, after source composition. Its private standard-library-only
   process does not import user site packages. The canonical upstream embedding
   keeps its distinct name; `ggml-wgsl-shaders.hpp` must not shadow it.
6. Every host source, shader header and the `source-preparation.json` receipt is
   prepared before replacing the previous output directory. Generation failure
   leaves the previous complete output intact. Unchanged outputs preserve their
   timestamps, including on warm configuration.

The shared output helper provides rollback for ordinary filesystem errors. It
is **not** a transaction across concurrent external readers or hostile writers.
Build isolation and immutable captured inputs remain required.

## Observations are not semantic compatibility claims

The receipt records actual original input identities, inventory identity, patch
order and before/after identities, per-header source identities, and final file
identities. It does not require the unrelated parts of a source file to retain a
hard-coded digest. A harmless comment outside a patch hunk is accepted. A changed
hunk, undeclared file edit or failed generator is not accepted.

`webgpu_provenance.py` prepares only the modes selected by built profiles. It
compares the final output and receipt with the compiled-input manifest through
the existing immutable capture/report pipeline. `config/webgpu-features.json`
documents every accepted patch exactly once and preserves the feature's purpose;
it is not another executable build registry. Per-feature compiled-copy fields
explicitly refer to the **shared final composition**, not to an imaginary
independent output of that feature or a historical intermediate stage.

The Q8 diagnostic's SHA-256 code is an owned header under `native/diagnostics/`.
It retains the public-domain implementation previously extracted from upstream
OpenCL source. It no longer depends on OpenCL source spelling or extraction
boundaries. A native test compiles the complete header and verifies known
vectors, padding boundaries and incremental updates, including under undefined
behavior instrumentation. This does not assert browser execution.

## Warm CMake configuration

The CMake hook selects exactly the upstream translation units it replaces and
fails if the upstream target layout changes. Shader-directory globs track newly
added or removed WGSL/template files. Module, patch and owned-header dependencies
also trigger preparation. Before writing Ninja's regeneration rule, dependency
paths are normalized and deduplicated. In particular, `/root/build_tools/x.py`
and `/root/runtimes/llama-cpp/../../build_tools/x.py` are the same input, not two
generated outputs.

The configuration test really runs CMake/Ninja and the production source recipe,
then adds, edits and removes an upstream shader input. It does **not** compile the
runtime: only that test's Python launcher omits the unavailable Dawn-header gate,
and its explicit configure-only targets never compile placeholder sources.
Complete prepared-source compilation is a separate test requiring real headers.

## Historical reference fixtures and independent model preparation

`tests/support/legacy_webgpu_source.py` retains the previous composer solely for
historical prefix/reference tests. Their byte guards identify those reference
fixtures. They are not production upstream-compatibility policy. Tests for the
new composer exercise actual final inputs/outputs, atomic failure handling,
private input capture, harmless source changes and exact patch scope.

Historical stage builders live only in
`tests/support/llama_cpp/webgpu_stages/`. Production modules and CMake no longer
import those helpers or enumerate their old dependency chain. CMake reads the
actual recipe receipt's input and patch inventory, then watches shader-directory
membership separately. Thus a changed recipe updates the next configuration's
dependencies without maintaining a parallel registry of historical stages.

Independent QSA-selection and file-backed-row preparation applies the existing
file-scoped patches with the common runner. It records actual input/output
identities in `source-preparation.json`, rather than reading test-owned hash
fixtures. File-backed preparation copies a coherent `src/` header/translation-unit
set: mixing its model layout with pristine sibling headers is not permitted.
The QSA source also requires its matching model header, and is excluded from a
unity build so that the source-specific include boundary remains effective.
Native numerical, address-layout and I/O tests remain separate from these
observations. A changed patch hunk still fails; harmless changes elsewhere are
not rejected merely because a digest differs.

The optional retained-BF16 mode retains its explicit reviewed-source gate. This
is an admission policy for that experimental memory-lifetime behavior, not a
claim that arbitrary source digests prove semantic compatibility. Removing it
requires equivalent ownership/lifetime coverage; directory cleanup alone does
not authorize weakening that boundary.

The final-preparation recorder test feeds current generated shaders and complete
factory headers to both Q8 and IQ3 recorders. Both require the real pinned Dawn
declarations. Their provider-dispatch functions are selected after full-unit
Clang type checking; this does not execute the entire backend or a GPU.
Historical fixed/forced and before/after comparisons use explicitly owned frozen
inputs and are not evidence for arbitrary current upstream compatibility. See
`validation-tiers.md` for the current/historical boundaries and remaining work.

Native source-bound models, complete translation-unit compilation, Wasm builds,
browser probes and real-model browser inference are distinct evidence. Do not
label one as another. Missing pinned upstreams, Dawn headers or SDKs must be
reported; an unrelated source archive or mock header is not a substitute.
