# Validation boundaries

Validation should establish useful properties of the runtime and its artifacts.
Source identity, successful compilation, native behavior, Wasm interoperability,
and model performance provide different evidence. Report the boundary exercised
by each check; do not infer a stronger result from an unrelated passing test.

## Source preparation and identity

Production preparation uses the selected checkout's typed tools and isolated
compiler inputs. Source preparation tests cover patch order and scope, coherent
headers and translation units, unchanged upstream checkouts, output preservation
on failure, and unchanged-input timestamps. These common properties do not need
separate source-text guards for every optimization.

Preparation receipts and compiler-input records identify the bytes that were
prepared and compiled. They do not prove numerical correctness or compatibility
merely because their hashes match. Ordinary compilation checks the actual
implementation rather than extracted copies of function bodies. Harmless source
comments must not require updating a semantic acceptance digest.

## Types and compiler facts

The ABI generator reads declarations, evaluated constants, record layouts, and
target properties from the actual compiler configuration. Its checks connect C
and C++ declarations, compiled exports, host conversions, generated TypeScript
interfaces, and package exports. Missing or unsupported public types must produce
an actionable error rather than silently disappearing from the public interface.

Generated package consumers are checked through public imports. Positive calls
must type-check, and wrong names or argument types must fail. Handwritten types
or consumer casts must not hide a mismatch in generated declarations.

These checks establish signatures and layout, not the behavior of every exported
function. A successful compiler invocation does not establish model quality or
GPU throughput.

## Native behavior

The native suites under `tests/runtimes/llama-cpp/` and
`tests/runtimes/stable-diffusion-cpp/` compile prepared implementation translation
units against the SDK declarations they use. Controlled external API observers
exercise host decisions, ownership, queue ordering, submission, cache reuse,
uploads, and readbacks without requiring a physical GPU.

Assertions must remain active in Release builds with `NDEBUG`. Observers must
preserve the relevant external API semantics, including work-completion prefixes
and deferred resource lifetimes. Passing a simplified model that behaves
differently from the API is not evidence about the real boundary.

Direct counts can establish useful host properties, such as no unnecessary wait
or allocation on a warm path. They do not establish hardware memory traffic or
model tokens per second. C++ compilation of an embedded WGSL string also does
not compile or execute that shader.

## Wasm and consumer boundaries

Representative Wasm tests exercise actual pointer widths, suspension modes,
integer and boolean conversions, records, callbacks, allocator conventions, and
host ownership gates. Native calls to the original C functions are a different
boundary and cannot replace these checks.

A local package containing selected profiles is useful for focused compiler,
public-import, and runtime checks. It must remain visibly local and incomplete;
do not fill missing profiles with substitute payloads or weaken publication
requirements to label it a complete distribution.

Browser smoke tests run the packaged files and record which scope was exercised.
A missing-adapter suspension test establishes suspension and failure handling,
not WebGPU numerical inference. A tiny synthetic model establishes its tested
path, not support or performance for unrelated production models.

## Artifact and publication checks

Package validation checks the actual file inventory, ABI and factory mapping,
licenses, profile and variant requirements, and source provenance. Publication
binds the validated tree to immutable Git objects and the resolved consumer
package. Independent saved input is decoded at its entry; validated typed values
are then passed directly through internal functions.

Tests cover missing or altered payloads, failed copies and rollback, source and
Git identity mismatches, publication conflicts, and consumer-resolution errors.
Repeated decoding or rescanning is not a substitute for keeping these operations
bound to the same validated inputs.

## Numerical and performance evidence

Use numerical tests that exercise the implementation whose result is claimed.
An independent CPU reference can supply expected values, but executing only that
reference does not validate a GPU shader. A translated shader model, source-byte
comparison, or host dispatch observer must not be reported as GPU execution.

Generation throughput and prefill measurements need representative inputs and
comparable execution conditions. Distinguish those measurements from build time,
isolated host-call overhead, and native correctness results. Unavailable hardware
or unsupported execution environments are limitations to report, not passing
performance checks.

## Historical references and test selection

Keep a historical reference only when it supports a useful comparison. Own its
input, expected result, license, and provenance explicitly. Its digest may
identify that fixed input, but must not become an acceptance condition for an
arbitrary current upstream source tree.

Tests are judged by the regressions they detect and their maintenance cost.
Remove redundant checks and tests of discarded implementation mechanisms when
they no longer protect a useful contract. Do not preserve an obsolete generator,
interpreter, or full reference environment solely to reproduce its old test
results. Retained behavior coverage and the reasons for removal should be
reviewable without requiring knowledge of a development conversation.
