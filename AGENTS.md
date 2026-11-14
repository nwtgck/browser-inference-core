# Browser inference runtimes

Runtime-specific source, bindings and profiles belong under `runtimes/`.
Shared development tools belong under `tools/`; runtime-specific typed adapters
belong under `runtimes/<runtime>/tools/`. Shared tests live under `tests/tools/`,
and runtime tests/harnesses under `tests/runtimes/`, not a second test tree inside
`runtimes/`. Use the targeted verification commands below. Local native/host
checks do not require Emscripten; leave the browser-build matrix to CI when that
toolchain is absent.
Do not merge Wasm instances or upstream version pins between runtimes.

Follow the runtime-specific instructions when modifying upstream code. Approval
for one runtime does not authorize changes to another runtime.

Never edit a checked-out upstream in place or misrepresent compilation/mock tests
as real-model browser inference. Record patch identities and upstream commits.
Publication must be append-only, source-bound, and contain runtime assets only.

Put disposable compiler probes and experiments in a dedicated directory under
`/tmp/`, with an explicit working directory and output paths. Do not let default
outputs such as `a.out*` appear in the repository root, or add ignore rules merely
to hide them. Ordinary documented build outputs still belong in their configured
build directories.

## Maintainability and source checks

Preserve functionality and performance while reducing the knowledge, coupling,
and repeated maintenance needed for upstream updates. Existing implementation
choices and green tests do not by themselves establish that a design is
appropriate. Do not make a refactor pass by silently narrowing its supported
behavior or weakening its meaningful tests.

Write repository documentation for readers who do not know the current
conversation. Describe the supported behavior, how to use it, and durable design
reasons. Keep temporary plans, abandoned approaches, and intermediate migration
history in working notes. Retain historical explanations only when readers need
them, such as migration instructions for an actually published interface.

Make temporary scaffolding mechanically discoverable while it is needed.
Before finishing a change, inspect its markers, forwarding files, and obsolete
documentation together. Remove transitional artifacts that no longer serve a
purpose; explain any deliberate ongoing compatibility or build boundary in the
code that owns it. Do not rely on remembering the conversation to distinguish
intentional structure from migration leftovers.

Before adding a check, state which property it establishes and which change it
should detect. Prefer ordinary types, compilation of the actual implementation,
and focused structural or behavioral tests. A harmless source edit should not
require updating constants or rewriting a test's copy of the implementation.
Do not use whole-file hashes, exact source text, function-body extraction, or
substring counts as substitutes for API, ownership, layout, or numerical checks.
Do not mechanically regenerate expected digests to accept an upstream update.

Hashes remain useful for source and artifact identity, provenance, and transfer
integrity; these are different responsibilities from proving behavior. A frozen
historical comparison belongs to its owned fixture, not to current-source
admission. If a whole-input hash is genuinely the last viable guard, explain
why ordinary checks cannot establish the required property. Keep original-source
acceptance hashes in one file-path-to-hash JSON inventory per upstream runtime,
with the rationale and conditions for re-auditing documented alongside it;
do not scatter them through scripts or tests.

Keep upstream modifications as individual, reviewable `.patch` files. Write
owned C/C++ implementation code as ordinary source files compiled normally;
do not assemble current implementation bodies in scripting-language strings or
extract upstream bodies into substitute current-source translation units.
Genuinely unavoidable source generation is a last resort: explain why ordinary
sources and build mechanisms cannot achieve the required result. Small templates
for data constants or externally required declarations need a concrete purpose
and must not become another hidden source-rewriting system. Historical helpers
must remain owned by their frozen fixtures, never used as the current
implementation. Choose techniques for the property being checked, rather than
mandating one technique everywhere.

## Static types and runtime boundaries

Connect internal stages with typed function calls and values. Do not introduce
child processes, intermediate JSON files, serialization round trips, or repeated
validation merely to separate implementation modules. Use the current checkout's
tools; running a different revision requires selecting that checkout first.

Runtime decoding needs a concrete input and failure model beyond the presence
of a file, JSON, or another process. External API and compiler/tool output, or
an independently supplied saved artifact, can require decoding. Owned static
configuration and values just constructed by typed producers do not. Decode a
necessary external input at its entry, then preserve its type internally. Do not
replace removed validation with unchecked casts or duplicate hidden checks.

DTOs describe structures, not domain restrictions. Keep their definitions in
the root `00-dto/` tree, expose types and named decoders through its public
entries, and keep raw schemas internal. ESLint permits Zod imports only within
`00-dto/`; do not maintain a second registration list of individual DTO files.
Do not scatter new schemas or local schema extensions through consumers. A new
schema still needs a justified decoding boundary. Check actual files, paths,
bytes, and publication conditions at the operations responsible for them.

## Runtime behavior and performance

Review ordinary generation throughput and prefill time, not only build and CI
duration. Preserve the intent of existing kernel selection, fusion, cache reuse,
uploads, readbacks, synchronization, file-backed storage, and profiling-off paths.
Check the applied sources, shader generation, build options, and host ABI calls;
retaining a patch filename alone does not prove that an optimization is active.
Do not add per-token Promise layers, conversions, allocations, logging, or waits
without checking their necessity and cost.

Keep C declarations, Wasm exports, JS adapters, generated TypeScript declarations,
and package exports consistent. Native calls to the original C functions do not
test the JS/Wasm boundary. Validate real generated package consumers rather than
handwritten substitute types. Distinguish source-level evidence, native
correctness, isolated JS overhead, and measured model inference performance;
never claim t/s or prefill preservation from indirect evidence alone.

## Verification

Before each commit, run the affected lint, strict type checks, and relevant
tests. Running the affected tests is mandatory before every commit. Limit lint
fixes and tests to the affected scope; directory targets and explicit file
targets are both appropriate. Update callers and test support when a shared
contract changes, and include the relevant consumers in verification.

For changes to source selection, compile definitions or CMake wiring, verify an
actual production configuration that activates the affected path and compile
the affected target or translation unit with its generated build settings.
Native fixtures or another profile do not establish coverage of different active
sources, platform branches or include paths. If prerequisites prevent this check,
identify the unverified path and missing inputs explicitly. Choose representative
configurations for the changed behavior; a full local matrix is not required.

- Use `npm run lint -- --fix <affected paths...>` for owned JS/TS changes.
- Use `npm run tooling:build` when emitting TypeScript for Node tests; it also
  performs strict type checking. Use `npm run tooling:check` when only checking
  types is needed. Avoid checking and emitting the same unchanged input twice.
  A targeted TypeScript configuration is appropriate for isolated changes.
- Run the affected emitted Node test files with `node --test <test paths...>`.
  Run `npm run test:lint` when changing local lint rules or their configuration.
- Run the related native build/tests for affected C/C++ code or build recipes;
  format owned C/C++ files as specified below. Do not substitute a broad build
  for identifying the checks that exercise the changed behavior.

Use non-interactive test commands that exit after completion; do not start watch
mode. Preserve meaningful test coverage during refactoring: adapt or migrate
tests to the new structure rather than silently dropping their guarantees.
When replacing a misleading implementation-text check, explain the property
that the replacement actually verifies. Report failed, skipped, and unavailable
checks accurately; an unavailable runtime is not a passing test.

Judge tests by the behavior and contracts they protect and the regressions they
can detect, not by test counts or the implementation history of the test suite.
Remove excessive, redundant, or obsolete tests when their maintenance cost is
not justified by a meaningful regression they detect. Checks of discarded
generators, fixed source text, or historical implementation details need not gain
replacement machinery. Explain retained guarantees and removal decisions; do not
silently discard useful behavior coverage merely to obtain a passing result.

## Shared code and change propagation

Decide whether to share code by asking whether a change is intended to propagate
to all consumers. Share an implementation when its consumers represent the same
rule or responsibility and should evolve together. Keep independent policies
separate when a change for one consumer should not affect the others, even if
their current implementations look similar.

Do not deduplicate code by matching similar patterns like cards in a memory game.
Syntactic similarity alone does not establish shared meaning. Merging unrelated
behavior creates unintended coupling: a local change can then break otherwise
independent features. Before extracting shared code, inspect its consumers and
identify which changes should, and should not, propagate between them.

## Test resource management

Actively use `using` and `await using` in Node.js-only `*.test.ts` files when a
resource's lifetime matches its lexical scope. These declarations are forbidden
in all other files, including `*.test-support.ts` and development/build tools.
Definitions and uses of `Symbol.dispose`, `Symbol.asyncDispose`, and disposal
runtime APIs are allowed only in `*.test.ts` and `*.test-support.ts`. Use the
exact `*.test-support.ts` suffix for test-only helpers, and never import test-only
modules from production code.

Do not use these declarations or disposal symbols in `.js`, `.mjs`, or `.cjs`
source files, browser-targeted TypeScript, or modules shared between Node.js and
browsers. Transpiling TypeScript does not by itself guarantee browser support.
Keep Node.js-only disposal helpers out of browser runtime dependencies.

Preserve cleanup timing, order, rollback, and earlier release where required;
keep explicit control flow when disposal alone does not express those semantics.
Do not add redundant scopes, thin callback adapters, or disposal polyfills just
to use this syntax.

## Function signatures and named arguments

Prefer `for...of` to `forEach` for iteration so that `continue`, `break`,
`return`, and `await` retain their ordinary control-flow meaning.

For new or refactored project-owned TypeScript callables, use one mandatory,
destructured object parameter, even for a single argument. Zero-argument
callables may use `()`. Avoid parameter and property defaults; pass explicit
values, including `undefined`, so the caller's intent remains visible.

Write an explicitly annotated outer argument shape inline. Do not hide it behind
an alias, interface, schema-inferred type, or utility type such as `Pick` or
`Parameters`. Shared property types may reference existing definitions. When the
function consumes one cohesive value, wrap that value under a meaningful name
instead of copying its fields into an argument bag. Contextually typed
implementations may infer parameters from a canonical inline signature.

```ts
function read({ path, signal }: { path: string; signal: AbortSignal | undefined }) {}
function copyPublication({ publication }: { publication: Publication }) {}
```

Preserve genuine external calling conventions. Type predicates may keep their
positional parameter when needed for narrowing. Platform callbacks and methods
implementing external interfaces may remain positional when their external
contract is verifiable from TypeScript types. An external parameter type or a
name such as `callback`, `resolve`, or `handler` is not sufficient evidence.
Prefer existing external callable types over handwritten callback signatures.

Do not remove annotations, add casts, or weaken a shared contract to bypass the
rule. Use focused lint suppressions only for a specific external, deprecated,
or runtime-only calling convention that cannot be expressed through its types,
and explain the reason. Keep argument-shape changes and callers consistent;
preserve optional fields, union correlations, behavior, and existing layout.
Prefer `undefined` for absent values unless an external contract requires
`null`. See `.codex/skills/named-args-lint/SKILL.md` for fix guidance.

## C and C++ formatting

When modifying project-owned C or C++ source files and headers, format them with
`clang-format-17` using the `.clang-format` configuration in the repository root.
Only format `.c`, `.cpp`, `.h`, and `.hpp` files in these directories:

- `runtimes/llama-cpp/native/`
- `runtimes/stable-diffusion-cpp/native/`
- `tests/runtimes/llama-cpp/`
- `tests/runtimes/stable-diffusion-cpp/`
- `tests/runtimes/llama_cpp/`
- `tests/runtimes/stable_diffusion_cpp/`

From the repository root, format all files in scope with:

```sh
find runtimes/llama-cpp/native runtimes/stable-diffusion-cpp/native tests/runtimes/llama-cpp tests/runtimes/stable-diffusion-cpp tests/runtimes/llama_cpp tests/runtimes/stable_diffusion_cpp -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' \) -exec clang-format-17 -i --style=file {} +
```

Do not format upstream sources, `.patch` files, or generated code. Avoid
unnecessary formatting differences from upstream.

The `.cpp.in` templates are excluded from this bulk command. Handle them
separately when needed, taking care to preserve their template syntax.

## Network boundaries

Network operations are intentionally kept inline in workflow YAML for auditability.
Keep destinations, credentials, cache transfers and remote writes visible there;
do not move them into repository Python or shell wrappers for deduplication.
Local preparation, validation and generation belong in typed tool functions.
Docker isolation must be explicit in YAML (`docker run --network none`), not a
custom shell wrapper. Do not add unrelated syscall or socket restrictions.

Upstream modifications remain individual, reviewable `.patch` files. Separate
observed input/patch identities from behavioral contracts: a changed file digest
is not by itself proof of an API, ownership, layout or numerical violation.

## Validation tiers

Keep current-source contracts separate from frozen historical comparisons; see
`docs/validation-tiers.md`. A historical digest belongs to its explicitly owned
fixture, never to the acceptance condition of the current build. Do not replace
current behavioral coverage with tests of a frozen input or silently skip it.
