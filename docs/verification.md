# Verification scopes and prerequisites

Run tool and test commands from the repository root. `npm run tooling:build`
performs strict TypeScript checking and emits the Node tools and tests under
`build/tooling/`. Use the selected runtime project when its independent build
scope is sufficient. `npm run tooling:check` checks types without emitting;
checking and emitting unchanged inputs twice is unnecessary.

| Scope | Command or entry point | Evidence |
| --- | --- | --- |
| Shared typed tools | `node --test build/tooling/tests/tools/<area>/*.test.js` | The selected preparation, DTO, package, publication or automation contracts |
| Llama tools | `node --test build/tooling/tests/runtimes/llama-cpp/tools/*.test.js` | Runtime configuration, preparation and build orchestration |
| Image tools | `node --test build/tooling/tests/runtimes/stable-diffusion-cpp/tools/*.test.js` | Independent image preparation and build contracts |
| Browser host helpers | `npm test` | JavaScript helpers and ABI adapters using the suites' declared native doubles |
| CLI adapters | `node --test tests/tools/cli/*.test.mjs` | Typed dispatch, argument validation, metrics and exit-status handling with command stubs |
| Local lint rules | `npm run test:lint` | Parser/configuration behavior and protected formatting boundaries |
| Generated package consumers | `node --test build/tooling/tests/tools/package/public-types.test.js` | Type resolution through a real assembled package's public exports |
| Current native integration | `node --test build/tooling/tests/runtimes/<runtime>/native-validation.test.js` | Prepared C/C++ sources and complete native fixtures |
| Independent CPU and CMake checks | The explicit unittest command below | Real quantization arithmetic, fixture boundaries and target-property snapshots |
| Native QSA parity | The [QSA runner](../runtimes/llama-cpp/docs/qsa-all-pools-selection.md) | Independent reference/optimized synthetic model traces and graph coverage |
| Saved runtime packages | `node build/tooling/tools/cli.js validation runtime --help` | Entry to the explicit Wasm/browser validation boundary used by CI |

Select affected files or directories instead of substituting a broad build.
Tests can require initialized upstream checkouts, Clang, CMake/Ninja, pinned
Dawn headers or already compiled runtime artifacts. Read those tests' input
requirements and report missing inputs and skips. A missing dependency is not
a passing contract check.

Choose a representative production configuration that activates the changed
path. Configure checks target setup; compiling the affected consumer checks its
include paths and platform-specific code. Native fixtures may use different
source selections or include directories, so their success does not substitute
for the corresponding Wasm compile. Use the generated build settings rather than
adding test-only include paths to make that consumer compile. This does not
require a full local matrix or browser execution.

Native checks do not establish Wasm pointer width, JSPI/Asyncify completion,
Embind marshalling or WebGPU execution. The CI matrix separately compiles the
actual targets and validates their matching generated schemas, declarations,
JavaScript and Wasm. Browser smoke uses bounded synthetic inputs; real-model
quality, ordinary generation throughput and prefill performance require their
own measurements. See [validation tiers](validation-tiers.md).

## Independent CPU and CMake checks

The host CI job runs these standard-library Python tests once inside its
existing offline execution boundary:

```sh
python3 -m unittest \
  tests.runtimes.llama_cpp.source.test_native_quantization \
  tests.runtimes.llama_cpp.source.test_q4_reference \
  tests.runtimes.llama_cpp.source.test_iq3_fixtures \
  tests.runtimes.llama_cpp.test_cmake_property_snapshot
```

The numerical tests compile the initialized llama upstream's real C sources
with small synthetic inputs. They require a native compiler, not model weights
or a GPU. The property tests use CMake's actual unset, empty and false-like
target-property values without compiling an upstream runtime. These tests do
not depend on the production tool pipeline or source-body extraction.

## Source and execution boundaries

Production preparation runs from the executing checkout's typed tools and
static configuration. Release builds require the selected source commit and
its Git objects; local builds explicitly capture working inputs and stay
unpublishable. An unrelated source archive cannot establish Git provenance.
Record its actual identity when using one for an isolated experiment.

Shared tests belong under `tests/tools/`; runtime tests and harnesses live under
`tests/runtimes/`. The underscore-named runtime directories contain C++/JavaScript
fixtures and independent CPU/CMake test helpers. Frozen source comparisons belong
to their named historical fixtures and do not admit the current upstream by a
whole-file hash.
Production tools never import test-only helpers.

Acquisition, credentials and remote writes remain visible in workflow YAML.
Local stages use `.github/scripts/run-offline.sh` for shared network-disabled
Docker settings and common environment. The executable script forwards extra
Docker options and an explicit image argument, then runs quoted stdin with strict
Bash.
See its header for calling syntax and isolation limits. Workflow-specific
environment, image selection, commands and redirections stay in YAML. A local test
directly establishes its functional result, not Docker isolation.
Saved-package validation binds runtime/test inputs and package bytes to its
result before recording browser-smoke provenance.
