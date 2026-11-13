# Verification scopes and prerequisites

Run Python commands from the repository root. Test code and native/browser
harnesses belong under `tests/runtimes/llama_cpp` and
`tests/runtimes/stable_diffusion_cpp`; a runtime source directory does not own a
second test discovery tree. Historical WebGPU stage composers are test support,
not production build tools. Do not import `tests` from a production generator.

| Scope | Local command or entry point | What it does not establish |
| --- | --- | --- |
| Shared tools | `python3 -m unittest discover -s tests/build_tools -t .` | Actual runtime compilation or GPU execution |
| Llama contracts | `python3 -m unittest discover -s tests/runtimes/llama_cpp -t .` | Every Emscripten/profile combination |
| Image contracts | `python3 -m unittest discover -s tests/runtimes/stable_diffusion_cpp -t .` | Browser/WebGPU inference |
| Browser host helpers | `npm test` | Execution of generated Wasm |
| All Python scopes | `npm run test:python` | Availability of optional native tools and upstream checkouts |
| Native integration | Runtime CMake with `LCB_NATIVE_CHECK` or `SDCB_NATIVE_CHECK` | Wasm address width, JavaScript glue or suspension behavior |
| Native QSA parity | `python3 tests/runtimes/llama_cpp/qsa-selection/run.py --source <llama> --work <output> --jobs 2` | Real-model performance or WebGPU execution |

Narrow any Python discovery directory further, or select fully-qualified test
modules, for a focused change. The all-scope command is optional locally: report
the selected modules, failures and skips instead of claiming whole-repository
success. C++ probes require an ordinary native compiler and, where indicated,
real pinned Dawn headers. Type checks use the pinned TypeScript compiler.

Emscripten is not required for these source/native scopes. Actual Emscripten
compilation, linkage, generated JavaScript and browser-runtime checks remain
mandatory in the CI build and publication graph. Local tests must not weaken or
replace those gates. Do not require a user-side duplicate Wasm build merely to
compensate for an offline agent lacking the compiler.

The production build captures selected Git objects. A supplied source-only
archive can be used to exercise preparation and native behavior, but it is not
proof of upstream Git-object identity. Keep the declared archive identity and
observed file hashes separate from verified Git provenance. Do not fabricate a
matching commit for a pruned archive.

Acquisition is visible inline in workflow YAML; execution of local source code
is directly wrapped by `docker run --network none`. Local verification without
Docker is not evidence that the isolation boundary ran. Runtime publication
also binds browser harness files, fixture/configuration inputs and package bytes
to the same source and result session; moved tests must remain in that inventory.
