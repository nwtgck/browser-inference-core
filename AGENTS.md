# Browser inference runtimes

Runtime-specific source, bindings and profiles belong under `runtimes/`.
Shared development code uses the ordinary `build_tools` Python package; shared
tests use the corresponding namespaced directories under `tests/`.
All runtime tests/harnesses belong to `tests/runtimes/<runtime_name>/`, not a
second tree inside `runtimes/`. Use `python3 -m unittest discover -s <scope> -t .`
from the repository root. Local native/host checks do not require Emscripten;
leave the mandatory browser-build matrix to CI when that toolchain is absent.
Do not merge Wasm instances or upstream version pins between runtimes.

- `runtimes/llama-cpp/`: the existing explicit-approval/last-resort policy still applies.
- `runtimes/stable-diffusion-cpp/upstream-patches/`: the user expressly authorized the
  implementing agent to choose browser/WebGPU enablement patches for this
  experiment on 2026-09-24. This does not authorize changes to llama.cpp's runtime.

Never edit a checked-out upstream in place or misrepresent compilation/mock tests
as real-model browser inference. Record patch identities and upstream commits.
Publication must be append-only, source-bound, and contain runtime assets only.

## C and C++ formatting

When modifying project-owned C or C++ source files and headers, format them with
`clang-format-17` using the `.clang-format` configuration in the repository root.
Only format `.c`, `.cpp`, `.h`, and `.hpp` files in these directories:

- `runtimes/llama-cpp/native/`
- `tests/runtimes/llama_cpp/`
- `tests/runtimes/stable_diffusion_cpp/`

From the repository root, format all files in scope with:

```sh
find runtimes/llama-cpp/native tests/runtimes/llama_cpp tests/runtimes/stable_diffusion_cpp -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' \) -exec clang-format-17 -i --style=file {} +
```

Do not format upstream sources, `.patch` files, or generated code. Avoid
unnecessary formatting differences from upstream.

The `.cpp.in` templates are excluded from this bulk command. Handle them
separately when needed, taking care to preserve their template syntax.

## Network boundaries

Network operations are intentionally kept inline in workflow YAML for auditability.
Keep destinations, credentials, cache transfers and remote writes visible there;
do not move them into repository Python or shell wrappers for deduplication.
Local preparation, validation and generation belong in ordinary Python modules.
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
