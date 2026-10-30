# Browser inference runtimes

Runtime-specific source, bindings, profiles and tests belong in their runtime directory.
Do not merge Wasm instances or upstream version pins between runtimes.

- `llama-cpp/`: the existing explicit-approval/last-resort policy still applies.
- `stable-diffusion-cpp/upstream-patches/`: the user expressly authorized the
  implementing agent to choose browser/WebGPU enablement patches for this
  experiment on 2026-09-24. This does not authorize changes to llama.cpp's runtime.

Never edit a checked-out upstream in place or misrepresent compilation/mock tests
as real-model browser inference. Record patch identities and upstream commits.
Publication must be append-only, source-bound, and contain runtime assets only.

## C and C++ formatting

When modifying project-owned C or C++ source files and headers, format them with
`clang-format-17` using the `.clang-format` configuration in the repository root.
Only format `.c`, `.cpp`, `.h`, and `.hpp` files in these directories:

- `llama-cpp/bridge/`
- `llama-cpp/tests/`
- `stable-diffusion-cpp/tests/`

From the repository root, format all files in scope with:

```sh
find llama-cpp/bridge llama-cpp/tests stable-diffusion-cpp/tests -type f \( -name '*.cpp' -o -name '*.hpp' -o -name '*.h' -o -name '*.c' \) -exec clang-format-17 -i --style=file {} +
```

Do not format upstream sources, `.patch` files, or generated code. Avoid
unnecessary formatting differences from upstream.

The `.cpp.in` templates are excluded from this bulk command. Handle them
separately when needed, taking care to preserve their template syntax.
