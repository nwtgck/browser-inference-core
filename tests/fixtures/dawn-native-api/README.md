# Native Dawn API declarations

These unmodified upstream C/C++ SDK declarations compile the native branch of
the current WebGPU backend against its real external API. They contain no GPU
implementation. Native contracts implement that external C API with observable
handles, queues and callbacks. The browser branch separately uses the pinned
Emdawnwebgpu package used by the production build.

Source: <https://dawn.googlesource.com/dawn>, commit
`7098c3b14768900c86c26a9a29dda467fd89bddf`. The included BSD license and upstream
copyright notices apply. These are test inputs, not runtime package payloads.
Do not format the generated or copied SDK headers.

From an existing clean checkout of that Dawn commit, regenerate with its ordinary
generators (Python with Jinja2 is required):

```sh
python3 generator/dawn_json_generator.py \
  --dawn-json src/dawn/dawn.json --targets headers,cpp_headers \
  --template-dir generator/templates --output-dir /tmp/dawn-native-api
python3 generator/dawn_version_generator.py gen \
  --dawn-dir . --version-json src/dawn/dawn_version.json \
  --template-dir generator/templates --output-dir /tmp/dawn-native-api
```

Copy the generated `include/dawn/{webgpu.h,webgpu_cpp.h,dawn_version.h}` and
`include/webgpu/webgpu_cpp_chained_struct.h` into this fixture, together with
upstream `include/webgpu/{webgpu.h,webgpu_cpp.h,webgpu_enum_class_bitmasks.h}` and
`LICENSE`. Keep these declarations from one coherent upstream revision. Updating
them is an API compatibility check; their bytes do not establish backend
selection, ownership, synchronization or shader correctness.
