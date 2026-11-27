# Q4_0 GEGLU shared storage regression

Build the current native WebGPU provider and run its `webgpu-provider` CTest.
It emits the real kernel variants into `webgpu-provider-shaders` in the
build directory. Replay the shader selected for the local Dawn adapter:

```sh
node tests/runtimes/llama_cpp/webgpu-q4-geglu/check-arena.mjs \
  /absolute/path/to/native-build/webgpu-provider-shaders \
  /absolute/path/to/webgpu/index.js
```

The Vulkan adapter must support shader-f16 and fixed subgroups of width 16
or 32. A software Vulkan adapter can run the same check. It submits the
shader with separate buffers, one shared activation/output buffer, and one
buffer for all four ranges. The ranges are disjoint in every layout. Output
is compared with an independent calculation, and the inputs and padding
must remain unchanged. The check installs no dependencies.
