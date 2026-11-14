# Scale and unary WGSL numerical checks

This optional Dawn check compares supplied fused and staged WGSL with an
independent F32 CPU reference. It validates finite results, observed nonfinite
classification parity, output guards and input preservation. It does not measure
GPU timestamps or establish model throughput or production shader provenance.

```sh
node tests/runtimes/llama_cpp/webgpu-scale-unary/check-gpu.mjs \
  /absolute/path/to/shaders /tmp/scale-unary-report.json \
  /absolute/path/to/webgpu/index.js forward
```

Supply `fused-0.wgsl` through `fused-7.wgsl`, `scale.wgsl`,
`scale-inplace.wgsl`, `SILU.wgsl`, `SIGMOID.wgsl`, `SILU-inplace.wgsl` and
`SIGMOID-inplace.wgsl`. The fused key bits select sigmoid, final scale and
in-place binding respectively. Supply the matching buffer/uniform layouts
consumed by this script. `reverse` reverses pipeline creation order.

Inputs and the local Dawn module are caller-owned; the script downloads nothing.
There is no repository producer for this experiment. Its reported input hashes
identify the supplied bytes; they do not prove agreement with a current factory.
Current host fusion admission and warm reuse are checked separately by the
complete-source native transfer target.
