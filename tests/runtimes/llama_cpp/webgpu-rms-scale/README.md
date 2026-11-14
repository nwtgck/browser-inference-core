# RMS and scale WGSL comparison

This optional Dawn experiment compares supplied fused and staged WGSL with a
Float64 CPU reference, output/input guards and observed nonfinite classification
parity. It does not establish production factory provenance or model throughput.

```sh
LCB_TEST_WEBGPU_MODULE=file:///absolute/path/to/webgpu/index.js \
node tests/runtimes/llama_cpp/webgpu-rms-scale/check_gpu.mjs \
  /absolute/path/to/shaders /tmp/rms-scale-report.json numerical 1024
```

Supply `fused-0.wgsl`, `fused-1.wgsl`, `rms.wgsl`, `rms-inplace.wgsl`,
`scale.wgsl` and `scale-inplace.wgsl` with the buffer/uniform layouts consumed
by the script. The final number is the supplied SCALE shader's workgroup size;
it must equal the adapter's maximum compute invocations and is requested as a
device limit. RMS and fused shaders use WG128. The staged layout performs SCALE
in place on the RMS output. The script uses Dawn’s default backend.

Use `benchmark` in place of `numerical` for paired AB/BA GPU timestamps; this
requires `timestamp-query`. Numerical mode alone makes no speed measurement.
The caller supplies all shaders and an installed Dawn module. No repository
producer is provided for this experiment. Input hashes identify those bytes,
without proving agreement with a current production factory. Current host fusion
admission and warm reuse are checked by the complete-source native transfer target.
