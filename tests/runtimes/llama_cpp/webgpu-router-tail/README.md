# Router-tail WGSL alias comparison

This optional Dawn script compares caller-supplied one-token fused and staged
WGSL using independent alias arenas. It checks exact output bits, special-value
classes, buffer guards, IDs and uniforms. It records no GPU timestamps and makes
no speed, graph integration or production provenance claim.

```sh
node tests/runtimes/llama_cpp/webgpu-router-tail/run_gpu.mjs \
  --work /absolute/path/to/inputs --output /tmp/router-report.json \
  --webgpu-module /absolute/path/to/webgpu/index.js --backend vulkan \
  --workgroups 256,512,1024
```

For each selected WG, supply `generated/<WG>/{get_rows,sum_rows,clamp,
clamp_inplace,div,router_tail,router_tail_merged}.wgsl` with the layouts consumed
by the script. `manifest.json` contains `shaderSha256[WG]` entries for the two
router shaders and `baselineShaderSha256[WG]` entries for the five staged
shaders, keyed by filename. Optional `source` metadata is copied into the report.
Hashes check the integrity of caller-supplied inputs; they do not prove their
relationship to a production factory. `--filter` and `--limit` select synthetic
cases; each WG must be supported by the selected adapter.

The caller supplies the shaders, manifest and an installed Dawn module. No
repository producer or external private lab is required or provided.
