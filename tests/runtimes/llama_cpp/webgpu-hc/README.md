# Optional HC Dawn numerical comparison

Supply `hc_pre.wgsl` and `hc_post.wgsl` in one directory, using the entry points and
buffer/uniform layout in `check-gpu.mjs`. This experiment compares those inputs
with separately staged GPU expressions and checks output bits and buffer guards.
It does not prove that the inputs match the current production factory.

With a compatible Dawn Node module already available locally:

```sh
LCB_TEST_WEBGPU_MODULE=/absolute/path/to/webgpu/index.js \
node tests/runtimes/llama_cpp/webgpu-hc/check-gpu.mjs \
  /absolute/path/to/shaders /absolute/path/to/report.json 0 288
```

The final two arguments select a start and positive case count within the 288
synthetic cases. The runner requests a Vulkan adapter. Set `LCB_HC_FORCE_GC=1`
and add `--expose-gc` to Node for owner-lifetime stress. The JSON report and its
`.jsonl` case stream record numerical/guard failures and adapter information.
This runner records no GPU timing and makes no speed claim. It does not build or
execute the inference runtime and installs no dependencies.
