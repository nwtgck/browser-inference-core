# Source-bound expert-sum validation

Run `python tests/webgpu-moe-sum/run_lab.py --llama-source /path/to/llama --lab /path/to/qwen-next-hc-lab --work /separate/output --benchmarks`.

Tests use the actual prepared admission header, actual production shader factory,
upstream ADD shader, real ggml graph allocator, and software WebGPU. This is not a
compiled-runtime or real-model benchmark. The lab's explicit Dawn instance remains
alive until device destruction and report serialization. The negative control must
fail with a numerical mismatch, not merely a launch error. No download is attempted.
