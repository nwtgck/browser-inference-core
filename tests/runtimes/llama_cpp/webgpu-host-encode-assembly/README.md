# Synchronous host assembly checks

The allocation fixture extracts the builders, arena and exact cache from the real
composed source. A pinned preceding-stage baseline supplies the comparison. Each
case changes parameter values, checks payload and slot padding, verifies bind keys
and dispatch order, destroys caller inputs, then flushes and executes a deferred
CPU queue model. Actual normal/profile pipeline structs are used, including the
profile diagnostic digest. Profile result names retain ownership.

The comparison checks reductions relative to the same compiler and standard
library; native object sizes and small-string allocation thresholds are not fixed
cross-platform requirements. Initialization, cold cache creation, command capacity,
flush and queue submission are outside the allocation window. These are native CPU
mock allocations, not GPU/browser timings or a model throughput claim.

The existing bind-group-cache fixture additionally runs its full key-change,
bounds, generation, free/recreate, pending queue and creation failure scenarios
with the new helpers in all normal/profile and batched/unbatched parameter modes.
Full composed translation-unit checks retain the pinned actual WebGPU headers.

Run the focused unittest modules from `llama-cpp/`:

```sh
python3 -m unittest discover -s tests -p test_webgpu_host_encode_assembly.py
python3 -m unittest discover -s tests -p test_webgpu_bind_group_cache.py
```
