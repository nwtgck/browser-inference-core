# Short Q4_K factory checks

`test_webgpu_q4_short_overlay.py` compiles the actual prepared selector, cache
keys, ordinary/indexed pipeline factories, dispatcher bodies, upstream shader
embedder and preprocessor. Only GPU resource/pipeline creation boundaries are
recorded. It covers all 24 construction orders for WG32/64/128/256, fallback
byte identity, subgroup ranges/device limits, shape exclusions and actual
parameter/binding/2D-dispatch decisions. The upstream shader sources are never
copied into production files or modified.

The optional standalone runner needs an existing Node 22+ `webgpu` package and
local native C++ compiler. It installs nothing and launches no browser:

```sh
python3 llama-cpp/tests/webgpu-q4-short/check_gpu.py \
  --llama-source /path/to/pinned/llama.cpp \
  --work /tmp/q4-short-validation \
  --webgpu-module /path/to/node_modules/webgpu/index.js
```

The runner compiles actual factory-emitted shaders supported by the adapter, checks 80 fixtures
against unchanged upstream WG256 outputs bit for bit, and compares both outputs
to a double dot computed from the actual pinned upstream Q4_K CPU dequantizer.
The dequantizer and its scale helper are extracted unchanged for a native test
executable; this reference does not simulate shader lane or reduction mapping.
The independent reference allows 64 F32 epsilons times the absolute product sum
for normal dequantized-dot rounding. Known blocks test low/high scale bits,
minimum terms, zero values and truncation errors in the CPU-only unit suite.

Fixtures cover complete K blocks up to 4096, 4352 fallback, row tails,
misaligned logical offsets inside aligned bindings, padded row/batch strides,
ordinary 2D/3D broadcast, per-slot/shared indexed inputs, repeated selected
experts including the final stored expert, guards and unchanged input buffers.
The actual dispatcher host tests independently verify four outputs per workgroup
and the unchanged grid. The selected workgroup for each subgroup maximum comes
from the compiled production factory, not a JavaScript selector copy.

`source.json` records source/patch/reference/shader identities and `report.json`
records adapter information, each case, error counts and numerical bounds.
These are standalone correctness checks, not browser/model inference or target
hardware performance evidence. A software adapter cannot establish physical-GPU
occupancy, stable cross-adapter bit identity or a throughput improvement.

`--benchmarks` additionally requires `timestamp-query` and records alternating
AB/BA GPU-pass timings for eight-slot indexed K512/M2048 and K2048/M512,
ordinary K2048/M2048, K2048/M10752 and K2048/M6144, and a K10752/M2048 unchanged-fallback
control. Four warmup pairs precede sixteen measured pairs. The timing path uses
the same buffers and the compiled factory's choice; the report retains every
pair and zero-duration count. Software-adapter timings describe that adapter
only.

`--wall-benchmarks` additionally records elapsed time for batches of twenty
separate submissions and queue completion without timestamp writes. It requires
`--benchmarks`; four warmup pairs precede eight measured AB/BA pairs. Both
outputs, independent CPU references, input bytes and guards are checked again
after every benchmark. Neither timing method is full-runtime decode latency.
