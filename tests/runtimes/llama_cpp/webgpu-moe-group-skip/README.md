# MoE matrix inactive groups

The ordinary Python test executes canonical and prepared factories with the
pinned include expander/preprocessor, checks complete generated WGSL byte
relations, cache hits, parameters and tiling, and syntax-checks the full backend
with normal and optional profiling definitions. It does not execute a GPU.

`check_gpu.py` is an explicit, opt-in native Dawn check. Supply a local upstream,
Node and the local `webgpu` package; it downloads nothing and uses no browser.
It generates self-authored F32/Q8_0/IQ3_S bytes, uses unchanged pinned CPU
dequantizers, executes the actual upstream gather, and validates every output
byte against canonical matrix shaders. It checks routing membership, guards,
source/metadata immutability, changing IDs, shared/slot-specific input, offsets,
partial groups, vectorized tails and zero-tile to normal reuse.

The default contains 40 small cases. `--large` adds bounded K256/1024/2560,
M32/E512/used10/tokens128 uniform and concentrated cases. No model weights or
fixture binaries are distributed. GPU work is serialized and buffers are freed
between cases. The gather uses the actual requested 256-invocation device limit;
the smaller test grid-planning bound exercises flattening without pretending
that it is the hardware's limit. All actual dispatch dimensions are checked.

Optional `--benchmarks` reports isolated kernel timestamps and wall intervals
separately. Repeated warm-input measurements and native software-adapter results
are not model throughput or browser/Mac performance claims.
