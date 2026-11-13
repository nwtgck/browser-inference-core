# IQ3_S validation boundaries

- `emit_factory.py` prepares the current source without a historical digest allow-list.
  It compiles the complete current shader library with the pinned Dawn declarations.
  Only low-level C WebGPU calls are recorded; selectors, cache keys, preprocessing
  and pipeline construction are the actual library implementation.
- The complete provider is type-checked by Clang before semantic source ranges
  select indexed dispatch functions. This still executes a native fragment, not
  the whole backend. It is not GPU execution or a device performance measurement.
- The unchanged complete upstream fallback factory consumes the same current
  common shader inputs via an include adapter. This keeps the comparison about
  factory behavior, not unrelated decoder revisions. Both factories are included
  without editing their source text.
- `tests/support/llama_cpp/historical_iq3.py` and the explicitly historical test
  classes own their frozen source and patch fixture. Their golden hashes and
  inverse-patch assertions check that history, not current-source compatibility.
- Current tests cover normal/diagnostic preparation, unchanged behavioral checks,
  acceptance of unrelated comments and rejection of invalid unused declarations.
- `check_gpu.py` remains opt-in. Its receipts record observed preparation, CPU
  reference and returned native executable metadata, rather than fixture hashes.
  Source-only input archives have no claimed Git commit. CI still owns actual
  WebAssembly/compiler/browser execution; native success is not a substitute.
