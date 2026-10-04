# Language and process boundaries

## Producer: Python for artifacts, Rust for binary algorithms

The producer runs one native Rust executable per requested pair. Within that
process `wasmparser`, function alignment, prediction training, exact matching and
WXP1 serialization use ordinary Rust data structures. They do not exchange a
parser index or a sequence file through another language. Native tables use
explicit little-endian byte reads and deterministic tie-breaking. Bounded search
can sacrifice compression ratio, never target byte identity.

Python retains source/package validation, hashes, codec policy, catalog assembly,
subprocess timeouts and atomic publication. This fits the existing source-aware
publication pipeline; converting that pipeline into Rust would add a second,
unrelated migration. Python requires no NumPy, pip-installed modules or separate
virtual environment for this feature.

The private process protocol is `wasm-encoder pair BASE TARGET NEW_DIRECTORY`.
It outputs a required plain recipe, an optional predicted recipe/rule set and a
small versioned JSON result. A predicted dictionary output is producer-only and
exists to compute its identity independently. The packer admits only documented
filenames and bounded output lengths, and publishes nothing until the shipped
JavaScript/Wasm decoder reconstructs every representation. `validate INPUT`
validates without allocating an instruction index for a full-only target.

The encoder validates real core modules. Function/shape similarity only proposes
compression hints; it does not prove semantic equivalence. Caps are 64 MiB per
input/output, 100,000 functions, 4,000,000 operators per function and 16,000,000
indexed operators per module. Parser state is released before allocating match
history. A parser index is built per pair, not persistently cached across a run;
large source sets may benefit from a bounded cache later. No global speed or
optimal compression claim follows from choosing Rust.

## Standard codecs are library reuse, not additional homemade algorithms

- gzip: Python's standard library.
- Brotli: the already-required Node.js runtime's `node:zlib`, through a small
  byte-in/byte-out command. This deliberately avoids a new Python Brotli package
  or another native binding. Node is also required for independent verification.
- Zstandard: documented C functions through one named, typed Python `ctypes`
  adapter. The producer is Linux; `libzstd.so.1` is an explicit prerequisite.
  The adapter is not an algorithm implementation, and C library code is not
  shipped to browsers. The library version is reported. Cross-machine compressed
  byte reproducibility additionally requires pinning that library/build image.

There is no project-owned C++ compression program. Inference C++ and approved
upstream patches are outside this change. No research-only sequence-extraction
API is used. Build dependencies retain their own licenses; the browser decoder
has its own small Cargo dependency graph rather than linking the encoder.

## Browser: JavaScript orchestration plus an optional Rust/Wasm codec

JavaScript validates the selected plan, reads assets supplied by the consumer,
checks hashes, applies PRD1 rules and reconstructs WXP1 recipes. Native gzip and
Brotli decoding remains a platform adapter. Rust/ruzstd only implements the
optional Zstandard expansion kernel; it does not parse inference Wasm, choose a
profile, fetch data, or perform the prediction/delta algorithm.

`selectWasmAssets` returns `plan.runtime.entry`. Import that entry from the
selected package, not a hard-coded codec implementation:

- `loader.mjs` supports gzip/Brotli and has no dependency on Zstandard JavaScript,
  decoder Wasm or third-party notices.
- `loader-zstd.mjs` adds the matching Zstandard codec to `loader-core.mjs`.

The current common core still includes the small prediction/delta modules in a
full-only plan. It does not pull another profile or codec into the bundle.
Artifact-owned code understands the format; consumers only choose targets and
codec policy and emit the returned file closure. Runtime API version 1 and final
WXP1/PRD1 formats remain unchanged. The new codec-entry catalog metadata travels
with the matching selector/decoder, not an old consumer-side format parser.

## Licenses and validation

The pack's own LICENSE is copied from the repository root during packaging; a
second hand-maintained identical source file is unnecessary. The unmodified
ruzstd/twox-hash notices remain and are selected only with Zstandard. This cleanup
does not replace dependency notices or change the repository's license.

Tests cross boundaries: Rust unit tests exercise algorithms, Node verifies every
produced pack alternative, and randomized valid modules are encoded by Rust,
restored by the shipped JavaScript and executed. Selector tests copy only selected
files into isolated directories before importing the selected runtime entry.
Standalone/native Brotli support, final Vite/file-protocol integration, Actions
and model inference are distinct validation tasks; Node tests do not establish
browser compatibility.
