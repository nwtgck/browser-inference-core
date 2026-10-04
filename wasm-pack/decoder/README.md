# Bounded browser Zstandard decoder

This cdylib uses the unmodified, locked `ruzstd` crate. It exposes a small memory
interface consumed by `../runtime/zstd-loader.mjs`. Input, window and output limits
are checked. Output is drained in bounded chunks instead of retaining a second
full output in Rust. The JavaScript wrapper returns bytes only after complete frame
processing and expected-length checks; the public loader additionally checks hashes.

Build with `python3 wasm-pack/build_tools.py`. The generated `.wasm` belongs in
artifact output, not in a consumer install hook or a checked-in host toolchain.
Cargo.lock and dependency notices are tracked. No inference engine or model is
linked into this decoder. Its linear-memory capacity is not the whole consumer's
memory use: JavaScript dictionaries, outputs and inference memories are separate.
