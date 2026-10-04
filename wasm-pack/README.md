# Selectable, lossless Wasm packaging

This build-time component adds compressed representations to already-built browser
Wasm. It never recompiles an inference engine, changes a raw input, patches an
upstream checkout, or installs a consumer build hook. Runtime API version 1 ships
with the exact data format it decodes.

## Selection before bundling

The producer supplies an independent full representation for every target and
codec it offers, plus useful direct deltas. A target has distinct `runtime`,
`source`, `profile` and `variant` coordinates. The initial encoder admits browser
variants only. It rejects input length/hash mismatches and linked paths.

The consumer imports `runtime/catalog.mjs` at build time and calls:

```js
const plan = selectWasmAssets(catalog, {
  targets: selectedTargetIds,
  codec: 'brotli', // or gzip / zstd; an explicit distribution policy
  fullTargets: [], // optionally keep frequently used targets independent
});
```

`plan.files` is the exact asset closure. Emit only those files and the matching
selected core glue, not the entire producer package. A delta base MUST also be a
selected target with a full representation. No hidden CPU/Asyncify base is added
to a GPU-only selection. Removing the base switches to an independent full.
Missing IDs/codecs fail rather than selecting another source. An empty target
selection emits no files or decoder. All choices are deterministic. For up to 12
targets, full-root subsets are enumerated; per-target alternatives are chosen by
bounded heuristics. This is not a global minimum-cost guarantee, especially when
assets are shared between alternatives. Larger selections use safe limited candidates.

An embedded distribution can select Brotli without any Zstandard loader, decoder
Wasm or dependency notice. Import `plan.runtime.entry` from the selected files;
the codec-specific entry delegates to a shared decoder core.
A hosted distribution can select another codec without changing source identity.
Development and release of the same distribution should select the same plan.
Do not silently substitute raw Wasm on decode failure or in development.

## Runtime contract

```js
// createWasmLoader is exported by the selected plan.runtime.entry.
const loader = createWasmLoader({
  plan,
  readAsset: async (asset, signal) => readBytesAtYourChosenLocation(asset.path, signal),
});
const originalWasm = await loader.load(targetId, { signal });
```

`readAsset` returns a `Uint8Array`; the consumer owns URL resolution or embedded
byte extraction. There is no built-in network fetch, global cache, raw fallback,
or implicit change of source/profile. `decompress` may be supplied for an embedding
platform, but compressed and expanded identities are still checked. Normally
Brotli/gzip use `DecompressionStream`. Brotli availability is a consumer capability
requirement, not inferred from the user-agent string. Zstandard uses the packaged,
hash-checked Wasm decoder with a 1 MiB maximum frame window.

The loader verifies compressed assets, full outputs, the temporary predicted
dictionary and the final original target using length plus SHA-256. These checks
assume a trusted manifest/decoder; they do not authenticate an attacker-controlled
manifest and payload together. Requests are serialized per loader; successful
full outputs are not retained as an unbounded cache. Abort is checked around
asynchronous operations and synchronous kernels, not cooperatively within every
prediction or copy loop. The consumer should run these kernels off its main UI
thread. A failed request does not poison later requests.

Caps: 64 MiB per raw/predicted/output item, 4 MiB for prediction rules, 128 targets,
4096 requested encoder pairs. Limitations apply explicitly, not by trusting a
frame's declared content size. Increasing limits requires memory testing.

## Encoder

`encoder/pack.py` owns input identities, representation metadata and atomic
publication. One Rust executable owns all binary algorithms:

1. `wasmparser` validation and instruction analysis.
2. Deterministic full/partial function alignment and PRD1 prediction training.
3. Bounded exact byte matching and WXP1 copy/additive-correction serialization.

Python then compresses full data/recipes using standard codecs and reconstructs
**every alternative** through the shipped JavaScript/Wasm decoder. See
[language and process boundaries](ARCHITECTURE.md) for selection rationale,
private process protocol, bounds and dependencies. There is no C++ matcher,
NumPy environment, WSI1 index file or SEQ1 sequence interchange.

A predicted dictionary is not executed or required to be valid Wasm. Structural
similarity is not semantic equivalence: the final delta preserves every differing
byte. The encoder does not call `ZSTD_generateSequences` or other research-only
sequence extraction APIs. It uses documented Zstandard functions; the system
`libzstd.so.1` version is recorded in the build report. Pin that library/build
image when byte-reproducible compressed output across machines is required.

## Build tools

Prerequisites: the Rust version in `toolchain.json` with `wasm32-unknown-unknown`,
Node.js, Python standard library, and libzstd for Zstandard. No Python package
installation or C++ compiler is required for the packing tool itself.
Place the exact recorded wasm-tools revision at `.tools/wasm-tools` (repository
root). `build_tools.py` accepts a Git checkout or the provided offline archive's
`WASM_TOOLS_COMMIT` marker. Cargo dependencies are locked, not installed by consumers.

```sh
python3 wasm-pack/build_tools.py --offline --test
python3 wasm-pack/encoder/pack.py \
  --root /path/to/raw-artifacts --spec /path/to/targets.json \
  --output /path/to/fresh-packed-directory \
  --codecs gzip brotli zstd \
  --encoder build/wasm-pack-tools/wasm-encoder \
  --zstd-decoder build/wasm-pack-tools/zstd-decoder.wasm \
  --pairs /path/to/explicit-pairs.json --report build/pack-report.json
```

Omit `--offline` only in a build environment allowed to obtain locked Cargo
sources. For offline builds configure Cargo's vendored source replacement first.
`--pairs` is an array of `[baseId, targetId]`. With no pair list all same-runtime
directions are considered, within the cap; this is build work, not an Actions job
matrix. `[]` produces only independent full representations. New outputs are
assembled in a temporary sibling and published only after reconstruction checks.
Raw originals remain at the caller's input paths.

`verify-pack.mjs PACK_DIR RAW_ROOT` validates all representations, not just those
selected by a convenient default plan. Runtime third-party notices are included
only when their codec is selected. Encoder/toolchain sources are never in a plan.

## Pipeline integration

The main `build.yml` now invokes the source-aware v4 assembler after verified
per-source builds or exact raw-package reuse. `wasm-pack/prepare_ci_tools.py`
prepares build-only dependencies, and `scripts/assemble_source_plan.py` validates
run-local inputs before encoding. Every compressed alternative is reconstructed
before append-only publication. The published receipt then binds consumer YAML.
The manual source-build workflow remains candidate-only and cannot publish an
incomplete root catalog. See [source publication](../docs/source-catalog-publication.md)
for the reuse trust boundary, cold migration and verification limits.
