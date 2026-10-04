# Source-aware runtime publication

The root `build.yml` now publishes a format-4 catalog. This is a source layout
change, not an instruction for consumers to upgrade a dependency without reviewing
its glue and bindings. Raw Wasm remains available. Compressed data is an optional
selection; never copy the entire producer package into a consumer bundle.

## Pipeline and trust boundaries

1. `pipeline_plan.py` resolves this repository's `artifacts` ref once, fetches that
   exact commit into a bare temporary Git repository, and extracts only regular
   runtime files. It never checks out or runs code from the previous snapshot.
   Missing artifacts or a legacy format-3 artifact cause a cold build. Corruption,
   unexpected formats, and failed network resolution stop; `cache_mode: cold` is
   the explicit way to skip prior-artifact acquisition as well as Actions caches.
2. Root file coverage, lengths, hashes and bound inner manifests are checked.
   Only source packages with matching pins, profile/variant sets, build and test
   fingerprints, clean provenance and successful recorded smoke scopes are admitted.
   A previous packed codec need not match today's decoder: only raw packages are
   reused, and current runtime validators recheck admitted packages before use.
3. The compile matrix contains only missing/changed llama sources. A nightly-only
   pin change normally needs its two configured profiles times browser/test: four
   compile jobs. The image build and native checks may be skipped only when the
   image package's own inputs and validation match. Host/publication tests still run.
4. The package job covers *all* configured sources, selecting fresh shard assembly
   or verified raw reuse. A flat CI shard download is passed explicitly as the
   build root; the nightly local build directory is not inferred for a download.
5. `assemble_source_plan.py` rechecks the selected source commit, admission receipt,
   package identities and current fingerprints. It assembles every configured source
   plus the image runtime, packs browser Wasm, reconstructs every representation,
   and validates the final npm file inventory. Reused inner files stay byte-exact;
   their build commit is not replaced by the aggregation commit. `reuseInputs`
   records the previous immutable artifact and bound reused package manifests.
6. Only the existing same-repository/non-Dependabot publication boundary receives
   write permissions. The append-only publisher returns the actual published
   commit and manifest digest; YAML generation uses that receipt. A report failure
   cannot undo publication. Privileged PR reporting and intentional push/PR
   duplicate runs are unchanged.

Compilation fingerprints are conservative: shared bridge, toolchain, build and
validation changes may rebuild multiple sources. Another track's pin or profile
list alone does not change a stable binary fingerprint. Changing the set of
provided profiles causes that source's package to be regenerated; removed
profiles are never silently retained. No ccache hit is accepted as build evidence.
Compiler caches have distinct source partitions and paths, even if two tracks
happen to share the same upstream commit. No `max-parallel` cap is introduced.

First migration and common build changes need full compilation. With the initial
configuration this is 14 llama compile jobs plus six image jobs; host/native and
orchestration work are additional. Four is the usual *nightly compile* count,
not a promise about the total workflow's runner usage or elapsed time.

## Configuration and patches

`llama-cpp/config/sources.json` owns source IDs, independent vendor/pin paths,
provided profiles and patch plans. Stable keeps its existing vendor path;
nightly has an independent submodule. Add/remove existing profiles here rather
than copying bridge/scripts or maintaining a second hard-coded build matrix.
The updater checks only its selected source and creates a candidate branch;
a human-created PR remains the handoff after `GITHUB_TOKEN` push.

The two accepted llama upstream exceptions remain byte-identical and apply only
to build-tree copies. If the nightly pin no longer accepts them, stop and review:
failed application is not evidence of an upstream fix. New patch scopes need the
existing explicit approval. No fork is invented or added by this implementation.

## Artifact and consumer boundaries

```
manifest.json
runtimes/llama-cpp/sources/<source>/manifest.json
runtimes/llama-cpp/sources/<source>/profiles/<profile>/<variant>/core.{mjs,wasm,d.ts}
runtimes/llama-cpp/sources/<source>/api/...
runtimes/stable-diffusion-cpp/sources/upstream-default/...
packed/llama-cpp/catalog.json
packed/llama-cpp/runtime/...
packed/llama-cpp/data/...
```

Consumers explicitly select target IDs and a codec using the matching catalog
module. The selected file closure must not introduce an unselected full Wasm as a
base. Independent full alternatives exist for every offered target/codec, so
removing a base remains possible. A delta is direct from a selected full target;
there are no target-to-target delta chains. Select only matching glue/bindings too.

Brotli selections need no Zstandard decoder Wasm. The browser's Brotli capability
and an embedded distribution's final Base64/HTML size must be tested by the consumer.
Hosted and embedded distributions can choose different plans, but development and
release of *the same* distribution should use the same plan/decoder, not a raw
Wasm bypass. The producer keeps raw Wasm without duplicating it in a selected plan.

## Build-only encoder

`wasm-pack/prepare_ci_tools.py` prepares pinned Rust, wasm-tools, Cargo dependencies
and Python dependencies on a hosted Linux runner. No encoder/compiler runs during
consumer installation. See [wasm-pack](../wasm-pack/README.md) for the offline path,
format caps, selected runtime API and per-representation verification.

The runner-provided libzstd and C++ compiler are not pinned to one binary build.
The used libzstd version is reported. Cross-machine byte reproducibility of
compressed assets therefore is not claimed; reconstructed raw Wasm is exact.
Packing all same-runtime directions is bounded work inside one job, not a large
Actions matrix. Timeout and memory are finite; future source growth may require
an explicit pair policy or persistent encoding cache.

## Validation limits

Local tests exercise synthetic packages, real local Git repositories, selection
and reconstruction of supplied real Wasm. Synthetic smoke metadata is fixture
evidence only. No local test is evidence that GitHub Actions ran, the fixed
nightly C++ tree compiled, or real GPU inference succeeded. Missing vendored
headers must not be hidden by disabling their existing tests. Actual hosted-runner
validation, pinned upstream patch compatibility and consumer integration remain
separate acceptance checks.
