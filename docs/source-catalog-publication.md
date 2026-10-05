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

Fingerprint record version 3 uses repository-relative paths for every input. A
runtime script such as `llama-cpp/scripts/package_runtime.py` and the root script
`scripts/package_runtime.py` are independent inputs; neither can overwrite the
other in the input inventory. Older records are deliberately invalidated once.
The publication manifest format and original raw Wasm identities are unchanged.

First migration and common build changes need full compilation. With the initial
configuration this is 14 llama compile jobs plus six image jobs; host/native and
orchestration work are additional. Four is the usual *nightly compile* count,
not a promise about the total workflow's runner usage or elapsed time.

## Configuration and patches

`llama-cpp/config/sources.json` owns source IDs, independent vendor/pin paths,
provided profiles and patch plans. Stable keeps its existing vendor path;
nightly has an independent submodule. Add/remove existing profiles here rather
than copying bridge/scripts or maintaining a second hard-coded build matrix.
The updater validates the committed registration and patch plan before checking
out only its selected source. It does not recursively fetch unrelated runtime
submodules. Both update and llama build workflows use `checkout_source.py`.
It checks the committed registration and the staged gitlink (the latter is what
`git submodule update` consumes), then rejects a dirty or unrelated populated
checkout **before** updating it. It explicitly selects `--checkout` rather than
inheriting a local `update=merge`, `rebase`, `none` or custom-command policy.
The resulting pinned HEAD and clean state are checked again. The helper neither
repairs a gitlink nor forces away local edits; it does not rewrite local config. The
updater creates a candidate branch; a human-created PR remains the handoff after
`GITHUB_TOKEN` push.

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

## Payload paths and toolchain notices

Artifact payload paths are not source/profile identifiers. Copied license trees
retain npm scope directories, for example
`licenses/1/node_modules/@jridgewell/gen-mapping/LICENSE`. The snapshot extractor
and both format-4 catalog validation paths use the same relative-path check, which
accepts `@` without renaming or dropping these original notices.

This is not a bypass for arbitrary archive input. Absolute paths, empty/dot/parent
components, backslashes, colon syntax, encoded separators and control characters
are rejected. Extraction independently rejects hidden control files, links,
case-colliding entries and size/count overruns. Format-4 reuse still requires full
manifest coverage, exact hashes and matching source identities. A safe legacy
format-3 snapshot is read but not reused; it requests a cold source build.

An `Unsafe package path` error during `fetch_snapshot` is a payload-validation
failure after source registration, not another request to stage or commit a
nightly gitlink. Rejected paths are included in bounded, escaped diagnostics.
Regression coverage collects scoped notices, publishes to a local Git remote,
fetches a fixed commit, validates npm's file inventory and verifies raw reuse.

## Build-only encoder

`wasm-pack/prepare_ci_tools.py` prepares pinned Rust, wasm-tools, Cargo dependencies
on a hosted Linux runner; Python needs no pip-installed dependencies. No encoder/compiler runs during
consumer installation. See [wasm-pack](../wasm-pack/README.md) for the offline path,
format caps, selected runtime API and per-representation verification.

The runner-provided libzstd is not pinned to one binary build.
The used libzstd version is reported; packing itself no longer requires a C++ compiler. Cross-machine byte reproducibility of
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


## Committed source registration and CMake selection

A source is identified by both its registered pin and a `160000` gitlink in the
**committed** superproject tree. A `.gitmodules` entry, an empty vendor directory,
a fetched checkout, and a staged-but-uncommitted gitlink are not substitutes.
The planner and updater share `llama-cpp/scripts/source_git.py`, which reports the
source, exact tree path, expected pin and observed entry on a mismatch. CI must
not repair its checkout or copy an observed vendor HEAD into the pin.

Git patch application is part of source registration, not just text editing.
A patch that changes submodule commits must be applied with `git apply --index`.
Plain `git apply` ignores the gitlink commit change; a later `git add -A` cannot
recover a commit ID that was never registered. `.gitmodules` alone is insufficient.
An incremental fix for a missing gitlink must actually contain the `160000`
addition. A diagnostics/helper-only patch does not repair that missing entry.

```sh
# After applying the appropriate patch with --index:
python3 scripts/register_source_gitlinks.py --index
git diff --cached --submodule=short
# Review and commit. The planner reads HEAD, not the staged index.
python3 scripts/register_source_gitlinks.py
```

`--index` is read-only. It verifies the staged registry/pins and `.gitmodules`
registration, refuses untracked or unstaged configuration, and checks each
staged gitlink. Its `commitRequired` result is not a successful HEAD check.
The default command remains a read-only check of HEAD. The repository's own
regression suite checks the actual committed source inventory too, not only
synthetic submodule fixtures.

For a previous file-only application, the explicit local migration helper remains
available: `python3 scripts/register_source_gitlinks.py --stage-missing` followed
by review and commit. This is an alternative to a gitlink-bearing repair patch,
not an extra prerequisite when the patch already staged that same gitlink. A
missing-gitlink patch is not applied on top of an already registered gitlink.

The mutation mode stages only absent entries. It verifies the selected registry,
staged pins and `.gitmodules` path/URL, rejects populated non-Git or dirty source
directories, refuses existing mismatching entries and deliberate removals, and
preflights all candidates before updating the index. It never fetches, commits or
changes an upstream pin. It is forbidden in GitHub Actions. Correct registrations
are a no-op. If an existing entry has the wrong commit, review the diagnostic;
`--stage-missing` deliberately cannot overwrite it. An uninitialized source can
remain an empty directory: Actions initializes the committed gitlink normally.

The real CMake build and the configure-only overlay harness both include
`llama-cpp/cmake/SourceConfig.cmake` before preparing copies. It resolves the
source ID, vendor path, pin and patch-plan dependencies from the same registry.
Unknown/empty source IDs fail early. An explicit `LCB_LLAMA_SOURCE` remains
available for native fixtures. Derived defaults are not cached, so a subsequent
source-ID change cannot silently retain a default from another source. A build
directory configured by the older implementation may already contain an explicit
cache entry; pass `-U LCB_LLAMA_SOURCE` or configure a fresh directory to clear it.
Do not silently discard intentional local overrides.

The vendor-independent regression suite runs real CMake with synthetic old-side
patch snippets. This checks default/source-switch/overlay wiring without a network
checkout; it is not evidence that the full pinned upstream compiles. The original
real-vendor overlay test is retained and prints the full configure diagnostic on
failure. Expected patch rejection tests capture and assert stderr so their output
cannot be mistaken for a failed positive test.

## Packed metadata is bound to original source identities

A catalog's internal round-trip is necessary but insufficient: it could correctly
reconstruct nightly bytes while labelling them stable. The v4 assembler and
validator derive the expected packed target map from each validated raw source
package, using the same function. If packing is present, it must contain exactly
all original llama browser targets, with the corresponding runtime/source/profile/
variant, raw path, length and digest. Missing/extra targets or altered coordinates
are rejected before invoking any decoder, including validation modes that skip
expensive packed reconstruction. The catalog bytes read for this check must still
match the root manifest's identity. JSON duplicates and non-JSON numbers are rejected
rather than interpreted differently by producer and consumer.

This validates producer coverage, not what a consumer must bundle. A consumer can
still choose any supported subset, independent full alternatives and codec-specific
entry points. The source-derived target map does not add unused profiles or change
raw Wasm, approved patches, default profiles, compression recipes or decoder APIs.
