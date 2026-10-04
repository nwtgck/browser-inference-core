# Independent upstream source tracks

`config/sources.json` owns the available upstream tracks and their profile lists.
`browser` and `test` remain variants; they are not release channels. A configured
nightly is a pinned released upstream commit, not a moving branch fetched at build
time. The initial pins come from the supplied artifact snapshots.

The default stable source retains `vendor/llama.cpp` and `config/toolchain.json`.
The independent nightly source uses `vendor/llama.cpp-nightly` and
`sources/upstream-nightly/pin.json`. Shared bridge/build logic is not copied.
Nightly currently offers `webgpu-wasm64-jspi` and `cpu-wasm32`; editing the source's
profile list changes the generated candidate matrix. Adding an unknown execution
profile still requires a real entry in `config/profiles.json` and corresponding
runtime capability support. Another fork would receive its own ID/repository/pin,
not overwrite an existing source.

## Patches

Each source has a separately reviewed `patches.json` plan referring to the existing
explicit-approval patch register by file and SHA-256. This change adds no upstream
patch and changes neither accepted patch's content. The plan and checksum checks
fail closed; an apply failure does not mean the upstream fixed the problem. Patch
preflight uses a temporary build copy, never the vendor checkout. Removing or
changing an approved exception remains an explicit review decision.

`build.py --source SOURCE --profile PROFILE --variant VARIANT` records source ID,
repository, patch series, original bicore commit and a content-based build input
fingerprint. Non-default sources use `build/sources/SOURCE/PROFILE/VARIANT`.
Staging verifies the chosen source and toolchain. Package smoke derives its exact
expected profile set from the source configuration and tests only those profiles.
An absent Asyncify profile is not reported as an Asyncify test success.

## Update and candidate build workflows

- `update-llama-cpp.yml`: stable manual entry plus the shared reusable updater.
- `update-llama-cpp-nightly.yml`: independent manual nightly update entry.
- `build-llama-source.yml`: manual/reusable candidate build for a configured source.

Update safety is shared: exact ref resolution, pin/gitlink agreement, clean working
tree, non-force push, source-specific patch preflight and manual pull-request
handoff. Release policy, profiles and patch plans are source-specific. The updater
changes exactly its selected pin file and gitlink. It neither creates a pull
request nor dispatches another workflow. There is no new periodic schedule.

The candidate build validates its source before creating the dynamic matrix. The
initial nightly has four compile shards (two profiles, two variants). There is
no artificial max-parallel cap and no image/stable compilation in this candidate
workflow. It uploads a source candidate only, never replaces a complete artifact.
Existing push/pull-request triggers and their intentional duplication are unchanged.

## Aggregate artifacts and safe reuse

`scripts/package_sources.py` assembles the configured source set into:

```
runtimes/llama-cpp/sources/SOURCE/
runtimes/stable-diffusion-cpp/sources/upstream-default/
packed/llama-cpp/                 # optional, plus its matching decoder
```

Each inner package retains its own build sourceCommit, source-specific API/glue,
raw Wasm and notices. The root v4 sourceCommit identifies the aggregation, not a
falsified rebuild of reused inputs. Raw Wasm remains available even when packed
alternatives are supplied. The matching consumer report is generated from bound
manifest snapshots rather than inspecting a different source checkout for reuse.

`plan_source_build.py --previous DIRECTORY --previous-manifest-sha256 DIGEST`
validates a caller-supplied immutable prior v4 snapshot, then admits a source only
if every selected profile/variant has the same build fingerprint and complete
clean compiled/browser validation evidence. Otherwise it schedules that source.
Shared code/toolchain changes propagate; other tracks' pin changes do not. This
planner does not fetch or authenticate a remote branch and does not perform final
publication. A trusted artifact identity must be obtained before calling it.

**Remaining integration:** the legacy `build.yml` still publishes v3. It has not
been changed to fetch prior v4 packages, apply this reuse plan, or publish the new
catalog. Those steps need end-to-end workflow and concurrency tests before switching
the existing publisher. Do not describe a manual nightly candidate as a complete
nightly-only update of the published root package.
