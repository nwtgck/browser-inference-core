# Build and artifact distribution

## Version boundaries

Source commits pin llama.cpp through a submodule gitlink. Use `git submodule update --init --recursive`; do not advance to a moving upstream tip with `--remote` during a build. The submodule HEAD must match `llamaCommit` in `config/toolchain.json`.

The pinned Emscripten Asyncify runtime needs its existing argument-preservation logic
enabled for Wasm32 bigint exports as well as Memory64. Toolchain setup changes only
the four relevant preprocessor guards; input and output SHA-256 pins are recorded
in repository-root `toolchain/config.json` and artifact provenance. Unknown source is rejected,
repeat setup is safe, and Asyncify builds verify that the selected compiler is patched.

The source commit, llama.cpp commit, and artifact commit are distinct identifiers. The manifest records the first two. The artifact commit is reported after publication through the Actions output and summary; embedding its own hash in the committed manifest would create a circular reference.

## Runtime package

The following layout describes the independently assembled llama component. The
public aggregate package places it under `runtimes/llama-cpp/`, uses a format-4
root manifest, and exposes it through `browser-inference-core/llama-cpp/*`. Each
component retains its format-2 manifest. See the [root package layout](../../../README.md).

The artifact root gets a separate `package.json`, not a copy of the development manifest. It has no build scripts, workspaces, dependencies, submodules, or build intermediates. `private: true` prevents accidental npm registry publication without changing the Git-based distribution model.

Each profile/variant runtime directory is copied as a unit to `profiles/<profile>/<variant>/`. `browser` disables Emscripten assertions and targets `web,worker`; `test` enables assertions and targets `web,worker,node`. Both are always included. Assertions can affect linked Wasm, so variants use independent build directories and their JavaScript/Wasm files must never be mixed. Do not assume every build produces exactly two files. Types, schemas, host helpers, a manifest, and license notices accompany the runtime. Manifest format 2 records separate provenance and validation under `profiles[profile].variants[variant]`. It covers every other file's path, size, and SHA-256; unexpected files are rejected. `npm pack` output is checked against the actual package tree.

Mixing source commits, upstream commits, or binding schemas across profiles is rejected. Dirty source builds cannot be published. A single file of 100 MiB or more is rejected before pushing. Exceeding that guard requires an explicit distribution change, not a silent conversion to Git LFS pointers.

## Publication

The first artifact commit has no parent. Later commits use the previous artifact tip as their parent. Each commit replaces the complete package tree so obsolete auxiliary files do not survive. Publication uses normal pushes only, never force-pushes.

When concurrent builds race, the losing publisher fetches the new tip and recreates its candidate commit on that parent. Existing commits and hashes remain intact. After 20 failed attempts, the job fails and can be rerun. A concurrency configuration that cancels earlier pending builds is deliberately not used.

Human pushes and `pull_request` opened/reopened/synchronize events build independently.
PR jobs check out the exact head commit, not the synthetic merge commit, and a
same-repository PR can publish before merge. Duplicate push/synchronize builds
are accepted; no cancellation or deduplication mechanism is added. Artifact
branches are excluded from source builds. The publisher checks the assembled
manifest against the selected source commit before writing to the remote.

The test, profile-build, and assembly jobs have read permission; only the publication job has write permission. Credentials are supplied through a process-local HTTP header rather than stored in the remote URL. Anyone who can push a source branch can change the resulting artifact; the workflow does not grant publication credentials to external pull requests or Dependabot-triggered runs. Fork PRs can still run the read-only build and test jobs.

## Parallel profile builds

Host/publication tests and the ten profile/variant builds run independently. Each pair
has its own runner, checkout, toolchain, and `build/<profile>/<variant>/` directory. The matrix
intentionally omits `max-parallel` so every pair can start as soon as a runner is
available, reducing build wait time. Do not add an artificial matrix parallelism
cap. A failing pair does not cancel the other profiles, so their diagnostics remain
available. Runner availability and account/repository limits still determine when
jobs actually start.

`python3 -m build_tools.runtimes.llama_cpp.staging` (from the repository root) transfers only the runtime directory, four generated API
files, and provenance for one profile/variant pair. CMake caches, object files, static libraries,
and SDK executables are not uploaded. The wasm64 WebGPU JSPI/browser job additionally supplies the
standalone Emscripten and Dawn notices using the same selection rules as local
packaging. Each shard has distinct paths; the assembly job merges them under
`build/`. That job checks out llama.cpp for its notices and browser-test templates,
but does not install Emscripten or compile the core again.

The existing `build` job remains the aggregate success check. Assembly requires
the host tests and every matrix build to succeed. It then uses
the existing packager, CPU Chromium tests (including the chat/tool surface),
transactional browser-validation recorder, and final package checks. No partial profile set is published.
Only the resulting `runtime-package` is passed to the single publication job.
The local, sequential build commands remain supported.

Parallel runners reduce elapsed time rather than the total amount of compilation.
They also repeat setup work and add artifact transfers; total runner usage can
increase. Profile build times and queueing determine the actual speedup.

## Release checks

Publication requires all five profiles in both variants, Chromium checks, and package verification. Chromium tests exercise both variants of the CPU profile and all four WebGPU profiles with JSPI/Asyncify suspension. Testing the assertions-enabled variant alone does not validate the browser variant; production files are exercised directly. The workflow compiles WebGPU but does not record GPU inference as verified. CPU smoke tests use a small untrained synthetic GGUF, not a quality benchmark or a multi-GiB model acceptance test.

An additional Node.js real-Wasm Asyncify regression uses the test variant and suspends a bigint-argument backend call
on an asynchronous mock adapter request, then verifies completion after rewind.
It requires neither a model nor a physical GPU and does not certify GPU inference.

Reproducible tests and CI verification logic belong in the source repository. Per-run logs and test-result JSON belong in ignored build output or CI artifacts, not committed documentation. Artifact manifests retain build provenance and the scope of checks actually performed.

## Retention

Consumers may depend on older artifact hashes, so do not delete or replace artifact history. Binary history still occupies space in the shared repository even when source branches contain no binaries. Limit fetch scope and depth where appropriate, and only compact storage in ways that preserve referenced commits.

## Update automation and consumer reporting

[Upstream updates and consumer reports](update-automation.md) describes the
browser-only stable/nightly/custom updater, artifact publication before PR
merge, exact npm lock fragments, compact consumer YAML and source-aware PR
comments. Reporting is outside the runtime package and does not change its format.
