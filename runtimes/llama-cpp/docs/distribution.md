# Build and artifact distribution

## Source and toolchain identities

The source commit pins llama.cpp through its submodule gitlink. Initialize the
selected revision with `git submodule update --init --recursive`; do not use
`--remote` during a release build. Runtime configuration and build tools are read
from the executing checkout. Private compiler inputs are prepared from the
selected Git objects, and publication rejects a dirty or mismatched source.

The pinned Emscripten Asyncify runtime needs its existing argument-preservation
logic enabled for Wasm32 bigint exports as well as Memory64. Toolchain preparation
applies the reviewed guard correction and records the source and result
identities. Unknown inputs are rejected and repeated setup is safe. This SDK
workaround is separate from the runtime's individual upstream patches.

Source, upstream and artifact commits have distinct meanings. Manifests record
the first two; publication reports the artifact commit after it exists. Embedding
that commit in its own package would create a circular identity.

## Runtime package

The aggregate package stores this independently assembled runtime under
`runtimes/llama-cpp-browser-core/` and exposes
`browser-inference-core/llama-cpp/*`. The root manifest uses format 5 and the
runtime manifest uses format 2. See the [package layout](../../../README.md).

The artifact has its own dependency-free `package.json`, without development
scripts, workspaces, submodules or compiler intermediates. `private: true`
prevents accidental registry publication; Git commit installation remains
supported. Each profile contains both `browser` and `test` variants. Browser
targets `web,worker` with assertions disabled; test also targets Node and retains
assertions. Their JavaScript and Wasm must stay together.

Typed staging and packaging functions copy the complete selected runtime output,
generated API/schema files, host helpers, documentation and notices. Do not assume
that every runtime consists of exactly two files. Each inner manifest inventories
its payload; the root inventory binds root files and the two inner manifests.
Final `npm pack` verification checks the actual aggregate package payload.

Mixing source commits, upstream commits or target schemas is rejected. Dirty
source builds cannot be published. A file of 100 MiB or more is rejected before
publication; changing this limit needs a deliberate distribution decision.

## Parallel builds and validation

`build.yml` has independent native/host checks and runtime build matrices: five
llama profiles and three image profiles, each in both variants. Matrix jobs emit
and check the selected runtime's tools, build private compiler inputs, and upload
staged runtime files and provenance. They do not upload SDK executables, object
files, static libraries or CMake caches as runtime artifacts.

The matrices use `fail-fast: false` and no artificial `max-parallel` cap. Runner
availability still limits concurrency. Each runtime assembly waits for its own
required checks and shards. The publication job joins the two validated runtime
packages and never publishes a partial profile set. Assembly does not compile the
core again. The native QSA comparison is a separate required gate.

`validation runtime` runs the saved package's browser matrix and writes a
source-bound receipt. CPU smoke uses a small synthetic GGUF, not trained-model
quality or large-model acceptance. WebGPU build coverage and mock-adapter
suspension checks do not establish physical GPU inference. The manifest records
what each variant actually exercised. Production browser files and test files
need their own checks; passing one does not validate the other.

The real-Wasm Asyncify regression suspends a bigint-argument backend call on an
asynchronous mock adapter and verifies rewind completion. It needs neither a
model nor a physical GPU and makes no GPU correctness claim. See
[verification](../../../docs/verification.md) for the separate evidence levels.

## Append-only publication

Publication freezes one validated aggregate snapshot, verifies its Git tree and
source binding, and creates an artifact commit. The initial commit has no parent;
later commits use the observed artifact tip as parent and replace the complete
package tree. Obsolete auxiliary files cannot survive from an older tree.

Pushes are ordinary fast-forward pushes. A publisher that loses a race fetches
the new tip and recreates its candidate on that parent, preserving existing
commits. After the bounded retry limit it fails; it never force-pushes.
Freeze/confirm/report reuse the same job's verified state as described in
[publication processing](../../../docs/publication-performance.md).

PR builds select the exact PR head commit; push and manual runs select their
event commit. Push and PR builds for the same source may both run and publish.
Artifact branches are excluded from source triggers. Test/build jobs have read
permissions; only the publisher has content writes. Publication excludes external
PR heads and Dependabot-triggered runs. Source writers can change the resulting
runtime build, so source-write access remains a trust boundary.

Network acquisition, cache/artifact transfers, npm resolution and remote writes
are explicit in workflow YAML. Local stages use literal
`docker run --network none` commands there. Reports and per-run logs stay outside
the installed runtime package. See [upstream updates and reports](update-automation.md)
for candidate branches, manual PR handoff and source-aware comments.

## Retention

Consumers can depend on old artifact commits. Preserve their objects and
contents; do not replace artifact history or rewrite an existing commit's bytes.
Binary history occupies repository storage even when source branches contain no
binaries. Limit fetch scope where appropriate and compact only while preserving
referenced commits.
