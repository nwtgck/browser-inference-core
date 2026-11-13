# Publication validation and packing

Publication keeps full source, provenance, license, size and payload validation
while packing the private publication snapshot once. The aggregate package is
`browser-inference-core`, with manifest format 4 and the two independent runtime
trees beneath `runtimes/`. It contains runtime assets and public type declarations;
development tools and intermediate build trees are excluded.

`npm pack --dry-run --json` still compresses the package to obtain its file list.
The pipeline therefore assigns npm packing to one boundary instead of repeating
it at every payload check. No alternative packlist implementation or dependency
is introduced, and no compression settings are changed.

## Validation ownership

| Boundary in the publish job | Full payload/provenance checks | npm packing |
| --- | --- | --- |
| Inputs and assembled temporary tree | Every time | Deferred explicitly by `--defer-npm-pack` |
| Publisher input | Yes, before copying can dereference links | Deferred to the private snapshot |
| Publisher's private snapshot | Yes, before Git initialization or remote access | Each runtime separately, then the root package (3 calls) |
| Staged immutable Git tree | Every raw blob and root manifest, before remote access | Not repeated |
| Lock fixture preparation and report input | Yes | Reused only with this publication's matching manifest digest |
| Report input, after lock resolution and provenance collection | Yes again, against the same digest | Not repeated |
| End of metadata construction, before report writes | Bound metadata snapshots and another full payload check | Not repeated |

The default assembly/verification commands and each runtime's validator still
perform npm packing. The publisher has **no skip-packing option**. Standalone
report generation without a publisher digest runs all three pack checks before
reading the separately resolved lock; it revalidates the same payload afterward without compressing
it again.

The publish job invokes npm packing 0 + 3 + 0 times across aggregate assembly,
publication snapshot validation and consumer reporting. This count does **not** include npm's separate work to generate the consumer
lock. Timings depend on actual payloads, runner capacity, Git transfer and npm's
network resolution, so the call-count reduction is not a wall-clock guarantee.

## Exact-tree handoff to reporting

The offline `build_tools.publication.artifacts freeze` command validates and
copies the package, performs all three npm checks and verifies every immutable
Git blob. Its required `--source-repo` input also checks the captured source and
prepared-source receipts before Git initialization. The nested llama manifest is
read from the private snapshot, and its exact parsed bytes must match the root
manifest before the collector reproduces the selected source preparation.
It writes a private receipt containing the frozen tree, source commit
and manifest identity. The literal online push loop in `build.yml` extends the
fetched artifact tip using this tree; it never force-pushes or re-packs on a retry.

After a reported push failure, the workflow checks remote ancestry before
classifying the failure. If the server accepted the commit but lost its response,
that original commit is recognized even when another publication has already
advanced the branch. Conflicting ordinary pushes retry against the new parent.
A permission or unrelated transport failure does not authorize a rewritten branch.

The offline `confirm` command checks the returned commit's tree and manifest
against the frozen receipt, then records the paired `commit` and
`manifest-sha256` outputs and the early installation summary. Those same-job
outputs feed the npm fixture and reporter. No receipt is added to the distributed
package. Later worktree changes cannot alter the already verified Git tree.

`--published-manifest-sha256` is a trusted same-job handoff, **not** a signature or
an independently verifiable proof of remote publication. Do not compute it from
an unverified package, copy it from another job or restore it from a cache. Omit
the option for independent/standalone verification.

The reporter rejects an empty, malformed or mismatching digest. A matching
manifest alone is insufficient: it still hashes and checks every payload, both
inner manifests, required files, runtime configuration, source identity and
cleanliness. The root manifest covers all files, including inner manifests.
Updating a payload and regenerating the manifests therefore also invalidates
the handed-off digest. Revalidation after lock/provenance work prevents a later
change from being silently reported as the earlier package.

The existing immutable source-commit check remains before publication. npm lock
resolution still uses npm itself with an isolated empty cache, the actual
published Git commit, and all existing lock-entry checks. Integrity values are
not reconstructed locally. A reporting failure still leaves the already
published commit and its job output intact; it never rewrites history.

## Checkout and diagnostic scope

Only the publish job changes its submodule checkout. It initializes
`runtimes/llama-cpp/vendor/llama.cpp` at the source commit's gitlink, shallowly and without
recursion. The pre-publication receipt gate and consumer provenance read the
selected llama source and source-preparation inputs from this checkout; image
provenance comes from the downloaded, fully checked manifests.
Compilation and test jobs retain their existing recursive checkouts. No upstream
pin or patch changes.

Wasm header checks read exactly eight bytes instead of loading the entire Wasm
file to slice its first eight bytes. Full streaming SHA-256 checks are retained.

The publication and reporting Python stages run in explicit offline containers.
The fresh npm fixture is prepared offline; the exact `npm install
--package-lock-only --ignore-scripts --no-audit --no-fund --lockfile-version=3`
command and its isolated cache/configuration are visible in YAML. Reading and
validating npm's resulting lockfile is offline. No Python report helper acquires
packages or calls GitHub.

## Local regression coverage

`python3 -m unittest tests.build_tools.packaging.test_publish_optimization` exercises
real npm packing, the workflow's selective Git checkout, local bare-Git pushes,
append-only history, old-commit installation and an empty-cache `npm ci`.
Fixtures contain synthetic Wasm headers: these tests do not establish compilation
or real-model inference. The byte-publication harness uses local Git remotes and
does not initialize real upstream sources. Focused gate tests inject a failing
source collector and a mutation during source verification, and prove that neither
can emit a publication receipt. Report tests substitute lock and provenance data;
the source collector has separate capture and preparation tests. The workflow
performs actual remote acquisition in its explicit online steps.

## Adversarial review of the publication boundaries

The staged filesystem is not necessarily the Git tree. Global Git ignore rules
can drop files and attributes/clean filters or line-ending normalization can
change bytes during `git add`. A file can also change after its npm check. After
`write-tree`, the publisher therefore checks the immutable tree before any remote
access: exact path set, regular-file entry types, and every blob's size/SHA-256,
including the root manifest. It uses NUL-framed `ls-tree` output and one raw
`cat-file --batch` process, with at most one MiB per payload read. On a mismatch
it fails rather than silently changing the user's Git settings. That same tree
is reused for every append-only retry, so later worktree changes do not change
what is committed. These checks add no npm compression.

Reporting captures small metadata files as byte snapshots, checks those bytes
against the validated root manifest, then parses them. The llama metadata builder
accepts the captured manifest and hashes exactly what it parses, rather than
reading a potentially different manifest a second time. Both unchanged and
changed-then-restored files are covered: a late replacement cannot poison the
in-memory metadata. The reporter also repeats payload validation at the end of
metadata construction, before writing report files. npm packing stays deferred
on both post-publication checks.

All three validators now reject directory symlinks, dangling symlinks and special
files before opening JSON metadata. A file-only listing previously ignored
these entries. Peer dependency and bundled dependency declarations (including
both bundle field spellings) are rejected alongside the existing dependency and
lifecycle-hook checks. Ordinary packages produced by this repository have none
of these fields, so their dependency-free installation behavior is retained.

`tests/build_tools/packaging/test_publish_adversarial.py` injects faults into Git staging, the final npm
check and report construction. It also checks real local Git push conflicts,
permission/packing failures, immutable-tree reuse, unusual filenames and bounded
blob reads. These checks defend the content handoffs of this trusted job, not a
runner where an attacker controls the Python/Git/npm executables, environment,
credentials or arbitrary concurrent code. The receipt remains a trusted same-job
handoff, not a signature. Lock acquisition remains an npm operation in YAML; fault tests use
fixed lock data and make no GitHub publication or inference claim.
