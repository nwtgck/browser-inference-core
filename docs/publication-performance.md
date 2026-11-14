# Publication validation and packing

The aggregate `browser-inference-core` package uses a format-5 root manifest.
It binds root files and the two format-2 runtime manifests; each runtime manifest
owns its complete payload inventory. The combined inventories cover the installed
tree. Source tools, SDK executables and compiler intermediates are excluded.

## Validation ownership

Runtime packaging verifies its selected profiles, generated ABI catalog, actual
exports, public declarations and notices. Aggregate assembly requires both
independent runtimes from the same source commit. Independently supplied saved
packages are decoded at their entry and checked against their actual files.
Typed values then pass between local stages without repeated parsing.

`npm pack --dry-run --json --ignore-scripts --offline` checks npm's actual package
file list and sizes. Ordinary package/verify commands perform their requested
pack check. CI can defer that check during assembly because publication checks
the final private snapshot. It does not maintain a second npm packlist algorithm.

The offline `publication freeze` entry:

1. Validates the package, source identity and both runtime inventories.
2. Copies the validated byte inventory into a fresh private snapshot.
3. Checks npm packing for that aggregate snapshot.
4. Collects source facts once, including prepared-source and actual compiler-input
   correspondence for both runtimes.
5. Stages the snapshot in a private Git repository and verifies the immutable
   tree's paths, entry types, sizes and blob identities.
6. Writes a private receipt containing the tree, manifest identities and typed
   report data with its canonical digest.

The receipt remains outside the runtime package and is not a cache admission
record. The workflow's visible online push loop appends a commit using that same
verified tree. Conflicts retry on the new parent; they do not recompile, repack or
force-push. An accepted push with a lost response is resolved by ancestry checks.

## Confirmation and reporting

`publication confirm` reads this job's private receipt, checks its source/report
binding, and compares the returned artifact commit's tree and manifest blobs.
It reuses the verified typed report data. It does not rescan all payloads or
reproduce source preparation at every later local step. A different job or lost
receipt requires fresh package/source validation through the revalidation API.
Private receipt reuse is not a signature or proof for arbitrary third-party data.

`publication npm-fixture` prepares an isolated dependency-free npm consumer.
The exact online lock-resolution command, credentials and fresh npm cache stay
in workflow YAML. The offline reader checks the resulting lock against the
expected repository, immutable artifact commit, package version and dependency/
install-hook restrictions. Integrity values come from npm's output.

`publication report` combines that resolution with the confirmed report data.
It writes package/source locators, runtime profile and validation facts, Markdown,
and the complete selected lock entry. Workflow run ID and attempt are explicit
paired inputs when producing a comment-consumer envelope. Reporting performs no
remote calls and leaves the already published artifact commit unchanged on error.

The root manifest digest, runtime manifest digests and report-data digest bind
different owned values. They do not replace semantic ABI or numerical checks.
Git-tree verification also protects the staging boundary against changed bytes,
filters, ignore rules and line-ending conversion. Local filesystem checks do not
claim to protect a runner whose tools or credentials are controlled by an attacker.

## Execution and regression scope

Local preparation, packing, source verification and reporting run as typed Node
tools in network-disabled containers started by `.github/scripts/run-offline.sh`.
Command bodies, network acquisition, Actions transfers, npm resolution and remote
Git writes remain inline in workflow YAML.

`tests/tools/publication/` covers source collection, immutable Git bytes, frozen
report reuse and npm-lock constraints. `tests/tools/package/` covers real npm
packing, package closure, notices and generated consumer types. These tests use
synthetic runtime payloads and local repositories; they do not establish a
hosted workflow run or model inference. Timings depend on payload size, runner
capacity, cache state and network transfer. Packing fewer times is not by itself
a measured end-to-end speed result.
