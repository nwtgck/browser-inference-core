# Publication validation and packing

The aggregate `browser-inference-core` package uses a format-6 root manifest.
It binds root files and the three format-2 runtime manifests; each runtime manifest
owns its complete payload inventory. Publication adds one separately validated
integrity YAML envelope around the exact root manifest. The combined inventories
and envelope cover the installed tree. Source tools, SDK executables and compiler
intermediates are excluded.

## Validation ownership

Runtime packaging verifies its selected profiles, generated ABI catalog, actual
exports, public declarations and notices. Aggregate assembly requires all three
independent runtimes from the same source commit. Independently supplied saved
packages are decoded at their entry and checked against their actual files.
Typed values then pass between local stages without repeated parsing.

`npm pack --dry-run --json --ignore-scripts --offline` checks npm's actual package
file list and sizes. Ordinary package/verify commands perform their requested
pack check. CI can defer that check during assembly because publication checks
the final private snapshot. It does not maintain a second npm packlist algorithm.

CI calls the offline `publication assemble-freeze` entry with the three saved
runtime packages. Local callers with an independently saved aggregate use
`publication freeze`. Both entries:

1. Validate the package, source identity and all three runtime inventories.
2. Write the validated byte inventory into a fresh private snapshot. The CI
   entry assembles directly in that snapshot.
3. Resolve the package's JavaScript and Wasm exports and type-check its public
   declarations from a temporary consumer. Collect source facts once, including
   prepared-source and actual compiler-input correspondence and the complete
   committed-source SHA-256 tree fingerprint.
4. Create `browser-inference-core-artifact-integrity.yaml` with the root JSON
   bytes and source fingerprint, then check npm packing for the complete snapshot.
5. Stage the snapshot in a private Git repository and verify the immutable
   tree's paths, entry types, sizes and blob identities.
6. Write a private receipt containing the tree, YAML/manifest identities and typed
   report data with its canonical digest.

The receipt remains outside the runtime package and is not a cache admission
record. The workflow attests the receipt's YAML digest before its visible online
push loop appends a commit using that same
verified tree. Conflicts retry on the new parent; they do not recompile, repack or
force-push. An accepted push with a lost response is resolved by ancestry checks.

## Confirmation and reporting

Once Git accepts the artifact commit, the publish job writes the four install
commands to its Summary and uploads the guide before the job finishes. A separate
`consumer-report` job uses that guide and its hash-bound publication facts for
npm resolution and reporting. Both reports refer to the original publication
attempt. The guide allows PR comments to link the immutable publication when
an independent check fails.

`publication confirm` reads this job's private receipt, checks its source/report
binding, and compares the returned artifact commit's tree, YAML and manifest blobs.
It reuses the verified typed report data. It does not rescan all payloads or
reproduce source preparation at every later local step. A different job or lost
receipt requires fresh package/source validation through the revalidation API.
Private receipt reuse is not a signature or proof for arbitrary third-party data.
The separately published YAML attestation supplies that external signature; see
[artifact integrity](artifact-integrity.md) for the verification contract.

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
packing, package closure, notices and public consumer imports. These tests use
synthetic runtime payloads and local repositories; they do not establish a
hosted workflow run or model inference. Timings depend on payload size, runner
capacity, cache state and network transfer. Packing fewer times is not by itself
a measured end-to-end speed result.
