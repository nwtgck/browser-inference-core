# Artifact integrity and source fingerprints

Published artifact commits contain
`browser-inference-core-artifact-integrity.yaml` at the package root. The same
file is included by npm packing. It records the source repository and commit,
an independent SHA-256 (Secure Hash Algorithm, 256-bit) fingerprint of committed
source content, and the complete root `manifest.json` as a YAML literal string.
YAML means YAML Ain't Markup Language; JSON means JavaScript Object Notation.

The existing root and runtime JSON manifests remain the consumer interface.
Their inventories bind payload paths, sizes and SHA-256 hashes. The YAML contains
the exact root JSON bytes, including its final newline, and its size and hash.
This lets one attestation cover all three runtime packages without signing every
Wasm (WebAssembly) file separately.

The YAML is the one explicit exception to the JSON inventory closure. Putting its
hash into the JSON that it contains would create a circular dependency. Package
validation permits this single named envelope only after decoding it and checking
its embedded JSON, source commit and identity against the actual root manifest.
Other unexpected files, symlinks, missing payloads and modified bytes still fail.
Ordinary assembly and local builds can omit the envelope; publication freeze
always creates it and later publication revalidation requires it.

## Publication sequence

1. All three runtime builds finish their existing validation gates.
2. The publish job validates source snapshots, prepared-source receipts, actual
   compiler inputs and package inventories, then creates a private package copy.
3. It fingerprints the selected committed source and recursively initialized
   submodules, creates the YAML, and checks the complete npm package list.
4. It stages the complete tree and verifies every Git blob against the validated
   SHA-256 inventory. A private version-3 receipt records the YAML's hash and the
   fixed tree identity.
5. `actions/attest` signs that saved YAML digest. Signing does not reread a mutable
   worktree file. The action records standard SLSA (Supply-chain Levels for
   Software Artifacts) build provenance for the workflow run.
6. Only a successful attestation permits the append-only artifact push. A push
   conflict changes the commit's parent, never the fixed artifact tree. The
   returned commit is confirmed against the receipt and the YAML Git blob.

The artifact commit is deliberately absent from the YAML: its tree includes the
YAML, so embedding that commit would introduce another self-reference. The
source commit and source fingerprint are already known before publication.

An attestation failure leaves the artifact branch unchanged. A successful
attestation followed by a failed push may leave an attestation for an unpublished
candidate; attestation existence alone does not establish publication. A later
reporting failure leaves the already published commit intact.

The publishing job needs `contents: write` for its existing Git push,
`id-token: write` to obtain its OpenID Connect (OIDC) signing identity, and
`attestations: write` to store the attestation. Storage records for container
registries are disabled, so `artifact-metadata: write` is unnecessary. See the
[action's inputs and permission contract](https://github.com/actions/attest).

## Verify the signer and exact YAML first

Use the repository and workflow you independently expect. Do not choose them
solely from the file being verified. For the upstream project:

```sh
gh attestation verify /path/to/package/browser-inference-core-artifact-integrity.yaml \
  --repo nwtgck/browser-inference-core \
  --signer-workflow nwtgck/browser-inference-core/.github/workflows/build.yml \
  --deny-self-hosted-runners
```

`gh` is the GitHub command-line interface. This verifies the signed subject
against the exact YAML bytes and the expected signing identity. Keep those bytes;
reformatting YAML or changing comments changes the subject hash. Review any
commands before running them, including the inline `reproduce` recipe in a signed
file. A valid signature establishes an origin, not the safety of arbitrary code.

The existing workflow permits ordinary same-repository pull requests as well as
source pushes; fork pull requests and Dependabot do not publish. Choose a stricter
source-ref or signer-digest policy when accepting only a particular branch or
reviewed workflow revision. The
[verification command reference](https://cli.github.com/manual/gh_attestation_verify)
describes those policy options.

### Pull-request source identity

For a pull-request event, GitHub's event/certificate source digest can describe
the synthetic merge commit. The workflow explicitly builds
`LCB_SOURCE_COMMIT`, which selects the pull-request **head** commit. Therefore
`source.commit` and `source.tree_fingerprint` in the signed YAML describe that
checked-out head. The certificate identifies the signing execution; the YAML
records the source statement verified by our workflow. Do not assume the two
commits are equal, or pass the YAML head to `gh --source-digest` on a pull-request
run. Inspect the verified certificate, workflow and signed source fields together.
The workflow file within the fingerprinted head can therefore differ from the
workflow definition used by the event. Audit the executed workflow at its signer
revision as well as the source head and its scripts.

## Verify the package and source

After verifying the YAML, use an audited source checkout at its `source.commit`.
The normal source-tool dependencies and TypeScript tooling emit must already be
available. For a clean extracted package, excluding Git checkout metadata:

```sh
(
  set -e
  test -f /path/to/package/browser-inference-core-artifact-integrity.yaml
  node build/tooling/tools/cli.js package verify \
    --checkout-root "$PWD" --directory /path/to/package
)
```

This validates the enclosing YAML against the root JSON and checks the complete
root/runtime payload closure, source fields, runtime contracts and npm packing.
It is local validation; it does not fetch or verify an attestation itself. Keep
the preceding signature check as a separate step. Local development packages
without an envelope remain supported by the package tool, hence the explicit
file requirement above when verifying a published package.

For the full source fingerprint, check out the recorded source commit and
initialize its submodules before going offline:

```sh
git submodule update --init --recursive --checkout
bash .github/scripts/treefingerprint.sh
```

Compare its output with the verified YAML's
`source.tree_fingerprint.value`. The YAML embeds the same committed script as
`source.tree_fingerprint.reproduce`, so the recipe can also be reviewed and
copied without depending on an external script download. Run it at the source
repository root. Bash, Git and GNU `sha256sum` are required; Python is not.

### Fingerprint format

`git-committed-tree-sha256-v1` hashes committed content, independently of Git's
SHA-1 object names. For each repository:

- Walk `git ls-tree -rz --full-tree` in Git's native recursive tree order.
- For modes `100644` and `100755`, hash the raw blob bytes with SHA-256. For mode
  `120000`, hash the symlink-target blob bytes without following the link.
- For mode `160000`, require an initialized submodule at the committed gitlink
  and recursively compute its committed-content fingerprint.
- Append `mode`, path bytes and the lowercase hexadecimal digest, each followed
  by a NUL byte. Hash the concatenation of those records with SHA-256.

The mode distinguishes executable files, ordinary files, symbolic links and
submodules. NUL separators preserve spaces, tabs, newlines and non-UTF-8 path
bytes. Missing objects, unavailable submodules, mismatching submodule heads and
unsupported entries fail without returning a successful fingerprint. The recipe
does not fetch missing objects.

The fingerprint includes committed workflow files, lockfiles and recursive
submodule content. It excludes working-tree modifications, index changes,
untracked files, Git administrative data, commit messages, timestamps and parent
history. Two commits with identical source trees can therefore have the same
fingerprint. Release-source verification separately requires the selected clean
checkout and binds the compiled input snapshots to that source. The fingerprint
does not describe an arbitrary dirty local build.

## Offline preservation

The signature bundle is stored by GitHub, separately from this YAML and artifact
tree. While online, download it and trusted root material using:

```sh
gh attestation download /path/to/package/browser-inference-core-artifact-integrity.yaml \
  --repo nwtgck/browser-inference-core
gh attestation trusted-root > trusted_root.jsonl
```

The download command prints/writes a digest-named JSON-lines bundle. Preserve that
file, the trusted roots obtained through a trusted channel, the exact YAML and
all payloads. Supply the downloaded bundle's actual path when offline:

```sh
gh attestation verify /path/to/package/browser-inference-core-artifact-integrity.yaml \
  --bundle /path/to/downloaded-bundle.jsonl \
  --custom-trusted-root /path/to/trusted_root.jsonl \
  --repo nwtgck/browser-inference-core \
  --signer-workflow nwtgck/browser-inference-core/.github/workflows/build.yml \
  --deny-self-hosted-runners
```

No source/package hashing needs internet access once the complete files and Git
objects are present. Offline signature verification uses preserved historical
trust information; it cannot determine whether a repository name has since
changed ownership. See the official
[bundle download](https://cli.github.com/manual/gh_attestation_download) and
[trusted root](https://cli.github.com/manual/gh_attestation_trusted-root) commands.

## Guarantees and limits

Under the trust assumption that GitHub's hosted runner, signing service and the
reviewed workflow behave correctly, a valid attestation binds this exact package
inventory and independently reproducible source fingerprint to the identified
workflow execution. Payload replacement without changing the signed YAML fails
hash verification. Changing the YAML requires a new accepted attestation.

The SHA-256 fingerprint strengthens independent source-content comparison. It does
not replace GitHub's Git-hash-based workflow or certificate identities, or make
the entire provenance chain independent of SHA-1. The audited workflow measures
and asserts this fingerprint; GitHub does not independently recompute and certify
the source tree's SHA-256 value. Expected signer/ref/revision policy and review of
the executed workflow remain trust assumptions.

This does not establish reproducible compilation, harmless source code, an
uncompromised compiler or dependency, current/latest status, or what a deployment
provider actually serves. A privileged malicious workflow can sign false claims;
review the signer revision and its transitive scripts/actions. The fingerprint
covers committed source content, while existing build receipts describe compiler
inputs, preparation and tests. Neither replaces behavioral validation.

For public repositories, signing metadata and subject names/digests may be public
through GitHub and Sigstore's transparency service. This design uploads a subject
name and digest, not the YAML contents as a predicate, but the YAML is itself
published in the public artifact branch. It contains source identities, the root
manifest and the recipe, not environment variables, credentials or local machine
paths. Treat public signing metadata as durable; deleting a branch or attestation
does not promise erasure from transparency records.
