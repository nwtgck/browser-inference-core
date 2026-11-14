# Upstream updates and consumer reports

## Responsibilities and network boundaries

Three workflows separate source updates, artifact publication and PR reporting:

| Workflow | Online operations visible in YAML | Local typed tools |
| --- | --- | --- |
| `update-llama-cpp.yml` | Release metadata, selected-ref fetch, branch lookup and ordinary candidate push | `tools/automation/upstream/` validates selection, prepares or verifies the candidate and renders a manual PR handoff. |
| `build.yml` | Tool acquisition, cache/artifact transfers, append-only Git publication and isolated npm resolution | Runtime tools, `tools/package/`, `tools/validation/` and `tools/publication/` build and verify the selected source and package. |
| `runtime-comments.yml` | PR/run/artifact reads, bounded report download, final freshness reads and comment POST/PATCH | `tools/automation/comments/` validates captured inputs and prepares one bounded comment body. |

Local workflow stages use literal `docker run --network none` commands in YAML.
Destinations, credentials, cache transfers and remote writes remain at their
online execution sites. Local typed tools do not hide GitHub API calls or remote
Git operations. No consumer checkout is cloned or edited.

The emitter and dependencies used by offline tools come from the executing
checkout. CI compiles the selected tool project before invoking its emitted
Node entry; it does not start another CLI to call an internal typed operation.

## Installation and permissions

The reporter and its helper code must be available on the trusted default
branch. It checks out that branch rather than executing a PR's code. The selected
source branch must contain the updater/build implementation used for that run.
An update candidate does not have to be merged before its artifacts are usable.

Repository rules must allow `feature/automation/llama-cpp/*` candidate branches,
same-repository PR workflows and the configured artifact branch. The updater
creates neither PRs nor approvals, so it does not need permission to create or
approve PRs. Its update job requests content writes for ordinary branch pushes;
read-only permission cannot push a candidate. The build workflow reserves content
writes for publication. The reporter separately requests the reads and comment
writes its YAML uses.

Publication excludes external PR heads and Dependabot-triggered runs. A source
writer can change the runtime build; the automation does not turn source-write
access into an untrusted-input boundary. Repository rules and token restrictions
can still deny an operation. Private repositories need their own authorized
retrieval path; public raw/archive locators do not confer access.

## Selecting an upstream revision

Open **Actions > Update llama.cpp > Run workflow** and select a source branch.

| Input | Meaning |
| --- | --- |
| `target: latest` | Latest published full release, required to be a non-draft, non-prerelease `vX.Y.Z` tag. |
| `target: latest-unstable` | First published `bNNNN` prerelease in the newest-first release listing, including pagination. |
| `target: custom`, `custom_ref: ...` | Explicit tag, branch or full lowercase 40-character commit. `refs/tags/...` and `refs/heads/...` disambiguate names. |
| `allow_non_fast_forward` | Explicit consent to select an older or divergent upstream commit; false by default. |

There is no stable-to-nightly fallback or implicit selection of `master`.
A named ref is resolved once, peeled to a commit and recorded by full identity.
Its name remains a locator rather than an immutability guarantee.

The clean source checkout supplies the existing pin through the llama.cpp
submodule gitlink. An unchanged target produces no candidate commit. A new
candidate changes only `runtimes/llama-cpp/vendor/llama.cpp`; runtime toolchain
configuration does not carry a duplicate upstream commit pin.

Candidate creation captures the clean parent commit, checks the complete index
and rechecks the staged gitlink after preflight. It uses that explicit parent
when constructing the commit. A concurrent HEAD/index mutation cannot silently
retarget the update. The overlay preflight prepares private source copies; it
never modifies the checked-out upstream. A failure remains visible in the
candidate report and can be repaired manually. Patch application alone is not a
compile, browser or model-inference result.

## Candidate reuse and manual PR handoff

Candidate names start with `feature/automation/llama-cpp/` and encode the base
identity, source commit and selected target. A rerun validates and reuses an
existing candidate, preserving manual repair commits. It does not rebase,
force-push or push the same existing candidate again. Different pins are rejected.
A changed source base creates a different candidate. Remote races fail safely
rather than overwriting a branch.

The report supplies **Open pull request form**, **Compare branches** and **Update
branch** links. The form suggests the title/body; a human submits it. Long
prefilled URLs fall back to the plain form with the full suggested text retained
in the summary/report. The updater does not create, reopen, close or dispatch a
PR or runtime build. Reusing a candidate alone therefore does not rerun CI.
A human push, PR reopen or manual build remains an independent trigger.

## Source-bound builds and append-only artifacts

Human-created PR events build the exact head SHA rather than the synthetic merge
commit. Push/manual events use their selected event SHA. The publication boundary
checks this identity against the assembled manifests before writing remotely.
These are head builds, not merge-integration tests. Draft status does not suppress
builds in this workflow.

A push and a PR synchronization can both build the same source. They are allowed
to complete independently, and can append distinct immutable artifact commits.
The publisher handles races with ordinary fast-forward retries. The artifact
branch tip can represent any successful source branch; consumers must pin the
reported artifact commit, not its moving tip or an upstream source commit.

Five llama profiles and three image profiles each have browser/test variants.
The two runtime build/validation paths remain independent until aggregate
publication. See [distribution](distribution.md) for the build graph and
[publication processing](../../../docs/publication-performance.md) for snapshot,
Git verification and same-job report reuse.

## Report contents and npm resolution

After publication, an isolated dependency-free npm fixture resolves the exact
Git dependency with install scripts, audit and funding requests disabled. Its
lock entry must match the expected repository, artifact commit and package
version, contain no unexpected dependencies or hooks, and retain npm's integrity
when supplied. The fixture's complete lock is not a replacement for a consumer's
other dependencies or root metadata.

The report records the distinct source, upstream and artifact identities; the
root manifest's format/bytes/digest; immutable retrieval locators; npm resolution;
source facts; and each runtime's profile configuration and actual validation.
The package is `browser-inference-core`. Physical runtime directories are
`runtimes/llama-cpp-browser-core/` and
`runtimes/stable-diffusion-cpp-browser-core/`; public imports retain their
`llama-cpp` and `stable-diffusion-cpp` prefixes.

`metadata.json`, `report.md` and `package-lock-entry.json` are report outputs.
When the caller supplies both the GitHub run ID and rerun attempt, reporting also
writes the source-bound `report.json` envelope, `consumer-update.md` and
`consumer-update.yaml` used by the comment workflow. This metadata belongs outside
the runtime package. It does not modify a consumer repository, approve API
compatibility, invent an artifact identity or turn a digest into a behavioral
check. See the actual emitted metadata for the facts available to a consumer.

Archive container bytes are not the payload identity: verify the extracted
manifest and its inventories. A source archive omits submodule contents; upstream
commits identify those inputs separately. Retrieval failures and unavailable
consumer dependencies remain real limits even when metadata identifies the
correct pins.

## Comment lifecycle and trust

The reporter operates on trusted default-branch code. It never installs PR
packages, evaluates downloaded YAML, executes uploaded scripts or extracts a
report over its checkout. It reads only bounded, named report entries and checks
the envelope's repository, source SHA, run ID and attempt against the selected
run. A malformed report is unavailable for that PR rather than a success.
Authenticated artifact-location reads and credential-free archive downloads are
separate steps; redirect downloads do not receive the GitHub API credential.

`workflow_run` events cover requested, in-progress and completed runs.
`pull_request_target` events refresh PRs opened after a build or whose heads
change, without running those heads. Eligible builds come from the same
repository's source push, PR or manual events. A synthetic merge SHA cannot be
relabeled as the PR head.

Reconciliation queries current open PRs, matching run attempts and bot-owned
comments. Only comments owned by `github-actions[bot]` with the report marker are
updated. A changed head, pending/failed run or missing report never inherits an
unqualified old success. Previous successful metadata is explicitly historical.
Run start time and attempt participate in freshness; a slow older completion
cannot simply overwrite a newer result. The final API reads reduce races, but
GitHub's comment API cannot atomically lock a PR head against a concurrent push.
A later event reconciles that remaining race.

Comment size is bounded. Large metadata stays in the report artifact with its
byte count/digest and a link rather than silent truncation. Artifact expiry can
make the full report unavailable later even while its PR comment remains.
Ambiguous comment writes are not blindly retried; reconciliation locates the
bot comment again.

## Failure recovery and local checks

An incompatible overlay requires a source repair. A denied/racing candidate push
can be rerun without force-pushing. A report or npm-resolution failure after
publication does not roll back the immutable artifact; its report is incomplete.
Comment reconciliation can be rerun without rebuilding the runtime. Expired
report data requires a fresh reporting/publication run with its inputs.

Typed tests under `tests/tools/automation/` and `tests/tools/publication/` exercise
captured API data, bounded archives, local Git repositories, npm packing,
selection/candidate mutation boundaries and report freshness. Workflow tests
under `tests/tools/ci/` check local orchestration. Run their emitted Node tests as
described in [verification](../../../docs/verification.md). Local stand-ins do not
establish actual hosted permissions, remote npm resolution, browser execution or
real-model performance.
