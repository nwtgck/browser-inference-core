# Upstream updates and consumer reports

## Scope and boundaries

The automation has three separate responsibilities:

1. `update-llama-cpp.yml` resolves an upstream revision, proposes the submodule
   gitlink and `config/toolchain.json` change, and pushes a candidate branch. It
   ends there, with a browser link to a prefilled PR form. PR creation is manual.
2. `build.yml` retains its ten profile/variant builds, tests, package validation,
   and append-only artifact publication. After publication, it generates the
   consumer report with npm-resolved lock data. A human-created PR starts this
   workflow through `pull_request`; ordinary pushes remain independent triggers.
3. `runtime-comments.yml` reports the latest source-bound state on open,
   same-repository PRs. It can run again without compiling or publishing again.

The runtime package, exported interfaces, manifest format, and profile policy do
not acquire consumer-specific inputs. No consumer repository is cloned or edited.
Consumer knowledge exists only in the reporting generator, with its intentional
policy exception documented immediately above that implementation. The generated
YAML describes artifact facts, correspondences and historical search hints; it
is not an instruction prompt for an assistant.

## Auditable GitHub operations

`scripts/github_api.py` owns the HTTP API boundary. `GitHub` exposes named
operations for releases, refs/tags/commits, comparisons, pull requests, build
runs, report artifacts and conversation comments. The only API mutations are
comment creation and editing, with explicit arguments and fixed payload shapes.
There is no PR-creation or workflow-dispatch method. The updater uses only
upstream read operations from this client. Calling scripts do not supply HTTP
methods, endpoint paths or arbitrary payload dictionaries.

The reporter still reads `/actions/workflows/build.yml/runs` to find build results.
This is a GET operation, not a call to `/dispatches` and not a build trigger.

`_request`, `_pages`, headers and the token are private implementation details.
Authenticated API requests do not follow redirects. Only the artifact download
operation follows GitHub's signed HTTPS storage redirect, using a new request
without the API credential and retaining the caller's archive size bound. These
are Python visibility conventions backed by tests, not a sandbox or a substitute
for GitHub token permissions. Git CLI fetch/push operations remain separate in
the updater/publisher and their existing Git helpers.

## One-time repository setup

The workflow files **and their Python helpers must be on the default branch**.
GitHub only discovers `workflow_dispatch` and `workflow_run` triggers there. The
source branch selected in the updater's **Run workflow** menu must also contain
this implementation. This is a one-time installation requirement; subsequent
upstream-update PRs do not need to be merged before their artifacts are usable.
When upgrading from the previous dispatch-based design, the trusted default-branch
`scripts/runtime_comments.py` must also receive its `pull_request` run support.
Changing only a feature branch's build workflow cannot change the reporter's
trusted implementation. The chosen source branch must contain the branch-only
updater and PR-capable build workflow as well.

Repository/organization rules must permit creation of the
`automation/llama-cpp/*` branches, ordinary same-repository PR workflows, and the
existing artifact publisher. **Allow GitHub Actions to create and approve pull
requests is not required**: the updater does not create or approve PRs.

The updater workflow defaults to `contents: read`, but its **update job overrides
that default with `contents: write`** for branch push. Workflow-level permissions
are defaults, not an upper bound on a job's explicit permissions; `write` includes
`read`. Changing only the job to `contents: read` would prevent candidate pushes.
The workflow does not need `actions: write` or `pull-requests: write`, and there is
no reason to grant write access to every job by changing the workflow default.
Repository rules can still deny a push despite the job's requested permissions.
In the build workflow only the
publisher has content writes. The reporter separately needs content/action reads
and PR-comment writes. No personal access token, GitHub App, auto-approval or
auto-merge is introduced. Fork PRs can run the read-only build/test jobs, but
publication is limited to same-repository heads and excludes Dependabot-triggered
runs, whose credentials may be read-only. This does not elevate a fork merely
because a maintainer opens or reruns its PR. The existing trust boundary remains:
someone able to push source code in this repository can change its runtime build.

Public GitHub artifact/raw URLs are provided in the consumer metadata. Private
forks need their own authorized retrieval path; a repository-scoped token cannot
be assumed to grant cross-repository package access.

## Browser-only upstream update

Open **Actions > Update llama.cpp > Run workflow** and choose the source base
branch. Inputs have these exact semantics:

| Input | Meaning |
| --- | --- |
| `target: latest` | GitHub's latest published full release; additionally required to be a non-draft, non-prerelease `vX.Y.Z` tag. |
| `target: latest-unstable` | First published `bNNNN` pre-release in GitHub's newest-first release-list order, following pagination. It is not `master` HEAD. |
| `target: custom`, `custom_ref: ...` | An upstream tag, branch, or full lowercase 40-character commit. `refs/tags/...` and `refs/heads/...` disambiguate colliding names. `master` is available only as an explicit custom ref. |
| `allow_non_fast_forward` | Explicit consent for a target older than or divergent from the existing upstream pin, including a switch back from nightly to stable. Defaults to false. |

Release status and tag shape are both checked. There is no fallback from stable
to nightly, from missing releases to branches, or from a failed lookup to cached
pins. Named refs (including annotated tags) resolve to a full commit **once**;
only that commit is fetched and recorded. A ref name is a locator, not a claim
that the upstream tag is immutable.

The source checkout must be clean and its two existing pins must agree. A no-op
update creates no commit or branch; no PR or build is ever created by the updater. Otherwise only these paths are staged:

- `vendor/llama.cpp` (the gitlink, not patched vendor source)
- `config/toolchain.json` (only `llamaCommit` changes)

The current overlay is applied to a temporary copy as a cheap preflight. Passing
this check means patch application passed, **not** that compilation, browser
execution, or model inference passed. A failed preflight still pushes the
candidate branch and reports diagnostics. It does not create a draft PR or
launch a build. The patch is not fuzzed, dropped, or rewritten automatically.
Opening a PR is the user's decision; a **draft PR is not a build-suppression
mechanism** in this workflow. It can start the same builds as a non-draft PR.

After the updater finishes, its summary has **Open pull request form**, **Compare
branches**, and **Update branch** links. The PR form has a suggested title and
body containing the resolved upstream pins and preflight scope. Visiting the
form does not create a PR; the human submits it in GitHub. Raw preflight errors
stay in the report rather than being copied into a long query URL. Very long
prefilled URLs fall back to the plain form, retaining the suggested text in the
summary and `upstream-update-report` artifact.

Candidate branch names contain the base branch identity, base source commit and
upstream target commit. A rerun reuses an existing candidate and preserves any
manual repair commits on that branch. It never rebases or force-pushes. Different
pins on an existing candidate are rejected. A changed base commit produces a new
candidate rather than overwriting an earlier one. Concurrent conflicting branch
creation fails safely and can be rerun. Branch existence is probed before fetch;
validation and the reported source commit use the actual fetched `FETCH_HEAD`.
A human commit made between those two reads is therefore included in that
snapshot, and changed pins are rejected rather than validated against an older
probe result. The branch can still advance after fetch; the reported commit is
an immutable observation, not a lock on the remote branch.

An existing candidate is read and validated, **not pushed again**. Its PRs are
not queried, created, reopened, edited, or closed; their lifecycle belongs to the
user. If a PR is already open, rerunning the updater alone does not rerun CI. A
human source push, a human PR reopen, or a manual run of **Build and publish
runtime** is available independently. No new dispatch call is hidden in retries.

## Event handoff and source identity

```text
Update llama.cpp (manual workflow run)
  -> resolve upstream, update two pins, preflight, push branch, show links
  -> END
Human submits the PR form
  -> pull_request opened
  -> existing tests and all ten builds
  -> assemble and validate
  -> publish artifacts before merge
  -> consumer report in the build summary and PR comment
```

A `GITHUB_TOKEN` push does not trigger another `push` workflow. Opening a PR as a
human creates a **new `pull_request` event**; it does not change who authenticated
the earlier push or replay that push event. This workflow listens to `opened`,
`reopened`, and `synchronize`, and does not filter out drafts or automation branch
names. There is no automatic build dispatch and no `expected_source` input. The
ordinary `workflow_dispatch` entry remains available for a human manual build,
including recovery when no PR can be opened.

GitHub's repository settings, approvals and merge-conflict rules still apply.
In particular, GitHub does not start a `pull_request` workflow for a PR with a
merge conflict, even though this workflow checks out the head. A human direct
build dispatch is independent of that PR event limitation. See GitHub's
[workflow triggering rules](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)
and [pull-request events](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#pull_request).

Every job checks out **`github.event.pull_request.head.sha` for PR builds**, and
`github.sha` for push/manual builds. It never uses a moving branch name or the
synthetic PR merge commit as its source. The test job checks the checkout, and
the publisher checks that the package manifest's source commit matches this
selected identity **before** pushing an artifact. This intentionally validates
the proposed source, not a hypothetical merge with the base; it is not a separate
merge-integration test. All generated provenance, raw-source locators and PR
comments therefore refer to the actual source head that was compiled.

A human push to an open same-repository PR can start both `push` and
`pull_request/synchronize` builds for the same commit. **Both are allowed.**
There is no open-PR query to suppress a push build, deduplication gate, or
cross-run cancellation group. Both runs may publish different immutable artifact
commits with the same source commit. The existing append-only publisher handles
simultaneous writes without force-pushing. Reporter freshness checks choose which
result to display; they do not suppress or cancel either build.

The ten build combinations and aggregate `build` job remain in place. The matrix
intentionally has no `max-parallel` cap, minimizing build wait time subject to
runner/account limits. The `artifacts` tip may belong to any successfully built
source branch and is not a stable-channel alias. Consumers pin the reported
artifact commit, not `artifacts` HEAD or the upstream commit.

## What the single YAML block contains

After the publisher reports its commit, a temporary dependency-free npm consumer
resolves that exact Git dependency using `--package-lock-only --ignore-scripts
--no-audit --no-fund --lockfile-version=3`. The complete generated package entry is
retained, including `integrity` when npm supplies it. Node/npm versions and lock
format are recorded. Unexpected transitive dependencies, install hooks, a changed
specifier, repository, version or commit cause an error rather than an incomplete
lock fragment. The consumer's other dependencies and root package metadata are
not represented and cannot be replaced by this isolated fixture's complete lock.

The report combines:

- Distinct artifact, source, and upstream commits; manifest format, bytes and
  SHA-256; commit-pinned raw and archive locators.
- Every browser profile's `core.mjs`, `core.wasm` and type declaration file
  identity, plus shared interface files. Values come from the verified manifest,
  not duplicated handwritten hash tables.
- Consumer dependency/lock entry correspondences, reviewed-generated-code pins,
  standalone Wasm pins, conditional profile/schema/binding compatibility, relevant
  tests, and an example of an unrelated fixed capability-probe hash. Historical
  paths and symbols are explicitly search hints, not a live consumer scan or a
  claim that those locations remain current.
- The upstream overlay's pristine source digest, patch digest, reconstructed
  patched-source digest, helper/hook/doc identities, and activation recorded in
  each compiled profile/variant. Every other `upstream-patches-only-as-a-last-resort-with-explicit-user-approval/**/*.patch` is inventoried
  as unclassified rather than silently omitted. Toolchain patching is separately
  identified as not being a llama.cpp source patch.
- Actual validation scope from the manifest. Synthetic CPU model checks and
  mocked GPU suspension are not relabelled as real GPU/model inference.

The overlay inventory is deliberately scoped. When overlay mechanisms change,
`upstream_provenance.py` needs corresponding maintenance. A new ad-hoc compiler
transformation elsewhere in the repository is not automatically understood by
this report. Its source commit and raw locators remain available for investigation.

The reporting checkout must match the built source and upstream commits, and its
toolchain must match each variant's provenance. The report does not alter the
package or embed its own artifact commit in that package, avoiding a self-hash
cycle. Metadata and its integrity envelope are separate Actions artifacts.

A report digest identifies bytes, not compatibility approval. In particular,
generated-source adapter hashes are review gates; changing their values alone
does not establish that an exact-source replacement still has the intended
meaning. Upstream API/template/model changes can require real integration code
changes that this automation cannot infer from the hash table.

## Retrieval without reinstalling all dependencies

The metadata makes an updated, large `node_modules` archive optional for this
specific package. It supplies the commit and manifest identities needed to obtain
and verify only the runtime package independently of the consumer's other
installed dependencies. A complete package includes both variants, interfaces,
examples and notices; a few downloaded `core.mjs` files are review inputs, not a
complete installation.

GitHub-rendered/raw text viewed through a text tool is not proof of byte-exact
retrieval. Download availability, size limits and the assistant's tools remain
external constraints. File hashes certify bytes **after** retrieval; they cannot
recreate unavailable bytes. Likewise, an old installation of other consumer
dependencies can limit local tests. The YAML alone supports identifying pin
changes, but does not certify an end-to-end consumer build without its inputs.
Archive container bytes are not pinned: the extracted manifest and payload digests
are authoritative. Source archives also omit submodule contents, whose upstream
commit is separately provided.

## Comment lifecycle and security

Actions logs and the step summary receive the same generated Markdown. The PR
comment wraps that body in a source-aware status header. The installation command
is immediately visible; the single English YAML block is inside `<details>`.

The reporter operates on default-branch code, never PR-head code. It does not
install PR dependencies, evaluate YAML, execute uploaded scripts, or extract an
archive into its checkout. Only a bounded archive containing three named report
files is read in memory. The envelope must be a JSON object with strictly typed
positive integer schema/run/attempt fields and full commit strings, and must
match the repository, run ID, rerun attempt and source SHA. A malformed envelope
is an unavailable report for that PR, not an exception that aborts reconciliation
of every later PR. Previous metadata remains labelled as historical rather than
being promoted to current success. Download redirects receive no GitHub
authorization header.
Fork PRs and non-source build events are excluded from comment reconciliation.
Same-repository `push`, `pull_request`, and human `workflow_dispatch` build runs
are eligible. A PR run report containing a synthetic merge SHA instead of its
head is rejected, not relabelled as a successful head build.

`workflow_run` requested/in-progress/completed events cover normal builds and
reruns. `pull_request_target` opened/reopened/synchronize events handle PRs opened
**after** a build, and refresh the status for changed heads without executing their
code. Only comments authored by `github-actions[bot]` with the private report
marker are edited. A matching user comment is not overwritten.

Every invocation reconciles open same-repository PRs under one reporter-only
concurrency group. Pending-event coalescing therefore does not discard another
PR's update. Event payloads are not used as the latest-state authority: the current
PR head and matching newest build attempt are queried again before a write. Run
start time accounts for reruns retaining an old run ID; completion order does not
make a slow older build newer. Matching runs are paginated up to GitHub's filtered
search limit; an incomplete history is rejected rather than silently selecting
from a partial page. GitHub has no compare-and-swap comment API, so a
head change after the final read can still briefly precede reconciliation by its
next event. The displayed source identity makes that boundary explicit.

A changed head, pending run, failed run or unavailable report never inherits an
unqualified success claim. Previously posted successful metadata is retained
inside a section labelled **not confirmation of the current head/run**. An older
completed event cannot blindly replace the current head's status.

The report artifact is named per rerun attempt and requests 90-day retention
(subject to repository limits). Existing PR comments persist independently of
artifact expiry. A much later PR may not be able to recover an expired report;
that state is explicit. When the full metadata exceeds the inline comment budget,
the PR body keeps the commits and install command and links to that attempt's
report artifact, with the full YAML's byte count and SHA-256. The artifact retains
the complete YAML without truncation. After artifact expiry, a compact comment
still identifies the publication but no longer provides the full YAML download.
The notifier reads only the bounded report envelope and Markdown, never the YAML;
the compressed archive download remains bounded.

## Failure recovery and non-goals

| Failure | Result / recovery |
| --- | --- |
| No valid stable/nightly release | Failed updater with no implicit channel change. |
| Overlay conflict | Candidate branch and diagnostic report; no automatic PR or heavy build. A human can repair it and decide when to open a PR. |
| Wrapper/API/compiler incompatibility | Existing build fails; no runtime package is published. Manual source repair is needed. |
| Branch push denied or racing | No force-push; partial candidate identity remains in the report. Correct permissions or rerun from the same base. |
| Human PR does not start a build | Check workflow presence, approvals, merge conflicts and repository rules. A human manual runtime build remains available. |
| Existing candidate/PR needs another build | Reusing the candidate does not push. Human push/reopen or manual runtime build starts an independent run. |
| npm resolution/report generation fails after publication | Published artifact remains valid and its commit/install command remains in the publisher summary. Publication is not rolled back; the new consumer report is incomplete. |
| Comment API failure | Published artifact and report remain; rerun **Report runtime artifacts on PRs** without rebuilding. |
| Metadata retention expired | Reporter states it cannot recover that report. A publication/report job rerun is possible without rerunning successful compile jobs, but may append a new artifact commit. |

This is half-automation, not an auto-merge bot or a general source migration
engine. It does not modify the consumer, automatically approve new generated-code
adapters, promise downloaded bytes are available to every assistant, or claim
real GPU inference succeeded when only compile/mock checks ran.

## Local verification

The new Python tests use fake API responses, bounded ZIP fixtures, npm subprocess
fixtures and real **local** Git repositories. They cover both-pin updates,
channel/ref validation, conflict branches, browser form links, no-op/retry behavior,
preserved manual commits and probe/fetch races, no PR/dispatch mutations,
immutable PR-head checkouts, workflow defaults versus job-scoped publication
permissions, npm lock constraints, actual overlay hashing, push/PR comment
selection, comment freshness, strictly typed report envelopes and archive safety. A YAML-parser round trip is additionally performed when PyYAML
is available; no third-party Python package is required by the automation.

Representative focused commands:

```sh
python3 -m unittest discover -s tests -p test_github_api.py
python3 -m unittest discover -s tests -p test_update_automation.py
python3 -m unittest discover -s tests -p test_consumer_metadata.py
python3 -m unittest discover -s tests -p test_runtime_comments.py
python3 -m unittest discover -s tests -p test_automation_workflows.py
```

These tests do not replace the first live human-created PR, token/ruleset checks,
public npm Git resolution, or the existing WebAssembly build/browser validation.
