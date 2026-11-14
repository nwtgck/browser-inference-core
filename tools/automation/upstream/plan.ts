import { createHash } from 'node:crypto';
import type { GitHubRelease } from '#dto/llama-cpp';
import { gitObjectId } from '#tools/publication/git';
import { repositoryName } from '#tools/publication/npm-lock';

export type UpdateTarget = { kind: 'latest' | 'latest_unstable' } | { kind: 'custom'; ref: string };
export type UpdateRequest = { repository: string; base: string; sourceCommit: string; previousCommit: string };
export type UpstreamSelection = { requested: UpdateTarget['kind']; ref: string; release: GitHubRelease | undefined };
export type PlannedUpdate = { status: 'planned'; request: UpdateRequest; selection: UpstreamSelection; targetCommit: string; relation: string; branch: string };
export type UpdatePlan = PlannedUpdate | { status: 'unchanged'; request: UpdateRequest; selection: UpstreamSelection; targetCommit: string };
export type Preflight = { status: 'passed' | 'failed' | 'not_repeated'; scope: string; error: string | undefined };
export type Candidate = { plan: PlannedUpdate; status: 'candidate_prepared' | 'branch_exists'; sourceCommit: string; preflight: Preflight };

export function selectUpstream({ target, releases }: { target: UpdateTarget; releases: readonly GitHubRelease[] }): UpstreamSelection {
  let release: GitHubRelease | undefined;
  switch (target.kind) {
  case 'custom':
    return { requested: target.kind, ref: target.ref, release: undefined };
  case 'latest':
    release = releases[0];
    if (releases.length !== 1 || !release || release.draft || release.prerelease || !release.published_at || !/^v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$/u.test(release.tag_name)) {
      throw new Error('Latest release is not a published stable vX.Y.Z release; no fallback');
    }
    break;
  case 'latest_unstable':
    release = releases.find(item => !item.draft && item.prerelease && item.published_at && /^b[0-9]+$/u.test(item.tag_name));
    if (!release) {
      throw new Error('No published bNNNN pre-release was found; no fallback');
    }
    break;
  default: {
    const exhaustive: never = target;
    throw new Error(String(exhaustive));
  }
  }
  return { requested: target.kind, ref: `refs/tags/${release.tag_name}`, release };
}

export function candidateBranch({ base, source, target }: { base: string; source: string; target: string }): string {
  const identity = createHash('sha256').update(base).digest('hex').slice(0, 12);
  return `feature/automation/llama-cpp/${identity}-${gitObjectId({ value: source }).slice(0, 12)}-${gitObjectId({ value: target }).slice(0, 12)}`;
}

export function planUpdate({ request, selection, resolvedCommit, resolvedRef, relation, allowNonFastForward }: { request: UpdateRequest; selection: UpstreamSelection; resolvedCommit: string; resolvedRef: string; relation: string | undefined; allowNonFastForward: boolean }): UpdatePlan {
  repositoryName({ value: request.repository });
  gitObjectId({ value: request.sourceCommit });
  gitObjectId({ value: request.previousCommit });
  const targetCommit = gitObjectId({ value: resolvedCommit });
  if (resolvedRef !== selection.ref) {
    throw new Error('Resolved upstream ref differs from selection');
  }
  if (request.previousCommit === targetCommit) {
    return { status: 'unchanged', request, selection, targetCommit };
  }
  if (!relation || !['ahead', 'identical', 'behind', 'diverged'].includes(relation)) {
    throw new Error('Unknown upstream comparison status');
  }
  if (!['ahead', 'identical'].includes(relation) && !allowNonFastForward) {
    throw new Error('Target is older or divergent; allow_non_fast_forward is required for this change');
  }
  return { status: 'planned', request, selection, targetCommit, relation, branch: candidateBranch({ base: request.base, source: request.sourceCommit, target: targetCommit }) };
}

export function pullRequestHandoff({ candidate }: { candidate: Candidate }): { compareUrl: string; createPullRequestUrl: string; title: string; body: string } {
  const { plan, preflight } = candidate;
  const { request, selection, targetCommit, relation, branch } = plan;
  const title = 'chore: update llama.cpp to ' + selection.ref.replace(/^refs\/tags\//u, '');
  const body = `\
## Upstream update

- Requested channel: \`${selection.requested}\`
- Resolved ref: \`${selection.ref}\`
- Previous llama.cpp: \`${request.previousCommit}\`
- Proposed llama.cpp: \`${targetCommit}\`
- Upstream relation: \`${relation}\`
- Overlay preflight: **${preflight.status}** (${preflight.scope})

[Upstream comparison](https://github.com/ggml-org/llama.cpp/compare/${request.previousCommit}...${targetCommit})

The automated commit changes only the llama.cpp submodule gitlink. A reused branch may also contain manual repairs. Preflight is not build or inference validation. Full diagnostics are in the updater summary and upstream-update-report artifact.

Opening this PR starts the ordinary runtime workflow. A successful same-repository build publishes before merge; its runtime comment contains the installation command and consumer metadata.
`;
  const compareUrl = `https://github.com/${repositoryName({ value: request.repository })}/compare/${encodeURIComponent(request.base)}...${encodeURIComponent(branch)}`;
  const query = new URLSearchParams({ quick_pull: '1', title, body });
  const filled = `${compareUrl}?${query.toString()}`;
  return { compareUrl, createPullRequestUrl: filled.length > 7000 ? `${compareUrl}?quick_pull=1` : filled, title, body };
}
