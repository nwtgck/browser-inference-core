import type { GitHubArtifact, GitHubIssueComment, GitHubPullRequest, GitHubWorkflowRun } from '#dto/llama-cpp';
import { gitObjectId } from '#tools/publication/git';
import { repositoryName } from '#tools/publication/npm-lock';

export const marker = '<!-- lcb-runtime-artifact-comment:v1 -->';
export const payloadStart = '<!-- lcb-published-runtime:start -->';
export const payloadEnd = '<!-- lcb-published-runtime:end -->';

export function positiveId({ value }: { value: number }): number {
  if (!Number.isSafeInteger(value) || value <= 0) {
    throw new Error('Expected a positive GitHub identifier');
  }
  return value;
}

export function validatePullRequest({ pull }: { pull: GitHubPullRequest }): void {
  positiveId({ value: pull.number });
  gitObjectId({ value: pull.head.sha });
  if (!pull.head.ref || /[\n\r\0\t]/u.test(pull.head.ref)) {
    throw new Error('Unsafe pull-request head ref');
  }
}

export function sameRepositoryPull({ pull, repository }: { pull: GitHubPullRequest; repository: string }): boolean {
  return pull.state === 'open' && pull.head.repo?.full_name === repository && pull.head.ref !== 'artifacts' && !pull.head.ref.startsWith('artifacts/');
}

export function selectPulls({ pulls, repository }: { pulls: readonly GitHubPullRequest[]; repository: string }): GitHubPullRequest[] {
  repositoryName({ value: repository });
  const selected: GitHubPullRequest[] = [];
  const seen = new Set<number>();
  for (const pull of pulls) {
    if (!sameRepositoryPull({ pull, repository })) {
      continue;
    }
    validatePullRequest({ pull });
    if (seen.has(pull.number)) {
      throw new Error('Duplicate pull request in downloaded pages');
    }
    seen.add(pull.number);
    selected.push(pull);
  }
  return selected;
}

function validateRun({ run, repository }: { run: GitHubWorkflowRun; repository: string }): void {
  positiveId({ value: run.id });
  positiveId({ value: run.run_attempt });
  gitObjectId({ value: run.head_sha });
  if (run.html_url !== `https://github.com/${repositoryName({ value: repository })}/actions/runs/${run.id}`) {
    throw new Error('Unexpected source workflow run URL');
  }
  if (!['queued', 'in_progress', 'completed', 'waiting', 'requested', 'pending'].includes(run.status)) {
    throw new Error('Unexpected source workflow status');
  }
  if (run.conclusion !== null && !['success', 'failure', 'cancelled', 'skipped', 'timed_out', 'action_required', 'neutral', 'stale', 'startup_failure'].includes(run.conclusion)) {
    throw new Error('Unexpected source workflow conclusion');
  }
}

export function latestRun({ runs, repository, pull }: { runs: readonly GitHubWorkflowRun[]; repository: string; pull: GitHubPullRequest }): GitHubWorkflowRun | undefined {
  if (runs.length > 1000) {
    throw new Error('Incomplete or oversized build history');
  }
  const seen = new Set<string>();
  let latest: GitHubWorkflowRun | undefined;
  for (const run of runs) {
    const identity = `${positiveId({ value: run.id })}/${positiveId({ value: run.run_attempt })}`;
    if (seen.has(identity)) {
      throw new Error('Duplicate workflow run in downloaded pages');
    }
    seen.add(identity);
    if (run.head_sha !== pull.head.sha || run.head_branch !== pull.head.ref || run.head_repository?.full_name !== repository || !['push', 'pull_request', 'workflow_dispatch'].includes(run.event)) {
      continue;
    }
    validateRun({ run, repository });
    const time = run.run_started_at || run.created_at || '';
    const previousTime = latest?.run_started_at || latest?.created_at || '';
    if (!latest || time > previousTime || time === previousTime && (run.id > latest.id || run.id === latest.id && run.run_attempt > latest.run_attempt)) {
      latest = run;
    }
  }
  return latest;
}

export function selectArtifact({ artifacts, run }: { artifacts: readonly GitHubArtifact[]; run: GitHubWorkflowRun }): number {
  const name = `consumer-update-${positiveId({ value: run.run_attempt })}`;
  const matches = artifacts.filter(item => item.name === name && !item.expired);
  const match = matches[0];
  if (matches.length !== 1 || !match) {
    throw new Error('No unique unexpired report artifact for this attempt');
  }
  return positiveId({ value: match.id });
}

function sameRun({ first, second }: { first: GitHubWorkflowRun | undefined; second: GitHubWorkflowRun | undefined }): boolean {
  return first?.id === second?.id && first?.run_attempt === second?.run_attempt && first?.status === second?.status && first?.conclusion === second?.conclusion;
}

export function previousPayload({ body }: { body: string }): string | undefined {
  const first = body.indexOf(payloadStart);
  const last = body.indexOf(payloadEnd);
  if (first < 0 || last <= first || body.indexOf(payloadStart, first + payloadStart.length) >= 0 || body.indexOf(payloadEnd, last + payloadEnd.length) >= 0) {
    return undefined;
  }
  return body.slice(first + payloadStart.length, last).trim() || undefined;
}

export function commentBody({ pull, run, payload, previous, error }: { pull: GitHubPullRequest; run: GitHubWorkflowRun | undefined; payload: string | undefined; previous: string; error: string | undefined }): string {
  const head = gitObjectId({ value: pull.head.sha });
  let text = `${marker}\n## Runtime publication status\n\n**Current PR head:** \`${head}\`\n\n`;
  if (run === undefined) {
    text += 'No matching runtime build is recorded for this head yet. Human PR creation normally starts it; workflow approval, merge conflicts, or a missing workflow can prevent a run from starting.\n';
  } else if (run.status !== 'completed') {
    text += `The latest runtime build/report run is **${run.status}**. No new complete consumer report is available yet.\n`;
  } else if (run.conclusion === 'success') {
    text += payload === undefined ? 'The runtime build succeeded, but its consumer report is unavailable or expired.\n' : 'The runtime build and consumer report succeeded for this PR head.\n';
  } else {
    text += `The latest runtime build/report run concluded **${run.conclusion}**. Publication may have completed before a reporting failure; this is not a new complete consumer report.\n`;
  }
  if (run) {
    text += `\n[Build/report run](${run.html_url}) (attempt ${run.run_attempt})\n`;
  }
  if (error) {
    text += '\nReport retrieval error: ' + error.replaceAll('`', "'").replace(/[\r\n]/gu, ' ').slice(0, 500) + '\n';
  }
  if (payload !== undefined) {
    text += `\n${payloadStart}\n${payload.trim()}\n${payloadEnd}\n`;
  } else {
    const old = previousPayload({ body: previous });
    if (old) {
      text += `\n<details>\n<summary>Previous successful publication — not confirmation of the current head/run</summary>\n\n${payloadStart}\n${old}\n${payloadEnd}\n\n</details>\n`;
    }
  }
  if (Buffer.byteLength(text) > 62000) {
    throw new Error('PR comment size limit exceeded');
  }
  return text;
}

export type CommentWrite = { kind: 'skip'; reason: string } | { kind: 'create'; body: string } | { kind: 'update'; commentId: number; body: string };

export function planCommentWrite({ original, current, selectedRun, currentRun, comments, repository, payload, error }: { original: GitHubPullRequest; current: GitHubPullRequest; selectedRun: GitHubWorkflowRun | undefined; currentRun: GitHubWorkflowRun | undefined; comments: readonly GitHubIssueComment[]; repository: string; payload: string | undefined; error: string | undefined }): CommentWrite {
  validatePullRequest({ pull: original });
  validatePullRequest({ pull: current });
  if (current.number !== original.number || !sameRepositoryPull({ pull: current, repository }) || current.head.sha !== original.head.sha || current.head.ref !== original.head.ref || !sameRun({ first: selectedRun, second: currentRun })) {
    return { kind: 'skip', reason: 'PR head, trust or latest workflow run changed' };
  }
  const previous = comments.find(comment => comment.user.login === 'github-actions[bot]' && comment.user.type === 'Bot' && comment.body?.startsWith(marker));
  const body = commentBody({ pull: current, run: currentRun, payload, previous: previous?.body ?? '', error });
  if (previous) {
    positiveId({ value: previous.id });
    return previous.body === body ? { kind: 'skip', reason: 'Comment already matches current status' } : { kind: 'update', commentId: previous.id, body };
  }
  return { kind: 'create', body };
}
