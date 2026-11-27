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

export function matchingRuns({ runs, repository, pull }: { runs: readonly GitHubWorkflowRun[]; repository: string; pull: GitHubPullRequest }): GitHubWorkflowRun[] {
  if (runs.length > 1000) {
    throw new Error('Incomplete or oversized build history');
  }
  const seen = new Set<string>();
  const matches = new Map<number, GitHubWorkflowRun>();
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
    const previousAttempt = matches.get(run.id);
    if (!previousAttempt || run.run_attempt > previousAttempt.run_attempt) {
      matches.set(run.id, run);
    }
  }
  return Array.from(matches.values()).sort((first, second) => {
    const firstTime = first.run_started_at || first.created_at || '';
    const secondTime = second.run_started_at || second.created_at || '';
    return firstTime === secondTime ? second.id - first.id : firstTime < secondTime ? 1 : -1;
  });
}

export function reportAttemptAvailable({ run, attempt }: { run: GitHubWorkflowRun; attempt: number }): boolean {
  return run.actor?.login !== 'dependabot[bot]' && Number.isSafeInteger(attempt) && attempt > 0 &&
    (attempt < run.run_attempt || attempt === run.run_attempt && run.status === 'completed');
}

export type ReportKind = 'guide' | 'details';
export type ReportArchive = { artifactId: number; attempt: number; kind: ReportKind; createdAt: string | undefined };
export type PublishedReport = { run: GitHubWorkflowRun; attempt: number; kind: ReportKind; markdown: string };

export function reportArchives({ artifacts, run }: { artifacts: readonly GitHubArtifact[]; run: GitHubWorkflowRun }): ReportArchive[] {
  const matches = new Map<string, ReportArchive | undefined>();
  for (const artifact of artifacts) {
    const match = /^(consumer-update|publication-guide)-([1-9][0-9]*)$/u.exec(artifact.name);
    const attempt = Number(match?.[2]);
    if (artifact.expired || !reportAttemptAvailable({ run, attempt })) {
      continue;
    }
    const kind = match?.[1] === 'consumer-update' ? 'details' : 'guide';
    matches.set(artifact.name, matches.has(artifact.name) ? undefined : { artifactId: positiveId({ value: artifact.id }), attempt, kind, createdAt: artifact.created_at });
  }
  return Array.from(matches.values()).filter(match => match !== undefined).sort((first, second) =>
    second.attempt - first.attempt || Number(second.kind === 'details') - Number(first.kind === 'details'));
}

function sameRunHistory({ selected, current }: { selected: readonly GitHubWorkflowRun[]; current: readonly GitHubWorkflowRun[] }): boolean {
  return selected.length === current.length && selected.every((first, index) => {
    const second = current[index];
    return first.id === second?.id && first.run_attempt === second.run_attempt && first.status === second.status &&
      first.conclusion === second.conclusion && first.actor?.login === second.actor?.login;
  });
}

export function previousPayload({ body }: { body: string }): string | undefined {
  const first = body.indexOf(payloadStart);
  const last = body.indexOf(payloadEnd);
  if (first < 0 || last <= first || body.indexOf(payloadStart, first + payloadStart.length) >= 0 || body.indexOf(payloadEnd, last + payloadEnd.length) >= 0) {
    return undefined;
  }
  return body.slice(first + payloadStart.length, last).trim() || undefined;
}

export function commentBody({ pull, run, publication, previous, error }: { pull: GitHubPullRequest; run: GitHubWorkflowRun | undefined; publication: PublishedReport | undefined; previous: string; error: string | undefined }): string {
  const head = gitObjectId({ value: pull.head.sha });
  let text = `${marker}\n## Runtime publication status\n\n**Current PR head:** \`${head}\`\n\n`;
  if (run === undefined) {
    text += 'No matching runtime build is recorded for this head yet. Human PR creation normally starts it; workflow approval, merge conflicts, or a missing workflow can prevent a run from starting.\n';
  } else if (run.status !== 'completed') {
    text += `The latest runtime build is **${run.status}**.\n`;
  } else {
    text += `The latest runtime build concluded **${run.conclusion ?? 'unknown'}**.\n`;
  }
  if (run) {
    text += `\n[Build run summary](${run.html_url}/attempts/${run.run_attempt}) (attempt ${run.run_attempt})\n`;
  }
  if (publication) {
    text += `\nRuntime publication confirmed for this PR head. [Publication run summary](${publication.run.html_url}/attempts/${publication.attempt}) (attempt ${publication.attempt}).\n`;
    const detailsStatus = { guide: '\nConsumer integration details are unavailable.\n', details: '' } satisfies Record<ReportKind, string>;
    text += detailsStatus[publication.kind];
    text += `\n${payloadStart}\n${publication.markdown.trim()}\n${payloadEnd}\n`;
  } else if (run) {
    text += '\nA confirmed publication report is unavailable or expired for this PR head. See the build run summary for publication information.\n';
  }
  if (error) {
    text += '\nReport retrieval error: ' + error.replaceAll('`', "'").replace(/[\r\n]/gu, ' ').slice(0, 500) + '\n';
  }
  if (!publication) {
    const old = previousPayload({ body: previous });
    if (old) {
      text += `\n<details>\n<summary>Previous publication (historical reference)</summary>\n\n${payloadStart}\n${old}\n${payloadEnd}\n\n</details>\n`;
    }
  }
  if (Buffer.byteLength(text) > 62000) {
    throw new Error('PR comment size limit exceeded');
  }
  return text;
}

export type CommentWrite = { kind: 'skip'; reason: string } | { kind: 'create'; body: string } | { kind: 'update'; commentId: number; body: string };

export function planCommentWrite({ original, current, selectedRuns, currentRuns, comments, repository, publication, error }: { original: GitHubPullRequest; current: GitHubPullRequest; selectedRuns: readonly GitHubWorkflowRun[]; currentRuns: readonly GitHubWorkflowRun[]; comments: readonly GitHubIssueComment[]; repository: string; publication: PublishedReport | undefined; error: string | undefined }): CommentWrite {
  validatePullRequest({ pull: original });
  validatePullRequest({ pull: current });
  if (current.number !== original.number || !sameRepositoryPull({ pull: current, repository }) || current.head.sha !== original.head.sha || current.head.ref !== original.head.ref || !sameRunHistory({ selected: selectedRuns, current: currentRuns })) {
    return { kind: 'skip', reason: 'PR head, trust or matching workflow runs changed' };
  }
  const previous = comments.find(comment => comment.user.login === 'github-actions[bot]' && comment.user.type === 'Bot' && comment.body?.startsWith(marker));
  const body = commentBody({ pull: current, run: currentRuns[0], publication, previous: previous?.body ?? '', error });
  if (previous) {
    positiveId({ value: previous.id });
    return previous.body === body ? { kind: 'skip', reason: 'Comment already matches current status' } : { kind: 'update', commentId: previous.id, body };
  }
  return { kind: 'create', body };
}
