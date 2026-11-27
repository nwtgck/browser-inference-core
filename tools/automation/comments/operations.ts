import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { decodeGitHubArtifacts, decodeGitHubIssueComments, decodeGitHubPullRequest, decodeGitHubPullRequests, decodeGitHubWorkflowRuns, type GitHubPullRequest, type GitHubWorkflowRun } from '#dto/llama-cpp';
import { optionalStat } from '#tools/shared/files';
import { readJson } from '#tools/shared/json';
import { errorDetails } from '#tools/automation/upstream/operations';
import { archiveRedirect, readConsumerReport } from './archive.js';
import { matchingRuns, planCommentWrite, positiveId, reportArchives, reportAttemptAvailable, selectPulls, type PublishedReport, type ReportArchive } from './plan.js';

async function pulls({ state, repository }: { state: string; repository: string }): Promise<GitHubPullRequest[]> {
  return selectPulls({ pulls: await readJson({ path: join(state, 'pulls.json'), decode: decodeGitHubPullRequests }), repository });
}

async function selectedPull({ state, repository, number }: { state: string; repository: string; number: number }): Promise<GitHubPullRequest> {
  positiveId({ value: number });
  const pull = (await pulls({ state, repository })).find(item => item.number === number);
  if (!pull) {
    throw new Error('Pull request was not in the selected same-repository set');
  }
  return pull;
}

async function runHistory({ directory, repository, pull, file }: { directory: string; repository: string; pull: GitHubPullRequest; file: string }): Promise<GitHubWorkflowRun[]> {
  return matchingRuns({ runs: await readJson({ path: join(directory, file), decode: decodeGitHubWorkflowRuns }), repository, pull });
}

function reportRuns({ runs }: { runs: readonly GitHubWorkflowRun[] }): GitHubWorkflowRun[] {
  // A queued rerun can still have a completed first attempt.
  return runs.filter(run => reportAttemptAvailable({ run, attempt: 1 }));
}

async function recordError({ directory, error }: { directory: string; error: unknown }): Promise<void> {
  await writeFile(join(directory, 'error.txt'), errorDetails({ error }).slice(0, 1000) + '\n');
}

async function assertRetrieved({ directory }: { directory: string }): Promise<void> {
  const path = join(directory, 'error.txt');
  if (await optionalStat({ path })) {
    throw new Error(await readFile(path, 'utf8'));
  }
}

async function listedArchives({ directory, run }: { directory: string; run: GitHubWorkflowRun }): Promise<ReportArchive[]> {
  await assertRetrieved({ directory });
  const artifacts = await readJson({ path: join(directory, 'artifacts.json'), decode: decodeGitHubArtifacts });
  return reportArchives({ artifacts, run });
}

export async function planPullRequests({ state, repository }: { state: string; repository: string }): Promise<void> {
  const selected = await pulls({ state, repository });
  for (const pull of selected) {
    await mkdir(join(state, String(pull.number)), { recursive: true });
  }
  // Only YAML consumes this projection; typed operations retain the API response.
  await writeFile(join(state, 'requests.tsv'), selected.map(pull => `${pull.number}\t${pull.head.ref}\t${pull.head.sha}\n`).join(''));
}

export async function planReportDownloads({ state, repository }: { state: string; repository: string }): Promise<void> {
  const rows: string[] = [];
  for (const pull of await pulls({ state, repository })) {
    const directory = join(state, String(pull.number));
    const runs = await runHistory({ directory, repository, pull, file: 'runs.json' });
    for (const run of reportRuns({ runs })) {
      await mkdir(join(directory, String(run.id)), { recursive: true });
      rows.push(`${pull.number}\t${run.id}\n`);
    }
  }
  await writeFile(join(state, 'downloads.tsv'), rows.join(''));
}

export async function planReportArchives({ state, repository }: { state: string; repository: string }): Promise<void> {
  const rows: string[] = [];
  for (const pull of await pulls({ state, repository })) {
    const directory = join(state, String(pull.number));
    const runs = await runHistory({ directory, repository, pull, file: 'runs.json' });
    const { candidates } = await reportCandidates({ directory, runs });
    for (const { run, archive: { artifactId } } of candidates) {
      rows.push(`${pull.number}\t${run.id}\t${artifactId}\n`);
    }
  }
  await writeFile(join(state, 'archives.tsv'), rows.join(''));
}

export async function prepareArchiveRedirect({ state, number, runId, artifactId }: { state: string; number: number; runId: number; artifactId: number }): Promise<void> {
  const directory = join(state, String(positiveId({ value: number })), String(positiveId({ value: runId })), String(positiveId({ value: artifactId })));
  let url: string;
  try {
    const status = (await readFile(join(directory, 'archive-status'), 'utf8')).trim();
    const headers = await readFile(join(directory, 'archive-headers'), 'utf8');
    url = archiveRedirect({ status, headers });
  } catch (error) {
    await recordError({ directory, error });
    return;
  }
  await writeFile(join(directory, 'download-url.txt'), url + '\n');
}

export async function validateReportArchive({ state, repository, number, runId, artifactId }: { state: string; repository: string; number: number; runId: number; artifactId: number }): Promise<void> {
  const pull = await selectedPull({ state, repository, number });
  const directory = join(state, String(pull.number));
  const runs = await runHistory({ directory, repository, pull, file: 'runs.json' });
  const run = reportRuns({ runs }).find(item => item.id === runId);
  if (!run) {
    throw new Error('Report archive run was not selected for this pull request');
  }
  const runDirectory = join(directory, String(run.id));
  const archive = (await listedArchives({ directory: runDirectory, run })).find(item => item.artifactId === artifactId);
  if (!archive) {
    throw new Error('Report archive was not selected for this workflow run');
  }
  const archiveDirectory = join(runDirectory, String(archive.artifactId));
  let markdown: string;
  try {
    markdown = await readConsumerReport({ path: join(archiveDirectory, 'report.zip'), repository, run, attempt: archive.attempt, kind: archive.kind });
  } catch (error) {
    await recordError({ directory: archiveDirectory, error });
    return;
  }
  // The online loop uses this file's presence to stop trying candidates.
  // Final reconciliation reads the verified Markdown from the same job.
  await writeFile(join(archiveDirectory, 'publication.md'), markdown);
}

async function reportCandidates({ directory, runs }: { directory: string; runs: readonly GitHubWorkflowRun[] }) {
  let error: string | undefined;
  const candidates: { run: GitHubWorkflowRun; archive: ReportArchive; recordedAt: string }[] = [];
  for (const run of reportRuns({ runs })) {
    const runDirectory = join(directory, String(run.id));
    let archives: ReportArchive[];
    try {
      archives = await listedArchives({ directory: runDirectory, run });
    } catch (failure) {
      error ??= errorDetails({ error: failure });
      continue;
    }
    // A rerun's start time would make old publication records appear new.
    // Details are listed before the guide, whose upload is closest to the push.
    const recordedAt = new Map<number, string>();
    for (const { attempt, createdAt } of archives) {
      if (createdAt) {
        recordedAt.set(attempt, createdAt);
      }
    }
    for (const archive of archives) {
      candidates.push({ run, archive, recordedAt: recordedAt.get(archive.attempt) ?? run.created_at ?? run.run_started_at ?? '' });
    }
  }
  candidates.sort((first, second) => first.recordedAt === second.recordedAt ? 0 : first.recordedAt < second.recordedAt ? 1 : -1);
  return { candidates, error };
}

async function readPublication({ directory, runs }: { directory: string; runs: readonly GitHubWorkflowRun[] }): Promise<{ publication: PublishedReport | undefined; error: string | undefined }> {
  const { candidates, error: listingError } = await reportCandidates({ directory, runs });
  let error = listingError;
  for (const { run, archive: { artifactId, attempt, kind } } of candidates) {
    const archiveDirectory = join(directory, String(run.id), String(artifactId));
    try {
      await assertRetrieved({ directory: archiveDirectory });
      const path = join(archiveDirectory, 'publication.md');
      if (await optionalStat({ path })) {
        const markdown = await readFile(path, 'utf8');
        return { publication: { run, attempt, kind, markdown }, error };
      }
    } catch (failure) {
      error ??= errorDetails({ error: failure });
    }
  }
  return { publication: undefined, error };
}

export async function prepareCommentWrite({ state, repository, number }: { state: string; repository: string; number: number }): Promise<void> {
  const original = await selectedPull({ state, repository, number });
  const directory = join(state, String(original.number));
  const current = await readJson({ path: join(directory, 'current-pr.json'), decode: decodeGitHubPullRequest });
  const selectedRuns = await runHistory({ directory, repository, pull: original, file: 'runs.json' });
  const currentRuns = await runHistory({ directory, repository, pull: current, file: 'current-runs.json' });
  const comments = await readJson({ path: join(directory, 'current-comments.json'), decode: decodeGitHubIssueComments });
  const { publication, error } = await readPublication({ directory, runs: selectedRuns });
  const plan = planCommentWrite({ original, current, selectedRuns, currentRuns, comments, repository, publication, error });
  switch (plan.kind) {
  case 'skip':
    await writeFile(join(directory, 'write.txt'), 'skip\n');
    break;
  case 'create':
    await writeFile(join(directory, 'body.json'), JSON.stringify({ body: plan.body }) + '\n');
    await writeFile(join(directory, 'write.txt'), 'create\n');
    break;
  case 'update':
    await writeFile(join(directory, 'body.json'), JSON.stringify({ body: plan.body }) + '\n');
    await writeFile(join(directory, 'write.txt'), `update\t${plan.commentId}\n`);
    break;
  default: {
    const exhaustive: never = plan;
    throw new Error(String(exhaustive));
  }
  }
}
