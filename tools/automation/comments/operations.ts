import { copyFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { decodeGitHubArtifacts, decodeGitHubIssueComments, decodeGitHubPullRequest, decodeGitHubPullRequests, decodeGitHubWorkflowRuns, type GitHubPullRequest, type GitHubWorkflowRun } from '#dto/llama-cpp';
import { optionalStat } from '#tools/shared/files';
import { readJson } from '#tools/shared/json';
import { errorDetails } from '#tools/automation/upstream/operations';
import { archiveRedirect, maximumArchiveBytes, readConsumerReport } from './archive.js';
import { latestRun, planCommentWrite, positiveId, selectArtifact, selectPulls } from './plan.js';

async function pulls({ state, repository }: { state: string; repository: string }): Promise<GitHubPullRequest[]> {
  return selectPulls({ pulls: await readJson({ path: join(state, 'pulls.json'), decode: decodeGitHubPullRequests }), repository });
}

async function selectedRun({ state, repository, pull, file }: { state: string; repository: string; pull: GitHubPullRequest; file: string }): Promise<GitHubWorkflowRun | undefined> {
  return latestRun({ runs: await readJson({ path: join(state, String(pull.number), file), decode: decodeGitHubWorkflowRuns }), repository, pull });
}

async function recordError({ directory, error }: { directory: string; error: unknown }): Promise<void> {
  await writeFile(join(directory, 'error.txt'), errorDetails({ error }).slice(0, 1000) + '\n');
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
    const run = await selectedRun({ state, repository, pull, file: 'runs.json' });
    if (run?.status === 'completed' && run.conclusion === 'success') {
      rows.push(`${pull.number}\t${run.id}\t${run.run_attempt}\n`);
    }
  }
  await writeFile(join(state, 'downloads.tsv'), rows.join(''));
}

export async function planReportArchives({ state, repository }: { state: string; repository: string }): Promise<void> {
  const rows: string[] = [];
  for (const pull of await pulls({ state, repository })) {
    const directory = join(state, String(pull.number));
    if (await optionalStat({ path: join(directory, 'error.txt') })) {
      continue;
    }
    const run = await selectedRun({ state, repository, pull, file: 'runs.json' });
    if (run?.status !== 'completed' || run.conclusion !== 'success') {
      continue;
    }
    try {
      const artifacts = await readJson({ path: join(directory, 'artifacts.json'), decode: decodeGitHubArtifacts });
      rows.push(`${pull.number}\t${selectArtifact({ artifacts, run })}\n`);
    } catch (error) {
      await recordError({ directory, error });
    }
  }
  await writeFile(join(state, 'archives.tsv'), rows.join(''));
}

export async function planArchiveDownloads({ state, repository }: { state: string; repository: string }): Promise<void> {
  const rows: string[] = [];
  for (const pull of await pulls({ state, repository })) {
    const directory = join(state, String(pull.number));
    if (!await optionalStat({ path: join(directory, 'archive-status') }) || await optionalStat({ path: join(directory, 'error.txt') })) {
      continue;
    }
    try {
      const status = (await readFile(join(directory, 'archive-status'), 'utf8')).trim();
      if (status === '200') {
        const initial = join(directory, 'archive-initial');
        const initialStat = await optionalStat({ path: initial });
        if (!initialStat?.isFile() || initialStat.size > maximumArchiveBytes) {
          throw new Error('Report archive exceeds the size limit');
        }
        await copyFile(initial, join(directory, 'report.zip'));
      } else {
        const url = archiveRedirect({ status, headers: await readFile(join(directory, 'archive-headers'), 'utf8') });
        if (url.includes('\t')) {
          throw new Error('Unexpected artifact download redirect');
        }
        rows.push(`${pull.number}\t${url}\n`);
      }
    } catch (error) {
      await recordError({ directory, error });
    }
  }
  await writeFile(join(state, 'urls.tsv'), rows.join(''));
}

export async function prepareCommentWrite({ state, repository, number }: { state: string; repository: string; number: number }): Promise<void> {
  positiveId({ value: number });
  const directory = join(state, String(number));
  const original = (await pulls({ state, repository })).find(pull => pull.number === number);
  if (!original) {
    throw new Error('Pull request was not in the selected same-repository set');
  }
  const current = await readJson({ path: join(directory, 'current-pr.json'), decode: decodeGitHubPullRequest });
  const selected = await selectedRun({ state, repository, pull: original, file: 'runs.json' });
  const currentRun = await selectedRun({ state, repository, pull: current, file: 'current-runs.json' });
  const comments = await readJson({ path: join(directory, 'current-comments.json'), decode: decodeGitHubIssueComments });
  let payload: string | undefined;
  let error: string | undefined;
  if (selected?.status === 'completed' && selected.conclusion === 'success') {
    try {
      const retrievalError = join(directory, 'error.txt');
      if (await optionalStat({ path: retrievalError })) {
        throw new Error(await readFile(retrievalError, 'utf8'));
      }
      payload = await readConsumerReport({ path: join(directory, 'report.zip'), repository, run: selected });
    } catch (failure) {
      error = errorDetails({ error: failure });
    }
  }
  const plan = planCommentWrite({ original, current, selectedRun: selected, currentRun, comments, repository, payload, error });
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
