import { createHash } from 'node:crypto';
import { mkdtemp, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { GitHubIssueComment, GitHubPullRequest, GitHubWorkflowRun } from '#dto/llama-cpp';
import { runProcess } from '#tools/shared/process';
import { marker } from '#tools/automation/comments/plan';

export const repository = 'example/core';
export const source = 'a'.repeat(40);
export const artifact = 'c'.repeat(40);

export function pullRequest({ head, number }: { head: string; number: number }): GitHubPullRequest {
  return { number, state: 'open', head: { sha: head, ref: 'feature/update', repo: { full_name: repository } } };
}

export function workflowRun({ head, id, attempt, status, conclusion }: { head: string; id: number; attempt: number; status: string; conclusion: string | null }): GitHubWorkflowRun {
  return { id, run_attempt: attempt, head_sha: head, head_branch: 'feature/update', head_repository: { full_name: repository }, event: 'push', status, conclusion, html_url: `https://github.com/${repository}/actions/runs/${id}` };
}

export function reportPayload({ head }: { head: string }): string {
  return `## Runtime artifact published\n\nSource: ${head}\n\nnpm install github:${repository}#${artifact}\n\n<details>\n<summary>Metadata</summary>\n\nfixture\n\n</details>\n`;
}

export function botComment({ body }: { body: string }): GitHubIssueComment {
  return { id: 42, user: { login: 'github-actions[bot]', type: 'Bot' }, body };
}

export async function reportArchive({ root, run, changes, body, envelope, yaml }: { root: string; run: GitHubWorkflowRun; changes: Record<string, unknown>; body: string | undefined; envelope: { input: unknown } | undefined; yaml: string }): Promise<{ path: string; directory: string }> {
  const directory = await mkdtemp(join(root, 'archive-'));
  const markdown = body ?? reportPayload({ head: run.head_sha });
  const fields = { schemaVersion: 1, repository, sourceCommit: run.head_sha, artifactCommit: artifact, runId: run.id, runAttempt: run.run_attempt, markdownSha256: createHash('sha256').update(markdown).digest('hex'), ...changes };
  await writeFile(join(directory, 'report.json'), JSON.stringify(envelope ? envelope.input : fields));
  await writeFile(join(directory, 'consumer-update.md'), markdown);
  await writeFile(join(directory, 'consumer-update.yaml'), yaml);
  const path = join(directory, 'report.zip');
  await runProcess({ command: 'zip', args: ['-q', '-X', path, 'report.json', 'consumer-update.md', 'consumer-update.yaml'], cwd: directory });
  return { path, directory };
}

export const spoofedComment: GitHubIssueComment = { id: 999, user: { login: 'someone', type: 'User' }, body: marker + '\nUser text' };
