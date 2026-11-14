import { mkdtemp, rm, writeFile, mkdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { PreparationOptions } from '#dto/llama-cpp';
import { gitObjectId } from '#tools/publication/git';
import { repositoryName } from '#tools/publication/npm-lock';
import { gitRead, selectedGitlinks } from '#tools/source/capture';
import { ProcessError } from '#tools/shared/process';
import { type Candidate, type PlannedUpdate, type Preflight, type UpdateRequest, pullRequestHandoff } from './plan.js';

export const upstreamPath = 'runtimes/llama-cpp/vendor/llama.cpp';

async function git({ root, args }: { root: string; args: readonly string[] }): Promise<string> {
  return (await gitRead({ sourceRoot: root, args })).toString('utf8').trimEnd();
}

export async function checkedRef({ root, value }: { root: string; value: string }): Promise<string> {
  if (!value || value.startsWith('-') || !/^[A-Za-z0-9_./-]+$/u.test(value)) {
    throw new Error('Custom ref must be a tag, branch, or full commit SHA');
  }
  await git({ root, args: ['check-ref-format', `refs/tags/${value}`] });
  return value;
}

async function selectedCommit({ root, revision }: { root: string; revision: string | undefined }): Promise<string> {
  const commits = await selectedGitlinks({ sourceRoot: root, paths: { llama: upstreamPath }, revision });
  const commit = commits['llama'];
  if (commit === undefined) {
    throw new Error('Missing llama.cpp gitlink');
  }
  return commit;
}

export async function assertSource({ checkoutRoot, request, clean }: { checkoutRoot: string; request: UpdateRequest; clean: boolean }): Promise<void> {
  if (await git({ root: checkoutRoot, args: ['rev-parse', 'HEAD'] }) !== gitObjectId({ value: request.sourceCommit })) {
    throw new Error('Source HEAD changed after update planning');
  }
  if (await selectedCommit({ root: checkoutRoot, revision: request.sourceCommit }) !== request.previousCommit) {
    throw new Error('Committed upstream pin changed after update planning');
  }
  if (clean && await git({ root: checkoutRoot, args: ['status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=none'] })) {
    throw new Error('Updater requires a clean source checkout');
  }
}

export async function createUpdateRequest({ checkoutRoot, repository, base }: { checkoutRoot: string; repository: string; base: string }): Promise<UpdateRequest> {
  repositoryName({ value: repository });
  await checkedRef({ root: checkoutRoot, value: base });
  if (base === 'artifacts' || ['artifacts/', 'automation/llama-cpp/', 'feature/automation/llama-cpp/'].some(prefix => base.startsWith(prefix))) {
    throw new Error('Choose a source base branch, not artifacts or an updater branch');
  }
  const sourceCommit = gitObjectId({ value: await git({ root: checkoutRoot, args: ['rev-parse', 'HEAD'] }) });
  const request = { repository, base, sourceCommit, previousCommit: await selectedCommit({ root: checkoutRoot, revision: sourceCommit }) };
  await assertSource({ checkoutRoot, request, clean: true });
  return request;
}

export async function checkoutUpstream({ checkoutRoot, plan }: { checkoutRoot: string; plan: PlannedUpdate }): Promise<void> {
  await assertSource({ checkoutRoot, request: plan.request, clean: true });
  const root = join(checkoutRoot, upstreamPath);
  if (await git({ root, args: ['rev-parse', '--verify', `${plan.targetCommit}^{commit}`] }) !== plan.targetCommit) {
    throw new Error('Selected upstream commit is not locally available');
  }
  await git({ root, args: ['checkout', '--detach', plan.targetCommit] });
}

export function errorDetails({ error }: { error: unknown }): string {
  return (error instanceof ProcessError && error.stderr.length ? error.stderr.toString('utf8') : error instanceof Error ? error.message : String(error)).trim().slice(0, 4000);
}

/** Exercise the actual composer, including pristine/composed QSA and profile patches. */
export async function overlayPreflight({ checkoutRoot }: { checkoutRoot: string }): Promise<Preflight> {
  const temporary = await mkdtemp(join(tmpdir(), 'bicore-update-overlay-'));
  try {
    const { prepareSources } = await import('#runtimes/llama-cpp/tools/prepare');
    for (const [index, options] of ([
      { target: 'wasm', webgpu: false, gpuProfile: false, fileRows: false, qsa: true, mtmdMode: 'cpu' },
      { target: 'wasm', webgpu: false, gpuProfile: false, fileRows: true, qsa: true, mtmdMode: 'cpu' },
      { target: 'wasm', webgpu: true, gpuProfile: false, fileRows: true, qsa: true, mtmdMode: 'webgpu' },
      { target: 'wasm', webgpu: true, gpuProfile: true, fileRows: true, qsa: true, mtmdMode: 'webgpu' },
    ] satisfies PreparationOptions[]).entries()) {
      await prepareSources({ inputs: { compileRoot: checkoutRoot, upstreamRoots: { llama: join(checkoutRoot, upstreamPath) } }, output: join(temporary, String(index)), options, python: 'python3' });
    }
    return { status: 'passed', scope: 'source preparation only; not compilation or inference', error: undefined };
  } catch (error) {
    return { status: 'failed', scope: 'source preparation only; not compilation or inference', error: errorDetails({ error }) };
  } finally {
    await rm(temporary, { recursive: true, force: true });
  }
}

async function assertCandidateIndex({ checkoutRoot, plan }: { checkoutRoot: string; plan: PlannedUpdate }): Promise<void> {
  await assertSource({ checkoutRoot, request: plan.request, clean: false });
  const changed = await git({ root: checkoutRoot, args: ['diff', '--cached', '--name-only', '--no-relative'] });
  if (changed !== upstreamPath) {
    throw new Error('Updater may change only the llama.cpp submodule gitlink');
  }
  if (await selectedCommit({ root: checkoutRoot, revision: undefined }) !== plan.targetCommit) {
    throw new Error('Candidate pin changed during overlay preflight');
  }
  if (await git({ root: checkoutRoot, args: ['diff', '--name-only', '--no-relative'] }) || await git({ root: checkoutRoot, args: ['ls-files', '--others', '--exclude-standard'] })) {
    throw new Error('Unstaged source changes appeared during candidate preparation');
  }
  if (await git({ root: join(checkoutRoot, upstreamPath), args: ['status', '--porcelain=v1', '--untracked-files=all', '--ignore-submodules=none'] })) {
    throw new Error('Selected upstream worktree changed during candidate preparation');
  }
}

export async function prepareCandidate({ checkoutRoot, plan, existingCommit, preflight }: { checkoutRoot: string; plan: PlannedUpdate; existingCommit: string | undefined; preflight: ({ checkoutRoot }: { checkoutRoot: string }) => Promise<Preflight> }): Promise<Candidate> {
  await assertSource({ checkoutRoot, request: plan.request, clean: false });
  if (existingCommit !== undefined) {
    const sourceCommit = gitObjectId({ value: existingCommit });
    if (await selectedCommit({ root: checkoutRoot, revision: sourceCommit }) !== plan.targetCommit) {
      throw new Error('Existing updater branch has different pins; refusing to overwrite it');
    }
    await assertSource({ checkoutRoot, request: plan.request, clean: true });
    return { plan, status: 'branch_exists', sourceCommit, preflight: { status: 'not_repeated', scope: 'existing branch preserved; full build is authoritative', error: undefined } };
  }
  if (await git({ root: join(checkoutRoot, upstreamPath), args: ['rev-parse', 'HEAD'] }) !== plan.targetCommit) {
    throw new Error('Local upstream checkout differs from the selected target');
  }
  await git({ root: checkoutRoot, args: ['add', '--', upstreamPath] });
  await assertCandidateIndex({ checkoutRoot, plan });
  const result = await preflight({ checkoutRoot });
  await assertCandidateIndex({ checkoutRoot, plan });
  const tree = gitObjectId({ value: await git({ root: checkoutRoot, args: ['write-tree'] }) });
  const sourceCommit = gitObjectId({ value: await git({ root: checkoutRoot, args: ['-c', 'user.name=github-actions[bot]', '-c', 'user.email=41898282+github-actions[bot]@users.noreply.github.com', 'commit-tree', tree, '-p', plan.request.sourceCommit, '-m', 'chore: update llama.cpp to ' + plan.selection.ref.replace(/^refs\/tags\//u, '')] }) });
  return { plan, status: 'candidate_prepared', sourceCommit, preflight: result };
}

/** These files are presentation artifacts, never inputs to another tool stage. */
export async function writeCandidateReport({ state, candidate }: { state: string; candidate: Candidate }): Promise<void> {
  const handoff = pullRequestHandoff({ candidate });
  const branchUrl = `https://github.com/${candidate.plan.request.repository}/tree/${candidate.plan.branch}`;
  let text = `[Open pull request form](${handoff.createPullRequestUrl}) · [Compare branches](${handoff.compareUrl}) · [Update branch](${branchUrl})\n\nThe links do not create a PR. Submitting the PR starts the ordinary runtime workflow.\n`;
  switch (candidate.preflight.status) {
  case 'failed':
    text += '\n**Overlay preflight failed.** The candidate is preserved for manual repair. Opening a draft PR can still start the build; draft status is not a build gate.\n';
    break;
  case 'passed':
  case 'not_repeated': break;
  default: {
    const exhaustive: never = candidate.preflight.status;
    throw new Error(String(exhaustive));
  }
  }
  switch (candidate.status) {
  case 'branch_exists':
    text += '\n**Existing branch preserved without a push.** Re-running this updater does not re-run its CI.\n';
    break;
  case 'candidate_prepared': break;
  default: {
    const exhaustive: never = candidate.status;
    throw new Error(String(exhaustive));
  }
  }
  await mkdir(state, { recursive: true });
  await writeFile(join(state, 'handoff.md'), text);
  const report = { ...candidate, handoff, branchUrl };
  await writeFile(join(state, 'report.json'), JSON.stringify(report, undefined, 2) + '\n');
  await writeFile(join(state, 'details.md'), '<details>\n<summary>Update details and suggested PR text</summary>\n\n```json\n' + JSON.stringify(report, undefined, 2) + '\n```\n\n</details>\n');
}
