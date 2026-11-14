import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import test from 'node:test';
import type { GitHubRelease } from '#dto/llama-cpp';
import { main } from '#tools/automation/cli';
import { assertSource, checkedRef, checkoutUpstream, createUpdateRequest, errorDetails, prepareCandidate, upstreamPath, writeCandidateReport } from '#tools/automation/upstream/operations';
import { candidateBranch, planUpdate, pullRequestHandoff, selectUpstream, type PlannedUpdate, type Preflight } from '#tools/automation/upstream/plan';
import { ProcessError, runProcess } from '#tools/shared/process';
import { commitFiles, git, temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { createUpdaterFixture as fixture } from '#tests/tools/automation/upstream.test-support';

const repository = 'example/core';
const passed: Preflight = { status: 'passed', scope: 'fixture preparation', error: undefined };

function release({ name, prerelease, draft }: { name: string; prerelease: boolean; draft: boolean }): GitHubRelease {
  return { tag_name: name, draft, prerelease, published_at: '2026-09-22T00:00:00Z', html_url: `https://github.com/ggml-org/llama.cpp/releases/tag/${name}` };
}

test('stable and released-nightly selection never fall back to another channel', () => {
  const stable = release({ name: 'v1.2.3', prerelease: false, draft: false });
  assert.equal(selectUpstream({ target: { kind: 'latest' }, releases: [stable] }).ref, 'refs/tags/v1.2.3');
  for (const invalid of [release({ name: 'b999', prerelease: true, draft: false }), { ...stable, draft: true }, { ...stable, prerelease: true }, { ...stable, tag_name: 'v01.2.3' }, { ...stable, published_at: null }]) {
    assert.throws(() => selectUpstream({ target: { kind: 'latest' }, releases: [invalid] }), /no fallback/u);
  }
  const releases = [release({ name: 'b100', prerelease: true, draft: true }), stable, release({ name: 'b99', prerelease: true, draft: false }), release({ name: 'b98', prerelease: true, draft: false })];
  assert.equal(selectUpstream({ target: { kind: 'latest_unstable' }, releases }).ref, 'refs/tags/b99');
  assert.throws(() => selectUpstream({ target: { kind: 'latest_unstable' }, releases: [stable] }), /no fallback/u);
});

test('custom refs are explicit Git names and candidates bind base, source and target', async () => {
  await using temporary = await temporaryDirectory();
  for (const value of ['a'.repeat(40), 'master', 'refs/heads/master', 'refs/tags/v1.0.0']) {
    assert.equal(await checkedRef({ root: temporary.root, value }), value);
  }
  for (const value of ['', '-x', 'HEAD~1', 'x^{}', 'x\ny', 'https://evil/x', 'x;id', 'x..y', 'x@{1}', 'a.lock']) {
    await assert.rejects(checkedRef({ root: temporary.root, value }));
  }
  const source = 'a'.repeat(40);
  const target = 'b'.repeat(40);
  const branch = candidateBranch({ base: 'feature/base', source, target });
  assert.match(branch, /^feature\/automation\/llama-cpp\//u);
  assert.equal(branch, candidateBranch({ base: 'feature/base', source, target }));
  assert.notEqual(branch, candidateBranch({ base: 'main', source, target }));
  assert.notEqual(branch, candidateBranch({ base: 'feature/base', source: target, target }));
});

test('no-op and rollback selection preserve the original request identity', async () => {
  await using temporary = await temporaryDirectory();
  const value = await fixture({ root: temporary.root });
  const unchanged = planUpdate({ request: value.request, selection: value.plan.selection, resolvedCommit: value.previous, resolvedRef: value.plan.selection.ref, relation: undefined, allowNonFastForward: false });
  assert.equal(unchanged.status, 'unchanged');
  assert.throws(() => planUpdate({ request: value.request, selection: value.plan.selection, resolvedCommit: value.target, resolvedRef: 'master', relation: 'ahead', allowNonFastForward: false }), /differs from selection/u);
  assert.throws(() => planUpdate({ request: value.request, selection: value.plan.selection, resolvedCommit: value.target, resolvedRef: value.target, relation: 'behind', allowNonFastForward: false }), /allow_non_fast_forward/u);
  assert.equal(planUpdate({ request: value.request, selection: value.plan.selection, resolvedCommit: value.target, resolvedRef: value.target, relation: 'behind', allowNonFastForward: true }).status, 'planned');
  for (const base of ['artifacts', 'artifacts/tree', 'automation/llama-cpp/old', 'feature/automation/llama-cpp/old']) {
    await assert.rejects(createUpdateRequest({ checkoutRoot: value.checkoutRoot, repository, base }), /source base/u);
  }
  await writeFile(join(value.checkoutRoot, 'dirty'), 'untracked');
  await assert.rejects(createUpdateRequest({ checkoutRoot: value.checkoutRoot, repository, base: 'feature/base' }), /clean/u);
});

test('real local candidate changes only the canonical gitlink and keeps HEAD as its parent', async () => {
  await using temporary = await temporaryDirectory();
  const value = await fixture({ root: temporary.root });
  await checkoutUpstream({ checkoutRoot: value.checkoutRoot, plan: value.plan });
  let calls = 0;
  const candidate = await prepareCandidate({
    checkoutRoot: value.checkoutRoot,
    plan: value.plan,
    existingCommit: undefined,
    preflight: async () => {
      calls++;
      return passed;
    },
  });
  assert.equal(calls, 1);
  assert.equal(candidate.status, 'candidate_prepared');
  assert.equal(await git({ root: value.checkoutRoot, args: ['diff', '--name-only', value.source, candidate.sourceCommit] }), upstreamPath);
  assert.equal(await git({ root: value.checkoutRoot, args: ['rev-parse', `${candidate.sourceCommit}^`] }), value.source);
  assert.equal(await git({ root: value.checkoutRoot, args: ['rev-parse', 'HEAD'] }), value.source);
  assert.deepEqual(JSON.parse(await readFile(join(value.checkoutRoot, 'runtimes/llama-cpp/config/toolchain.json'), 'utf8')), { artifactBranch: 'artifacts' });
  await writeCandidateReport({ state: join(temporary.root, 'report'), candidate });
  assert.match(await readFile(join(temporary.root, 'report/handoff.md'), 'utf8'), /The links do not create a PR/u);
});

test('existing fetched repair commits are preserved without repeating preflight', async () => {
  await using temporary = await temporaryDirectory();
  const value = await fixture({ root: temporary.root });
  await checkoutUpstream({ checkoutRoot: value.checkoutRoot, plan: value.plan });
  const first = await prepareCandidate({ checkoutRoot: value.checkoutRoot, plan: value.plan, existingCommit: undefined, preflight: async () => passed });
  await writeFile(join(value.checkoutRoot, 'manual-repair'), 'human repair');
  await git({ root: value.checkoutRoot, args: ['add', 'manual-repair'] });
  const tree = await git({ root: value.checkoutRoot, args: ['write-tree'] });
  const repair = await git({ root: value.checkoutRoot, args: ['commit-tree', tree, '-p', first.sourceCommit, '-m', 'human repair'] });
  await git({ root: value.checkoutRoot, args: ['reset', '--hard', value.source] });
  await git({ root: join(value.checkoutRoot, upstreamPath), args: ['checkout', '--detach', value.previous] });
  const candidate = await prepareCandidate({
    checkoutRoot: value.checkoutRoot,
    plan: value.plan,
    existingCommit: repair,
    preflight: async () => {
      throw new Error('Reused branches must not repeat preflight');
    },
  });
  assert.equal(candidate.status, 'branch_exists');
  assert.equal(candidate.sourceCommit, repair);
  assert.equal(candidate.preflight.status, 'not_repeated');
  await assert.rejects(prepareCandidate({ checkoutRoot: value.checkoutRoot, plan: value.plan, existingCommit: value.source, preflight: async () => passed }), /refusing to overwrite/u);
});

test('preflight failures preserve the candidate while source and index races prevent a commit', async () => {
  await using temporary = await temporaryDirectory();
  const value = await fixture({ root: temporary.root });
  await checkoutUpstream({ checkoutRoot: value.checkoutRoot, plan: value.plan });
  const candidate = await prepareCandidate({ checkoutRoot: value.checkoutRoot, plan: value.plan, existingCommit: undefined, preflight: async () => ({ status: 'failed', scope: 'fixture patch application', error: 'conflicting patch' }) });
  assert.equal(candidate.status, 'candidate_prepared');
  assert.equal(candidate.preflight.status, 'failed');
  const state = join(temporary.root, 'report');
  await writeCandidateReport({ state, candidate });
  assert.match(await readFile(join(state, 'handoff.md'), 'utf8'), /draft status is not a build gate/u);
  await assert.rejects(prepareCandidate({
    checkoutRoot: value.checkoutRoot,
    plan: value.plan,
    existingCommit: undefined,
    preflight: async () => {
      await writeFile(join(value.checkoutRoot, 'outside-runtime'), 'unexpected');
      await git({ root: value.checkoutRoot, args: ['add', 'outside-runtime'] });
      return passed;
    },
  }), /only the llama.cpp/u);
});

test('HEAD and same-path staged pin races cannot retarget the planned candidate', async () => {
  for (const race of ['head', 'pin'] as const) {
    await using temporary = await temporaryDirectory();
    const value = await fixture({ root: temporary.root });
    await checkoutUpstream({ checkoutRoot: value.checkoutRoot, plan: value.plan });
    await assert.rejects(prepareCandidate({
      checkoutRoot: value.checkoutRoot,
      plan: value.plan,
      existingCommit: undefined,
      preflight: async () => {
        switch (race) {
        case 'head': await git({ root: value.checkoutRoot, args: ['commit', '--allow-empty', '-qm', 'unexpected move'] });
          break;
        case 'pin': await git({ root: value.checkoutRoot, args: ['update-index', '--cacheinfo', `160000,${value.previous},${upstreamPath}`] });
          break;
        default: {
          const exhaustive: never = race;
          throw new Error(String(exhaustive));
        }
        }
        return passed;
      },
    }), /HEAD changed|only the llama.cpp|pin changed/u);
  }
});

test('handoff URLs encode names, bound query length and exclude diagnostics', () => {
  const plan: PlannedUpdate = { status: 'planned', request: { repository, base: 'feature/base', sourceCommit: 'a'.repeat(40), previousCommit: 'b'.repeat(40) }, selection: { requested: 'custom', ref: 'refs/heads/topic/new', release: undefined }, targetCommit: 'c'.repeat(40), relation: 'ahead', branch: 'feature/automation/llama-cpp/candidate' };
  const candidate = { plan, status: 'candidate_prepared' as const, sourceCommit: 'd'.repeat(40), preflight: { status: 'failed' as const, scope: 'patch application', error: 'diagnostic '.repeat(1000) } };
  const result = pullRequestHandoff({ candidate });
  const url = new URL(result.createPullRequestUrl);
  assert.equal(url.searchParams.get('title'), 'chore: update llama.cpp to refs/heads/topic/new');
  assert.ok(decodeURIComponent(url.pathname).includes('feature/base...feature/automation/llama-cpp/candidate'));
  assert.ok(!url.search.includes('diagnostic+diagnostic'));
  const long = pullRequestHandoff({ candidate: { ...candidate, plan: { ...plan, selection: { ...plan.selection, ref: 'refs/tags/' + 'x'.repeat(8000) } } } });
  assert.equal(new URL(long.createPullRequestUrl).search, '?quick_pull=1');
  const failure = new ProcessError({ command: 'git', exitCode: 1, stderr: Buffer.from([255, 10]), reason: 'failed' });
  assert.equal(errorDetails({ error: failure }), '\ufffd');
  assert.equal(errorDetails({ error: new Error('x'.repeat(5000)) }).length, 4000);
});

test('the real upstream CLI carries workflow primitives through plan, checkout and candidate without plan files', async () => {
  await using temporary = await temporaryDirectory();
  const value = await fixture({ root: temporary.root });
  const state = join(temporary.root, 'state');
  const output = join(temporary.root, 'output');
  const env = { GITHUB_OUTPUT: output };
  const args = ['--checkout-root', value.checkoutRoot, '--state', state, '--repository', repository, '--base', value.request.base, '--source', value.source, '--previous', value.previous, '--ref', 'refs/tags/v1.0.0', '--resolved-ref', 'refs/tags/v1.0.0', '--commit', value.target, '--requested', 'custom', '--allow-non-fast-forward', 'false', '--relation', 'ahead'];
  await main({ args: ['upstream-plan', ...args], outputPath: env.GITHUB_OUTPUT });
  await main({ args: ['upstream-checkout', ...args], outputPath: env.GITHUB_OUTPUT });
  await main({ args: ['upstream-candidate', ...args], outputPath: env.GITHUB_OUTPUT });
  const text = await readFile(output, 'utf8');
  assert.match(text, /changed=true/u);
  assert.match(text, /needs-push=true/u);
  assert.match(text, /status=candidate_prepared/u);
  assert.match(await readFile(join(state, 'details.md'), 'utf8'), /sourceCommit/u);
  assert.match(await readFile(join(state, 'handoff.md'), 'utf8'), /Overlay preflight failed/u);
  // This fixture lacks native sources. The real preflight failure still preserves
  // its canonical-pin candidate, rather than substituting a mocked CLI result.
  const unchangedArgs = args.map((arg, index) => args[index - 1] === '--commit' ? value.previous : arg).filter((_, index) => args[index] !== '--relation' && args[index - 1] !== '--relation');
  await git({ root: value.checkoutRoot, args: ['reset', '--hard', value.source] });
  await git({ root: join(value.checkoutRoot, upstreamPath), args: ['checkout', '--detach', value.previous] });
  await main({ args: ['upstream-plan', ...unchangedArgs, '--comparison', join(temporary.root, 'absent-comparison.json')], outputPath: env.GITHUB_OUTPUT });
  assert.match(await readFile(output, 'utf8'), /changed=false/u);
  await assertSource({ checkoutRoot: value.checkoutRoot, request: value.request, clean: true });
});

test('actual Git patch failures retain bounded text in the saved candidate report', async () => {
  await using temporary = await temporaryDirectory();
  const value = await fixture({ root: temporary.root });
  const patch = join(temporary.root, 'conflict.patch');
  await writeFile(patch, `\
--- a/source
+++ b/source
@@ -1 +1 @@
-missing original line
+replacement line
`);
  let diagnostic: string | undefined;
  await assert.rejects(runProcess({ command: 'git', args: ['apply', '--no-index', patch], cwd: join(value.checkoutRoot, upstreamPath) }), (error: unknown) => {
    assert.ok(error instanceof ProcessError);
    diagnostic = errorDetails({ error });
    assert.match(diagnostic, /patch does not apply/u);
    return true;
  });
  const state = join(temporary.root, 'report');
  await writeCandidateReport({ state, candidate: { plan: value.plan, status: 'candidate_prepared', sourceCommit: value.source, preflight: { status: 'failed', scope: 'patch application', error: diagnostic } } });
  assert.match(await readFile(join(state, 'details.md'), 'utf8'), /patch does not apply/u);
  assert.match(await readFile(join(state, 'report.json'), 'utf8'), /patch does not apply/u);
});

test('request validation and selection share checks and preserve the source identity across acquisition', async () => {
  await using temporary = await temporaryDirectory();
  const value = await fixture({ root: temporary.root });
  const output = join(temporary.root, 'output');
  const args = ['--checkout-root', value.checkoutRoot, '--state', join(temporary.root, 'state'), '--repository', repository, '--base', value.request.base, '--target', 'custom', '--custom-ref', value.target];
  const env = { GITHUB_OUTPUT: output };
  await main({ args: ['upstream-validate', ...args], outputPath: env.GITHUB_OUTPUT });
  assert.match(await readFile(output, 'utf8'), new RegExp(`source=${value.source}`, 'u'));
  assert.match(await readFile(output, 'utf8'), new RegExp(`previous=${value.previous}`, 'u'));
  await main({ args: ['upstream-select', ...args, '--expected-source', value.source, '--expected-previous', value.previous], outputPath: env.GITHUB_OUTPUT });
  assert.match(await readFile(output, 'utf8'), new RegExp(`ref=${value.target}`, 'u'));
  await commitFiles({ root: value.checkoutRoot });
  await assert.rejects(main({ args: ['upstream-select', ...args, '--expected-source', value.source, '--expected-previous', value.previous], outputPath: env.GITHUB_OUTPUT }), /Source identity changed/u);
});
