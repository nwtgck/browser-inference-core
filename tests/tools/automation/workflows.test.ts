import { READ_ENV_PATH } from '#tools/shared/node-env';
import assert from 'node:assert/strict';
import { chmod, mkdir, readFile, symlink, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test from 'node:test';
import { load } from 'js-yaml';
import { decodeGitHubReleases } from '#dto/llama-cpp';
import { selectUpstream } from '#tools/automation/upstream/plan';
import { main } from '#tools/automation/cli';
import { checkoutUpstream, prepareCandidate, upstreamPath } from '#tools/automation/upstream/operations';
import { optionalStat } from '#tools/shared/files';
import { readJson } from '#tools/shared/json';
import { ProcessError, runProcess } from '#tools/shared/process';
import { commitFiles, git, initializeRepository, temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { botComment, pullRequest, reportArchive, repository, source, workflowRun } from '#tests/tools/automation/comments.test-support';
import type { Scenario } from '#tests/tools/automation/service.test-support';
import { createUpdaterFixture } from '#tests/tools/automation/upstream.test-support';

const checkoutRoot = dirname(dirname(fileURLToPath(import.meta.resolve('#source/toolchain.json'))));
type ObjectValue = Record<string, unknown>;
type Step = { name: string; run: string | undefined; env: ObjectValue; raw: ObjectValue };
type Call = { tool: string; args: string[]; hasToken: boolean };

function isObject(value: unknown): value is ObjectValue {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function object({ value }: { value: unknown }): ObjectValue {
  assert.ok(isObject(value));
  return value;
}

async function workflow({ name, job }: { name: string; job: string }): Promise<{ document: ObjectValue; job: ObjectValue; steps: Step[] }> {
  const document = object({ value: load(await readFile(join(checkoutRoot, '.github/workflows', name), 'utf8')) });
  const selected = object({ value: object({ value: document['jobs'] })[job] });
  const steps = selected['steps'];
  assert.ok(Array.isArray(steps));
  return {
    document,
    job: selected,
    steps: steps.map((value: unknown) => {
      const raw = object({ value });
      return { name: typeof raw['name'] === 'string' ? raw['name'] : '', run: typeof raw['run'] === 'string' ? raw['run'] : undefined, env: raw['env'] === undefined ? {} : object({ value: raw['env'] }), raw };
    }),
  };
}

function scenario(): Scenario {
  return { pulls: [pullRequest({ head: source, number: 7 })], runs: [workflowRun({ head: source, id: 123, attempt: 1, status: 'completed', conclusion: 'success' })], comments: [], currentPulls: undefined, currentRuns: undefined, totalCount: undefined, currentTotalCount: undefined, redirect: 'https://storage.invalid/signed-report', directArchive: false, failArtifactListing: false, failWrite: false, latest: undefined, releases: [], failRelease: false, relation: 'ahead' };
}

async function serviceEnvironment({ root, value }: { root: string; value: Scenario }): Promise<NodeJS.ProcessEnv> {
  await writeFile(join(root, 'scenario.json'), JSON.stringify(value));
  const bin = join(root, 'bin');
  await mkdir(bin);
  const service = fileURLToPath(import.meta.resolve('#tests/tools/automation/service.test-support'));
  await chmod(service, 0o755);
  for (const name of ['gh', 'curl', 'docker', 'git']) {
    await symlink(service, join(bin, name));
  }
  return { PATH: `${bin}:${READ_ENV_PATH() ?? ''}`, BICORE_FIXTURE_ORIGINAL_PATH: READ_ENV_PATH(), BICORE_AUTOMATION_FIXTURE: root, GITHUB_WORKSPACE: checkoutRoot, GITHUB_REPOSITORY: repository, RUNNER_TEMP: root, REPORT_STATE: join(root, 'state'), UPDATE_STATE: join(root, 'update'), GITHUB_OUTPUT: join(root, 'outputs'), GITHUB_STEP_SUMMARY: join(root, 'summary'), BIC_OFFLINE_IMAGE: 'test-stand-in-with-no-isolation-claim' };
}

async function recorded({ root, name }: { root: string; name: string }): Promise<Call[]> {
  const path = join(root, name);
  return await optionalStat({ path }) ? (await readFile(path, 'utf8')).trim().split('\n').filter(Boolean).map(line => JSON.parse(line) as Call) : [];
}

async function executeReporter({ root, value }: { root: string; value: Scenario }): Promise<void> {
  const env = await serviceEnvironment({ root, value });
  const report = await reportArchive({ root, run: value.runs.at(-1) ?? workflowRun({ head: source, id: 123, attempt: 1, status: 'completed', conclusion: 'success' }), changes: {}, body: undefined, envelope: undefined, yaml: '' });
  await writeFile(join(root, 'report.zip'), await readFile(report.path));
  const { steps } = await workflow({ name: 'runtime-comments.yml', job: 'report' });
  for (const step of steps) {
    if (!step.run || ['Acquire trusted tooling dependencies (online)', 'Acquire offline automation image (online)', 'Emit trusted default-branch tooling'].includes(step.name)) {
      continue;
    }
    const environment = { ...env };
    environment['GH_TOKEN'] = undefined;
    if (step.env['GH_TOKEN'] !== undefined) {
      environment['GH_TOKEN'] = 'fixture-token';
    }
    await runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', step.run], cwd: checkoutRoot, env: { kind: 'inherit', removePrefixes: [], overrides: environment } });
  }
}

test('parsed workflow permissions, event trust and all literal shell bodies remain valid', async () => {
  const update = await workflow({ name: 'update-llama-cpp.yml', job: 'update' });
  const report = await workflow({ name: 'runtime-comments.yml', job: 'report' });
  assert.deepEqual(update.document['permissions'], { contents: 'read' });
  assert.deepEqual(update.job['permissions'], { contents: 'write' });
  assert.deepEqual(report.document['permissions'], { contents: 'read', actions: 'read', 'pull-requests': 'write' });
  assert.deepEqual(report.document['concurrency'], { group: 'runtime-pr-comment-reconciliation', 'cancel-in-progress': false });
  const checkout = report.steps.find(step => step.raw['uses'] === 'actions/checkout@v7.0.1');
  assert.ok(checkout);
  assert.deepEqual(checkout.raw['with'], { ref: '${{ github.event.repository.default_branch }}', 'persist-credentials': false });
  const events = object({ value: report.document['on'] });
  assert.deepEqual(object({ value: events['pull_request_target'] })['types'], ['opened', 'reopened', 'synchronize']);
  assert.deepEqual(object({ value: events['workflow_run'] })['types'], ['requested', 'in_progress', 'completed']);
  assert.ok(Object.hasOwn(events, 'workflow_dispatch'));
  assert.equal(report.job['if'], "(github.event_name != 'pull_request_target' || github.event.pull_request.head.repo.full_name == github.repository) && (github.event_name != 'workflow_run' || github.event.workflow_run.head_repository.full_name == github.repository)");
  for (const selected of [update, report]) {
    for (const step of selected.steps) {
      if (!step.run) {
        continue;
      }
      assert.doesNotMatch(step.run, /python3 -m build_tools|\$\{\{/u);
      await runProcess({ command: 'bash', args: ['-n'], cwd: checkoutRoot, input: step.run });
      if (step.run.includes('build/tooling/tools/automation/cli.js')) {
        assert.match(step.run, /bash \.github\/scripts\/run-offline\.sh/u);
        assert.doesNotMatch(step.run.split("<<'BICORE_OFFLINE'")[0] ?? '', /-e GH_TOKEN/u);
      }
    }
  }
});

test('literal reporter shell separates API credentials, consumes the attempt-bound report and posts once', async () => {
  await using temporary = await temporaryDirectory();
  await executeReporter({ root: temporary.root, value: scenario() });
  const writes = await recorded({ root: temporary.root, name: 'writes.jsonl' });
  assert.equal(writes.length, 1);
  const calls = await recorded({ root: temporary.root, name: 'calls.jsonl' });
  assert.deepEqual(calls.filter(call => call.tool === 'curl').map(call => call.hasToken), [true, false]);
  assert.match(await readFile(join(temporary.root, 'state/7/body.json'), 'utf8'), /succeeded for this PR head/u);
});

test('literal reporter handles direct archives, bot updates and all open PRs', async () => {
  await using temporary = await temporaryDirectory();
  const value = scenario();
  value.directArchive = true;
  value.pulls.push(pullRequest({ head: source, number: 8 }));
  value.comments.push(botComment({ body: '<!-- lcb-runtime-artifact-comment:v1 -->\nOld' }));
  await executeReporter({ root: temporary.root, value });
  const writes = await recorded({ root: temporary.root, name: 'writes.jsonl' });
  assert.equal(writes.length, 2);
  assert.ok(writes.every(write => JSON.stringify(write).includes('PATCH')));
  assert.equal((await recorded({ root: temporary.root, name: 'calls.jsonl' })).filter(call => call.tool === 'curl').length, 2);
});

test('literal reporter never follows a fork, a stale head or a newer rerun into a write', async () => {
  for (const failure of ['fork', 'head', 'rerun'] as const) {
    await using temporary = await temporaryDirectory();
    const value = scenario();
    const original = value.pulls[0]!;
    switch (failure) {
    case 'fork': value.pulls = [{ ...original, head: { ...original.head, repo: { full_name: 'attacker/core' } } }];
      break;
    case 'head': value.currentPulls = [{ ...original, head: { ...original.head, sha: 'b'.repeat(40) } }];
      break;
    case 'rerun': value.currentRuns = [workflowRun({ head: source, id: 123, attempt: 2, status: 'queued', conclusion: null })];
      break;
    default: {
      const exhaustive: never = failure;
      throw new Error(String(exhaustive));
    }
    }
    await executeReporter({ root: temporary.root, value });
    assert.equal((await recorded({ root: temporary.root, name: 'writes.jsonl' })).length, 0);
  }
});

test('literal reporter pagination selects a later-page rerun and rejects incomplete histories', async () => {
  await using temporary = await temporaryDirectory();
  const value = scenario();
  value.runs = Array.from({ length: 100 }, (_, index) => ({ ...workflowRun({ head: source, id: 200 + index, attempt: 1, status: 'completed', conclusion: 'success' }), run_started_at: '2026-09-22T10:00:00Z' }));
  value.runs.push({ ...workflowRun({ head: source, id: 100, attempt: 3, status: 'completed', conclusion: 'success' }), run_started_at: '2026-09-22T12:00:00Z' });
  await executeReporter({ root: temporary.root, value });
  assert.match(await readFile(join(temporary.root, 'state/7/body.json'), 'utf8'), /attempt 3/u);
  const histories = (await recorded({ root: temporary.root, name: 'calls.jsonl' })).filter(call => call.args.some(arg => arg.endsWith('/actions/workflows/build.yml/runs')));
  assert.equal(histories.length, 4);
  for (const total of [2, 101, 1001]) {
    await using incomplete = await temporaryDirectory();
    await assert.rejects(executeReporter({ root: incomplete.root, value: { ...scenario(), totalCount: total } }), ProcessError);
    assert.equal((await recorded({ root: incomplete.root, name: 'writes.jsonl' })).length, 0);
  }
});

test('literal reporter preserves per-PR retrieval errors, rejects unsafe redirects and never retries writes', async () => {
  for (const failure of ['listing', 'redirect', 'write'] as const) {
    await using temporary = await temporaryDirectory();
    const value = scenario();
    switch (failure) {
    case 'listing': value.failArtifactListing = true;
      break;
    case 'redirect': value.redirect = 'http://storage.invalid/untrusted';
      break;
    case 'write': value.failWrite = true;
      break;
    default: {
      const exhaustive: never = failure;
      throw new Error(String(exhaustive));
    }
    }
    switch (failure) {
    case 'write': await assert.rejects(executeReporter({ root: temporary.root, value }), ProcessError);
      break;
    case 'listing':
    case 'redirect': await executeReporter({ root: temporary.root, value });
      assert.match(await readFile(join(temporary.root, 'state/7/body.json'), 'utf8'), /Report retrieval error/u);
      break;
    default: {
      const exhaustive: never = failure;
      throw new Error(String(exhaustive));
    }
    }
    assert.equal((await recorded({ root: temporary.root, name: 'writes.jsonl' })).length, 1);
  }
});

test('literal updater acquisition peels annotated tags, rejects ambiguity and binds full SHAs locally', async () => {
  await using temporary = await temporaryDirectory();
  const upstream = join(temporary.root, 'upstream');
  await initializeRepository({ root: upstream });
  await writeFile(join(upstream, 'source'), 'original');
  const old = await commitFiles({ root: upstream });
  await writeFile(join(upstream, 'source'), 'updated');
  const target = await commitFiles({ root: upstream });
  await git({ root: upstream, args: ['tag', '-a', 'v1.0.0', '-m', 'annotated fixture', target] });
  await git({ root: upstream, args: ['tag', 'shared', target] });
  await git({ root: upstream, args: ['branch', 'shared', old] });
  const sourceRoot = join(temporary.root, 'source');
  const vendor = join(sourceRoot, 'runtimes/llama-cpp/vendor/llama.cpp');
  await mkdir(dirname(vendor), { recursive: true });
  await runProcess({ command: 'git', args: ['clone', '-q', upstream, vendor], cwd: temporary.root });
  await git({ root: vendor, args: ['checkout', '--detach', old] });
  const env = await serviceEnvironment({ root: temporary.root, value: scenario() });
  env['GITHUB_WORKSPACE'] = sourceRoot;
  env['BICORE_FIXTURE_UPSTREAM'] = upstream;
  env['BICORE_FIXTURE_REAL_GIT'] = (await runProcess({ command: 'which', args: ['git'], cwd: checkoutRoot })).stdout.toString().trim();
  env['PREVIOUS_COMMIT'] = old;
  await mkdir(join(temporary.root, 'update'));
  const { steps } = await workflow({ name: 'update-llama-cpp.yml', job: 'update' });
  const fetch = steps.find(step => step.name === 'Fetch the exact upstream and comparison (online)');
  assert.ok(fetch?.run);
  for (const ref of ['refs/tags/v1.0.0', target]) {
    await runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', fetch.run], cwd: sourceRoot, env: { kind: 'inherit', removePrefixes: [], overrides: { ...env, SELECTED_REF: ref } } });
    assert.match(await readFile(join(temporary.root, 'outputs'), 'utf8'), new RegExp(`commit=${target}`, 'u'));
    assert.equal(await git({ root: vendor, args: ['rev-parse', 'HEAD'] }), old);
  }
  await assert.rejects(runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', fetch.run], cwd: sourceRoot, env: { kind: 'inherit', removePrefixes: [], overrides: { ...env, SELECTED_REF: 'shared' } } }), /missing or ambiguous/u);
  const blob = await git({ root: upstream, args: ['rev-parse', `${target}:source`] });
  await git({ root: upstream, args: ['tag', 'not-a-commit', blob] });
  await assert.rejects(runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', fetch.run], cwd: sourceRoot, env: { kind: 'inherit', removePrefixes: [], overrides: { ...env, SELECTED_REF: 'refs/tags/not-a-commit' } } }));
});

test('literal nightly pagination stops at the first published match and stable failures have no fallback', async () => {
  await using temporary = await temporaryDirectory();
  const value = scenario();
  value.releases = Array.from({ length: 100 }, (_, index) => ({ draft: false, prerelease: false, published_at: '2026-09-22', tag_name: `v1.0.${index}`, html_url: 'https://github.com/ggml-org/llama.cpp/releases' }));
  value.releases.push({ draft: false, prerelease: true, published_at: '2026-09-22', tag_name: 'b99', html_url: 'https://github.com/ggml-org/llama.cpp/releases' });
  const env = await serviceEnvironment({ root: temporary.root, value });
  const { steps } = await workflow({ name: 'update-llama-cpp.yml', job: 'update' });
  const release = steps.find(step => step.name === 'Acquire the requested release metadata (online)');
  assert.ok(release?.run);
  await runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', release.run], cwd: checkoutRoot, env: { kind: 'inherit', removePrefixes: [], overrides: { ...env, TARGET: 'latest-unstable' } } });
  const releases = await readJson({ path: join(temporary.root, 'update/releases.json'), decode: decodeGitHubReleases });
  assert.equal(selectUpstream({ target: { kind: 'latest_unstable' }, releases }).ref, 'refs/tags/b99');
  assert.equal((await recorded({ root: temporary.root, name: 'calls.jsonl' })).length, 2);
  value.failRelease = true;
  await writeFile(join(temporary.root, 'scenario.json'), JSON.stringify(value));
  await assert.rejects(runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', release.run], cwd: checkoutRoot, env: { kind: 'inherit', removePrefixes: [], overrides: { ...env, TARGET: 'latest' } } }));
  assert.equal((await recorded({ root: temporary.root, name: 'calls.jsonl' })).length, 3);
  await assert.rejects(main({ args: ['upstream-select', '--state', join(temporary.root, 'failed'), '--repository', repository, '--checkout-root', checkoutRoot, '--base', 'feature/base', '--target', 'custom', '--custom-ref', 'bad..ref'], outputPath: undefined }));
  assert.match(await readFile(join(temporary.root, 'failed/details.md'), 'utf8'), /Upstream update failed/u);
});

test('literal final updater reporting exposes the push outcome without reading its saved JSON', async () => {
  await using temporary = await temporaryDirectory();
  const state = join(temporary.root, 'state');
  await mkdir(state);
  await writeFile(join(state, 'handoff.md'), 'Reviewable handoff link\n');
  await writeFile(join(state, 'details.md'), 'Candidate source identity and diagnostics\n');
  // A presentation-only saved artifact is never reparsed by final reporting.
  await writeFile(join(state, 'report.json'), 'not JSON');
  const { steps } = await workflow({ name: 'update-llama-cpp.yml', job: 'update' });
  const final = steps.find(step => step.name === 'Render the manual pull request handoff');
  assert.ok(final?.run);
  for (const [status, outcome, changed, handoff] of [['candidate_prepared', 'success', 'true', true], ['branch_exists', 'skipped', 'true', true], ['candidate_prepared', 'failure', 'true', false], ['', 'skipped', 'false', false]] as const) {
    const summary = join(temporary.root, `summary-${outcome}-${changed}`);
    await runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', final.run], cwd: checkoutRoot, env: { kind: 'inherit', removePrefixes: [], overrides: { UPDATE_STATE: state, GITHUB_STEP_SUMMARY: summary, CANDIDATE_STATUS: status, PUSH_OUTCOME: outcome, CHANGED: changed } } });
    const text = await readFile(summary, 'utf8');
    assert.equal(text.includes('Reviewable handoff link'), handoff);
    assert.match(text, /Candidate source identity/u);
    assert.equal(await readFile(join(state, 'push-outcome.txt'), 'utf8'), outcome + '\n');
  }
});

test('the first literal offline phase rejects invalid requests before any release API call', async () => {
  const { steps } = await workflow({ name: 'update-llama-cpp.yml', job: 'update' });
  const first = steps.find(step => step.name === 'Emit the current checkout tooling');
  const release = steps.find(step => step.name === 'Acquire the requested release metadata (online)');
  assert.ok(first?.run);
  assert.ok(release?.run);
  assert.ok(steps.indexOf(first) < steps.indexOf(release));
  assert.equal(release.raw['if'], undefined);
  for (const [target, customRef, base, message] of [
    ['latest', 'master', 'feature/base', /custom_ref is only valid/u],
    ['latest-unstable', 'master', 'feature/base', /custom_ref is only valid/u],
    ['custom', 'bad..ref', 'feature/base', /git: exited with 1/u],
    ['latest', '', 'artifacts', /Choose a source base branch/u],
  ] as const) {
    await using temporary = await temporaryDirectory();
    const env = await serviceEnvironment({ root: temporary.root, value: scenario() });
    const offlineBin = join(temporary.root, 'offline-bin');
    await mkdir(offlineBin);
    // The test already emits the actual CLI. This stand-in avoids repeating a
    // repository-wide compiler run while exercising the literal command order.
    await writeFile(join(offlineBin, 'npm'), `\
#!/bin/sh
test "$#" = 2 && test "$1" = run && test "$2" = tooling:build
`);
    await chmod(join(offlineBin, 'npm'), 0o755);
    await assert.rejects(runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', first.run + '\n' + release.run], cwd: checkoutRoot, env: { kind: 'inherit', removePrefixes: [], overrides: { ...env, TARGET: target, CUSTOM_REF: customRef, BASE: base, GH_TOKEN: 'fixture-token', BICORE_FIXTURE_ORIGINAL_PATH: `${offlineBin}:${READ_ENV_PATH() ?? ''}` } } }), (error: unknown) => error instanceof ProcessError && message.test(error.message));
    const calls = await recorded({ root: temporary.root, name: 'calls.jsonl' });
    assert.equal(calls.filter(call => call.tool === 'docker').length, 1);
    assert.equal(calls.filter(call => call.tool === 'gh' || call.tool === 'curl').length, 0);
    assert.match(await readFile(join(temporary.root, 'update/details.md'), 'utf8'), /Upstream update failed/u);
  }
});

test('literal branch inspection preserves the fetched repair tip when a branch advances after its probe', async () => {
  await using temporary = await temporaryDirectory();
  const value = await createUpdaterFixture({ root: temporary.root });
  await checkoutUpstream({ checkoutRoot: value.checkoutRoot, plan: value.plan });
  const first = await prepareCandidate({ checkoutRoot: value.checkoutRoot, plan: value.plan, existingCommit: undefined, preflight: async () => ({ status: 'passed', scope: 'fixture source preparation', error: undefined }) });
  await writeFile(join(value.checkoutRoot, 'manual-repair'), 'human repair\n');
  await git({ root: value.checkoutRoot, args: ['add', 'manual-repair'] });
  const tree = await git({ root: value.checkoutRoot, args: ['write-tree'] });
  const repair = await git({ root: value.checkoutRoot, args: ['commit-tree', tree, '-p', first.sourceCommit, '-m', 'human repair'] });
  await git({ root: value.checkoutRoot, args: ['update-ref', `refs/heads/${value.plan.branch}`, first.sourceCommit] });
  await git({ root: value.checkoutRoot, args: ['update-ref', 'refs/heads/feature/repair-fixture', repair] });
  await git({ root: value.checkoutRoot, args: ['reset', '--hard', value.source] });
  await git({ root: join(value.checkoutRoot, upstreamPath), args: ['checkout', '--detach', value.previous] });
  const remote = join(temporary.root, 'remote.git');
  await runProcess({ command: 'git', args: ['clone', '--bare', '-q', value.checkoutRoot, remote], cwd: temporary.root });
  const env = await serviceEnvironment({ root: temporary.root, value: scenario() });
  env['GITHUB_WORKSPACE'] = value.checkoutRoot;
  env['BICORE_FIXTURE_REPOSITORY'] = remote;
  env['BICORE_FIXTURE_ADVANCE'] = repair;
  env['BICORE_FIXTURE_REAL_GIT'] = (await runProcess({ command: 'which', args: ['git'], cwd: checkoutRoot })).stdout.toString().trim();
  env['BRANCH'] = value.plan.branch;
  env['GH_TOKEN'] = 'fixture-token';
  await mkdir(join(temporary.root, 'update'));
  const { steps } = await workflow({ name: 'update-llama-cpp.yml', job: 'update' });
  const inspection = steps.find(step => step.name === 'Inspect the existing candidate branch (online)');
  assert.ok(inspection?.run);
  await runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', inspection.run], cwd: value.checkoutRoot, env: { kind: 'inherit', removePrefixes: [], overrides: env } });
  const output = await readFile(join(temporary.root, 'outputs'), 'utf8');
  assert.match(output, /exists=true/u);
  assert.match(output, new RegExp(`commit=${repair}`, 'u'));
  const candidate = await prepareCandidate({
    checkoutRoot: value.checkoutRoot,
    plan: value.plan,
    existingCommit: repair,
    preflight: async () => {
      throw new Error('A fetched repair must not repeat preparation');
    },
  });
  assert.equal(candidate.sourceCommit, repair);
  assert.equal(candidate.status, 'branch_exists');
  assert.equal(await git({ root: value.checkoutRoot, args: ['rev-parse', 'HEAD'] }), value.source);
  await runProcess({ command: 'bash', args: ['-euo', 'pipefail', '-c', inspection.run], cwd: value.checkoutRoot, env: { kind: 'inherit', removePrefixes: [], overrides: { ...env, BICORE_FIXTURE_ADVANCE: value.source } } });
  const changedOutput = await readFile(join(temporary.root, 'outputs'), 'utf8');
  assert.match(changedOutput, new RegExp(`commit=${value.source}`, 'u'));
  await assert.rejects(prepareCandidate({
    checkoutRoot: value.checkoutRoot,
    plan: value.plan,
    existingCommit: value.source,
    preflight: async () => {
      throw new Error('A conflicting fetched pin must not reach preparation');
    },
  }), /refusing to overwrite/u);
});
