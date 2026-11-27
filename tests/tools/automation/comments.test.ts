import assert from 'node:assert/strict';
import { mkdir, readFile, symlink, unlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import test from 'node:test';
import { archiveRedirect, maximumArchiveBytes, readConsumerReport } from '#tools/automation/comments/archive';
import { commentBody, matchingRuns, marker, payloadStart, planCommentWrite, previousPayload, reportArchives, selectPulls } from '#tools/automation/comments/plan';
import { planPullRequests, planReportDownloads, planReportArchives, prepareCommentWrite, validateReportArchive } from '#tools/automation/comments/operations';
import { runProcess } from '#tools/shared/process';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { botComment, pullRequest, reportArchive, reportPayload, repository, source, spoofedComment, workflowRun } from '#tests/tools/automation/comments.test-support';

const pull = pullRequest({ head: source, number: 7 });
const run = workflowRun({ head: source, id: 123, attempt: 1, status: 'completed', conclusion: 'success' });
const payload = reportPayload({ head: source });
const publication = { run, attempt: 1, kind: 'details', markdown: payload } as const;

test('pull selection covers late PRs while excluding forks and artifact branches', () => {
  const other = pullRequest({ head: source, number: 8 });
  const fork = { ...pull, head: { ...pull.head, repo: { full_name: 'attacker/core' } } };
  const artifact = { ...pull, head: { ...pull.head, ref: 'artifacts/branch' } };
  assert.deepEqual(selectPulls({ pulls: [pull, other, fork, artifact], repository }), [pull, other]);
  assert.throws(() => selectPulls({ pulls: [pull, pull], repository }), /Duplicate/u);
  assert.throws(() => selectPulls({ pulls: [{ ...pull, number: 1.5 }], repository }), /identifier/u);
});

test('latest attempts use start time, source head, branch, event and repository rather than completion order', () => {
  const recent = { ...workflowRun({ head: source, id: 100, attempt: 3, status: 'completed', conclusion: 'success' }), run_started_at: '2026-09-22T12:00:00Z' };
  const older = Array.from({ length: 100 }, (_, index) => ({ ...workflowRun({ head: source, id: 200 + index, attempt: 1, status: 'completed', conclusion: 'success' }), run_started_at: '2026-09-22T10:00:00Z' }));
  assert.equal(matchingRuns({ runs: [...older, recent], repository, pull })[0], recent);
  for (const irrelevant of [{ ...run, head_sha: 'b'.repeat(40) }, { ...run, head_branch: 'another' }, { ...run, event: 'pull_request_target' }, { ...run, head_repository: { full_name: 'fork/core' } }]) {
    assert.deepEqual(matchingRuns({ runs: [irrelevant], repository, pull }), []);
  }
  const prRun = { ...workflowRun({ head: source, id: 124, attempt: 1, status: 'completed', conclusion: 'success' }), event: 'pull_request' };
  const rerun = { ...run, run_attempt: 2, status: 'in_progress', conclusion: null, run_started_at: '2026-09-22T12:00:00Z' };
  assert.deepEqual(matchingRuns({ runs: [run, prRun], repository, pull }), [prRun, run]);
  assert.deepEqual(matchingRuns({ runs: [run, prRun, rerun], repository, pull }), [rerun, prRun]);
  assert.throws(() => matchingRuns({ runs: [run, run], repository, pull }), /Duplicate/u);
  assert.throws(() => matchingRuns({ runs: Array.from({ length: 1001 }, () => run), repository, pull }), /oversized/u);
  assert.throws(() => matchingRuns({ runs: [{ ...run, html_url: 'https://attacker.invalid' }], repository, pull }), /URL/u);
});

test('each completed attempt considers unique, unexpired details first and then its guide', () => {
  const details = { id: 91, name: 'consumer-update-1', expired: false };
  const artifacts = [
    details,
    { id: 92, name: 'publication-guide-1', expired: false },
    { id: 93, name: 'publication-guide-2', expired: false },
    { id: 94, name: 'consumer-update-2', expired: true },
    { id: 95, name: 'consumer-update-0', expired: false },
  ];
  const first = [
    { artifactId: 91, attempt: 1, kind: 'details', createdAt: undefined },
    { artifactId: 92, attempt: 1, kind: 'guide', createdAt: undefined },
  ];
  assert.deepEqual(reportArchives({ artifacts, run }), first);
  assert.deepEqual(reportArchives({ artifacts, run: { ...run, run_attempt: 2, status: 'in_progress', conclusion: null } }), first);
  assert.deepEqual(reportArchives({ artifacts, run: { ...run, run_attempt: 2, conclusion: 'failure' } }), [{ artifactId: 93, attempt: 2, kind: 'guide', createdAt: undefined }, ...first]);
  assert.deepEqual(reportArchives({ artifacts: [...artifacts, { ...details, id: 96 }], run }), first.slice(1));
  assert.deepEqual(reportArchives({ artifacts, run: { ...run, actor: { login: 'dependabot[bot]' } } }), []);
});

test('comment plans update only an owned bot comment and suppress identical writes', () => {
  const base = { original: pull, current: pull, selectedRuns: [run], currentRuns: [run], repository, publication, error: undefined };
  const create = planCommentWrite({ ...base, comments: [spoofedComment] });
  assert.equal(create.kind, 'create');
  const old = botComment({ body: marker + '\nOld' });
  const update = planCommentWrite({ ...base, comments: [old] });
  assert.equal(update.kind, 'update');
  const body = commentBody({ pull, run, publication, previous: '', error: undefined });
  assert.equal(planCommentWrite({ ...base, comments: [botComment({ body })] }).kind, 'skip');
  for (const current of [{ ...pull, head: { ...pull.head, sha: 'b'.repeat(40) } }, { ...pull, head: { ...pull.head, ref: 'another' } }, { ...pull, head: { ...pull.head, repo: null } }, { ...pull, state: 'closed' }]) {
    assert.equal(planCommentWrite({ ...base, current, comments: [old] }).kind, 'skip');
  }
  assert.equal(planCommentWrite({ ...base, currentRuns: [{ ...run, run_attempt: 2, status: 'queued', conclusion: null }], comments: [old] }).kind, 'skip');
  const older = workflowRun({ head: source, id: 121, attempt: 1, status: 'completed', conclusion: 'success' });
  assert.equal(planCommentWrite({ ...base, selectedRuns: [run, older], currentRuns: [run, { ...older, run_attempt: 2 }], comments: [old] }).kind, 'skip');
});

test('pending, missing and failed reports preserve one clearly previous publication section', () => {
  const old = commentBody({ pull, run, publication, previous: '', error: undefined });
  const current = pullRequest({ head: 'b'.repeat(40), number: 7 });
  const pending = commentBody({ pull: current, run: { ...run, head_sha: current.head.sha, status: 'queued', conclusion: null }, publication: undefined, previous: old, error: undefined });
  const again = commentBody({ pull: current, run: { ...run, head_sha: current.head.sha, status: 'in_progress', conclusion: null }, publication: undefined, previous: pending, error: undefined });
  assert.equal(again.split(payloadStart).length - 1, 1);
  assert.equal(again.split('Previous publication (historical reference)').length - 1, 1);
  assert.doesNotMatch(again, /Runtime publication confirmed for this PR head/u);
  assert.equal(previousPayload({ body: again }), payload.trim());
  assert.equal(previousPayload({ body: `${old}${payloadStart}` }), undefined);
  assert.match(commentBody({ pull, run: undefined, publication: undefined, previous: '', error: undefined }), /Human PR creation/u);
  assert.match(commentBody({ pull, run, publication: undefined, previous: old, error: 'invalid report' }), /unavailable or expired/u);
  assert.match(commentBody({ pull, run: { ...run, conclusion: 'failure' }, publication: undefined, previous: old, error: undefined }), /\*\*failure\*\*/u);
});

test('standard unzip reads bounded JSON and Markdown while leaving large download-only YAML compressed', async () => {
  await using temporary = await temporaryDirectory();
  const archive = await reportArchive({ root: temporary.root, run, changes: {}, body: undefined, envelope: undefined, yaml: 'x'.repeat(1000000) });
  assert.equal(await readConsumerReport({ path: archive.path, repository, run }), payload);
  // Unrelated files are not extracted or executed; only named report members are read.
  await writeFile(join(archive.directory, 'untrusted.sh'), 'exit 99\n');
  await runProcess({ command: 'zip', args: ['-q', archive.path, 'untrusted.sh'], cwd: archive.directory });
  assert.equal(await readConsumerReport({ path: archive.path, repository, run }), payload);
});

test('archive identities, declared field types, digest and delimiter injection are checked at receipt', async () => {
  await using temporary = await temporaryDirectory();
  const invalids: Record<string, unknown>[] = [{ repository: 'fork/core' }, { sourceCommit: 'b'.repeat(40) }, { runId: 124 }, { runAttempt: 2 }, { schemaVersion: 2 }, { artifactCommit: 'moving-branch' }, { markdownSha256: '0'.repeat(64) }, { runId: '123' }, { runAttempt: true }, { sourceCommit: [] }];
  for (const changes of invalids) {
    const archive = await reportArchive({ root: temporary.root, run, changes, body: undefined, envelope: undefined, yaml: 'fixture' });
    await assert.rejects(readConsumerReport({ path: archive.path, repository, run }));
  }
  for (const input of [null, [], 'invalid', 1]) {
    const archive = await reportArchive({ root: temporary.root, run, changes: {}, body: undefined, envelope: { input }, yaml: '' });
    await assert.rejects(readConsumerReport({ path: archive.path, repository, run }));
  }
  for (const body of [payload + payloadStart, 'x'.repeat(1000000), payload + 'x'.repeat(56000)]) {
    const archive = await reportArchive({ root: temporary.root, run, changes: {}, body, envelope: undefined, yaml: '' });
    await assert.rejects(readConsumerReport({ path: archive.path, repository, run }));
  }
});

test('bad ZIPs, duplicate report members, symlinks, CRC errors and oversized archives fail without extraction', async () => {
  await using temporary = await temporaryDirectory();
  const archive = await reportArchive({ root: temporary.root, run, changes: {}, body: undefined, envelope: undefined, yaml: '' });
  await writeFile(join(archive.directory, 'consumer-xxxxxx.md'), payload);
  await runProcess({ command: 'zip', args: ['-q', archive.path, 'consumer-xxxxxx.md'], cwd: archive.directory });
  const duplicate = Buffer.from((await readFile(archive.path)).toString('latin1').replaceAll('consumer-xxxxxx.md', 'consumer-update.md'), 'latin1');
  await writeFile(archive.path, duplicate);
  await assert.rejects(readConsumerReport({ path: archive.path, repository, run }), /duplicate/u);
  const linked = await reportArchive({ root: temporary.root, run, changes: {}, body: undefined, envelope: undefined, yaml: '' });
  await unlink(join(linked.directory, 'consumer-update.md'));
  await symlink('report.json', join(linked.directory, 'consumer-update.md'));
  await runProcess({ command: 'zip', args: ['-q', '-y', linked.path, 'consumer-update.md'], cwd: linked.directory });
  await assert.rejects(readConsumerReport({ path: linked.path, repository, run }), /Unsafe/u);
  const malformed = join(temporary.root, 'malformed.zip');
  await writeFile(malformed, 'not a zip');
  await assert.rejects(readConsumerReport({ path: malformed, repository, run }));
  await writeFile(malformed, Buffer.alloc(maximumArchiveBytes + 1));
  await assert.rejects(readConsumerReport({ path: malformed, repository, run }), /size limit/u);
  const corrupt = await reportArchive({ root: temporary.root, run, changes: {}, body: undefined, envelope: undefined, yaml: '' });
  // A stored member's bytes are covered by unzip's real CRC verification.
  await runProcess({ command: 'zip', args: ['-q', '-0', corrupt.path, 'consumer-update.md'], cwd: corrupt.directory });
  const raw = await readFile(corrupt.path);
  const offset = raw.indexOf(Buffer.from(payload));
  assert.ok(offset > 0);
  raw[offset] = raw[offset]! ^ 1;
  await writeFile(corrupt.path, raw);
  await assert.rejects(readConsumerReport({ path: corrupt.path, repository, run }));
});

test('artifact redirect policy allows only a single credential-free HTTPS location', () => {
  assert.equal(archiveRedirect({ status: '302', headers: 'HTTP/2 302\r\nLocation: https://storage.invalid/archive?sig=abc\r\n' }), 'https://storage.invalid/archive?sig=abc');
  for (const location of ['http://storage.invalid/archive', 'https://user:password@storage.invalid/x', 'https://storage.invalid:444/x', 'https://storage.invalid/a\tb']) {
    assert.throws(() => archiveRedirect({ status: '302', headers: `Location: ${location}\r\n` }));
  }
  assert.throws(() => archiveRedirect({ status: '302', headers: 'Location: https://a.invalid\nLocation: https://b.invalid\n' }));
  assert.throws(() => archiveRedirect({ status: '404', headers: '' }));
});

test('real file operations isolate malformed reports per PR and recheck current trust before writing', async () => {
  await using temporary = await temporaryDirectory();
  const state = join(temporary.root, 'state');
  await mkdir(state);
  const second = pullRequest({ head: source, number: 8 });
  await writeFile(join(state, 'pulls.json'), JSON.stringify([pull, second]));
  await planPullRequests({ state, repository });
  for (const number of [7, 8]) {
    const directory = join(state, String(number));
    await writeFile(join(directory, 'runs.json'), JSON.stringify([run]));
    await writeFile(join(directory, 'current-runs.json'), JSON.stringify([run]));
    await writeFile(join(directory, 'current-pr.json'), JSON.stringify(pullRequest({ head: source, number })));
    await writeFile(join(directory, 'current-comments.json'), JSON.stringify([botComment({ body: commentBody({ pull, run, publication, previous: '', error: undefined }) })]));
  }
  await planReportDownloads({ state, repository });
  assert.equal(await readFile(join(state, 'downloads.tsv'), 'utf8'), '7\t123\n8\t123\n');
  for (const number of [7, 8]) {
    await writeFile(join(state, String(number), '123/artifacts.json'), JSON.stringify([{ id: 91, name: 'consumer-update-1', expired: false }]));
  }
  await planReportArchives({ state, repository });
  assert.equal(await readFile(join(state, 'archives.tsv'), 'utf8'), '7\t123\t91\n8\t123\t91\n');
  for (const number of [7, 8]) {
    await mkdir(join(state, String(number), '123/91'));
  }
  await writeFile(join(state, '7/123/91/report.zip'), 'malformed');
  const archive = await reportArchive({ root: temporary.root, run, changes: {}, body: undefined, envelope: undefined, yaml: '' });
  await writeFile(join(state, '8/123/91/report.zip'), await readFile(archive.path));
  for (const number of [7, 8]) {
    await validateReportArchive({ state, repository, number, runId: 123, artifactId: 91 });
  }
  await prepareCommentWrite({ state, repository, number: 7 });
  assert.match(await readFile(join(state, '7/body.json'), 'utf8'), /Previous publication \(historical reference\)/u);
  assert.match(await readFile(join(state, '7/body.json'), 'utf8'), /Report retrieval error/u);
  await prepareCommentWrite({ state, repository, number: 8 });
  assert.equal(await readFile(join(state, '8/write.txt'), 'utf8'), 'skip\n');
  await writeFile(join(state, '8/current-pr.json'), JSON.stringify({ ...second, head: { ...second.head, sha: 'b'.repeat(40) } }));
  await prepareCommentWrite({ state, repository, number: 8 });
  assert.equal(await readFile(join(state, '8/write.txt'), 'utf8'), 'skip\n');
});
