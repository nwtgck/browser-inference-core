import test from 'node:test';
import assert from 'node:assert/strict';
import { lstat, readdir, readFile, writeFile, rm, cp } from 'node:fs/promises';
import { join } from 'node:path';
import { decodeFrozenPublication, type SourceFacts } from '#dto/shared';
import { decodeConsumerReportEnvelope } from '#dto/llama-cpp';
import { bytesIdentity } from '#tools/package/files';
import { packageFixture, fixtureSource, fixtureTreeFingerprint, sourceRoot } from '#tests/tools/package/fixtures.test-support';
import { assembleAndFreezePackage, freezePackage, confirmPublication, checkFrozen, revalidatePublication } from '#tools/publication/artifacts';
import type { MeasurePublication } from '#tools/publication/metrics';
import { artifactIntegrityPath, readArtifactIntegrity } from '#tools/package/integrity';
import { validatePackage } from '#tools/package/aggregate';
import { readConfirmedGuide, writeConfirmedGuide, writeConfirmedReport } from '#tools/publication/consumer';
import { extractNpmResolution, prepareNpmResolution } from '#tools/publication/npm-lock';
import { localGit, verifyGitTree } from '#tools/publication/git';
import { runProcess } from '#tools/shared/process';
import { readConsumerReport } from '#tools/automation/comments/archive';
import { createCli } from '#tools/cli';

function fixtureSourceFacts({ sourceCommit }: { sourceCommit: string }): SourceFacts {
  return { sourceCommit, treeFingerprint: fixtureTreeFingerprint, runtimes: { 'llama-cpp': { upstream: 'b'.repeat(40), preparations: {}, verifiedVariants: [] }, 'stable-diffusion-cpp': { upstreams: { 'stable-diffusion': { path: 'fixture', repository: 'fixture', commit: 'b'.repeat(40) }, ggml: { path: 'fixture', repository: 'fixture', sourceSubdirectory: 'ggml', commit: 'c'.repeat(40) } }, preparation: { files: 0, sha256: '0'.repeat(64) }, patches: [], verifiedVariants: [] }, 'speech-cpp': { upstreams: { speech: { path: 'fixture', repository: 'fixture', commit: 'd'.repeat(40) }, ggml: { path: 'fixture', repository: 'fixture', commit: 'e'.repeat(40) } }, preparation: { files: 0, sha256: '0'.repeat(64) }, patches: [], verifiedVariants: [] } } };
}

test('freeze validates source once and reuses the same report after immutable Git confirmation', async () => {
  await using fixture = await packageFixture();
  let checks = 0;
  const work = join(fixture.root, 'publication');
  const receipt = join(fixture.root, 'frozen.json');
  const frozen = await freezePackage({
    sourceRoot,
    packageDirectory: fixture.directory,
    work,
    receipt,
    branch: 'artifacts',
    expectedSource: fixtureSource,
    repository: 'fixture/browser-inference-core',
    verifySource: async ({ sourceCommit }) => {
      checks++;
      return fixtureSourceFacts({ sourceCommit });
    },
  });
  assert.equal(checks, 1);
  const commit = (await localGit({ cwd: work, args: ['commit-tree', frozen.tree], input: 'Synthetic packaging test\n' })).toString('utf8').trim();
  const publication = await confirmPublication({ work, receipt, commit });
  const integrity = await readArtifactIntegrity({ directory: work, manifest: await readFile(join(work, 'manifest.json')), sourceCommit: fixtureSource });
  assert.deepEqual(integrity?.identity, frozen.integrity);
  assert.deepEqual(integrity?.document.source.tree_fingerprint, fixtureTreeFingerprint);
  assert.equal(integrity?.document.source.repository, 'https://github.com/fixture/browser-inference-core');
  // Reports read the confirmed Git-bound private receipt, with no payload scan.
  await rm(join(work, 'runtimes'), { recursive: true });
  const repository = 'fixture/browser-inference-core';
  const run = { id: 123, run_attempt: 2, head_sha: fixtureSource, head_branch: 'feature/runtime', head_repository: { full_name: repository }, event: 'push', status: 'completed', conclusion: 'failure', html_url: `https://github.com/${repository}/actions/runs/123` };
  const guide = join(fixture.root, 'guide');
  const identity = { id: run.id, attempt: run.run_attempt };
  await writeConfirmedGuide({ publication, repository, output: guide, run: identity });
  assert.deepEqual((await readdir(guide)).sort(), ['consumer-update.md', 'frozen.json', 'report.json']);
  const guideArchive = join(fixture.root, 'publication-guide-2.zip');
  await runProcess({ command: 'zip', args: ['-q', guideArchive, 'report.json', 'consumer-update.md', 'frozen.json'], cwd: guide });
  const guideReader = { path: guideArchive, repository, run };
  const guideMarkdown = await readConsumerReport({ ...guideReader, kind: 'guide' });
  assert.deepEqual(guideMarkdown.split('\n').filter(line => line.startsWith('npm install ')), [
    `npm install github:${repository}#${commit}`,
    `npm install llama-cpp-browser-core@github:${repository}#${commit}`,
    `npm install stable-diffusion-cpp-browser-core@github:${repository}#${commit}`,
    `npm install speech-cpp-browser-core@github:${repository}#${commit}`,
  ]);
  assert.ok(guideMarkdown.includes(fixtureSource));
  await assert.rejects(readConsumerReport(guideReader), /archive member/u);
  await assert.rejects(readConsumerReport({ ...guideReader, kind: 'guide', attempt: 1 }), /identity/u);
  const rerun = { ...run, run_attempt: 3, status: 'in_progress', conclusion: null };
  assert.equal(await readConsumerReport({ ...guideReader, run: rerun, attempt: 2, kind: 'guide' }), guideMarkdown);
  await assert.rejects(readConsumerReport({ ...guideReader, run: rerun, attempt: 3, kind: 'guide' }), /eligible/u);
  await assert.rejects(readConsumerReport({ ...guideReader, run: { ...run, actor: { login: 'dependabot[bot]' } }, kind: 'guide' }), /eligible/u);
  const handoff = { directory: guide, repository, commit, sourceCommit: fixtureSource, run: identity };
  assert.deepEqual(await readConfirmedGuide(handoff), publication);
  for (const mismatch of [{ repository: 'fixture/other' }, { commit: 'd'.repeat(40) }, { sourceCommit: 'e'.repeat(40) }, { run: { ...identity, id: 124 } }, { run: { ...identity, attempt: 3 } }]) {
    await assert.rejects(readConfirmedGuide({ ...handoff, ...mismatch }), /guide identity/u);
  }
  const markdownPath = join(guide, 'consumer-update.md');
  await writeFile(markdownPath, guideMarkdown + 'different contents');
  await assert.rejects(readConfirmedGuide(handoff), /Markdown digest/u);
  await writeFile(markdownPath, guideMarkdown);
  const frozenPath = join(guide, 'frozen.json');
  const originalReceipt = await readFile(frozenPath);
  await writeFile(frozenPath, JSON.stringify({ ...frozen, tree: 'f'.repeat(40) }));
  await assert.rejects(readConfirmedGuide(handoff), /frozen receipt digest/u);

  const guideEnvelopePath = join(guide, 'report.json');
  const originalEnvelope = await readFile(guideEnvelopePath);
  const guideEnvelope = decodeConsumerReportEnvelope({ input: JSON.parse(originalEnvelope.toString('utf8')) });
  const inconsistent = structuredClone(frozen);
  inconsistent.report.data.package.version = '9.0.0';
  const inconsistentReceipt = Buffer.from(JSON.stringify(inconsistent));
  await writeFile(frozenPath, inconsistentReceipt);
  await writeFile(guideEnvelopePath, JSON.stringify({ ...guideEnvelope, frozenSha256: bytesIdentity({ data: inconsistentReceipt }).sha256 }));
  await assert.rejects(readConfirmedGuide(handoff), /report data changed/u);
  await writeFile(frozenPath, originalReceipt);
  await writeFile(guideEnvelopePath, originalEnvelope);

  const version = frozen.report.data.package.version;
  const lock = { lockfileVersion: 3, packages: { '': { dependencies: { 'browser-inference-core': `github:${repository}#${commit}` } }, 'node_modules/browser-inference-core': { version, resolved: `git+https://github.com/${repository}.git#${commit}` } } };
  const resolution = extractNpmResolution({ repository, commit, version, lock });
  const npmFixture = join(fixture.root, 'npm-fixture');
  await prepareNpmResolution({ directory: npmFixture, repository, commit, version });
  await writeFile(join(npmFixture, 'package-lock.json'), JSON.stringify(lock));
  const summary = join(fixture.root, 'job summary.md');
  const initialGuide = `\
## Runtime artifact commit

\`\`\`sh
npm install github:fixture/browser-inference-core#${commit}
\`\`\`

\`\`\`sh
npm install llama-cpp-browser-core@github:fixture/browser-inference-core#${commit}
\`\`\`

\`\`\`sh
npm install stable-diffusion-cpp-browser-core@github:fixture/browser-inference-core#${commit}
\`\`\`

\`\`\`sh
npm install speech-cpp-browser-core@github:fixture/browser-inference-core#${commit}
\`\`\`
`;
  await writeFile(summary, initialGuide);
  const result = await writeConfirmedReport({ publication, repository: 'fixture/browser-inference-core', resolution, output: join(fixture.root, 'report'), run: { id: 123, attempt: 2 }, summary });
  assert(result.markdown.includes(commit));
  assert.ok(result.markdown.startsWith(guideMarkdown));
  const reportFromGuide = join(fixture.root, 'report-from-guide');
  const base = ['publication', 'report', '--guide', guide, '--commit', commit, '--repository', repository, '--npm-fixture', npmFixture, '--output', reportFromGuide];
  const sourceArgs = ['--source-commit', fixtureSource];
  const runArgs = ['--run-id', String(identity.id), '--run-attempt', String(identity.attempt)];
  async function runReport({ args }: { args: readonly string[] }): Promise<unknown> {
    let emitted: unknown;
    const cli = createCli({
      modules: undefined,
      writeJson: async ({ value }) => {
        emitted = value;
      },
    });
    await cli.parseAsync([...args], {}, error => {
      if (error) {
        throw error;
      }
    });
    return emitted;
  }
  await assert.rejects(runReport({ args: [...base, ...runArgs] }), /Missing required arguments for --guide/u);
  await assert.rejects(runReport({ args: [...base, ...sourceArgs] }), /Missing required arguments for --guide/u);
  await assert.rejects(runReport({ args: [...base, ...sourceArgs, ...runArgs, '--work-root', work] }), /mutually exclusive/u);
  assert.deepEqual(await runReport({ args: [...base, ...sourceArgs, ...runArgs] }), { output: reportFromGuide, artifactCommit: commit, sourceCommit: fixtureSource });
  assert.deepEqual(await readFile(join(reportFromGuide, 'report.json')), await readFile(join(fixture.root, 'report/report.json')));
  assert.equal(await readFile(join(reportFromGuide, 'consumer-update.md'), 'utf8'), result.markdown);
  const displayed = await readFile(summary, 'utf8');
  assert.ok(displayed.startsWith(initialGuide));
  const details = displayed.slice(initialGuide.length);
  assert.ok(details.includes('## Consumer integration details'));
  assert.ok(details.includes('| llama-cpp |'));
  assert.ok(details.includes('| stable-diffusion-cpp |'));
  assert.ok(!details.includes('npm install'));
  const markdown = await readFile(join(fixture.root, 'report/consumer-update.md'));
  const envelope = decodeConsumerReportEnvelope({ input: JSON.parse(await readFile(join(fixture.root, 'report/report.json'), 'utf8')) });
  assert.deepEqual(envelope, { schemaVersion: 1, repository: 'fixture/browser-inference-core', sourceCommit: fixtureSource, artifactCommit: commit, runId: 123, runAttempt: 2, markdownSha256: bytesIdentity({ data: markdown }).sha256 });
  assert.deepEqual(JSON.parse(await readFile(join(fixture.root, 'report/consumer-update.yaml'), 'utf8')), result.metadata);
  const archive = join(fixture.root, 'consumer-update-2.zip');
  await runProcess({ command: 'zip', args: ['-q', archive, 'report.json', 'consumer-update.md', 'consumer-update.yaml', 'metadata.json', 'report.md', 'package-lock-entry.json'], cwd: join(fixture.root, 'report') });
  const decoded = await readConsumerReport({ path: archive, repository, run });
  assert.equal(decoded, result.markdown);
  await writeConfirmedReport({ publication, repository: 'fixture/browser-inference-core', resolution, output: join(fixture.root, 'report'), run: undefined, summary: undefined });
  await assert.rejects(readFile(join(fixture.root, 'report/report.json')), { code: 'ENOENT' });
  assert.equal(await readFile(join(fixture.root, 'report/report.md'), 'utf8'), result.markdown);
  assert.equal(checks, 1);
  const tampered = structuredClone(frozen);
  tampered.report.data.sourceCommit = 'c'.repeat(40);
  assert.throws(() => checkFrozen({ frozen: tampered }), /source mismatch/u);
  const changed = structuredClone(frozen);
  changed.report.data.package.version = '9.0.0';
  assert.throws(() => checkFrozen({ frozen: changed }), /data changed/u);
  const wrongCommit = (await localGit({ cwd: work, args: ['commit-tree', frozen.tree], input: 'Another commit, same validated bytes\n' })).toString('utf8').trim();
  const wrongResolution = { ...resolution, specifier: `github:fixture/browser-inference-core#${wrongCommit}` };
  if (wrongCommit !== commit) {
    await assert.rejects(writeConfirmedReport({ publication, repository: 'fixture/browser-inference-core', resolution: wrongResolution, output: join(fixture.root, 'bad-report'), run: undefined, summary }), /npm resolution/u);
    assert.equal(await readFile(summary, 'utf8'), displayed);
  }
});

test('Git publication rejects different staged blob bytes despite a valid source package', async () => {
  await using fixture = await packageFixture();
  const work = join(fixture.root, 'publication');
  const receipt = join(fixture.root, 'frozen.json');
  const frozen = await freezePackage({ sourceRoot, packageDirectory: fixture.directory, work, receipt, branch: 'artifacts/test', expectedSource: fixtureSource, repository: 'fixture/browser-inference-core', verifySource: async ({ sourceCommit }) => fixtureSourceFacts({ sourceCommit }) });
  await writeFile(join(work, 'README.md'), Buffer.alloc((await readFile(join(work, 'README.md'))).length, 120));
  await localGit({ cwd: work, args: ['add', '--all'] });
  const changedTree = (await localGit({ cwd: work, args: ['write-tree'] })).toString('utf8').trim();
  const files = new Map(fixture.validated.files);
  files.set(artifactIntegrityPath, { path: artifactIntegrityPath, ...frozen.integrity });
  await assert.rejects(verifyGitTree({ work, tree: changedTree, files }), /Git blob differs/u);
  const commit = (await localGit({ cwd: work, args: ['commit-tree', changedTree], input: 'Wrong publication tree\n' })).toString('utf8').trim();
  await assert.rejects(confirmPublication({ work, receipt, commit }), /frozen Git tree/u);
  const raw = decodeFrozenPublication({ input: JSON.parse(await readFile(receipt, 'utf8')) });
  raw.runtimeManifests['llama-cpp'].sha256 = '0'.repeat(64);
  await writeFile(receipt, JSON.stringify(raw));
  const originalCommit = (await localGit({ cwd: work, args: ['commit-tree', frozen.tree], input: 'Original publication tree\n' })).toString('utf8').trim();
  await assert.rejects(confirmPublication({ work, receipt, commit: originalCommit }), /runtime manifest/u);
});

test('npm lock validation rejects mutable, wrong-repository and install-script resolutions', () => {
  const commit = 'b'.repeat(40);
  const repository = 'fixture/browser-inference-core';
  const entry = { version: '0.1.0', resolved: `git+ssh://git@github.com/${repository}.git#${commit}` };
  const lock = { lockfileVersion: 3, packages: { '': { dependencies: { 'browser-inference-core': `github:${repository}#${commit}` } }, 'node_modules/browser-inference-core': entry } };
  assert.equal(extractNpmResolution({ lock, repository, commit, version: '0.1.0' }).packageEntry.resolved, entry.resolved);
  for (const extra of [{ resolved: entry.resolved.replace(repository, 'other/repository') }, { resolved: entry.resolved.replace(commit, 'main') }, { hasInstallScript: true }, { dependencies: { extra: '1' } }]) {
    assert.throws(() => extractNpmResolution({ lock: { ...lock, packages: { ...lock.packages, 'node_modules/browser-inference-core': { ...entry, ...extra } } }, repository, commit, version: '0.1.0' }));
  }
});

test('source failures and mutations during freeze never produce a publication receipt', async () => {
  await using fixture = await packageFixture();
  let checks = 0;
  const failedWork = join(fixture.root, 'source-failure');
  const failedReceipt = join(fixture.root, 'source-failure.json');
  await assert.rejects(freezePackage({
    sourceRoot,
    packageDirectory: fixture.directory,
    work: failedWork,
    receipt: failedReceipt,
    branch: 'artifacts',
    expectedSource: fixtureSource,
    repository: 'fixture/browser-inference-core',
    verifySource: async () => {
      checks++;
      throw new Error('Selected source receipt mismatch');
    },
  }), /source receipt mismatch/u);
  await assert.rejects(readFile(failedReceipt), { code: 'ENOENT' });
  await assert.rejects(lstat(failedWork), { code: 'ENOENT' });
  assert.equal(checks, 1);
  for (const [name, relative] of [['root_manifest', 'manifest.json'], ['runtime_manifest', 'runtimes/llama-cpp-browser-core/manifest.json'], ['payload', 'runtimes/llama-cpp-browser-core/profiles/cpu-wasm32/browser/core.wasm']] as const) {
    const work = join(fixture.root, name);
    const receipt = join(fixture.root, name + '.json');
    await assert.rejects(freezePackage({
      sourceRoot,
      packageDirectory: fixture.directory,
      work,
      receipt,
      branch: 'artifacts',
      expectedSource: fixtureSource,
      repository: 'fixture/browser-inference-core',
      verifySource: async ({ sourceCommit }) => {
        checks++;
        const [staging] = (await readdir(fixture.root)).filter(name => name.startsWith('bicore-output-'));
        assert(staging);
        await assert.rejects(lstat(work), { code: 'ENOENT' });
        const path = join(fixture.root, staging, 'prepared', relative);
        const data = await readFile(path);
        data[0] = (data[0]! + 1) & 255;
        await writeFile(path, data);
        return fixtureSourceFacts({ sourceCommit });
      },
    }), /Git blob|Artifact hash\/size mismatch/u);
    await assert.rejects(readFile(receipt), { code: 'ENOENT' });
    await assert.rejects(lstat(work), { code: 'ENOENT' });
  }
  assert.equal(checks, 4);
});

test('direct publication checks the selected source and binds copied files to the runtime inputs', async () => {
  await using fixture = await packageFixture();
  const inputs = join(fixture.root, 'package-inputs');
  const work = join(fixture.root, 'publication');
  const receipt = join(fixture.root, 'frozen.json');
  const common = { sourceRoot, inputs, work, receipt, branch: 'artifacts', repository: 'fixture/browser-inference-core', verifySource: async () => assert.fail('Source preparation must follow successful validation and copy') };
  await assert.rejects(assembleAndFreezePackage({ ...common, expectedSource: 'f'.repeat(40) }), /differs from selected source/u);

  const metadata = join(inputs, 'llama-cpp/package.json');
  const contents = await readFile(metadata, 'utf8');
  await writeFile(metadata, contents.replace('"0.1.0"', '"0.2.0"'));
  await assert.rejects(assembleAndFreezePackage({ ...common, expectedSource: fixtureSource }), /Artifact hash\/size mismatch: package\.json/u);
  await writeFile(metadata, contents);

  const input = join(inputs, 'llama-cpp/profiles/cpu-wasm32/browser/core.wasm');
  const original = await readFile(input);
  const changed = Buffer.from(original);
  changed[0] = 1;
  const measure: MeasurePublication = async ({ phase, operation }) => {
    if (phase === 'publication.freeze.copy') {
      await writeFile(input, changed);
    }
    return operation();
  };
  await assert.rejects(assembleAndFreezePackage({ ...common, expectedSource: fixtureSource, measure }), /Copied input changed/u);
  await assert.rejects(lstat(work), { code: 'ENOENT' });
  await assert.rejects(lstat(receipt), { code: 'ENOENT' });
});

test('receipt activation rolls back the snapshot if another writer claims the receipt', async () => {
  await using fixture = await packageFixture();
  const work = join(fixture.root, 'publication');
  const receipt = join(fixture.root, 'frozen.json');
  const common = { sourceRoot, inputs: join(fixture.root, 'package-inputs'), work, receipt, branch: 'artifacts', expectedSource: fixtureSource, repository: 'fixture/browser-inference-core' };
  await assert.rejects(assembleAndFreezePackage({
    ...common,
    verifySource: async ({ sourceCommit }) => {
      await writeFile(receipt, 'concurrent receipt', { flag: 'wx' });
      return fixtureSourceFacts({ sourceCommit });
    },
  }), { code: 'EEXIST' });
  await assert.rejects(lstat(work), { code: 'ENOENT' });
  assert.equal(await readFile(receipt, 'utf8'), 'concurrent receipt');
  assert.equal((await readdir(fixture.root)).some(name => name.startsWith('bicore-output-') || name.startsWith('bicore-receipt-')), false);
});

test('the attested envelope stays in the exact Git/npm closure and cannot be dropped on revalidation', async () => {
  await using fixture = await packageFixture();
  const work = join(fixture.root, 'publication');
  const receipt = join(fixture.root, 'frozen.json');
  const verifySource = async ({ sourceCommit }: { sourceCommit: string }) => fixtureSourceFacts({ sourceCommit });
  const frozen = await freezePackage({ sourceRoot, packageDirectory: fixture.directory, work, receipt, branch: 'artifacts', expectedSource: fixtureSource, repository: 'fixture/browser-inference-core', verifySource });
  const commit = (await localGit({ cwd: work, args: ['commit-tree', frozen.tree], input: 'Attested publication\n' })).toString('utf8').trim();
  // Independently downloaded package files are validated without the publisher's .git directory.
  const downloaded = join(fixture.root, 'downloaded');
  await cp(fixture.directory, downloaded, { recursive: true });
  await cp(join(work, artifactIntegrityPath), join(downloaded, artifactIntegrityPath));
  const validated = await validatePackage({ sourceRoot, directory: downloaded, requireClean: true, checkNpmPack: true });
  assert.deepEqual(validated.files.get(artifactIntegrityPath), { path: artifactIntegrityPath, ...frozen.integrity });
  assert.ok(!validated.manifest.files.some(file => file.path === artifactIntegrityPath));
  assert.deepEqual((await revalidatePublication({ sourceRoot, packageDirectory: downloaded, work, commit, verifySource })).source.treeFingerprint, fixtureTreeFingerprint);
  await assert.rejects(revalidatePublication({ sourceRoot, packageDirectory: downloaded, work, commit, verifySource: async ({ sourceCommit }) => ({ ...fixtureSourceFacts({ sourceCommit }), treeFingerprint: { ...fixtureTreeFingerprint, value: 'c'.repeat(64) } }) }), /tree fingerprint differs/u);
  await assert.rejects(freezePackage({ sourceRoot, packageDirectory: downloaded, work: join(fixture.root, 'refrozen'), receipt: join(fixture.root, 'refrozen.json'), branch: 'artifacts', expectedSource: fixtureSource, repository: 'fixture/browser-inference-core', verifySource }), /previous integrity envelope/u);
  await writeFile(join(downloaded, 'extra.yaml'), 'unexpected metadata');
  await assert.rejects(validatePackage({ sourceRoot, directory: downloaded, requireClean: true, checkNpmPack: false }), /complete package tree/u);
  await rm(join(downloaded, 'extra.yaml'));
  await rm(join(downloaded, artifactIntegrityPath));
  await assert.rejects(revalidatePublication({ sourceRoot, packageDirectory: downloaded, work, commit, verifySource }), /missing its integrity envelope/u);

  // A later worktree change cannot replace the frozen blob or its attestation digest.
  await writeFile(join(work, artifactIntegrityPath), 'changed after signing');
  assert.deepEqual((await confirmPublication({ work, receipt, commit })).frozen.integrity, frozen.integrity);
  await writeFile(receipt, JSON.stringify({ ...frozen, integrity: { ...frozen.integrity, sha256: '0'.repeat(64) } }));
  await assert.rejects(confirmPublication({ work, receipt, commit }), /integrity envelope differs/u);
});
