import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { decodeFrozenPublication, type SourceFacts } from '#dto/shared';
import { decodeConsumerReportEnvelope } from '#dto/llama-cpp';
import { bytesIdentity } from '#tools/package/files';
import { packageFixture, fixtureSource, sourceRoot } from '#tests/tools/package/fixtures.test-support';
import { freezePackage, confirmPublication, checkFrozen } from '#tools/publication/artifacts';
import { writeConfirmedReport } from '#tools/publication/consumer';
import { extractNpmResolution } from '#tools/publication/npm-lock';
import { localGit, verifyGitTree } from '#tools/publication/git';
import { runProcess } from '#tools/shared/process';
import { readConsumerReport } from '#tools/automation/comments/archive';

function fixtureSourceFacts({ sourceCommit }: { sourceCommit: string }): SourceFacts {
  return { sourceCommit, runtimes: { 'llama-cpp': { upstream: 'b'.repeat(40), preparations: {}, verifiedVariants: [] }, 'stable-diffusion-cpp': { upstreams: { 'stable-diffusion': { path: 'fixture', repository: 'fixture', commit: 'b'.repeat(40) }, ggml: { path: 'fixture', repository: 'fixture', sourceSubdirectory: 'ggml', commit: 'c'.repeat(40) } }, preparation: { files: 0, sha256: '0'.repeat(64) }, patches: [], verifiedVariants: [] } } };
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
    verifySource: async ({ sourceCommit }) => {
      checks++;
      return fixtureSourceFacts({ sourceCommit });
    },
  });
  assert.equal(checks, 1);
  const commit = (await localGit({ cwd: work, args: ['commit-tree', frozen.tree], input: 'Synthetic packaging test\n' })).toString('utf8').trim();
  const publication = await confirmPublication({ work, receipt, commit });
  // Reports read the confirmed Git-bound private receipt, with no payload scan.
  await rm(join(work, 'runtimes'), { recursive: true });
  const resolution = extractNpmResolution({ repository: 'fixture/browser-inference-core', commit, version: '0.1.0', lock: { lockfileVersion: 3, packages: { '': { dependencies: { 'browser-inference-core': `github:fixture/browser-inference-core#${commit}` } }, 'node_modules/browser-inference-core': { version: '0.1.0', resolved: `git+https://github.com/fixture/browser-inference-core.git#${commit}` } } } });
  const result = await writeConfirmedReport({ publication, repository: 'fixture/browser-inference-core', resolution, output: join(fixture.root, 'report'), run: { id: 123, attempt: 2 } });
  assert(result.markdown.includes(commit));
  assert.ok(result.markdown.includes(`npm install github:fixture/browser-inference-core#${commit}`));
  const markdown = await readFile(join(fixture.root, 'report/consumer-update.md'));
  const envelope = decodeConsumerReportEnvelope({ input: JSON.parse(await readFile(join(fixture.root, 'report/report.json'), 'utf8')) });
  assert.deepEqual(envelope, { schemaVersion: 1, repository: 'fixture/browser-inference-core', sourceCommit: fixtureSource, artifactCommit: commit, runId: 123, runAttempt: 2, markdownSha256: bytesIdentity({ data: markdown }).sha256 });
  assert.deepEqual(JSON.parse(await readFile(join(fixture.root, 'report/consumer-update.yaml'), 'utf8')), result.metadata);
  const archive = join(fixture.root, 'consumer-update-2.zip');
  await runProcess({ command: 'zip', args: ['-q', archive, 'report.json', 'consumer-update.md', 'consumer-update.yaml', 'metadata.json', 'report.md', 'package-lock-entry.json'], cwd: join(fixture.root, 'report') });
  const decoded = await readConsumerReport({ path: archive, repository: 'fixture/browser-inference-core', run: { id: 123, run_attempt: 2, head_sha: fixtureSource, head_branch: 'feature/runtime', head_repository: { full_name: 'fixture/browser-inference-core' }, event: 'push', status: 'completed', conclusion: 'success', html_url: 'https://github.com/fixture/browser-inference-core/actions/runs/123' } });
  assert.equal(decoded, result.markdown);
  await writeConfirmedReport({ publication, repository: 'fixture/browser-inference-core', resolution, output: join(fixture.root, 'report'), run: undefined });
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
    await assert.rejects(writeConfirmedReport({ publication, repository: 'fixture/browser-inference-core', resolution: wrongResolution, output: join(fixture.root, 'bad-report'), run: undefined }), /npm resolution/u);
  }
});

test('Git publication rejects different staged blob bytes despite a valid source package', async () => {
  await using fixture = await packageFixture();
  const work = join(fixture.root, 'publication');
  const receipt = join(fixture.root, 'frozen.json');
  const frozen = await freezePackage({ sourceRoot, packageDirectory: fixture.directory, work, receipt, branch: 'artifacts/test', expectedSource: fixtureSource, verifySource: async ({ sourceCommit }) => fixtureSourceFacts({ sourceCommit }) });
  await writeFile(join(work, 'README.md'), Buffer.alloc((await readFile(join(work, 'README.md'))).length, 120));
  await localGit({ cwd: work, args: ['add', '--all'] });
  const changedTree = (await localGit({ cwd: work, args: ['write-tree'] })).toString('utf8').trim();
  await assert.rejects(verifyGitTree({ work, tree: changedTree, files: fixture.validated.files }), /Git blob differs/u);
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

test('source failures and mutations after the private npm pack never produce a publication receipt', async () => {
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
    verifySource: async () => {
      checks++;
      throw new Error('Selected source receipt mismatch');
    },
  }), /source receipt mismatch/u);
  await assert.rejects(readFile(failedReceipt), { code: 'ENOENT' });
  await assert.rejects(readFile(join(failedWork, '.git/config')), { code: 'ENOENT' });
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
      verifySource: async ({ sourceCommit }) => {
        checks++;
        const path = join(work, relative);
        const data = await readFile(path);
        data[0] = (data[0]! + 1) & 255;
        await writeFile(path, data);
        return fixtureSourceFacts({ sourceCommit });
      },
    }), /Git blob/u);
    await assert.rejects(readFile(receipt), { code: 'ENOENT' });
  }
  assert.equal(checks, 4);
});
