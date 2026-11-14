import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile, readdir, rename, rm, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { extractNpmResolution, prepareNpmResolution, readNpmResolution } from '#tools/publication/npm-lock';
import type { NpmLock } from '#dto/shared';

const expected = { repository: 'fixture/core', commit: 'c'.repeat(40), version: '0.1.0' };
const specifier = `github:${expected.repository}#${expected.commit}`;

function fixtureLock(): NpmLock {
  return { lockfileVersion: 3, packages: { '': { dependencies: { 'browser-inference-core': specifier } }, 'node_modules/browser-inference-core': { version: '0.1.0', resolved: `git+ssh://git@github.com/${expected.repository}.git#${expected.commit}`, integrity: 'sha512-explicit-synthetic-fixture', license: 'MIT' } } };
}

test('npm resolution preparation owns a fresh direct-dependency fixture with no request roundtrip', async () => {
  await using temporary = await temporaryDirectory();
  const directory = join(temporary.root, 'lock');
  const request = await prepareNpmResolution({ directory, ...expected });
  assert.deepEqual(request, { ...expected, package: 'browser-inference-core', specifier });
  assert.deepEqual((await readdir(directory)).sort(), ['empty-global.npmrc', 'empty.npmrc', 'package.json']);
  const before = await readFile(join(directory, 'package.json'));
  assert.deepEqual(JSON.parse(before.toString()), { name: 'runtime-consumer-lock-fixture', version: '0.0.0', private: true, dependencies: { 'browser-inference-core': specifier } });
  assert.equal((await readFile(join(directory, 'empty.npmrc'))).length, 0);
  assert.equal((await readFile(join(directory, 'empty-global.npmrc'))).length, 0);
  await assert.rejects(prepareNpmResolution({ directory, ...expected }), /fresh directory/u);
  assert.deepEqual(await readFile(join(directory, 'package.json')), before);
  await symlink(directory, join(temporary.root, 'linked'));
  await assert.rejects(prepareNpmResolution({ directory: join(temporary.root, 'linked'), ...expected }), /fresh directory/u);
  await assert.rejects(prepareNpmResolution({ directory: join(temporary.root, 'linked/new-child'), ...expected }), /Linked/u);
  for (const values of [{ ...expected, commit: 'main' }, { ...expected, commit: expected.commit.slice(0, 12) }, { ...expected, commit: expected.commit.toUpperCase() }, { ...expected, repository: 'fixture/../other' }, { ...expected, repository: 'https://github.com/fixture/core' }, { ...expected, version: '' }]) {
    await assert.rejects(prepareNpmResolution({ directory: join(temporary.root, 'invalid'), ...values }));
  }
});

test('external npm lock preserves integrity and future metadata and rejects changed request inputs', async () => {
  await using temporary = await temporaryDirectory();
  const directory = join(temporary.root, 'lock');
  await prepareNpmResolution({ directory, ...expected });
  const lock = fixtureLock();
  lock.packages['node_modules/browser-inference-core']!['futureNpmField'] = { preserved: true };
  await writeFile(join(directory, 'package-lock.json'), JSON.stringify(lock));
  const result = await readNpmResolution({ directory, expected });
  assert.deepEqual(result.packageEntry, lock.packages['node_modules/browser-inference-core']);
  assert.deepEqual(result.rootDependencies, { 'browser-inference-core': specifier });
  const withoutIntegrity = fixtureLock();
  delete withoutIntegrity.packages['node_modules/browser-inference-core']!['integrity'];
  assert(!('integrity' in extractNpmResolution({ lock: withoutIntegrity, ...expected }).packageEntry));
  for (const name of ['package.json', 'package-lock.json']) {
    const path = join(directory, name);
    const saved = path + '.saved';
    await rename(path, saved);
    await assert.rejects(readNpmResolution({ directory, expected }), /missing/u);
    await symlink(saved, path);
    await assert.rejects(readNpmResolution({ directory, expected }), /Linked/u);
    await rm(path);
    await rename(saved, path);
  }
  const fixture = JSON.parse(await readFile(join(directory, 'package.json'), 'utf8')) as { scripts?: object };
  fixture.scripts = { preinstall: 'unexpected' };
  await writeFile(join(directory, 'package.json'), JSON.stringify(fixture));
  await assert.rejects(readNpmResolution({ directory, expected }), /fixture changed/u);
});

test('npm lock admission rejects mutable, wrong-version, transitive and install-time inputs', () => {
  for (const change of [
    ({ lock }: { lock: NpmLock }) => {
      lock.lockfileVersion = 2;
    },
    ({ lock }: { lock: NpmLock }) => {
      lock.packages['']!.dependencies = { 'browser-inference-core': 'latest' };
    },
    ({ lock }: { lock: NpmLock }) => {
      lock.packages['node_modules/browser-inference-core']!.version = '0.2.0';
    },
    ({ lock }: { lock: NpmLock }) => {
      lock.packages['node_modules/another'] = {};
    },
  ]) {
    const lock = fixtureLock();
    change({ lock });
    assert.throws(() => extractNpmResolution({ lock, ...expected }));
  }
  for (const [field, value] of [['dependencies', { another: '1' }], ['optionalDependencies', { another: '1' }], ['peerDependencies', { another: '1' }], ['peerDependenciesMeta', { another: {} }], ['bundleDependencies', ['another']], ['bundledDependencies', ['another']], ['hasInstallScript', true]] as const) {
    const lock = fixtureLock();
    Object.assign(lock.packages['node_modules/browser-inference-core']!, { [field]: value });
    assert.throws(() => extractNpmResolution({ lock, ...expected }), /install-time/u);
  }
});

test('resolved npm URLs bind the exact GitHub repository and complete commit without credentials or extra URL state', () => {
  for (const resolved of [
    `git+https://example.invalid/${expected.repository}.git#${expected.commit}`,
    `git+https://github.com/fixture/other.git#${expected.commit}`,
    `git+https://github.com/${expected.repository}.git?query=1#${expected.commit}`,
    `git+https://git:secret@github.com/${expected.repository}.git#${expected.commit}`,
    `git+https://other@github.com/${expected.repository}.git#${expected.commit}`,
    `git+https://github.com:443/${expected.repository}.git#${expected.commit}`,
    `https://github.com/${expected.repository}/archive/${expected.commit}.tar.gz#${expected.commit}`,
    `git+file:///tmp/${expected.repository}.git#${expected.commit}`,
    `git+https://github.com/${expected.repository}.git#main`,
  ]) {
    const lock = fixtureLock();
    lock.packages['node_modules/browser-inference-core']!.resolved = resolved;
    assert.throws(() => extractNpmResolution({ lock, ...expected }));
  }
  for (const protocol of ['git+ssh://git@', 'git+https://']) {
    const lock = fixtureLock();
    lock.packages['node_modules/browser-inference-core']!.resolved = `${protocol}github.com/${expected.repository}.git#${expected.commit}`;
    assert.equal(extractNpmResolution({ lock, ...expected }).packageEntry.resolved, lock.packages['node_modules/browser-inference-core']!.resolved);
  }
});
