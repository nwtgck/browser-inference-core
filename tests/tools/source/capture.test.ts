import assert from 'node:assert/strict';
import { chmod, lstat, mkdir, readFile, readlink, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import test from 'node:test';
import { exportCommit, gitEnvironment, observeSource, selectedGitlinks, snapshotCompileInputs } from '#tools/source/capture';
import { runProcess } from '#tools/shared/process';
import { commitFiles, git, initializeRepository, temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

test('selected commit objects ignore worktree, staged changes, replacement refs and attributes', async () => {
  await using temporary = await temporaryDirectory();
  const sourceRoot = join(temporary.root, 'source');
  await initializeRepository({ root: sourceRoot });
  await writeFile(join(sourceRoot, 'input'), 'committed\n');
  await writeFile(join(sourceRoot, '.gitattributes'), 'input export-ignore\ntext export-subst\ntool.bat text eol=crlf\n');
  await writeFile(join(sourceRoot, 'text'), '$Format:%H$\n');
  await writeFile(join(sourceRoot, 'tool.bat'), '@echo off\r\necho fixture\r\n');
  const revision = await commitFiles({ root: sourceRoot });
  await writeFile(join(sourceRoot, 'input'), 'new HEAD\n');
  const replacement = await commitFiles({ root: sourceRoot });
  await git({ root: sourceRoot, args: ['replace', revision, replacement] });
  await writeFile(join(sourceRoot, 'input'), 'new staged input\n');
  await git({ root: sourceRoot, args: ['add', 'input'] });
  await git({ root: sourceRoot, args: ['update-index', '--assume-unchanged', 'input'] });
  await writeFile(join(sourceRoot, 'input'), 'hidden working-tree edit\n');
  const saved = { GIT_DIR: process.env.GIT_DIR, GIT_WORK_TREE: process.env.GIT_WORK_TREE, GIT_INDEX_FILE: process.env.GIT_INDEX_FILE };
  try {
    process.env.GIT_DIR = '/missing/repository';
    process.env.GIT_WORK_TREE = '/missing/worktree';
    process.env.GIT_INDEX_FILE = '/missing/index';
    const destination = join(temporary.root, 'output');
    await exportCommit({ sourceRoot, revision, destination });
    assert.equal(await readFile(join(destination, 'input'), 'utf8'), 'committed\n');
    assert.equal(await readFile(join(destination, 'text'), 'utf8'), '$Format:%H$\n');
    assert.equal(await readFile(join(destination, 'tool.bat'), 'utf8'), '@echo off\necho fixture\n');
  } finally {
    for (const [name, value] of Object.entries(saved)) {
      if (value === undefined) {
        delete process.env[name];
      } else {
        process.env[name] = value;
      }
    }
  }
});

test('binary large blobs, executable mode and contained links are preserved without hardlinks', async () => {
  await using temporary = await temporaryDirectory();
  const sourceRoot = join(temporary.root, 'source');
  await initializeRepository({ root: sourceRoot });
  const bytes = Buffer.from(Array.from({ length: 256 }, (_, index) => index));
  const large = Buffer.concat(Array.from({ length: 10000 }, () => bytes));
  await writeFile(join(sourceRoot, 'bytes'), large);
  await writeFile(join(sourceRoot, 'tool'), 'run\n');
  await chmod(join(sourceRoot, 'tool'), 0o755);
  await symlink('tool', join(sourceRoot, 'alias'));
  await symlink('missing-in-tree', join(sourceRoot, 'dangling'));
  const revision = await commitFiles({ root: sourceRoot });
  const destination = join(temporary.root, 'output');
  await exportCommit({ sourceRoot, revision, destination });
  assert.deepEqual(await readFile(join(destination, 'bytes')), large);
  assert.equal((await stat(join(destination, 'tool'))).mode & 0o777, 0o755);
  assert.equal(await readlink(join(destination, 'alias')), 'tool');
  assert.equal(await readlink(join(destination, 'dangling')), 'missing-in-tree');
  assert.notEqual((await stat(join(destination, 'tool'))).ino, (await stat(join(sourceRoot, 'tool'))).ino);
});

test('nested repositories use the parent gitlink and missing nested repositories never inherit parent objects', async () => {
  await using temporary = await temporaryDirectory();
  const sourceRoot = join(temporary.root, 'source');
  await initializeRepository({ root: sourceRoot });
  await writeFile(join(sourceRoot, 'input'), 'parent');
  await commitFiles({ root: sourceRoot });
  const nested = join(sourceRoot, 'nested');
  await initializeRepository({ root: nested });
  await writeFile(join(nested, 'data'), 'pinned child');
  const pin = await commitFiles({ root: nested });
  await git({ root: sourceRoot, args: ['update-index', '--add', '--cacheinfo', `160000,${pin},nested`] });
  await git({ root: sourceRoot, args: ['commit', '-qm', 'nested pin'] });
  const revision = await git({ root: sourceRoot, args: ['rev-parse', 'HEAD'] });
  await writeFile(join(nested, 'data'), 'different child HEAD');
  await commitFiles({ root: nested });
  const destination = join(temporary.root, 'output');
  await exportCommit({ sourceRoot, revision, destination });
  assert.equal(await readFile(join(destination, 'nested', 'data'), 'utf8'), 'pinned child');
  assert.deepEqual(await selectedGitlinks({ sourceRoot, revision, paths: { upstream: 'nested' } }), { upstream: pin });
  await rm(join(nested, '.git'), { recursive: true });
  await assert.rejects(exportCommit({ sourceRoot, revision, destination: join(temporary.root, 'missing') }), /Initialize the exact/u);
  assert.equal(await lstat(join(temporary.root, 'missing')).catch(() => undefined), undefined);
  await exportCommit({ sourceRoot, revision, destination: join(temporary.root, 'selected'), include: ['input'], exclude: ['nested'] });
});

test('invalid selection, noncommit identities and escaping/cyclic links leave no output', async () => {
  await using temporary = await temporaryDirectory();
  const sourceRoot = join(temporary.root, 'source');
  await initializeRepository({ root: sourceRoot });
  await writeFile(join(sourceRoot, 'input'), 'source');
  const revision = await commitFiles({ root: sourceRoot });
  const destination = join(temporary.root, 'output');
  for (const include of [[], ['missing'], ['../outside'], ['.git'], ['input', 'input']]) {
    await assert.rejects(exportCommit({ sourceRoot, revision, destination, include }));
  }
  for (const value of ['HEAD', revision.slice(0, 12), await git({ root: sourceRoot, args: ['rev-parse', 'HEAD^{tree}'] }), 'f'.repeat(40)]) {
    await assert.rejects(exportCommit({ sourceRoot, revision: value, destination }));
  }
  assert.equal(await lstat(destination).catch(() => undefined), undefined);
  await symlink('../outside', join(sourceRoot, 'alias'));
  const escaping = await commitFiles({ root: sourceRoot });
  await assert.rejects(exportCommit({ sourceRoot, revision: escaping, destination }), /escapes/u);
  await rm(join(sourceRoot, 'alias'));
  await symlink('alias', join(sourceRoot, 'alias'));
  const cyclic = await commitFiles({ root: sourceRoot });
  await assert.rejects(exportCommit({ sourceRoot, revision: cyclic, destination }), /cyclic/u);
  assert.equal(await lstat(destination).catch(() => undefined), undefined);
});

test('release snapshots require the expected clean checkout and retain previous inputs on failure', async () => {
  await using temporary = await temporaryDirectory();
  const checkoutRoot = join(temporary.root, 'checkout');
  await initializeRepository({ root: checkoutRoot });
  await mkdir(join(checkoutRoot, 'native'));
  await writeFile(join(checkoutRoot, 'native', 'input.txt'), 'committed input\n');
  await writeFile(join(checkoutRoot, '.gitignore'), 'build/\n');
  const head = await commitFiles({ root: checkoutRoot });
  await assert.rejects(observeSource({ checkoutRoot, mode: { kind: 'release', expectedCommit: 'f'.repeat(40) } }), /expected release commit/u);
  const source = await observeSource({ checkoutRoot, mode: { kind: 'release', expectedCommit: head } });
  assert.equal(source.publishable, true);
  const destination = join(checkoutRoot, 'build', 'compile-inputs');
  const snapshot = await snapshotCompileInputs({ source, destination, include: ['native'], exclude: [], replace: false });
  assert.deepEqual(Object.keys(snapshot.files), ['native/input.txt']);
  assert.equal(snapshot.compileRoot, destination);
  const before = await stat(join(destination, 'native', 'input.txt'));
  await snapshotCompileInputs({ source, destination, include: ['native'], exclude: [], replace: true });
  assert.equal((await stat(join(destination, 'native', 'input.txt'))).ino, before.ino);
  assert.equal((await stat(join(destination, 'native', 'input.txt'))).mtimeMs, before.mtimeMs);
  await writeFile(join(checkoutRoot, 'native', 'input.txt'), 'local change\n');
  await assert.rejects(observeSource({ checkoutRoot, mode: { kind: 'release', expectedCommit: head } }), /dirty/u);
  await assert.rejects(snapshotCompileInputs({ source, destination, include: ['native'], exclude: [], replace: true }), /dirty/u);
  assert.equal(await readFile(join(destination, 'native', 'input.txt'), 'utf8'), 'committed input\n');
  const local = await observeSource({ checkoutRoot, mode: { kind: 'local' } });
  assert.equal(local.publishable, false);
  await snapshotCompileInputs({ source: local, destination: join(temporary.root, 'local'), include: ['native'], exclude: [], replace: false });
  assert.equal(await readFile(join(temporary.root, 'local', 'native', 'input.txt'), 'utf8'), 'local change\n');
  await assert.rejects(snapshotCompileInputs({ source: local, destination: join(checkoutRoot, 'native', 'nested'), include: ['native'], exclude: [], replace: false }), /overlap selected/u);
  await commitFiles({ root: checkoutRoot });
  await assert.rejects(snapshotCompileInputs({ source: local, destination, include: ['native'], exclude: [], replace: true }), /HEAD changed/u);
  assert.equal(await readFile(join(destination, 'native', 'input.txt'), 'utf8'), 'committed input\n');
});

test('local snapshots include dirty/untracked inputs and deletions but omit ignored files', async () => {
  await using temporary = await temporaryDirectory();
  const checkoutRoot = join(temporary.root, 'checkout');
  await initializeRepository({ root: checkoutRoot });
  await mkdir(join(checkoutRoot, 'native'));
  await writeFile(join(checkoutRoot, '.gitignore'), 'native/ignored.txt\n');
  await writeFile(join(checkoutRoot, 'native', 'deleted.txt'), 'old');
  await commitFiles({ root: checkoutRoot });
  await rm(join(checkoutRoot, 'native', 'deleted.txt'));
  await writeFile(join(checkoutRoot, 'native', 'new.txt'), 'local input');
  await writeFile(join(checkoutRoot, 'native', 'ignored.txt'), 'build cache');
  const source = await observeSource({ checkoutRoot, mode: { kind: 'local' } });
  const snapshot = await snapshotCompileInputs({ source, destination: join(temporary.root, 'snapshot'), include: ['native'], exclude: [], replace: false });
  assert.deepEqual(Object.keys(snapshot.files), ['native/new.txt']);
});

test('local gitlinks use stage zero while release pins use the committed tree', async () => {
  await using temporary = await temporaryDirectory();
  const sourceRoot = join(temporary.root, 'source');
  await initializeRepository({ root: sourceRoot });
  const first = await commitFiles({ root: sourceRoot });
  await writeFile(join(sourceRoot, 'input'), 'next');
  const second = await commitFiles({ root: sourceRoot });
  await git({ root: sourceRoot, args: ['update-index', '--add', '--cacheinfo', `160000,${first},upstream`] });
  await git({ root: sourceRoot, args: ['commit', '-qm', 'pin'] });
  const revision = await git({ root: sourceRoot, args: ['rev-parse', 'HEAD'] });
  await git({ root: sourceRoot, args: ['update-index', '--cacheinfo', `160000,${second},upstream`] });
  assert.deepEqual(await selectedGitlinks({ sourceRoot, paths: { source: 'upstream' }, revision }), { source: first });
  assert.deepEqual(await selectedGitlinks({ sourceRoot, paths: { source: 'upstream' }, revision: undefined }), { source: second });
  await git({ root: sourceRoot, args: ['update-index', '--force-remove', 'upstream'] });
  await runProcess({ command: 'git', args: ['-C', sourceRoot, 'update-index', '--index-info'], cwd: sourceRoot, input: `160000 ${first} 1\tupstream\n160000 ${second} 2\tupstream\n`, env: gitEnvironment({ ceiling: undefined }) });
  await assert.rejects(selectedGitlinks({ sourceRoot, paths: { source: 'upstream' }, revision: undefined }), /Missing or ambiguous/u);
});
