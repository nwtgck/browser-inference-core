import test from 'node:test';
import assert from 'node:assert/strict';
import { chmod, lstat, mkdtemp, readFile, rm, symlink, truncate } from 'node:fs/promises';
import filesystem from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { copyRegular, copyRegularTree, packageFiles, readMetadataBytes, maxArtifactBytes } from '#tools/package/files';
import { write } from './fixtures.test-support.js';
import { runProcess } from '#tools/shared/process';

test('regular copy preserves executable permissions and never follows destination links', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-package-files-'));
  await using cleanup = {
    async [Symbol.asyncDispose]() {
      await rm(root, { recursive: true, force: true });
    },
  };
  const source = join(root, 'source/tool.mjs');
  const destination = join(root, 'copy/tool.mjs');
  const outside = join(root, 'outside.txt');
  await write({ path: source, data: 'fixture bytes' });
  await chmod(source, 0o755);
  await copyRegular({ source, destination });
  assert.equal((await lstat(destination)).mode & 0o777, 0o755);
  await write({ path: outside, data: 'keep outside unchanged' });
  await rm(destination);
  await symlink(outside, destination);
  await assert.rejects(copyRegular({ source, destination }), /Linked/u);
  assert.equal(await readFile(outside, 'utf8'), 'keep outside unchanged');
  await assert.rejects(copyRegularTree({ source: join(root, 'copy'), destination: join(root, 'other') }), /Linked/u);
  await symlink(join(root, 'source'), join(root, 'linked-parent'));
  await assert.rejects(copyRegularTree({ source: join(root, 'source'), destination: join(root, 'linked-parent/new-copy') }), /Linked/u);
  await assert.rejects(lstat(join(root, 'source/new-copy')), { code: 'ENOENT' });
});

test('tree paths reject case collisions and metadata allocation obeys the published file limit', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-package-files-'));
  await using cleanup = {
    async [Symbol.asyncDispose]() {
      await rm(root, { recursive: true, force: true });
    },
  };
  await write({ path: join(root, 'first/A.txt'), data: 'a' });
  await write({ path: join(root, 'first/a.txt'), data: 'b' });
  await assert.rejects(packageFiles({ root: join(root, 'first'), exclude: [] }), /case-colliding/u);
  await write({ path: join(root, 'backslash/bad\\path'), data: 'unsafe name' });
  await assert.rejects(packageFiles({ root: join(root, 'backslash'), exclude: [] }), /relative source path/u);
  await write({ path: join(root, 'hidden/.git/config'), data: 'unlisted Git configuration' });
  await assert.rejects(packageFiles({ root: join(root, 'hidden'), exclude: [] }), /relative source path/u);
  const metadata = join(root, 'metadata.json');
  await write({ path: metadata, data: '{}' });
  await truncate(metadata, maxArtifactBytes);
  await assert.rejects(readMetadataBytes({ path: metadata }), /Oversized/u);
});

test('input roots, parents, leaf links and named pipes are rejected before opening', async () => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-package-files-'));
  await using cleanup = {
    async [Symbol.asyncDispose]() {
      await rm(root, { recursive: true, force: true });
    },
  };
  await write({ path: join(root, 'source/file'), data: 'original' });
  const link = join(root, 'source/link');
  for (const target of [join(root, 'source'), join(root, 'source/file'), join(root, 'missing')]) {
    await symlink(target, link);
    await assert.rejects(packageFiles({ root: join(root, 'source'), exclude: [] }), /Linked/u);
    await assert.rejects(readMetadataBytes({ path: link }), /Linked/u);
    await rm(link);
  }
  await symlink(join(root, 'source'), join(root, 'parent-link'));
  await assert.rejects(packageFiles({ root: join(root, 'parent-link'), exclude: [] }), /Linked/u);
  await assert.rejects(readMetadataBytes({ path: join(root, 'parent-link/file') }), /Linked/u);
  const pipe = join(root, 'source/pipe');
  await runProcess({ command: 'mkfifo', args: [pipe], cwd: root });
  await assert.rejects(packageFiles({ root: join(root, 'source'), exclude: [] }), /Non-regular/u);
  await assert.rejects(readMetadataBytes({ path: pipe }), /non-regular/u);
});

test('copy detects same-size changes at the copy boundary', async t => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-package-files-'));
  await using cleanup = {
    async [Symbol.asyncDispose]() {
      await rm(root, { recursive: true, force: true });
    },
  };
  await write({ path: join(root, 'source/file'), data: 'abcdef' });
  const original = filesystem.cp;
  const replacement = t.mock.method(filesystem, 'cp', async (...args: Parameters<typeof original>) => {
    await original(...args);
    await filesystem.writeFile(args[1], 'ghijkl');
  });
  using restore = {
    [Symbol.dispose]() {
      replacement.mock.restore();
      syncBuiltinESMExports();
    },
  };
  syncBuiltinESMExports();
  await assert.rejects(copyRegularTree({ source: join(root, 'source'), destination: join(root, 'output') }), /Copied input changed/u);
});

test('copy-time source substitution preserves the link and never reads its target', async t => {
  const root = await mkdtemp(join(tmpdir(), 'bicore-package-files-'));
  await using cleanup = {
    async [Symbol.asyncDispose]() {
      await rm(root, { recursive: true, force: true });
    },
  };
  await write({ path: join(root, 'source/file'), data: 'abcdef' });
  const secret = join(root, 'secret');
  await write({ path: secret, data: 'never dereference' });
  const original = filesystem.cp;
  const replacement = t.mock.method(filesystem, 'cp', async (...args: Parameters<typeof original>) => {
    await filesystem.rm(args[0]);
    await filesystem.symlink(secret, args[0]);
    await original(...args);
  });
  using restore = {
    [Symbol.dispose]() {
      replacement.mock.restore();
      syncBuiltinESMExports();
    },
  };
  syncBuiltinESMExports();
  await assert.rejects(copyRegularTree({ source: join(root, 'source'), destination: join(root, 'output') }), /Linked/u);
  assert((await lstat(join(root, 'output/file'))).isSymbolicLink());
  assert.equal(await readFile(secret, 'utf8'), 'never dereference');
});
