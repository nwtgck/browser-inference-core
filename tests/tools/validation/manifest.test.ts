import assert from 'node:assert/strict';
import { chmod, lstat, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import path from 'node:path';
import test from 'node:test';
import { manifestIo, withValidationManifest } from '#tools/validation/manifest';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

test('manifest activation is complete, preserves mode, and rolls back final failure', async () => {
  await using temporary = await temporaryDirectory();
  const manifestPath = path.join(temporary.root, 'manifest.json');
  const before = Buffer.from('original complete manifest');
  const after = Buffer.from('validated complete manifest');
  await writeFile(manifestPath, before);
  await chmod(manifestPath, 0o640);
  await assert.rejects(withValidationManifest({
    manifestPath,
    before,
    after,
    io: manifestIo,
    finish: async () => {
      assert.deepEqual(await readFile(manifestPath), after);
      assert.equal((await lstat(manifestPath)).mode & 0o777, 0o640);
      throw new Error('npm failed');
    },
  }), /npm failed/u);
  assert.deepEqual(await readFile(manifestPath), before);
  assert.equal((await lstat(manifestPath)).mode & 0o777, 0o640);
  await withValidationManifest({ manifestPath, before, after, io: manifestIo, finish: async () => undefined });
  assert.deepEqual(await readFile(manifestPath), after);
});

test('partial backup/candidate writes and failed activation leave the old inode intact', async () => {
  for (const failure of ['original', 'next', 'activate']) {
    await using temporary = await temporaryDirectory();
    const manifestPath = path.join(temporary.root, 'manifest.json');
    const before = Buffer.from('complete original');
    await writeFile(manifestPath, before);
    const inode = (await lstat(manifestPath)).ino;
    const write: typeof writeFile = async (file, data, options) => {
      if (String(file).endsWith(`manifest.${failure}.json`)) {
        await writeFile(file, 'partial');
        throw new Error('partial write');
      }
      await writeFile(file, data, options);
    };
    const move: typeof rename = async (from, to) => {
      if (failure === 'activate') {
        throw new Error('activation failed');
      }
      await rename(from, to);
    };
    await assert.rejects(withValidationManifest({ manifestPath, before, after: Buffer.from('next'), io: { writeFile: write, rename: move }, finish: async () => assert.fail('must not finalize') }), /partial write|activation failed/u);
    assert.deepEqual(await readFile(manifestPath), before);
    assert.equal((await lstat(manifestPath)).ino, inode);
  }
});

test('failed recovery retains and identifies the exact original outside the package', async () => {
  await using temporary = await temporaryDirectory();
  const manifestPath = path.join(temporary.root, 'manifest.json');
  const before = Buffer.from('retained complete original');
  await writeFile(manifestPath, before);
  const move: typeof rename = async (from, to) => {
    if (String(from).endsWith('manifest.original.json')) {
      throw new Error('restore blocked');
    }
    await rename(from, to);
  };
  let retained: string | undefined;
  await assert.rejects(withValidationManifest({
    manifestPath,
    before,
    after: Buffer.from('next'),
    io: { writeFile, rename: move },
    finish: async () => {
      throw new Error('npm failed');
    },
  }), (error: unknown) => {
    assert.ok(error instanceof Error);
    retained = /original retained at (.+)$/u.exec(error.message)?.[1];
    return retained !== undefined;
  });
  assert.ok(retained);
  assert.notEqual(path.dirname(retained), temporary.root);
  assert.deepEqual(await readFile(retained), before);
  assert.deepEqual(await readdir(path.dirname(retained)), ['manifest.original.json']);
  // This retained backup is outside the disposable fixture's directory.
  const { rm } = await import('node:fs/promises');
  await rm(path.dirname(retained), { recursive: true });
});
