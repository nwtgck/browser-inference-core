import assert from 'node:assert/strict';
import filesystem from 'node:fs/promises';
import { mkdir, readFile, readlink, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { basename, join } from 'node:path';
import test from 'node:test';
import { withOutputDirectories } from '#tools/shared/output';
import { runProcess } from '#tools/shared/process';
import { temporaryDirectory } from './fixtures.test-support.js';

test('preparation and later activation failures preserve all previous outputs', async () => {
  await using temporary = await temporaryDirectory();
  const a = join(temporary.root, 'a');
  const b = join(temporary.root, 'b');
  for (const path of [a, b]) {
    await mkdir(path);
    await writeFile(join(path, 'payload'), `old-${path}`);
  }
  const inode = (await stat(a)).ino;
  await assert.rejects(withOutputDirectories({
    destinations: [a, b],
    replace: true,
    prepare: async ({ paths }) => {
      await writeFile(join(paths[0] ?? '', 'partial'), 'unfinished');
      throw new Error('preparation failure');
    },
  }), /preparation failure/u);
  assert.equal((await stat(a)).ino, inode);
  await assert.rejects(withOutputDirectories({
    destinations: [a, b],
    replace: true,
    prepare: async ({ paths }) => {
      for (const path of paths) {
        await writeFile(join(path, 'new'), 'candidate');
      }
      await rm(paths[1] ?? '', { recursive: true });
    },
  }), /Prepared output disappeared/u);
  for (const path of [a, b]) {
    assert.equal(await readFile(join(path, 'payload'), 'utf8'), `old-${path}`);
  }
});

test('unchanged outputs retain inode/mtime and changed trees retain equal file mtimes', async () => {
  await using temporary = await temporaryDirectory();
  const output = join(temporary.root, 'output');
  async function prepare({ add }: { add: boolean }): Promise<void> {
    await withOutputDirectories({
      destinations: [output],
      replace: true,
      preserveUnchanged: true,
      prepare: async ({ paths }) => {
        await writeFile(join(paths[0] ?? '', 'payload'), 'same');
        await symlink('payload', join(paths[0] ?? '', 'alias'));
        if (add) {
          await writeFile(join(paths[0] ?? '', 'new'), 'added');
        }
      },
    });
  }
  await prepare({ add: false });
  await runProcess({ command: 'python3', args: ['-c', 'import os,sys; os.utime(sys.argv[1],ns=(123456789,123456789))', join(output, 'payload')], cwd: temporary.root });
  const before = await stat(join(output, 'payload'));
  const precise = (await stat(join(output, 'payload'), { bigint: true })).mtimeNs;
  await prepare({ add: false });
  assert.equal((await stat(join(output, 'payload'))).ino, before.ino);
  await prepare({ add: true });
  const after = await stat(join(output, 'payload'));
  assert.equal(after.mtimeMs, before.mtimeMs);
  assert.equal((await stat(join(output, 'payload'), { bigint: true })).mtimeNs, precise);
  assert.notEqual(after.ino, before.ino);
  assert.equal(await readlink(join(output, 'alias')), 'payload');
});

test('overlap, linked parents and destinations changed by preparation are rejected', async () => {
  await using temporary = await temporaryDirectory();
  const output = join(temporary.root, 'output');
  await mkdir(output);
  await symlink(output, join(temporary.root, 'link'));
  for (const destinations of [[], [output, output], [output, join(output, 'nested')], [join(temporary.root, 'link', 'new')]]) {
    await assert.rejects(withOutputDirectories({ destinations, replace: true, prepare: async () => undefined }));
  }
  const absent = join(temporary.root, 'absent');
  await assert.rejects(withOutputDirectories({
    destinations: [absent],
    replace: true,
    prepare: async () => {
      await mkdir(absent);
      await writeFile(join(absent, 'independent'), 'keep');
    },
  }), /appeared/u);
  assert.equal(await readFile(join(absent, 'independent'), 'utf8'), 'keep');
});

test('a later backup failure restores an already activated output', async (context) => {
  await using temporary = await temporaryDirectory();
  const a = join(temporary.root, 'a');
  const b = join(temporary.root, 'b');
  for (const path of [a, b]) {
    await mkdir(path);
    await writeFile(join(path, 'payload'), `old-${path}`);
  }
  const rename = filesystem.rename;
  const intercepted = context.mock.method(filesystem, 'rename', async (...args: Parameters<typeof rename>) => {
    if (args[0] === b) {
      throw new Error('second backup failed');
    }
    return rename(...args);
  });
  syncBuiltinESMExports();
  try {
    await assert.rejects(withOutputDirectories({
      destinations: [a, b],
      replace: true,
      prepare: async ({ paths }) => {
        for (const path of paths) {
          await writeFile(join(path, 'new'), 'candidate');
        }
      },
    }), /second backup failed/u);
    for (const path of [a, b]) {
      assert.equal(await readFile(join(path, 'payload'), 'utf8'), `old-${path}`);
    }
  } finally {
    intercepted.mock.restore();
    syncBuiltinESMExports();
  }
});

test('failed recovery retains and names the complete previous output', async (context) => {
  await using temporary = await temporaryDirectory();
  const a = join(temporary.root, 'a');
  const b = join(temporary.root, 'b');
  for (const path of [a, b]) {
    await mkdir(path);
    await writeFile(join(path, 'payload'), `old-${path}`);
  }
  const rename = filesystem.rename;
  const intercepted = context.mock.method(filesystem, 'rename', async (...args: Parameters<typeof rename>) => {
    if (args[1] === b && ['prepared', 'previous'].includes(basename(args[0].toString()))) {
      throw new Error('activation and recovery failed');
    }
    return rename(...args);
  });
  syncBuiltinESMExports();
  try {
    let retained: string | undefined;
    await assert.rejects(withOutputDirectories({
      destinations: [a, b],
      replace: true,
      prepare: async ({ paths }) => {
        for (const path of paths) {
          await writeFile(join(path, 'new'), 'candidate');
        }
      },
    }), (error: unknown) => {
      assert.ok(error instanceof Error);
      retained = /previous output retained at (.+)$/u.exec(error.message)?.[1];
      return true;
    });
    assert.ok(retained);
    assert.equal(await readFile(join(retained, 'payload'), 'utf8'), `old-${b}`);
    assert.equal(await readFile(join(a, 'payload'), 'utf8'), `old-${a}`);
  } finally {
    intercepted.mock.restore();
    syncBuiltinESMExports();
  }
});
