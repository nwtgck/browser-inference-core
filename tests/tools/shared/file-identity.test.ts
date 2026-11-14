import assert from 'node:assert/strict';
import { appendFileSync, chmodSync, constants, statSync, utimesSync } from 'node:fs';
import filesystem from 'node:fs/promises';
import { chmod, rm, symlink, writeFile, type FileHandle } from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { join } from 'node:path';
import test from 'node:test';
import { fileIdentity } from '#tools/shared/files';
import { runProcess } from '#tools/shared/process';
import { temporaryDirectory } from './fixtures.test-support.js';

test('an input replaced by a symlink between inspection and open is never followed', async (context) => {
  await using temporary = await temporaryDirectory();
  const input = join(temporary.root, 'input');
  const outside = join(temporary.root, 'outside');
  await writeFile(input, 'original');
  await writeFile(outside, 'must not be read');
  const open = filesystem.open;
  const intercepted = context.mock.method(filesystem, 'open', async (...args: Parameters<typeof open>) => {
    if (args[0] === input) {
      await rm(input);
      await symlink(outside, input);
    }
    return open(...args);
  });
  syncBuiltinESMExports();
  try {
    await assert.rejects(fileIdentity({ path: input }), /ELOOP|symbolic link/u);
  } finally {
    intercepted.mock.restore();
    syncBuiltinESMExports();
  }
});

test('an input replaced by a FIFO opens without blocking and rejects the actual descriptor type', async (context) => {
  await using temporary = await temporaryDirectory();
  const input = join(temporary.root, 'input');
  await writeFile(input, 'original');
  const open = filesystem.open;
  let opened: FileHandle | undefined;
  const intercepted = context.mock.method(filesystem, 'open', async (...args: Parameters<typeof open>) => {
    if (args[0] === input) {
      // Fail before a blocking syscall if the required FIFO protection regresses.
      assert.equal(typeof args[1], 'number');
      assert.ok(typeof args[1] === 'number' && (args[1] & constants.O_NONBLOCK) !== 0);
      await rm(input);
      await runProcess({ command: 'mkfifo', args: [input], cwd: temporary.root });
      opened = await open(...args);
      return opened;
    }
    return open(...args);
  });
  syncBuiltinESMExports();
  try {
    await assert.rejects(fileIdentity({ path: input }), /Non-regular opened/u);
    assert.ok(opened);
    assert.equal(opened.fd, -1);
  } finally {
    intercepted.mock.restore();
    syncBuiltinESMExports();
    if (opened && opened.fd !== -1) {
      await opened.close();
    }
  }
});

for (const mutation of ['size', 'mtime', 'ctime'] as const) {
  test(`descriptor ${mutation} changes during the real stream are rejected`, async (context) => {
    await using temporary = await temporaryDirectory();
    const input = join(temporary.root, 'input');
    await writeFile(input, Buffer.alloc(128 * 1024, 0x61));
    await chmod(input, 0o640);
    utimesSync(input, 0, 0);
    const open = filesystem.open;
    let opened: FileHandle | undefined;
    let mutated = false;
    const intercepted = context.mock.method(filesystem, 'open', async (...args: Parameters<typeof open>) => {
      const handle = await open(...args);
      if (args[0] === input) {
        opened = handle;
        const createReadStream = handle.createReadStream.bind(handle);
        context.mock.method(handle, 'createReadStream', (...streamArgs: Parameters<typeof createReadStream>) => {
          const stream = createReadStream(...streamArgs);
          stream.once('readable', () => {
            switch (mutation) {
            case 'size':
              appendFileSync(input, 'changed');
              break;
            case 'mtime':
              utimesSync(input, 100, 100);
              break;
            case 'ctime': {
              const before = statSync(input, { bigint: true });
              chmodSync(input, 0o600);
              const after = statSync(input, { bigint: true });
              assert.equal(after.size, before.size);
              assert.equal(after.mtimeNs, before.mtimeNs);
              assert.notEqual(after.ctimeNs, before.ctimeNs);
              break;
            }
            default: {
              const _ex: never = mutation;
              throw new Error(`Unhandled mutation: ${_ex}`);
            }
            }
            mutated = true;
          });
          return stream;
        });
      }
      return handle;
    });
    syncBuiltinESMExports();
    try {
      await assert.rejects(fileIdentity({ path: input }), /changed while reading/u);
      assert.equal(mutated, true);
      assert.ok(opened);
      assert.equal(opened.fd, -1);
    } finally {
      intercepted.mock.restore();
      syncBuiltinESMExports();
      if (opened && opened.fd !== -1) {
        await opened.close();
      }
    }
  });
}
