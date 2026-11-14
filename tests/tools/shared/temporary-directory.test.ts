import assert from 'node:assert/strict';
import filesystem, { mkdir, readFile, stat, symlink, unlink, writeFile } from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { temporaryDirectory } from '#tools/shared/temporary-directory';

test('temporary directory ownership ends at scope exit without deleting sibling paths', async () => {
  await using parent = await temporaryDirectory({ prefix: join(tmpdir(), 'bicore-temporary-test-') });
  const sibling = join(parent.root, 'preserved');
  await writeFile(sibling, 'keep');
  let owned: string;
  {
    await using temporary = await temporaryDirectory({ prefix: join(parent.root, 'owned-') });
    owned = temporary.root;
    assert.equal(dirname(owned), parent.root);
    assert.equal((await stat(owned)).isDirectory(), true);
    await mkdir(join(owned, 'nested'));
    await writeFile(join(owned, 'nested', 'payload'), 'temporary');
    await symlink(sibling, join(owned, 'link'));
  }
  await assert.rejects(stat(owned), { code: 'ENOENT' });
  assert.equal(await readFile(sibling, 'utf8'), 'keep');
});

test('temporary directory cleanup finishes before a failed operation rejects', async () => {
  await using parent = await temporaryDirectory({ prefix: join(tmpdir(), 'bicore-temporary-test-') });
  const failure = new Error('operation failed');
  let owned: string | undefined;
  await assert.rejects(async () => {
    await using temporary = await temporaryDirectory({ prefix: join(parent.root, 'failed-') });
    owned = temporary.root;
    await writeFile(join(owned, 'partial'), 'unfinished');
    throw failure;
  }, error => error === failure);
  assert.ok(owned);
  await assert.rejects(stat(owned), { code: 'ENOENT' });
});

test('cleanup keeps the allocated directory identity when its parent alias changes', async () => {
  await using parent = await temporaryDirectory({ prefix: join(tmpdir(), 'bicore-temporary-test-') });
  const original = join(parent.root, 'original');
  const redirected = join(parent.root, 'redirected');
  const alias = join(parent.root, 'alias');
  await mkdir(original);
  await mkdir(redirected);
  await symlink(original, alias);
  let owned: string;
  let other: string;
  {
    await using temporary = await temporaryDirectory({ prefix: join(alias, 'owned-') });
    owned = temporary.root;
    assert.equal(dirname(owned), original);
    other = join(redirected, owned.slice(original.length + 1));
    await mkdir(other);
    await writeFile(join(other, 'preserved'), 'keep');
    await unlink(alias);
    await symlink(redirected, alias);
  }
  await assert.rejects(stat(owned), { code: 'ENOENT' });
  assert.equal(await readFile(join(other, 'preserved'), 'utf8'), 'keep');
});

for (const outcome of ['cleanup_success', 'cleanup_failure'] as const) {
  test(`canonicalization failure attempts cleanup and preserves diagnostics after ${outcome}`, async context => {
    await using parent = await temporaryDirectory({ prefix: join(tmpdir(), 'bicore-temporary-test-') });
    const failure = new Error('canonicalization failed');
    const cleanupFailure = new Error('cleanup failed');
    const rm = filesystem.rm;
    let allocated: string | undefined;
    const canonicalize = context.mock.method(filesystem, 'realpath', async (...args: Parameters<typeof filesystem.realpath>) => {
      assert.equal(typeof args[0], 'string');
      assert.ok(typeof args[0] === 'string');
      allocated = args[0];
      throw failure;
    });
    const remove = context.mock.method(filesystem, 'rm', async (...args: Parameters<typeof rm>) => {
      switch (outcome) {
      case 'cleanup_success': return rm(...args);
      case 'cleanup_failure': throw cleanupFailure;
      default: {
        const exhaustive: never = outcome;
        throw new Error(`Unhandled cleanup outcome: ${exhaustive}`);
      }
      }
    });
    syncBuiltinESMExports();
    try {
      await assert.rejects(temporaryDirectory({ prefix: join(parent.root, 'failed-') }), error => {
        switch (outcome) {
        case 'cleanup_success': return error === failure;
        case 'cleanup_failure':
          assert.ok(error instanceof AggregateError);
          assert.deepEqual(error.errors, [failure, cleanupFailure]);
          return true;
        default: {
          const exhaustive: never = outcome;
          throw new Error(`Unhandled cleanup outcome: ${exhaustive}`);
        }
        }
      });
    } finally {
      canonicalize.mock.restore();
      remove.mock.restore();
      syncBuiltinESMExports();
    }
    assert.ok(allocated);
    switch (outcome) {
    case 'cleanup_success': await assert.rejects(stat(allocated), { code: 'ENOENT' });
      break;
    case 'cleanup_failure': assert.equal((await stat(allocated)).isDirectory(), true);
      break;
    default: {
      const exhaustive: never = outcome;
      throw new Error(`Unhandled cleanup outcome: ${exhaustive}`);
    }
    }
  });
}
