import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import filesystem from 'node:fs/promises';
import { chmod, mkdir, readFile, readdir, stat, symlink, writeFile } from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { isContained, regularFileIdentities } from '#tools/shared/files';
import { ProcessError, runProcess } from '#tools/shared/process';
import { prepareDawn } from '#tools/toolchain/prepare';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

function sha({ value }: { value: string | Buffer }): string {
  return createHash('sha256').update(value).digest('hex');
}

test('Dawn preparation preserves validated files and prior output across filesystem boundaries and failures', async (context) => {
  await using temporary = await temporaryDirectory();
  const work = temporary.root;
  const tools = join(work, 'tools');
  const fixture = join(work, 'fixture');
  const packageRoot = join(fixture, 'emdawnwebgpu_pkg');
  await mkdir(join(tools, 'downloads'), { recursive: true });
  await mkdir(join(packageRoot, 'webgpu/include'), { recursive: true });
  await writeFile(join(packageRoot, 'emdawnwebgpu.port.py'), 'fixture port');
  await writeFile(join(packageRoot, 'README.md'), 'Dawn fixture notice');
  await writeFile(join(packageRoot, 'webgpu/include/webgpu.h'), 'fixture declarations');
  await chmod(join(packageRoot, 'emdawnwebgpu.port.py'), 0o755);
  await chmod(join(packageRoot, 'README.md'), 0o640);
  await chmod(join(packageRoot, 'webgpu'), 0o750);
  const expectedFiles = await regularFileIdentities({ root: packageRoot });
  const archive = join(tools, 'downloads/emdawnwebgpu_pkg-test.zip');
  await runProcess({ command: 'zip', args: ['-q', '-r', archive, 'emdawnwebgpu_pkg'], cwd: fixture });
  const dawnSha256 = sha({ value: await readFile(archive) });
  const rename = filesystem.rename;
  // Model a workspace mount distinct from the system temporary filesystem.
  const intercepted = context.mock.method(filesystem, 'rename', async (...args: Parameters<typeof rename>) => {
    const sourceInTools = isContained({ root: tools, path: args[0].toString() });
    const destinationInTools = isContained({ root: tools, path: args[1].toString() });
    if (sourceInTools !== destinationInTools) {
      throw Object.assign(new Error('Cross-device rename between system temporary files and workspace'), { code: 'EXDEV' });
    }
    return rename(...args);
  });
  syncBuiltinESMExports();
  try {
    const port = await prepareDawn({ toolsRoot: tools, dawnTag: 'test', dawnSha256 });
    assert.deepEqual(await regularFileIdentities({ root: port }), expectedFiles);
    assert.equal((await stat(port)).mode & 0o777, 0o700);
    for (const name of ['emdawnwebgpu.port.py', 'README.md', 'webgpu', 'webgpu/include/webgpu.h']) {
      assert.equal((await stat(join(port, name))).mode & 0o777, (await stat(join(packageRoot, name))).mode & 0o777, name);
    }
    const originalPort = await stat(join(port, 'emdawnwebgpu.port.py'));
    await writeFile(join(port, 'stale.py'), 'old extraction state');
    await prepareDawn({ toolsRoot: tools, dawnTag: 'test', dawnSha256 });
    await assert.rejects(stat(join(port, 'stale.py')), { code: 'ENOENT' });
    assert.equal((await stat(join(port, 'emdawnwebgpu.port.py'))).mtimeMs, originalPort.mtimeMs);
    const unchanged = await stat(port);
    await prepareDawn({ toolsRoot: tools, dawnTag: 'test', dawnSha256 });
    assert.equal((await stat(port)).ino, unchanged.ino);
    assert.deepEqual(await regularFileIdentities({ root: port }), expectedFiles);
    await assert.rejects(prepareDawn({ toolsRoot: tools, dawnTag: '../outside', dawnSha256 }), /Invalid Dawn release tag/);
    await assert.rejects(prepareDawn({ toolsRoot: tools, dawnTag: 'test', dawnSha256: '0'.repeat(64) }), /checksum/);
    assert.equal(await readFile(join(port, 'emdawnwebgpu.port.py'), 'utf8'), 'fixture port');
    const corrupted = join(tools, 'downloads/emdawnwebgpu_pkg-corrupt.zip');
    await runProcess({ command: 'zip', args: ['-q', '-0', '-r', corrupted, 'emdawnwebgpu_pkg'], cwd: fixture });
    const corruptedBytes = await readFile(corrupted);
    const payload = corruptedBytes.indexOf('fixture port');
    const first = corruptedBytes[payload];
    assert.ok(payload >= 0 && first !== undefined);
    corruptedBytes[payload] = first ^ 0xff;
    await writeFile(corrupted, corruptedBytes);
    await assert.rejects(prepareDawn({ toolsRoot: tools, dawnTag: 'corrupt', dawnSha256: sha({ value: corruptedBytes }) }), (error: unknown) => error instanceof ProcessError && error.command === 'unzip' && error.exitCode !== 0);
    assert.equal((await stat(port)).ino, unchanged.ino);
    assert.deepEqual(await regularFileIdentities({ root: port }), expectedFiles);
    await writeFile(join(fixture, 'outside'), 'outside');
    await symlink('../outside', join(fixture, 'emdawnwebgpu_pkg/link'));
    const linked = join(tools, 'downloads/emdawnwebgpu_pkg-link.zip');
    await runProcess({ command: 'zip', args: ['-q', '-r', '-y', linked, 'emdawnwebgpu_pkg'], cwd: fixture });
    await assert.rejects(prepareDawn({ toolsRoot: tools, dawnTag: 'link', dawnSha256: sha({ value: await readFile(linked) }) }), /unsupported/);
    const escaping = join(tools, 'downloads/emdawnwebgpu_pkg-escape.zip');
    await writeFile(join(work, 'outside'), 'outside the archive root');
    await runProcess({ command: 'zip', args: ['-q', escaping, '../outside'], cwd: fixture });
    await assert.rejects(prepareDawn({ toolsRoot: tools, dawnTag: 'escape', dawnSha256: sha({ value: await readFile(escaping) }) }), /Invalid relative source path/);
    assert.equal(await readFile(join(port, 'emdawnwebgpu.port.py'), 'utf8'), 'fixture port');
    assert.deepEqual((await readdir(tools)).sort(), ['downloads', 'emdawnwebgpu_pkg']);
  } finally {
    intercepted.mock.restore();
    syncBuiltinESMExports();
  }
});
