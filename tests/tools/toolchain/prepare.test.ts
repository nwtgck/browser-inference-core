import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, stat, symlink, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { runProcess } from '#tools/shared/process';
import { applyAsyncifyBigIntPatch, prepareDawn, verifyAsyncifyBigIntPatch } from '#tools/toolchain/prepare';

function sha({ value }: { value: string | Buffer }): string {
  return createHash('sha256').update(value).digest('hex');
}

const original = `\
    //
    // Asyncify code that is shared between mode 1 (original) and mode 2 (JSPI).
    //
#if ASYNCIFY == 1 && MEMORY64
    rewindArguments: new Map(),
#endif
    instrumentWasmImports(imports) {
        }
      }
    },
#if ASYNCIFY == 1 && MEMORY64
    saveRewindArguments(func, passedArguments) {
      return Asyncify.rewindArguments.set(func, Array.from(passedArguments));
    },
#endif
        Asyncify.exportCallStack.push(original);
        try {
#if MEMORY64
          Asyncify.saveRewindArguments(original, args);
#endif
          return original(...args);
      // Once we have rewound and the stack we no longer need to artificially
      // keep the runtime alive.
      {{{ runtimeKeepalivePop(); }}}
#if MEMORY64
      // When re-winding, the arguments to a function are ignored.  For i32 arguments we
      // can just call the function with no args at all since the engine will produce zeros
      // for all arguments.  However, for i64 arguments we get \`undefined cannot be converted to
`;
const expected = original.replaceAll('#if ASYNCIFY == 1 && MEMORY64', '#if ASYNCIFY == 1 && (MEMORY64 || WASM_BIGINT)').replaceAll('#if MEMORY64', '#if MEMORY64 || WASM_BIGINT');

test('the approved patch modifies four guards, is idempotent and rejects unknown sources', async () => {
  const work = await mkdtemp(join(tmpdir(), 'asyncify-test-'));
  try {
    const source = join(work, 'src/lib/libasync.js');
    await mkdir(join(work, 'src/lib'), { recursive: true });
    const patchConfig = { sourceSha256: sha({ value: original }), patchedSha256: sha({ value: expected }) };
    await writeFile(source, original);
    await applyAsyncifyBigIntPatch({ checkoutRoot: process.cwd(), emscriptenRoot: work, patchConfig });
    assert.equal(await readFile(source, 'utf8'), expected);
    const first = await stat(source);
    await applyAsyncifyBigIntPatch({ checkoutRoot: process.cwd(), emscriptenRoot: work, patchConfig });
    assert.equal((await stat(source)).mtimeMs, first.mtimeMs);
    await verifyAsyncifyBigIntPatch({ emscriptenRoot: work, patchConfig });
    await writeFile(source, `${original}different`);
    await assert.rejects(applyAsyncifyBigIntPatch({ checkoutRoot: process.cwd(), emscriptenRoot: work, patchConfig }), /Unexpected Emscripten/);
    assert.equal(await readFile(source, 'utf8'), `${original}different`);
  } finally {
    await rm(work, { recursive: true, force: true });
  }
});

test('a verified Dawn archive is extracted afresh and a bad checksum preserves the existing port', async () => {
  const work = await mkdtemp(join(tmpdir(), 'dawn-test-'));
  try {
    const tools = join(work, 'tools');
    const fixture = join(work, 'fixture');
    await mkdir(join(tools, 'downloads'), { recursive: true });
    await mkdir(join(fixture, 'emdawnwebgpu_pkg'), { recursive: true });
    await writeFile(join(fixture, 'emdawnwebgpu_pkg/emdawnwebgpu.port.py'), 'fixture port');
    const archive = join(tools, 'downloads/emdawnwebgpu_pkg-test.zip');
    await runProcess({ command: 'zip', args: ['-q', '-r', archive, 'emdawnwebgpu_pkg'], cwd: fixture });
    const dawnSha256 = sha({ value: await readFile(archive) });
    const port = await prepareDawn({ toolsRoot: tools, dawnTag: 'test', dawnSha256 });
    assert.equal(await readFile(join(port, 'emdawnwebgpu.port.py'), 'utf8'), 'fixture port');
    const originalPort = await stat(join(port, 'emdawnwebgpu.port.py'));
    await writeFile(join(port, 'stale.py'), 'old extraction state');
    await prepareDawn({ toolsRoot: tools, dawnTag: 'test', dawnSha256 });
    await assert.rejects(stat(join(port, 'stale.py')), { code: 'ENOENT' });
    assert.equal((await stat(join(port, 'emdawnwebgpu.port.py'))).mtimeMs, originalPort.mtimeMs);
    await assert.rejects(prepareDawn({ toolsRoot: tools, dawnTag: '../outside', dawnSha256 }), /Invalid Dawn release tag/);
    await assert.rejects(prepareDawn({ toolsRoot: tools, dawnTag: 'test', dawnSha256: '0'.repeat(64) }), /checksum/);
    assert.equal(await readFile(join(port, 'emdawnwebgpu.port.py'), 'utf8'), 'fixture port');
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
  } finally {
    await rm(work, { recursive: true, force: true });
  }
});
