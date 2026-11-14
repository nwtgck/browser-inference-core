import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import test from 'node:test';
import { decodeCompilationDatabase, decodeRootManifest } from '#dto/shared';
import { directory, fileIdentities, fileIdentity, regularFile, regularFileIdentities, relativePath } from '#tools/shared/files';
import { parseJson } from '#tools/shared/json';
import { temporaryDirectory } from './fixtures.test-support.js';

test('canonical paths allow Unicode and spaces and reject aliases or Git metadata', () => {
  assert.equal(relativePath({ value: 'some dir/資料.cpp' }), 'some dir/資料.cpp');
  for (const value of ['', '.', '..', '../outside', '/absolute', 'a/../b', 'a//b', 'a/./b', './a', 'a/', '.git/config', 'a/.GIT/b', 'a\\b', 'a\nb']) {
    assert.throws(() => relativePath({ value }));
  }
});

test('file identities stream binary bytes and do not follow linked files or parents', async () => {
  await using temporary = await temporaryDirectory();
  const bytes = Buffer.from([0, 255, 128, 10]);
  await writeFile(join(temporary.root, 'input'), bytes);
  assert.deepEqual(await fileIdentity({ path: join(temporary.root, 'input') }), { bytes: bytes.length, sha256: createHash('sha256').update(bytes).digest('hex') });
  await symlink('input', join(temporary.root, 'linked'));
  await mkdir(join(temporary.root, '.git'));
  await writeFile(join(temporary.root, '.git', 'metadata'), 'not source');
  assert.deepEqual(Object.keys(await fileIdentities({ root: temporary.root })), ['input', 'linked']);
  await assert.rejects(regularFileIdentities({ root: temporary.root }), /regular captured file/u);
  await assert.rejects(regularFile({ path: join(temporary.root, 'linked') }), /Linked/u);
  await symlink(temporary.root, join(temporary.root, 'parent'));
  await assert.rejects(directory({ path: join(temporary.root, 'parent', 'absent'), create: true }), /Linked/u);
});

test('JSON rejects duplicate decoded keys before the ordinary external decoder runs', () => {
  const row = { directory: '/source', file: 'fixture.cpp', arguments: ['clang++', '-c', 'fixture.cpp'] };
  assert.deepEqual(parseJson({ text: JSON.stringify([row]), decode: decodeCompilationDatabase }), [row]);
  for (const text of ['[{"directory":"/a","directory":"/b","file":"a.cpp"}]', '[{"directory":"/a","di\\u0072ectory":"/b","file":"a.cpp"}]', '[{"directory":"/a","file":"a.cpp",}]', '[/*comment*/]']) {
    assert.throws(() => parseJson({ text, decode: decodeCompilationDatabase }));
  }
  assert.throws(() => parseJson({ text: '[{"directory":5,"file":"a.cpp"}]', decode: decodeCompilationDatabase }));
  assert.deepEqual(parseJson({ text: JSON.stringify([{ ...row, externalMetadata: 'accepted' }]), decode: decodeCompilationDatabase }), [row]);
});

test('identity decoding accepts and preserves added observation metadata', () => {
  const identity = { path: 'index.mjs', bytes: 4, sha256: 'a'.repeat(64) };
  const manifest = decodeRootManifest({
    input: {
      formatVersion: 5,
      sourceCommit: 'b'.repeat(40),
      files: [{ ...identity, mode: 0o644 }],
      runtimes: {
        'llama-cpp': { manifest: 'runtimes/llama-cpp-browser-core/manifest.json', manifestFormatVersion: 2 },
        'stable-diffusion-cpp': { manifest: 'runtimes/stable-diffusion-cpp-browser-core/manifest.json', manifestFormatVersion: 2 },
      },
    },
  });
  assert.deepEqual(manifest.files, [{ ...identity, mode: 0o644 }]);
});
