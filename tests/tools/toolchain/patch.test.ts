import assert from 'node:assert/strict';
import filesystem from 'node:fs/promises';
import { mkdir, readFile, readdir, stat, symlink, writeFile } from 'node:fs/promises';
import { syncBuiltinESMExports } from 'node:module';
import { join } from 'node:path';
import test from 'node:test';
import { fileIdentity } from '#tools/shared/files';
import { applySdkPatch, verifySdkPatch } from '#tools/toolchain/patch';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';

const patchText = `\
--- a/tools/emscripten.py
+++ b/tools/emscripten.py
@@ -1 +1 @@
-generator = "raw"
+generator = "converted"
`;

test('SDK declaration patch uses context, preserves unrelated edits and is idempotent', async () => {
  await using temporary = await temporaryDirectory();
  const checkoutRoot = join(temporary.root, 'checkout');
  const emscriptenRoot = join(temporary.root, 'sdk');
  await mkdir(join(checkoutRoot, 'patches'), { recursive: true });
  await mkdir(join(emscriptenRoot, 'tools'), { recursive: true });
  const patchFile = join(checkoutRoot, 'patches/tsd.patch');
  const target = join(emscriptenRoot, 'tools/emscripten.py');
  await writeFile(patchFile, patchText);
  const patch = { file: 'patches/tsd.patch', ...await fileIdentity({ path: patchFile }) };
  const original = 'generator = "raw"\n# unrelated SDK edit\n';
  await writeFile(target, original);
  await assert.rejects(verifySdkPatch({ source: 'tools/emscripten.py', checkoutRoot, emscriptenRoot, patch }), /not applied/u);
  assert.equal(await readFile(target, 'utf8'), original);
  await applySdkPatch({ source: 'tools/emscripten.py', checkoutRoot, emscriptenRoot, patch });
  assert.equal(await readFile(target, 'utf8'), 'generator = "converted"\n# unrelated SDK edit\n');
  const before = await stat(target);
  await applySdkPatch({ source: 'tools/emscripten.py', checkoutRoot, emscriptenRoot, patch });
  assert.equal((await stat(target)).mtimeMs, before.mtimeMs);
  await verifySdkPatch({ source: 'tools/emscripten.py', checkoutRoot, emscriptenRoot, patch });
  await writeFile(target, 'generator = "other"\n');
  await assert.rejects(applySdkPatch({ source: 'tools/emscripten.py', checkoutRoot, emscriptenRoot, patch }), /context/u);
  assert.equal(await readFile(target, 'utf8'), 'generator = "other"\n');
  assert.deepEqual(await readdir(join(emscriptenRoot, 'tools')), ['emscripten.py']);
  await writeFile(target, 'generator = "raw"\ngenerator = "converted"\n');
  await assert.rejects(applySdkPatch({ source: 'tools/emscripten.py', checkoutRoot, emscriptenRoot, patch }), /ambiguous/u);
  await writeFile(patchFile, patchText + '# changed recipe\n');
  await assert.rejects(applySdkPatch({ source: 'tools/emscripten.py', checkoutRoot, emscriptenRoot, patch }), /patch changed/u);
  const linked = join(temporary.root, 'linked-sdk');
  await symlink(emscriptenRoot, linked);
  await assert.rejects(applySdkPatch({ source: 'tools/emscripten.py', checkoutRoot, emscriptenRoot: linked, patch }), /Linked/u);
});

test('a failed atomic install preserves the SDK generator and removes staging files', async context => {
  await using temporary = await temporaryDirectory();
  const checkoutRoot = join(temporary.root, 'checkout');
  const emscriptenRoot = join(temporary.root, 'sdk');
  await mkdir(join(checkoutRoot, 'patches'), { recursive: true });
  await mkdir(join(emscriptenRoot, 'tools'), { recursive: true });
  const patchFile = join(checkoutRoot, 'patches/tsd.patch');
  const target = join(emscriptenRoot, 'tools/emscripten.py');
  await writeFile(patchFile, patchText);
  await writeFile(target, 'generator = "raw"\n');
  const patch = { file: 'patches/tsd.patch', ...await fileIdentity({ path: patchFile }) };
  const rename = filesystem.rename;
  const failure = new Error('Install failed');
  const intercepted = context.mock.method(filesystem, 'rename', async (...args: Parameters<typeof rename>) => {
    if (args[1].toString() === target) {
      throw failure;
    }
    return rename(...args);
  });
  syncBuiltinESMExports();
  try {
    await assert.rejects(applySdkPatch({ source: 'tools/emscripten.py', checkoutRoot, emscriptenRoot, patch }), error => error === failure);
    assert.equal(await readFile(target, 'utf8'), 'generator = "raw"\n');
    assert.deepEqual(await readdir(join(emscriptenRoot, 'tools')), ['emscripten.py']);
  } finally {
    intercepted.mock.restore();
    syncBuiltinESMExports();
  }
});
