import assert from 'node:assert/strict';
import { mkdir, readFile, symlink, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import test from 'node:test';
import { applyPatch, applyRecipe, type Patch } from '#tools/source/patches';
import { git, initializeRepository, temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { stubEnvironment } from '#tests/tools/shared/node-env.test-support';

const patchText = `\
diff --git a/a.txt b/a.txt
--- a/a.txt
+++ b/a.txt
@@ -1,2 +1,2 @@
 one
-before
+after
`;
const declaration: Patch = { file: 'fix.patch', target: 'upstream', reason: 'Exercise a source change', files: ['a.txt'] };

test('transparent patches observe actual bytes, preserve declared order and ignore parent Git configuration', async () => {
  await using temporary = await temporaryDirectory();
  await initializeRepository({ root: temporary.root });
  await writeFile(join(temporary.root, 'a.txt'), 'parent must not change');
  await git({ root: temporary.root, args: ['add', 'a.txt'] });
  const index = await readFile(join(temporary.root, '.git', 'index'));
  const sourceRoot = join(temporary.root, 'private');
  const patchRoot = join(temporary.root, 'patches');
  await mkdir(sourceRoot);
  await mkdir(patchRoot);
  for (const name of ['a.txt', 'b.txt']) {
    await writeFile(join(sourceRoot, name), 'one\nbefore\n');
  }
  await writeFile(join(patchRoot, 'fix.patch'), patchText + patchText.replaceAll('a.txt', 'b.txt'));
  using environment = stubEnvironment({ values: { GIT_DIR: join(temporary.root, '.git') } });
  const receipt = await applyPatch({ sourceRoot, patchRoot, patch: { ...declaration, files: ['b.txt', 'a.txt'] } });
  assert.deepEqual(receipt.files.map((file) => file.path), ['b.txt', 'a.txt']);
  assert.notEqual(receipt.files[0]?.beforeSha256, receipt.files[0]?.afterSha256);
  assert.equal(await readFile(join(sourceRoot, 'a.txt'), 'utf8'), 'one\nafter\n');
  assert.equal(await readFile(join(temporary.root, 'a.txt'), 'utf8'), 'parent must not change');
  assert.deepEqual(await readFile(join(temporary.root, '.git', 'index')), index);
  await applyPatch({ sourceRoot, patchRoot, patch: { ...declaration, files: ['b.txt', 'a.txt'], mode: 'reverse' } });
  assert.equal(await readFile(join(sourceRoot, 'a.txt'), 'utf8'), 'one\nbefore\n');
});

test('scope mismatch, linked inputs and context errors fail without changing the source', async () => {
  await using temporary = await temporaryDirectory();
  const sourceRoot = join(temporary.root, 'private');
  const patchRoot = join(temporary.root, 'patches');
  await mkdir(sourceRoot);
  await mkdir(patchRoot);
  for (const name of ['a.txt', 'b.txt']) {
    await writeFile(join(sourceRoot, name), 'one\nbefore\n');
  }
  await writeFile(join(patchRoot, 'fix.patch'), patchText + patchText.replaceAll('a.txt', 'b.txt'));
  await assert.rejects(applyPatch({ sourceRoot, patchRoot, patch: declaration }), /outside its inventory/u);
  assert.equal(await readFile(join(sourceRoot, 'a.txt'), 'utf8'), 'one\nbefore\n');
  await writeFile(join(patchRoot, 'fix.patch'), patchText);
  await writeFile(join(sourceRoot, 'a.txt'), 'one\nchanged contract\n');
  await assert.rejects(applyPatch({ sourceRoot, patchRoot, patch: declaration }));
  await symlink(sourceRoot, join(sourceRoot, 'alias'));
  await writeFile(join(patchRoot, 'fix.patch'), patchText.replaceAll('a.txt', 'alias/b.txt'));
  await assert.rejects(applyPatch({ sourceRoot, patchRoot, patch: { ...declaration, files: ['alias/b.txt'] } }), /Linked/u);
  await writeFile(join(patchRoot, 'fix.patch'), `diff --git a/link b/link\nnew file mode 120000\n--- /dev/null\n+++ b/link\n@@ -0,0 +1 @@\n+outside\n`);
  await assert.rejects(applyPatch({ sourceRoot, patchRoot, patch: { ...declaration, files: ['link'] } }), /Unsupported/u);
});

test('typed recipes reject unknown targets and duplicate entries before applying anything', async () => {
  await using temporary = await temporaryDirectory();
  await assert.rejects(applyRecipe({ sources: {}, patchRoot: temporary.root, patches: [declaration] }), /Unknown patch target/u);
  await assert.rejects(applyRecipe({ sources: { upstream: temporary.root }, patchRoot: temporary.root, patches: [declaration, declaration] }), /Duplicate patch declaration/u);
});

test('added and deleted files appear explicitly as missing sides of observed identities', async () => {
  await using temporary = await temporaryDirectory();
  const sourceRoot = join(temporary.root, 'source');
  const patchRoot = join(temporary.root, 'patches');
  await mkdir(sourceRoot);
  await mkdir(patchRoot);
  const added = `\
diff --git a/new.h b/new.h
new file mode 100644
--- /dev/null
+++ b/new.h
@@ -0,0 +1 @@
+int value;
`;
  await writeFile(join(patchRoot, 'fix.patch'), added);
  const receipt = await applyPatch({ sourceRoot, patchRoot, patch: { ...declaration, files: ['new.h'] } });
  assert.equal(receipt.files[0]?.beforeSha256, undefined);
  assert.equal(typeof receipt.files[0]?.afterSha256, 'string');
  const removed = await applyPatch({ sourceRoot, patchRoot, patch: { ...declaration, files: ['new.h'], mode: 'reverse' } });
  assert.equal(removed.files[0]?.afterSha256, undefined);
  assert.equal(typeof removed.files[0]?.beforeSha256, 'string');
});
