import assert from 'node:assert/strict';
import { readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import { prepareSources, resolveSources, verifyPreparedSources } from '#runtimes/stable-diffusion-cpp/tools/prepare';
import { fileIdentity } from '#tools/shared/files';
import { sourceFixture, write } from './source.test-support.js';

test('independent selected git objects produce a coherent patched tree and observed receipt', async () => {
  await using fixture = await sourceFixture();
  await writeFile(path.join(fixture.sources['stable-diffusion'], 'a.txt'), 'hidden latest worktree input\n');
  const selected = await resolveSources({ declarations: fixture.declarations, source: fixture.source });
  assert.deepEqual(selected.upstreams, fixture.upstreams);
  const facts = await prepareSources({ ...selected, output: fixture.output, patchRoot: fixture.patchRoot, recipe: fixture.recipe, replace: false, selection: undefined });
  assert.equal(await readFile(path.join(fixture.output, 'stable-diffusion/a.txt'), 'utf8'), 'one\nafter\n');
  assert.equal(await readFile(path.join(fixture.output, 'ggml/ggml/include/ggml.h'), 'utf8'), '// real selected header\n');
  assert.equal(await readFile(path.join(fixture.sources['stable-diffusion'], 'a.txt'), 'utf8'), 'hidden latest worktree input\n');
  assert.equal(facts.receipt.patches.length, 1);
  assert.deepEqual(facts.receipt.upstreams, fixture.upstreams);
  await verifyPreparedSources({ preparedRoot: fixture.output, facts });
  await writeFile(path.join(fixture.output, 'ggml/ggml/include/ggml.h'), '// mutated prepared header\n');
  await assert.rejects(verifyPreparedSources({ preparedRoot: fixture.output, facts }), /Prepared image sources changed/);
});

test('failed preparation retains previous output and identical regeneration retains timestamps', async () => {
  await using fixture = await sourceFixture();
  const options = { sources: fixture.sources, upstreams: fixture.upstreams, output: fixture.output, patchRoot: fixture.patchRoot, recipe: fixture.recipe, replace: true, selection: undefined };
  const facts = await prepareSources(options);
  const before = await stat(path.join(fixture.output, 'stable-diffusion/a.txt'));
  await prepareSources(options);
  const after = await stat(path.join(fixture.output, 'stable-diffusion/a.txt'));
  assert.equal(after.mtimeMs, before.mtimeMs);
  await writeFile(path.join(fixture.patchRoot, 'fix.patch'), 'invalid patch\n');
  await assert.rejects(prepareSources(options));
  await verifyPreparedSources({ preparedRoot: fixture.output, facts });
});

test('a newly added file records absence and actual output identity without touching upstream', async () => {
  await using fixture = await sourceFixture();
  await write({
    root: fixture.patchRoot,
    name: 'new.patch',
    text: `\
diff --git a/src/core/fixture.h b/src/core/fixture.h
new file mode 100644
--- /dev/null
+++ b/src/core/fixture.h
@@ -0,0 +1 @@
+fixture
`,
  });
  const facts = await prepareSources({ sources: fixture.sources, upstreams: fixture.upstreams, output: fixture.output, patchRoot: fixture.patchRoot, recipe: [...fixture.recipe, { file: 'new.patch', target: 'stable-diffusion', reason: 'New fixture header', files: ['src/core/fixture.h'] }], replace: false, selection: undefined });
  const output = path.join(fixture.output, 'stable-diffusion/src/core/fixture.h');
  assert.equal(await readFile(output, 'utf8'), 'fixture\n');
  const record = facts.receipt.patches[1]?.files[0];
  assert.ok(record);
  assert.equal(record.afterSha256, (await fileIdentity({ path: output })).sha256);
  assert.equal(record.beforeSha256, undefined);
  await assert.rejects(stat(path.join(fixture.sources['stable-diffusion'], 'src/core/fixture.h')), /ENOENT/);
});

test('undeclared patch changes and aliased sources fail before replacing output', async () => {
  await using fixture = await sourceFixture();
  await assert.rejects(prepareSources({ sources: fixture.sources, upstreams: fixture.upstreams, output: fixture.output, patchRoot: fixture.patchRoot, recipe: [{ ...fixture.recipe[0]!, files: ['other.txt'] }], replace: false, selection: undefined }), /inventory|scope|declared/i);
  await assert.rejects(prepareSources({ sources: { ...fixture.sources, ggml: fixture.sources['stable-diffusion'] }, upstreams: fixture.upstreams, output: fixture.output, patchRoot: fixture.patchRoot, recipe: fixture.recipe, replace: false, selection: undefined }), /alias/);
});
