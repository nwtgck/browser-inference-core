import assert from 'node:assert/strict';
import { readFile, stat, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { test } from 'node:test';
import { prepareSources, resolveSources, verifyPreparedSources } from '#runtimes/speech-cpp/tools/prepare';
import { sourceFixture } from './source.test-support.js';

test('independent selected git objects produce a coherent patched tree and observed receipt', async () => {
  await using fixture = await sourceFixture();
  await writeFile(path.join(fixture.sources['speech'], 'a.txt'), 'hidden latest worktree input\n');
  const selected = await resolveSources({ declarations: fixture.declarations, source: fixture.source });
  assert.deepEqual(selected.upstreams, fixture.upstreams);
  const facts = await prepareSources({ ...selected, output: fixture.output, patchRoot: fixture.patchRoot, recipe: fixture.recipe, replace: false, selection: undefined });
  assert.equal(await readFile(path.join(fixture.output, 'speech/a.txt'), 'utf8'), 'one\nafter\n');
  assert.equal(await readFile(path.join(fixture.output, 'ggml/include/ggml.h'), 'utf8'), '// real selected header\n');
  assert.equal(await readFile(path.join(fixture.sources['speech'], 'a.txt'), 'utf8'), 'hidden latest worktree input\n');
  assert.equal(facts.receipt.patches.length, 1);
  assert.deepEqual(facts.receipt.upstreams, fixture.upstreams);
  await verifyPreparedSources({ preparedRoot: fixture.output, facts });
  await writeFile(path.join(fixture.output, 'ggml/include/ggml.h'), '// mutated prepared header\n');
  await assert.rejects(verifyPreparedSources({ preparedRoot: fixture.output, facts }), /Prepared speech sources changed/);
});

test('failed preparation retains previous output and identical regeneration retains timestamps', async () => {
  await using fixture = await sourceFixture();
  const options = { sources: fixture.sources, upstreams: fixture.upstreams, output: fixture.output, patchRoot: fixture.patchRoot, recipe: fixture.recipe, replace: true, selection: undefined };
  const facts = await prepareSources(options);
  const before = await stat(path.join(fixture.output, 'speech/a.txt'));
  await prepareSources(options);
  const after = await stat(path.join(fixture.output, 'speech/a.txt'));
  assert.equal(after.mtimeMs, before.mtimeMs);
  await writeFile(path.join(fixture.patchRoot, 'fix.patch'), 'invalid patch\n');
  await assert.rejects(prepareSources(options));
  await verifyPreparedSources({ preparedRoot: fixture.output, facts });
});

test('undeclared patch changes and aliased sources fail before replacing output', async () => {
  await using fixture = await sourceFixture();
  await assert.rejects(prepareSources({ sources: fixture.sources, upstreams: fixture.upstreams, output: fixture.output, patchRoot: fixture.patchRoot, recipe: [{ ...fixture.recipe[0]!, files: ['other.txt'] }], replace: false, selection: undefined }), /inventory|scope|declared/i);
  await assert.rejects(prepareSources({ sources: { ...fixture.sources, ggml: fixture.sources['speech'] }, upstreams: fixture.upstreams, output: fixture.output, patchRoot: fixture.patchRoot, recipe: fixture.recipe, replace: false, selection: undefined }), /alias/);
});
