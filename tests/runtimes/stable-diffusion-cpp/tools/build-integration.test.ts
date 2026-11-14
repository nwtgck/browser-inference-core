import assert from 'node:assert/strict';
import { chmod, mkdir, readFile, stat, symlink, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { test } from 'node:test';
import { decodeStableDiffusionProvenance } from '#dto/stable-diffusion-cpp';
import { buildRuntime } from '#runtimes/stable-diffusion-cpp/tools/build';
import { READ_ENV_PATH } from '#tools/shared/node-env';
import { temporaryDirectory } from '#tests/tools/shared/fixtures.test-support';
import { stubEnvironment } from '#tests/tools/shared/node-env.test-support';

const fixtures = path.resolve('tests/runtimes/stable-diffusion-cpp/tools/fixtures');

test('same-process build connects real private compiler inputs and patched sources to typed provenance', async () => {
  await using temporary = await temporaryDirectory();
  const checkoutRoot = process.cwd();
  const binaries = path.join(temporary.root, 'bin');
  await mkdir(binaries);
  // Import the checkout-owned fixture so package imports keep their original scope.
  const fixtureUrl = pathToFileURL(path.join(fixtures, 'cmake.mjs')).href;
  await writeFile(path.join(binaries, 'cmake.mjs'), `\
#!/usr/bin/env node
await import(${JSON.stringify(fixtureUrl)});
`);
  await chmod(path.join(binaries, 'cmake.mjs'), 0o755);
  await symlink('cmake.mjs', path.join(binaries, 'cmake'));
  const context = { checkoutRoot, toolingRoot: path.join(checkoutRoot, 'build/tooling'), outputRoot: path.join(temporary.root, 'work'), toolchainRoot: path.join(checkoutRoot, '.tools') };
  const selection = { profile: 'webgpu-wasm32-jspi', variant: 'browser', kind: 'native' as const, jobs: 2, fresh: false, includeToolchainNotices: false };
  const options = { context, selection, mode: { kind: 'local' } as const };
  const buildRoot = path.join(context.outputRoot, 'build', 'native', selection.profile, selection.variant);
  const stageRoot = path.join(context.outputRoot, 'stage', selection.profile, selection.variant);
  using environment = stubEnvironment({ values: { PATH: `${binaries}${path.delimiter}${READ_ENV_PATH() ?? ''}`, BIC_TEST_CHECKOUT: checkoutRoot } });
  const built = await buildRuntime(options);
  assert.equal(built.stagedRoot, undefined);
  const provenance = decodeStableDiffusionProvenance({ input: JSON.parse(await readFile(built.provenance, 'utf8')) });
  assert.equal(provenance.sourceDirty, true);
  assert.equal(provenance.sourceInputs.mode, 'local');
  assert.ok(Object.keys(provenance.compiledInputs.translationUnits).some((name) => name.startsWith('compile/')));
  assert.ok(!Object.keys(provenance.sourceInputs.files).some((name) => name.includes('node_modules/') || name === 'package.json' || name.endsWith('/tools/build.ts')));
  await mkdir(stageRoot, { recursive: true });
  await writeFile(path.join(stageRoot, 'keep'), 'previous candidate');
  {
    using failure = stubEnvironment({ values: { BIC_TEST_COMPILER_FAILURE: '1' } });
    await assert.rejects(buildRuntime(options), /Synthetic compiler failure/);
  }
  await assert.rejects(stat(path.join(buildRoot, 'provenance.json')), /ENOENT/);
  assert.equal(await readFile(path.join(stageRoot, 'keep'), 'utf8'), 'previous candidate');
  for (const mutation of ['prepared', 'snapshot']) {
    using changed = stubEnvironment({ values: { BIC_TEST_MUTATE: mutation } });
    await assert.rejects(buildRuntime(options), /Prepared image sources changed|Image-runtime compiler inputs changed/);
    await assert.rejects(stat(path.join(buildRoot, 'provenance.json')), /ENOENT/);
    assert.equal(await readFile(path.join(stageRoot, 'keep'), 'utf8'), 'previous candidate');
  }
  const sibling = path.join(context.outputRoot, 'build/native/webgpu-wasm64-jspi/browser');
  await mkdir(sibling, { recursive: true });
  await writeFile(path.join(sibling, 'keep'), 'sibling profile');
  await writeFile(path.join(buildRoot, 'stale'), 'old build');
  await buildRuntime({ ...options, selection: { ...selection, fresh: true } });
  await assert.rejects(stat(path.join(buildRoot, 'stale')), /ENOENT/);
  assert.equal(await readFile(path.join(sibling, 'keep'), 'utf8'), 'sibling profile');
  assert.equal(await readFile(path.join(stageRoot, 'keep'), 'utf8'), 'previous candidate');
});
