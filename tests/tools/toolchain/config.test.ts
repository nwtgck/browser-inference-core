import assert from 'node:assert/strict';
import { resolve } from 'node:path';
import { test } from 'node:test';
import ts from 'typescript';
import { loadToolchainConfiguration } from '#tools/toolchain/config';
import config from '#source/toolchain.json' with { type: 'json' };
import { fileIdentity } from '#tools/shared/files';

test('toolchain values belong to the executing checkout and report the loaded compiler version', async () => {
  const checkoutRoot = process.cwd();
  const actual = await loadToolchainConfiguration({ checkoutRoot });
  assert.equal(actual.emscriptenRelease, config.emscriptenRelease);
  assert.equal(actual.typescriptVersion, ts.version);
  assert.deepEqual(actual.emscriptenTypeScriptPatch, { file: config.emscriptenTypeScriptPatch, ...await fileIdentity({ path: resolve(checkoutRoot, config.emscriptenTypeScriptPatch) }) });
  assert.deepEqual(actual.emscriptenAsyncifyBigIntPatch, { file: config.emscriptenAsyncifyBigIntPatch, ...await fileIdentity({ path: resolve(checkoutRoot, config.emscriptenAsyncifyBigIntPatch) }) });
  await assert.rejects(loadToolchainConfiguration({ checkoutRoot: resolve(checkoutRoot, 'other-checkout') }), /executing checkout/);
});
